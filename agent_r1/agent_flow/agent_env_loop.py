import json
import logging
import os
from typing import Any
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import (
    AgentFlowBase,
    AgentFlowOutput,
    AgentFlowStep,
    register,
)
from agent_r1.env import AgentEnv
from agent_r1.env.base import Action, Observation
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("agent_env_loop")
class AgentEnvLoop(AgentFlowBase):
    """Generic agent-environment interaction loop.

    Each turn of the loop produces an independent ``AgentFlowStep`` whose
    ``prompt_ids`` come from the environment observation and whose
    ``response_ids`` come from the LLM generation, following the same pattern
    as ``SingleStepSingleTurnAgentFlow``.

    Environment creation is driven by ``env_kwargs``:

    - **Global defaults** come from this flow's constructor ``**kwargs``
      (typically provided via a separate YAML loaded from
      ``config.actor_rollout_ref.rollout.agent.agent_flow_config_path``).
    - **Per-sample overrides** come from the ``env_kwargs`` field in the
      dataset row (passed through ``**kwargs``).
    - The two dicts are merged (per-sample wins), and ``env_type`` is popped
      to look up the ``AgentEnv`` subclass via its registry.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.max_steps: int = self.config.actor_rollout_ref.rollout.agent.get("max_steps", 10)
        self.skip_special_tokens: bool = self.config.actor_rollout_ref.rollout.agent.get("skip_special_tokens", True)
        self.env_kwargs: dict[str, Any] = kwargs

    def _create_env(self, **kwargs) -> AgentEnv:
        """Create an environment instance for a single trajectory.

        Merges global ``env_kwargs`` from config with per-sample ``env_kwargs``
        from the dataset, then delegates to ``AgentEnv.from_config``.

        Args:
            **kwargs: Dataset fields from ``verl.utils.dataset.RLHFDataset``.

        Returns:
            AgentEnv: A fresh environment instance.
        """
        env_kwargs = kwargs.get("env_kwargs", {})
        if isinstance(env_kwargs, str):
            env_kwargs = json.loads(env_kwargs)

        merged = {**self.env_kwargs, **env_kwargs}

        env_type = merged.pop("env_type")
        return AgentEnv.from_config(env_type, **merged)

    async def _obs_to_prompt(self, obs: Observation, tools: list[dict] | None = None) -> tuple[list[int], dict]:
        """Convert an observation to prompt token ids and multi-modal data.

        Handles all three observation formats:

        - ``token_ids``: used directly.
        - ``messages``: processed through ``apply_chat_template``.
        - ``text``: tokenised with the raw tokenizer.

        Args:
            obs (Observation): The environment observation.
            tools (list[dict] | None): Optional tool schemas passed to the
                chat template.

        Returns:
            prompt_ids (list[int]): Token ids for the LLM prompt.
        """
        if obs.token_ids is not None:
            return obs.token_ids
        if obs.messages is not None:
            prompt_ids = await self.apply_chat_template(
                obs.messages,
                tools=tools,
            )
            return prompt_ids
        if obs.text is not None:
            prompt_ids = self.tokenizer.encode(obs.text)
            return prompt_ids
        raise ValueError("Observation must have at least one field set")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        """Run the agent-environment interaction loop.

        Args:
            sampling_params (dict[str, Any]): LLM sampling parameters.
            **kwargs: Dataset fields from ``verl.utils.dataset.RLHFDataset``.

        Returns:
            AgentFlowOutput: Output containing one ``AgentFlowStep`` per turn.
        """
        env = self._create_env(**kwargs)
        obs = env.reset(**kwargs)
        tools = getattr(env, "tool_schemas", None)

        steps: list = []
        metrics = {}

        for step_idx in range(self.max_steps):
            prompt_ids = await self._obs_to_prompt(obs, tools=tools)

            if len(prompt_ids) > self.prompt_length:
                logger.warning(
                    "Prompt length (%d) exceeds configured prompt_length (%d) at step %d. "
                    "Stopping rollout early. Consider increasing "
                    "actor_rollout_ref.rollout.prompt_length in your config.",
                    len(prompt_ids),
                    self.prompt_length,
                    step_idx,
                )
                break

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )

            response_ids = output.token_ids[: self.response_length]

            response_text = await self.loop.run_in_executor(
                None,
                lambda _ids=response_ids: self.tokenizer.decode(_ids, skip_special_tokens=self.skip_special_tokens),
            )

            action = Action(text=response_text, token_ids=response_ids)

            # TODO: rename tool_calls to env_step
            with simple_timer("tool_calls", metrics):
                next_obs, reward, done, info = await env.step(action)

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=(output.log_probs[: self.response_length] if output.log_probs else None),
                routed_experts=(
                    output.routed_experts[: len(prompt_ids) + self.response_length]
                    if output.routed_experts is not None
                    else None
                ),
                reward_score=reward,
            )
            step = await self._postprocess(step, **kwargs)
            steps.append(step)

            if done:
                break
            obs = next_obs

        return AgentFlowOutput(steps=steps, metrics=metrics)
