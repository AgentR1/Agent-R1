from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.reward_loop.reward_loop import RewardLoopWorker
from recipes.alfworld.env.tool_executor import INVALID_TOOL_CALL_ACTION, AlfworldToolExecutor
from recipes.alfworld.prompts import ALFWORLD_TOOL_SCHEMAS
from recipes.alfworld.utils import (
    build_alfworld_messages,
    build_invalid_tool_call_observation,
    extract_task_text,
)
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _recover_tool_calls_from_text(text: str) -> list[FunctionCall]:
    recovered: list[FunctionCall] = []
    for raw in _TOOL_CALL_BLOCK.findall(text):
        try:
            payload = json.loads(raw.strip())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str):
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                continue
        if not isinstance(arguments, dict):
            continue
        recovered.append(
            FunctionCall(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
        )
    return recovered


@register("alfworld_agent")
class AlfworldAgentFlow(AgentFlowBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls,
        dataset_config,
        **kwargs,
    ):
        super().__init__(
            trainer_config,
            server_manager,
            reward_loop_worker,
            tokenizer,
            processor,
            dataset_cls,
            dataset_config,
            **kwargs,
        )
        self.max_steps = kwargs.get("max_steps", 20)
        self.max_parallel_calls = 1
        self.max_episode_steps = kwargs.get("max_episode_steps", 50)

        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format,
            self.tokenizer,
        )
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = ALFWORLD_TOOL_SCHEMAS

        self.executor = AlfworldToolExecutor(max_episode_steps=self.max_episode_steps)
        self.current_observation: str = ""
        self.current_admissible_commands: list[str] = []
        self.history_actions: list[str] = []
        self.steps: list[AgentFlowStep] = []

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        extra_info = kwargs.get("extra_info") or {}
        task_id = extra_info.get("task_id")
        split = extra_info.get("split", "train")
        task_type_raw = extra_info.get("task_type_raw")
        task_family = extra_info.get("task_family")
        game_relative_path = extra_info.get("game_relative_path")
        if not game_relative_path:
            raise ValueError("ALFWorld sample is missing extra_info.game_relative_path")

        self.current_observation, reset_info = self.executor.reset_with_info(
            game_relative_path=game_relative_path,
            task_id=task_id,
        )
        admissible_commands = reset_info.get("admissible_commands")
        self.current_admissible_commands = admissible_commands if isinstance(admissible_commands, list) else []
        task_text = extract_task_text(self.current_observation, extra_info.get("goal_text"))
        self.history_actions = []
        self.steps = []

        metrics: dict[str, Any] = {}
        num_steps = 0
        done = False
        final_success_flag: bool | None = None
        dense_reward_sum = 0.0
        end_status = {"terminated": False, "truncated": True, "termination_reason": "max_steps"}

        def build_reward_extra_info(step_env_reward: float = 0.0) -> dict[str, Any]:
            return {
                "score": 0.0,
                "step_env_reward": float(step_env_reward),
                "dense_reward_sum": float(dense_reward_sum),
                "success": bool(final_success_flag),
                "num_steps": int(num_steps),
                "task_id": str(task_id or ""),
                "split": str(split or ""),
                "task_type_raw": str(task_type_raw or ""),
                "task_family": str(task_family or ""),
                "is_action_valid": False,
            }

        while num_steps < self.max_steps and not done:
            num_steps += 1
            observation_before_action = self.current_observation

            messages = build_alfworld_messages(
                task_text=task_text,
                observation=observation_before_action,
                history_actions=self.history_actions,
                admissible_commands=self.current_admissible_commands,
            )

            prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas)

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )

            response_ids = output.token_ids[: self.response_length]
            generation_info = self._generation_metadata(output)
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                tool_calls = _recover_tool_calls_from_text(response_text)

            env_reward = 0.0
            is_action_valid = False
            invalid_reason: str | None = None

            if not tool_calls:
                invalid_reason = "missing env_step tool call"
            else:
                tool_call = tool_calls[0]
                if tool_call.name != "env_step":
                    invalid_reason = f"expected env_step tool, got {tool_call.name!r}"
                else:
                    command = ""
                    try:
                        tool_args = json.loads(tool_call.arguments)
                        command = str(tool_args.get("command", "")).strip()
                    except Exception as e:
                        logger.warning("Failed to parse env_step arguments: %s", e)
                        invalid_reason = f"failed to parse env_step arguments: {e}"

                    if not command and invalid_reason is None:
                        invalid_reason = "missing command argument"

                    if command:
                        is_action_valid = command in self.current_admissible_commands
                        result = self.executor.step(command)
                        self.current_observation = result["observation"]
                        env_reward = float(result["reward"])
                        done = bool(result["done"])
                        info = result.get("info", {}) or {}
                        if done:
                            env_truncated = bool(info.get("truncated", info.get("TimeLimit.truncated", False)))
                            end_status = {
                                "terminated": not env_truncated,
                                "truncated": env_truncated,
                                "termination_reason": "env_truncated" if env_truncated else "env_done",
                            }
                        admissible_commands = info.get("admissible_commands")
                        self.current_admissible_commands = (
                            admissible_commands if isinstance(admissible_commands, list) else []
                        )
                        self.history_actions = result.get("history_actions", self.history_actions)
                        if "success" in info:
                            final_success_flag = bool(info["success"])
                        elif "won" in info:
                            final_success_flag = bool(info["won"])
                        dense_reward_sum += env_reward

            if invalid_reason is not None:
                self.current_observation = build_invalid_tool_call_observation(
                    self.current_observation,
                    invalid_reason,
                )
                self.history_actions.append(f"{INVALID_TOOL_CALL_ACTION}: {invalid_reason}")

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=0.0,
                extra_fields={
                    "anchor_obs": observation_before_action,
                    "reward_extra_info": {
                        **build_reward_extra_info(env_reward),
                        "is_action_valid": bool(is_action_valid),
                    },
                },
            )
            step.extra_fields.update(generation_info)
            step = await self._postprocess(step, **kwargs)
            self.steps.append(step)

            if done:
                final_step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=None,
                    extra_fields={
                        "anchor_obs": observation_before_action,
                        "reward_extra_info": build_reward_extra_info(),
                    },
                )
                final_step.extra_fields.update(generation_info)
                final_step = await self._postprocess(final_step, **kwargs)
                self.steps.append(final_step)
                break

        return AgentFlowOutput(steps=self.steps, metrics=metrics, **end_status)
