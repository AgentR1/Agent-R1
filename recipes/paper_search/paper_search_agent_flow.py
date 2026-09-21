import logging
import os
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.agent_flow.rollout_utils import terminal_status
from agent_r1.reward_loop.reward_loop import RewardLoopWorker
from recipes.paper_search.env.paper_client import PaperSearchClient, SelectorClient
from recipes.paper_search.prompts import PAPERSEARCH_TOOL_SCHEMAS
from recipes.paper_search.runtime import PaperSearchRuntime, PaperSearchRuntimeConfig
from recipes.paper_search.utils import recover_tool_calls_from_text
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("paper_search_agent")
class PaperSearchAgentFlow(AgentFlowBase):
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
        self.max_steps = kwargs.get("max_steps", 5)
        self.max_parallel_calls = kwargs.get("max_parallel_calls", 5)
        runtime_config = PaperSearchRuntimeConfig(
            max_steps=self.max_steps,
            max_parallel_calls=self.max_parallel_calls,
            reward_top_k=kwargs.get("reward_top_k", 3),
            score_threshold=kwargs.get("score_threshold", 0.4),
            search_cost=kwargs.get("search_cost", 0),
            expand_cost=kwargs.get("expand_cost", 0),
            use_discrete_reward=kwargs.get("use_discrete_reward", False),
            search_top_k=kwargs.get("search_top_k", 10),
            citations_limit=kwargs.get("citations_limit", 30),
            references_limit=kwargs.get("references_limit", -1),
        )

        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        )
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = PAPERSEARCH_TOOL_SCHEMAS
        self.runtime = PaperSearchRuntime(
            config=runtime_config,
            paper_client=PaperSearchClient(timeout=30.0),
            selector_client=SelectorClient(timeout=30.0),
            logger_=logger,
        )
        self.steps: list[AgentFlowStep] = []

    def _make_anchor_obs(self) -> str:
        return self.runtime.make_user_prompt()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        self.runtime.reset(raw_prompt[0]["content"])
        self.steps = []

        metrics: dict[str, Any] = {}
        total_search_action_count = 0
        total_expand_action_count = 0
        num_steps = 0
        end_status = {"terminated": False, "truncated": True, "termination_reason": "max_steps"}

        while num_steps < self.max_steps:
            num_steps += 1
            anchor_obs = self._make_anchor_obs()

            prompt_ids = await self.apply_chat_template(self.runtime.make_messages(), tools=self.tool_schemas)

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )

            response_ids = output.token_ids[: self.response_length]
            generation_info = self._generation_metadata(output)
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            if not tool_calls:
                tool_calls = recover_tool_calls_from_text(response_text)

            if not tool_calls:
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=0.0,
                    extra_fields={
                        "anchor_obs": anchor_obs,
                        "reward_extra_info": {
                            "search_actions_total": total_search_action_count,
                            "expand_actions_total": total_expand_action_count,
                        },
                    },
                )
                step.extra_fields.update(generation_info)
                step = await self._postprocess(step, **kwargs)
                self.steps.append(step)
                end_status = terminal_status(generation_info, reason="no_tool_calls")
                break

            with simple_timer("tool_calls", metrics):
                step_reward_score, _, counters = await self.runtime.execute_tool_calls(tool_calls)

            total_search_action_count += counters["search"]
            total_expand_action_count += counters["expand"]
            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=step_reward_score,
                extra_fields={
                    "anchor_obs": anchor_obs,
                    "reward_extra_info": {
                        "search_actions_total": total_search_action_count,
                        "expand_actions_total": total_expand_action_count,
                    },
                },
            )
            step.extra_fields.update(generation_info)
            step = await self._postprocess(step, **kwargs)
            self.steps.append(step)

        return AgentFlowOutput(steps=self.steps, metrics=metrics, **end_status)
