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
from recipes.webshop.env.client import WebShopEnvClient
from recipes.webshop.prompts import WEBSHOP_TOOL_SCHEMAS
from recipes.webshop.utils import build_invalid_tool_call_observation, build_webshop_messages
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
        recovered.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
    return recovered


def _metadata_str(value: Any) -> str:
    return "" if value is None else str(value)


@register("webshop_agent")
class WebShopAgentFlow(AgentFlowBase):
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
        self.max_steps = int(kwargs.get("max_steps", 15))
        self.max_parallel_calls = 1
        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format,
            self.tokenizer,
        )
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = WEBSHOP_TOOL_SCHEMAS
        self.client = WebShopEnvClient(timeout=float(kwargs.get("env_timeout", 30.0)))
        self.invalid_tool_call_penalty = float(kwargs.get("invalid_tool_call_penalty", 0.1))
        self.success_reward = float(kwargs.get("success_reward", 10.0))
        self.steps: list[AgentFlowStep] = []

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        extra_info = kwargs.get("extra_info") or {}
        raw_prompt = list(kwargs.get("raw_prompt") or kwargs.get("prompt") or [])
        instruction = str(extra_info.get("instruction") or (raw_prompt[0]["content"] if raw_prompt else "")).strip()
        goal_index = int(extra_info.get("goal_index"))
        split = extra_info.get("split", "train")
        asin = extra_info.get("asin")

        reset_payload = await self.client.reset(goal_index)
        current_observation = str(reset_payload["observation"])
        env_state = reset_payload["env_state"]
        available_actions = (reset_payload.get("info") or {}).get("available_actions") or []
        recent_history: list[dict[str, str]] = []
        self.steps = []

        metrics: dict[str, Any] = {}
        done = False
        final_reward = 0.0
        final_task_score = 0.0
        final_info: dict[str, Any] = {}
        num_steps = 0
        end_status = {"terminated": False, "truncated": True, "termination_reason": "max_steps"}

        while num_steps < self.max_steps and not done:
            num_steps += 1
            observation_before_action = current_observation
            messages = build_webshop_messages(
                instruction=instruction,
                observation=current_observation,
                recent_history=recent_history,
                available_actions=available_actions,
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

            command = ""
            invalid_reason: str | None = None
            if not tool_calls:
                invalid_reason = "missing env_step tool call"
            elif len(tool_calls) > 1:
                invalid_reason = f"expected exactly one env_step tool call, got {len(tool_calls)}"
            else:
                tool_call = tool_calls[0]
                if tool_call.name != "env_step":
                    invalid_reason = f"expected env_step tool, got {tool_call.name!r}"
                else:
                    try:
                        command = str(json.loads(tool_call.arguments).get("command", "")).strip()
                    except Exception as exc:
                        logger.warning("Failed to parse env_step arguments: %r", exc)
                        invalid_reason = f"failed to parse env_step arguments: {exc}"
                    if not command and invalid_reason is None:
                        invalid_reason = "missing command argument"

            env_reward = 0.0
            step_reward = 0.0
            invalid_tool_call = False
            success = False
            step_info: dict[str, Any] = {}
            if command:
                try:
                    result = await self.client.step(goal_index, env_state, command)
                    current_observation = str(result["observation"])
                    env_state = result["env_state"]
                    env_reward = float(result["reward"])
                    done = bool(result["done"])
                    step_info = result.get("info") or {}
                    if done:
                        env_truncated = bool(step_info.get("truncated", step_info.get("TimeLimit.truncated", False)))
                        end_status = {
                            "terminated": not env_truncated,
                            "truncated": env_truncated,
                            "termination_reason": "env_truncated" if env_truncated else "env_done",
                        }
                    success = bool(step_info.get("success", env_reward >= 0.999))
                    step_reward = self.success_reward if done and success else 0.0
                    available_actions = step_info.get("available_actions", available_actions)
                    recent_history.append({"observation": observation_before_action, "action": command})
                    if done:
                        final_reward = step_reward
                        final_task_score = float(step_info.get("task_score", env_reward))
                        final_info = step_info
                except Exception as exc:
                    logger.warning("WebShop env step failed: %r", exc)
                    step_info = {"error": str(exc)}
            else:
                reason = invalid_reason or "missing command argument"
                current_observation = build_invalid_tool_call_observation(current_observation, reason)
                invalid_tool_call = True
                step_reward = -self.invalid_tool_call_penalty
                recent_history.append(
                    {"observation": observation_before_action, "action": f"INVALID_TOOL_CALL: {reason}"}
                )
                step_info = {"error": reason, "available_actions": available_actions}

            reward_extra_info = {
                "step_env_reward": env_reward,
                "step_reward": step_reward,
                "final_reward": final_reward if done else 0.0,
                "task_score": final_task_score if done else float(step_info.get("task_score", 0.0) or 0.0),
                "success": success or bool(final_info.get("success", False)),
                "success_reward": self.success_reward if success else 0.0,
                "invalid_tool_call": invalid_tool_call,
                "invalid_tool_call_penalty": self.invalid_tool_call_penalty if invalid_tool_call else 0.0,
                "num_steps": num_steps,
                "goal_index": goal_index,
                "split": _metadata_str(split),
                "asin": _metadata_str(asin),
                "selected_asin": _metadata_str(step_info.get("selected_asin", final_info.get("selected_asin"))),
            }
            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=step_reward,
                extra_fields={
                    "anchor_obs": observation_before_action,
                    "reward_extra_info": reward_extra_info,
                },
            )
            step.extra_fields.update(generation_info)
            step = await self._postprocess(step, **kwargs)
            self.steps.append(step)

            if done:
                break

        return AgentFlowOutput(steps=self.steps, metrics=metrics, **end_status)
