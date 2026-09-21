"""
HotpotQA AgentFlow — multi-step search agent for multi-hop QA.

Architecture follows recipes/paper_search style:
- Each step re-builds messages from current state (not multi-turn message accumulation)
- Passages and action history are maintained as structured state, rendered into prompt each step
- Prompt length is bounded: passages are truncated to fit within budget
- Tool / answer format: `tools=` in apply_chat_template **plus** user prompt aligned with
  `recipes/paper_search/prompts.py` (`<analysis>`, `<tool_call>`, final `<answer>`); see
  `recipes.hotpotqa.prompts.HOTPOTQA_USER_PROMPT`.
- Search tool backed by local FAISS + BGE (HotpotQASearchToolLegacy)
- Reward: tool steps get reward_score=0.0; final step gets reward_score=None (→ custom EM reward)
"""

import ast
import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.agent_flow.rollout_utils import terminal_status
from agent_r1.reward_loop.reward_loop import RewardLoopWorker
from recipes.hotpotqa.env.search_tool import (
    DEFAULT_HOTPOTQA_EMBEDDING_MODEL,
    HotpotQASearchToolLegacy,
    parse_legacy_tool_result,
    resolve_hotpotqa_embedding_devices,
)
from recipes.hotpotqa.prompts import (
    HOTPOTQA_SYSTEM_PROMPT,
    HOTPOTQA_TOOL_SCHEMAS,
    HOTPOTQA_USER_PROMPT,
)
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_RETRIEVAL_TOOL_NAMES = frozenset({"search", "wiki_search"})
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _json_or_python_dict(s: str) -> Any:
    """Parse JSON; on failure try Python literal dict (common small-model mistake)."""
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    if len(s) > 8192 or "__" in s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _normalize_tool_call_dict(obj: Any) -> dict[str, Any] | None:
    """Return dict with str name and dict arguments, or None."""
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    name = obj["name"]
    args = obj.get("arguments")
    if args is None:
        return None
    if isinstance(args, str):
        inner = _json_or_python_dict(args)
        if not isinstance(inner, dict):
            return None
        args = inner
    if not isinstance(args, dict):
        return None
    return {"name": str(name), "arguments": args}


def _recover_tool_calls_from_text(text: str) -> list[FunctionCall]:
    """
    Fallback when Hermes json.loads fails (e.g. single-quoted dicts, trailing commas).
    Agent-R1 NousToolEnv still fails JSON but feeds 'Error: JSONDecodeError' into the next
    user turn; here we try to salvage calls and additionally add text feedback (see run loop).
    """
    out: list[FunctionCall] = []
    for raw in _TOOL_CALL_BLOCK.findall(text):
        obj = _json_or_python_dict(raw.strip())
        if obj is None:
            continue
        norm = _normalize_tool_call_dict(obj)
        if norm is None:
            continue
        try:
            out.append(
                FunctionCall(
                    name=norm["name"],
                    arguments=json.dumps(norm["arguments"], ensure_ascii=False),
                )
            )
        except Exception:
            continue
    return out


def _decode_tool_arguments(arguments: str) -> dict[str, Any] | None:
    try:
        obj: Any = json.loads(arguments)
    except Exception:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def _format_passage_list(passages: list[tuple[str, str]], max_chars: int = 0) -> str:
    """Format accumulated passages for prompt. Each entry is (query, text)."""
    if not passages:
        return "None"
    lines: list[str] = []
    total = 0
    for i, (query, text) in enumerate(passages, start=1):
        snippet = text[:1200].replace("\n", " ")
        line = f"[{i}] (query: {query}) {snippet}"
        if max_chars > 0 and total + len(line) > max_chars:
            lines.append(f"... ({len(passages) - i + 1} more passages truncated)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _format_history_actions(actions: list[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Search] {q}" for q in actions)


@register("hotpotqa_agent")
class HotpotQAAgentFlow(AgentFlowBase):
    """
    Multi-step HotpotQA agent (paper_search style state management).

    Each step re-builds [system, user] from current state:
    - user_query (fixed)
    - passage_list (accumulated, truncated to fit)
    - history_actions (list of past queries)
    This avoids prompt length explosion from multi-turn message accumulation.
    """

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

        self.max_steps = int(kwargs.get("max_steps", 5))
        self.max_parallel_calls = int(kwargs.get("max_parallel_calls", 4))
        self.force_first_search = bool(kwargs.get("force_first_search", True))

        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format,
            self.tokenizer,
        )
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = HOTPOTQA_TOOL_SCHEMAS

        corpus_data_dir = kwargs.get("corpus_data_dir")
        embedding_model_name = kwargs.get("embedding_model_name", DEFAULT_HOTPOTQA_EMBEDDING_MODEL)
        embedding_devices = resolve_hotpotqa_embedding_devices(
            kwargs.get("embedding_devices"),
            kwargs.get("agent_flow_worker_index"),
        )
        self.search_tool = HotpotQASearchToolLegacy(
            embedding_model_name=embedding_model_name,
            embedding_devices=embedding_devices,
            corpus_data_dir=corpus_data_dir,
        )
        self.enable_tool_parse_feedback = bool(kwargs.get("enable_tool_parse_feedback", True))

    def _build_messages(
        self,
        question: str,
        passages: list[tuple[str, str]],
        actions: list[str],
        tool_feedback: str,
        *,
        max_passage_chars: int | None = None,
    ) -> list[dict]:
        """Build [system, user] messages from current state, with passage truncation for safety.

        Must stay in sync with `HOTPOTQA_USER_PROMPT.format(...)` keys:
        user_query, history_actions, passage_list, tool_feedback.
        """
        if max_passage_chars is None:
            max_passage_chars = self.prompt_length * 3
        fb = tool_feedback.strip() if tool_feedback.strip() else "None"
        user_content = HOTPOTQA_USER_PROMPT.format(
            user_query=question,
            passage_list=_format_passage_list(passages, max_chars=max_passage_chars),
            history_actions=_format_history_actions(actions),
            tool_feedback=fb,
        )
        return [
            {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def _prompt_ids_within_budget(
        self,
        question: str,
        passages: list[tuple[str, str]],
        history_actions: list[str],
        fb_text: str,
        *,
        num_step: int,
    ) -> list[int]:
        """
        Tokenize prompt with tools= so the model always sees tool definitions (chat template prefix).

        PaperSearchAgentFlow does not truncate the tokenized prompt. Here we only shrink the
        variable-length passage block; we must NOT use tail slicing on prompt_ids — that drops
        the tool schema and breaks <tool_call> JSON adherence.
        """
        max_chars = self.prompt_length * 3
        min_chars = 400
        last_ids: list[int] = []
        while True:
            messages = self._build_messages(question, passages, history_actions, fb_text, max_passage_chars=max_chars)
            last_ids = await self.apply_chat_template(messages, tools=self.tool_schemas)
            if len(last_ids) <= self.prompt_length:
                return last_ids
            if max_chars <= min_chars:
                break
            max_chars = max(min_chars, int(max_chars * 0.72))

        logger.warning(
            "[hotpotqa_agent][step=%d] prompt still too long (%d tokens, limit %d) after shrinking "
            "passages; keeping PREFIX so tool definitions remain.",
            num_step,
            len(last_ids),
            self.prompt_length,
        )
        return last_ids[: self.prompt_length]

    def _make_anchor_obs(
        self,
        question: str,
        passages: list[tuple[str, str]],
        history_actions: list[str],
        tool_feedback: str,
    ) -> str:
        fb = tool_feedback.strip() if tool_feedback.strip() else "None"
        return HOTPOTQA_USER_PROMPT.format(
            user_query=question,
            passage_list=_format_passage_list(passages, max_chars=self.prompt_length * 3),
            history_actions=_format_history_actions(history_actions),
            tool_feedback=fb,
        )

    def _make_extra_fields(self, anchor_obs: str, history_actions: list[str], acc: float = 0.0) -> dict[str, Any]:
        """Build extra_fields with consistent reward_extra_info keys across all steps."""
        return {
            "anchor_obs": anchor_obs,
            "reward_extra_info": {
                "num_tool_steps": len(history_actions),
                "acc": acc,
            },
        }

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        question = raw_prompt[0]["content"]

        metrics: dict[str, Any] = {}
        steps: list[AgentFlowStep] = []
        passages: list[tuple[str, str]] = []
        history_actions: list[str] = []

        if self.force_first_search:
            self._do_search(question, passages, history_actions)

        tool_feedback_lines: list[str] = []
        end_status = {"terminated": False, "truncated": True, "termination_reason": "max_steps"}

        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1

            fb_text = "\n".join(tool_feedback_lines[-3:]) if tool_feedback_lines else ""
            if not self.enable_tool_parse_feedback:
                fb_text = ""
            anchor_obs = self._make_anchor_obs(question, passages, history_actions, fb_text)
            prompt_ids = await self._prompt_ids_within_budget(
                question, passages, history_actions, fb_text, num_step=num_steps
            )

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
                recovered = _recover_tool_calls_from_text(response_text)
                if recovered:
                    tool_calls = recovered
                elif (
                    self.enable_tool_parse_feedback
                    and "<tool_call>" in response_text
                    and "</tool_call>" in response_text
                ):
                    tool_feedback_lines.append(
                        "Previous assistant message contained <tool_call>...</tool_call> but the JSON "
                        "could not be parsed. Use strict JSON with double-quoted keys and string values, "
                        "comma between fields: name must be search and arguments must include query."
                    )

            if not tool_calls:
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=None,
                    extra_fields=self._make_extra_fields(anchor_obs, history_actions),
                )
                step.extra_fields.update(generation_info)
                step = await self._postprocess(step, **kwargs)
                ri = step.extra_fields.get("reward_extra_info", {})
                step.extra_fields["reward_extra_info"] = {
                    "num_tool_steps": len(history_actions),
                    "acc": ri.get("acc", 0.0),
                }
                steps.append(step)
                end_status = terminal_status(generation_info)
                break

            tool_calls = tool_calls[: self.max_parallel_calls]

            queries: list[str] = []
            for tc in tool_calls:
                if tc.name not in _RETRIEVAL_TOOL_NAMES:
                    continue
                tool_args = _decode_tool_arguments(tc.arguments)
                if not tool_args:
                    continue
                query = tool_args.get("query")
                if query:
                    queries.append(str(query))

            if queries:
                with simple_timer("tool_calls", metrics):
                    self._do_search_batch(queries, passages, history_actions)
            elif self.enable_tool_parse_feedback and tool_calls:
                tool_feedback_lines.append(
                    "Previous tool call was recognized but arguments were invalid. "
                    "For search, arguments must be a JSON object with a string field query."
                )

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=0.0,
                extra_fields=self._make_extra_fields(anchor_obs, history_actions),
            )
            step.extra_fields.update(generation_info)
            step = await self._postprocess(step, **kwargs)
            steps.append(step)

        return AgentFlowOutput(steps=steps, metrics=metrics, **end_status)

    def _do_search(self, query: str, passages: list[tuple[str, str]], history_actions: list[str]) -> None:
        """Execute a single search query and update state."""
        try:
            results = self.search_tool.batch_execute([{"query": query}])
            self._ingest_results(query, results, passages)
            history_actions.append(query)
        except Exception as e:
            logger.warning("[hotpotqa_agent] search failed for query=%r: %s", query, e)
            history_actions.append(query)

    def _do_search_batch(self, queries: list[str], passages: list[tuple[str, str]], history_actions: list[str]) -> None:
        """Execute multiple search queries and update state."""
        try:
            results = self.search_tool.batch_execute([{"query": q} for q in queries])
            for query, item in zip(queries, results, strict=False):
                self._ingest_results(query, [item], passages)
                history_actions.append(query)
        except Exception as e:
            logger.warning("[hotpotqa_agent] batch search failed: %s", e)
            for q in queries:
                history_actions.append(q)

    @staticmethod
    def _ingest_results(
        query: str,
        results: list[dict[str, Any]],
        passages: list[tuple[str, str]],
    ) -> None:
        """Parse search results and deduplicate into the passage list."""
        for item in results:
            if not item.get("success", False):
                continue
            content = str(item.get("content", ""))
            for p in parse_legacy_tool_result(content):
                if not any(existing_text == p.text for _, existing_text in passages):
                    passages.append((query, p.text))
