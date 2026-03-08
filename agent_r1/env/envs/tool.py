import asyncio
from typing import Any

from agent_r1.tool import BaseTool

from ..base import Action, AgentEnv, Observation
from ..tool_format import ToolCallAction, ToolFormatWrapper


@AgentEnv.register("tool")
class ToolEnv(AgentEnv):
    """Stateful tool-calling environment that manages its own conversation history.

    Delegates format handling to a ``ToolFormatWrapper`` so that tool execution
    logic is decoupled from model-specific wire formats (e.g. Hermes XML vs
    GPT-OSS special tokens).

    Args:
        tools: List of registered tool names to instantiate.
        tool_format: Registered ``ToolFormatWrapper`` name (default ``"hermes"``).
        tools_kwargs: Extra keyword arguments to pass to tools.
        **kwargs: Reserved for future extension.
    """

    def __init__(
        self, tools: list[str], tool_format: str = "hermes", tools_kwargs: dict[str, Any] | None = None, **kwargs
    ):
        self.tools: dict[str, BaseTool] = {name: BaseTool.from_name(name) for name in tools}
        self.format_wrapper: ToolFormatWrapper = ToolFormatWrapper.from_name(tool_format)
        self.tools_kwargs: dict[str, Any] = tools_kwargs if tools_kwargs is not None else {}
        self._messages: list[dict] = []

    def reset(self, **kwargs) -> Observation:
        """Reset the environment and return the initial observation.

        The initial messages are taken from ``kwargs["raw_prompt"]`` if
        present; otherwise the conversation starts empty.

        Args:
            **kwargs: Task keyword arguments.  ``raw_prompt`` (list[dict]) is
                used as the initial messages when available.

        Returns:
            Observation: Initial observation with ``messages`` populated.
        """
        self._messages = list(kwargs.get("raw_prompt", []))
        return Observation(messages=list(self._messages))

    def parse_response(self, llm_response: str) -> tuple[list[ToolCallAction], bool]:
        """Parse tool calls from LLM response.

        Args:
            llm_response (str): The decoded LLM output string.

        Returns: actions, is_valid
            actions: List of parsed ``ToolCallAction`` objects (empty if none found).
            is_valid: Always ``True``; individual malformed calls are silently skipped.
        """
        _, actions = self.format_wrapper.parse_response(llm_response)
        return actions, True

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Process the LLM response: parse tool calls, execute them, and update history.

        Args:
            action (Action): The LLM response.

        Returns: observation, reward, done, info
            observation (Observation): Updated messages including the assistant
                reply and tool responses.
            reward (float): Accumulated reward from all tool executions.
            done (bool): ``True`` when no tool calls are found (final answer).
            info (dict[str, Any]): Merged extra info from tool executions.
        """
        if not isinstance(action, Action) or action.text is None:
            raise TypeError("ToolEnv only accepts Action with text")

        _, tool_calls = self.format_wrapper.parse_response(action.text)
        self._messages.append({"role": "assistant", "content": action.text})

        if not tool_calls:
            return Observation(messages=list(self._messages)), None, True, {}

        async def _execute_one(
            tc: ToolCallAction,
        ) -> tuple[str, float | None]:
            if tc.name not in self.tools:
                available = ", ".join(self.tools.keys())
                return (
                    f"Error: tool '{tc.name}' not found. Available tools: [{available}]",
                    None,
                )
            tool_response, reward_score, _ = await self.tools[tc.name].run(tc.arguments, tools_kwargs=self.tools_kwargs)
            return tool_response.text or "", reward_score

        results = await asyncio.gather(*[_execute_one(tc) for tc in tool_calls])

        total_reward = None
        observation_parts: list[str] = []
        for obs_text, reward_score in results:
            observation_parts.append(self.format_wrapper.format_observation(obs_text))
            if reward_score is not None:
                total_reward = reward_score if total_reward is None else total_reward + reward_score

        self._messages.append({"role": "user", "content": "\n".join(observation_parts)})

        # TODO: support thinking model, filter out the thinking content
        return Observation(messages=list(self._messages)), total_reward, False, {}

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool schemas for all registered tools."""
        return [tool.tool_schema for tool in self.tools.values()]
