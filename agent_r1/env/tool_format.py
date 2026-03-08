import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ToolCallAction:
    """Parsed tool call action from LLM response."""

    name: str
    """Tool name to invoke."""

    arguments: dict[str, Any]
    """Parsed arguments dict."""


class ToolFormatWrapper(ABC):
    """Base class for tool call format wrappers.

    Handles bidirectional format conversion between LLM text and structured tool calls:
    - ``parse_response``: extract tool calls from LLM-generated text.
    - ``format_observation``: wrap tool execution results back into LLM-consumable text.

    Subclasses should register themselves via the ``@ToolFormatWrapper.register`` decorator.
    """

    _registry: dict[str, type["ToolFormatWrapper"]] = {}

    @abstractmethod
    def parse_response(self, llm_response: str) -> tuple[str, list[ToolCallAction]]:
        """Extract tool calls from LLM response text.

        Args:
            llm_response (str): The decoded LLM output string.

        Returns: content, tool_calls
            content: Remaining text after stripping tool call tokens.
            tool_calls: List of extracted ToolCallAction objects (empty if none found).
        """
        raise NotImplementedError

    @abstractmethod
    def format_observation(self, observation: str) -> str:
        """Wrap a single tool execution result for LLM input.

        Args:
            observation (str): Raw tool execution output.

        Returns:
            str: Formatted observation string ready to be appended to conversation.
        """
        raise NotImplementedError

    @classmethod
    def from_name(cls, name: str) -> "ToolFormatWrapper":
        """Create an instance by registered name.

        Args:
            name (str): Registered name (e.g. "hermes", "gpt-oss").

        Returns:
            ToolFormatWrapper: A new instance.

        Raises:
            ValueError: If the name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown tool format wrapper: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]()

    @classmethod
    def register(cls, name: str):
        """Decorator to register a ToolFormatWrapper subclass under a given name.

        Args:
            name (str): The name to register under.
        """

        def decorator(subclass: type["ToolFormatWrapper"]) -> type["ToolFormatWrapper"]:
            cls._registry[name] = subclass
            return subclass

        return decorator


# ---------------------------------------------------------------------------
# Hermes (XML)
# ---------------------------------------------------------------------------


@ToolFormatWrapper.register("hermes")
class HermesFormatWrapper(ToolFormatWrapper):
    """Hermes-style XML tool call format.

    - Parse: ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
    - Observe: ``<tool_response>...</tool_response>``
    """

    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def parse_response(self, llm_response: str) -> tuple[str, list[ToolCallAction]]:
        if "<tool_call>" not in llm_response or "</tool_call>" not in llm_response:
            return llm_response, []

        matches = self._TOOL_CALL_RE.findall(llm_response)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                if not isinstance(arguments, dict):
                    arguments = json.loads(arguments) if isinstance(arguments, str) else {}
                function_calls.append(ToolCallAction(name=name, arguments=arguments))
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")

        content = self._TOOL_CALL_RE.sub("", llm_response)
        return content, function_calls

    def format_observation(self, observation: str) -> str:
        return f"<tool_response>\n{observation}\n</tool_response>"


# ---------------------------------------------------------------------------
# GPT-OSS (OpenAI Harmony)
# ---------------------------------------------------------------------------


@ToolFormatWrapper.register("gpt-oss")
class GptOssFormatWrapper(ToolFormatWrapper):
    """GPT-OSS tool call format using OpenAI Harmony special tokens.

    Expects decoded text with special tokens preserved (``skip_special_tokens=False``).
    See https://cookbook.openai.com/articles/openai-harmony for format details.

    Adapted from
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/gpt_oss_detector.py
    """

    _COT_PATTERN = re.compile(r"<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>", re.DOTALL)
    _PARTIAL_COT_PATTERN = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", re.DOTALL)
    _TOOL_CALL_PATTERN = re.compile(
        r"<\|start\|>assistant<\|channel\|>[^<]* to=functions\.([^<]+) "
        r"<\|constrain\|>json<\|message\|>(.*?)<\|call\|>",
        re.DOTALL,
    )

    def parse_response(self, llm_response: str) -> tuple[str, list[ToolCallAction]]:
        text = llm_response
        # Strip COT blocks that may contain tool-call-like tokens
        text = self._COT_PATTERN.sub("", text)
        text = self._PARTIAL_COT_PATTERN.sub("", text)

        matches = self._TOOL_CALL_PATTERN.findall(text)
        if not matches:
            return text, []

        function_calls = []
        for match in matches:
            try:
                name, arguments_str = match[0], match[1]
                arguments = json.loads(arguments_str)
                if not isinstance(arguments, dict):
                    arguments = {}
                function_calls.append(ToolCallAction(name=name, arguments=arguments))
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")

        content = self._TOOL_CALL_PATTERN.sub("", text)
        return content, function_calls

    def format_observation(self, observation: str) -> str:
        return f"<|start|>tool<|message|>{observation}<|end|>"
