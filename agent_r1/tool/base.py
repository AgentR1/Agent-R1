from abc import ABC, abstractmethod
from typing import Any

from .schema import OpenAIFunctionSchema, OpenAIFunctionToolSchema, ToolResponse, normalize_parameters_schema


class BaseTool(ABC):
    """Abstract base class for tools.

    Subclasses should register via ``@BaseTool.register(name)`` so that
    ``BaseTool.from_name(name)`` can instantiate them.
    """

    _registry: dict[str, type["BaseTool"]] = {}

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] | None = None

    @classmethod
    def register(cls, name: str):
        """Decorator to register a BaseTool subclass under *name*."""

        def decorator(subclass: type["BaseTool"]) -> type["BaseTool"]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "BaseTool":
        """Instantiate a registered tool by *name*.

        Raises:
            ValueError: If *name* is not registered.
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown tool: {name!r}. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    def __init__(self):
        if not self.name:
            raise ValueError("Tool name must be provided")

        # Normalize to a stable dict form and validate via Pydantic.
        params = normalize_parameters_schema(self.parameters).model_dump(exclude_none=True)
        self._function_schema = OpenAIFunctionSchema(
            name=self.name,
            description=self.description or "",
            parameters=normalize_parameters_schema(params),
        )
        self.parameters = params
        self._tool_schema = OpenAIFunctionToolSchema(
            type="function",
            function=self._function_schema,
        )

    def normalize_args(self, args: Any) -> dict[str, Any]:
        """Key defense: tool args must be an object; otherwise fallback to {}."""
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        return {}

    def normalize_response(self, value: Any) -> ToolResponse:
        """Key defense: ensure tool response shape (image/video must be list)."""
        if isinstance(value, ToolResponse):
            return value
        if isinstance(value, str):
            return ToolResponse(text=value)
        return ToolResponse.model_validate(value)

    async def run(self, args: Any = None, **kwargs) -> tuple[ToolResponse, float | None, dict]:
        """Recommended entrypoint: normalize args, execute, normalize response."""
        tool_response, reward_score, extra_info = await self.execute(self.normalize_args(args), **kwargs)
        return self.normalize_response(tool_response), reward_score, extra_info

    @abstractmethod
    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        """Execute the tool.

        Args:
            args: The arguments to the tool.
            kwargs: The keyword arguments to the tool.
        Returns:
            tool_response: The tool response.
            reward_score: The reward score.
            extra_info: The extra information.
        """
        raise NotImplementedError

    @property
    def function_schema(self) -> dict[str, Any]:
        return self._function_schema.model_dump(exclude_none=True)

    @property
    def tool_schema(self) -> dict[str, Any]:
        return self._tool_schema.model_dump(exclude_none=True)
