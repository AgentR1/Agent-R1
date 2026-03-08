import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


class OpenAIFunctionPropertySchema(BaseModel):
    """The schema of a parameter in OpenAI format."""

    type: str
    description: str | None = None
    enum: list[Any] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    """The schema of parameters in OpenAI format."""

    type: str = "object"
    properties: dict[str, OpenAIFunctionPropertySchema] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class OpenAIFunctionSchema(BaseModel):
    """The schema of a function in OpenAI format."""

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema = Field(
        default_factory=lambda: OpenAIFunctionParametersSchema(type="object", properties={}, required=[])
    )


class OpenAIFunctionToolSchema(BaseModel):
    """The schema of a tool in OpenAI format."""

    type: Literal["function"] = "function"
    function: OpenAIFunctionSchema


class OpenAIFunctionParsedSchema(BaseModel):
    """The parsed schema of a tool call in OpenAI format (arguments as JSON string)."""

    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    """The decoded tool call schema."""

    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(
        parsed_schema: OpenAIFunctionParsedSchema,
    ) -> tuple["OpenAIFunctionCallSchema", bool]:
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments), has_decode_error


class ToolResponse(BaseModel):
    """The response from a tool execution."""

    text: str | None = None
    image: list[Any] | None = None
    video: list[Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not isinstance(values, dict):
            raise ValueError("ToolResponse must be a dict.")
        if "image" in values and values["image"] is not None and not isinstance(values["image"], list):
            raise ValueError(
                "Image must be a list. For single images, wrap in a list: [image]. "
                "Example: {'image': [img1]} or {'image': [img1, img2, ...]}."
            )
        if "video" in values and values["video"] is not None and not isinstance(values["video"], list):
            raise ValueError(
                "Video must be a list. For single videos, wrap in a list: [video]. "
                "Example: {'video': [video1]} or {'video': [video1, video2, ...]}."
            )
        return values

    def is_empty(self) -> bool:
        return not self.text and not self.image and not self.video

    def is_text_only(self) -> bool:
        return bool(self.text) and not self.image and not self.video


def normalize_parameters_schema(parameters: dict[str, Any] | None) -> OpenAIFunctionParametersSchema:
    """Accept None as 'no-arg' schema and normalize to OpenAI parameters schema."""

    if not parameters:
        return OpenAIFunctionParametersSchema(type="object", properties={}, required=[])
    return OpenAIFunctionParametersSchema.model_validate(parameters)


def is_tool_schema(tool_like: dict[str, Any]) -> bool:
    """Return True if `tool_like` can be represented as an OpenAI tool schema."""

    try:
        function = OpenAIFunctionSchema(
            name=tool_like.get("name", ""),
            description=tool_like.get("description", "") or "",
            parameters=normalize_parameters_schema(tool_like.get("parameters")),
        )
        OpenAIFunctionToolSchema(function=function)
        return True
    except (ValidationError, ValueError):
        return False
