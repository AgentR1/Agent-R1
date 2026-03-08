from typing import Any

from verl.utils.reward_score import gsm8k

from ..base import BaseTool
from ..schema import ToolResponse


@BaseTool.register("calc_gsm8k_reward")
class GSM8KTool(BaseTool):
    name: str = "calc_gsm8k_reward"
    description: str = "A tool for calculating the reward of gsm8k"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question",
            },
        },
        "required": ["answer"],
    }

    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        answer = args.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        if not answer.startswith("#### "):
            answer = "#### " + answer

        tools_kwargs = kwargs.get("tools_kwargs") or {}
        # Support both flat and tool-namespaced layouts:
        # - tools_kwargs["ground_truth"] = "..."
        # - tools_kwargs["calc_gsm8k_reward"]["ground_truth"] = "..."

        ground_truth = None
        if "ground_truth" in tools_kwargs:
            ground_truth = tools_kwargs["ground_truth"]
        elif "calc_gsm8k_reward" in tools_kwargs and "ground_truth" in tools_kwargs["calc_gsm8k_reward"]:
            ground_truth = tools_kwargs["calc_gsm8k_reward"]["ground_truth"]
        else:
            raise ValueError("ground_truth is required in tools_kwargs for calc_gsm8k_reward")

        reward = gsm8k.compute_score(
            solution_str=answer,
            ground_truth=ground_truth,
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

        extra_info = {
            "answer": answer,
            "ground_truth": ground_truth,
            "reward": float(reward),
        }
        return ToolResponse(text=f"Current parsed {answer=} {reward=}"), None, extra_info
