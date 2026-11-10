import numpy as np
import pytest
import torch

from agent_r1.agent_flow.agent_flow import (
    AgentFlowMetrics,
    AgentFlowOutput,
    AgentFlowWorkerBase,
    _InternalAgentFlowStep,
)
from agent_r1.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _flow(response_mask, reward, device):
    mask = torch.tensor([response_mask], dtype=torch.long, device=device)
    length = mask.shape[1]
    step = _InternalAgentFlowStep(
        prompt_ids=torch.ones((1, 2), dtype=torch.long, device=device),
        response_ids=torch.ones((1, length), dtype=torch.long, device=device),
        response_mask=mask,
        attention_mask=torch.ones((1, length + 2), dtype=torch.long, device=device),
        input_ids=torch.ones((1, length + 2), dtype=torch.long, device=device),
        position_ids=torch.arange(length + 2, device=device).unsqueeze(0),
        reward_score=reward,
    )
    return AgentFlowOutput(steps=[step], metrics=AgentFlowMetrics())


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ]
)
def device(request):
    return request.param


@pytest.mark.parametrize(
    ("mask", "reward_index"),
    [
        ([1, 1, 1, 0, 0, 0], 2),
        ([1, 0, 0, 1, 1, 0], 4),
        ([1, 1, 0, 1, 0, 0], 3),
        ([0, 1, 0, 0, 0, 0], 1),
        ([0, 0, 0, 0, 0, 0], 5),
    ],
)
def test_reward_placed_at_last_action_token(mask, reward_index, device):
    data = AgentFlowWorkerBase._postprocess(None, [_flow(mask, 3.0, device)])
    expected = torch.zeros((1, len(mask)), device=device)
    expected[0, reward_index] = 3.0
    torch.testing.assert_close(data.batch["rm_scores"], expected)


def test_masked_tool_tokens_preserve_grpo_reward(device):
    masks = [[1, 1, 0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]
    data = AgentFlowWorkerBase._postprocess(None, [_flow(masks[0], 2.0, device), _flow(masks[1], 0.0, device)])
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["rm_scores"],
        response_mask=data.batch["response_mask"],
        index=np.array(["prompt", "prompt"]),
        trajectory_uids=data.non_tensor_batch["trajectory_uids"],
        norm_adv_by_std_in_grpo=False,
    )
    expected = torch.tensor(masks, dtype=torch.float32, device=device) * torch.tensor([[1.0], [-1.0]], device=device)
    torch.testing.assert_close(advantages, expected)
