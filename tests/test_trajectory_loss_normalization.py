import numpy as np
import pytest
import torch

from agent_r1.trainer.ppo.core_algos import agg_loss
from agent_r1.trainer.ppo.trajectory_batching import get_mini_batch_global_info, prepare_trajectory_mini_batch
from verl.trainer.ppo.core_algos import agg_loss as upstream_agg_loss


class _Data:
    def __init__(self, batch, uids):
        self.batch = batch
        self.non_tensor_batch = {"trajectory_uids": np.asarray(uids)}
        self.meta_info = {}

    def __len__(self):
        return self.batch["attention_mask"].shape[0]

    def select_idxs(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        result = _Data(
            {key: value[indices.to(value.device)].clone() for key, value in self.batch.items()},
            self.non_tensor_batch["trajectory_uids"][indices.cpu().numpy()],
        )
        result.meta_info = dict(self.meta_info)
        return result


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ]
)
def device(request):
    return request.param


@pytest.mark.parametrize("loss_mode", ["seq-mean-token-mean", "seq-mean-token-sum"])
@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_sequence_loss_excludes_real_empty_responses(device, loss_mode, micro_batch_size):
    response_mask = torch.tensor([[1, 1], [0, 0], [1, 1]], device=device)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]], device=device)
    source = _Data({"response_mask": response_mask, "attention_mask": attention_mask}, ["a", "b", "c"])
    prepared = prepare_trajectory_mini_batch(source, mini_batch_size=3, dp_size=2)

    local_size = len(prepared) // 2
    rank_gradients = []
    for rank in range(2):
        local = prepared.select_idxs(range(rank * local_size, (rank + 1) * local_size))
        info = get_mini_batch_global_info(local)
        assert info["global_batch_size"] == 2
        assert info["batch_num_tokens"] == 4
        assert sorted(info["global_token_num"]) == [2, 4, 4]
        theta = torch.tensor(1.0, device=device, requires_grad=True)
        for start in range(0, len(local), micro_batch_size):
            mask = local.batch["response_mask"][start : start + micro_batch_size]
            loss = agg_loss(
                theta * torch.ones_like(mask), mask, loss_mode, dp_size=2, global_batch_size=info["global_batch_size"]
            )
            loss.backward()
        rank_gradients.append(theta.grad)

    actual = torch.stack(rank_gradients).mean()
    expected = 1.0 if loss_mode == "seq-mean-token-mean" else 2.0
    torch.testing.assert_close(actual, torch.tensor(expected, device=device))


def test_each_trajectory_minibatch_counts_its_own_valid_rows(device):
    mask = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 0], [0, 0]], device=device)
    source = _Data(
        {"response_mask": mask, "attention_mask": torch.ones(5, 4, device=device)}, ["a", "a", "b", "c", "d"]
    )
    prepared = prepare_trajectory_mini_batch(source, mini_batch_size=2, dp_size=2)
    ids = prepared.batch["mini_batch_id"]
    sizes = prepared.batch["mini_batch_global_size"]
    assert sizes[ids == 0].unique().tolist() == [2]
    assert sizes[ids == 1].min().item() > 0


def test_fully_masked_batch_keeps_upstream_sequence_loss_finite(device):
    mask = torch.zeros(2, 2, device=device)
    source = _Data({"response_mask": mask, "attention_mask": torch.ones(2, 4, device=device)}, ["a", "b"])
    prepared = prepare_trajectory_mini_batch(source, mini_batch_size=2, dp_size=2)
    info = get_mini_batch_global_info(prepared)
    assert info["global_batch_size"] > 0
    theta = torch.tensor(1.0, device=device, requires_grad=True)
    loss = upstream_agg_loss(
        theta * torch.ones_like(mask),
        mask,
        "seq-mean-token-mean",
        dp_size=2,
        global_batch_size=info["global_batch_size"],
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.0, device=device))
    torch.testing.assert_close(theta.grad, torch.tensor(0.0, device=device))


def test_missing_response_mask_retains_all_real_rows(device):
    source = _Data({"attention_mask": torch.ones(3, 4, device=device)}, ["a", "b", "c"])
    prepared = prepare_trajectory_mini_batch(source, mini_batch_size=3, dp_size=2)
    assert prepared.batch["mini_batch_global_size"].unique().tolist() == [3]
