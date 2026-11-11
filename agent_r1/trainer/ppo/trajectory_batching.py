"""Trajectory-aware PPO mini-batch helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class _Entry:
    source_idx: int
    mini_batch_id: int
    is_padding: bool


def prepare_trajectory_mini_batch(data: Any, mini_batch_size: int, dp_size: int) -> Any:
    """Build an update batch whose PPO mini-batches preserve whole trajectories.

    Args:
        data (Any): A DataProto-like object with `batch`, `non_tensor_batch`, and `select_idxs`.
        mini_batch_size (int): Target number of trajectories per PPO mini-batch.
        dp_size (int): Data parallel size used by the training worker dispatch.

    Returns:
        Any: A DataProto-like object with mini-batch metadata and update-only padding rows.
    """
    if mini_batch_size <= 0:
        raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}")
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if len(data) == 0:
        return data

    valid_indices = _valid_indices(data)
    if not valid_indices:
        raise ValueError("trajectory mini-batching requires at least one valid row")

    mini_batches = _build_trajectory_batches(data, valid_indices, mini_batch_size)
    entries = _build_rank_ordered_entries(mini_batches, dp_size)
    source_indices = [entry.source_idx for entry in entries]
    prepared = data.select_idxs(source_indices)

    device = _batch_device(prepared.batch)
    mini_batch_ids = torch.tensor([entry.mini_batch_id for entry in entries], dtype=torch.long, device=device)
    padding_mask = torch.tensor([entry.is_padding for entry in entries], dtype=torch.bool, device=device)

    prepared.batch["mini_batch_id"] = mini_batch_ids
    prepared.batch["sample_mask"] = ~padding_mask
    _zero_padding_loss_masks(prepared.batch, padding_mask)
    _assign_global_mini_batch_info(prepared, data.batch, mini_batches, mini_batch_ids, device)

    prepared.meta_info = dict(getattr(prepared, "meta_info", {}))
    prepared.meta_info["num_mini_batch"] = len(mini_batches)
    return prepared


def split_data_proto_by_mini_batch_id(data: Any, *, shuffle: bool = False, seed: int = 42) -> list[Any]:
    """Split a local worker batch using precomputed `mini_batch_id` values.

    Args:
        data (Any): A DataProto-like object containing `batch["mini_batch_id"]`.
        shuffle (bool): Whether to shuffle mini-batch id order.
        seed (int): Deterministic seed for shuffling.

    Returns:
        list[Any]: DataProto-like mini-batches, each containing exactly one mini-batch id.
    """
    if "mini_batch_id" not in data.batch:
        raise KeyError("mini_batch_id is required for trajectory-aware mini-batch splitting")

    mini_batch_ids = data.batch["mini_batch_id"].detach().cpu()
    num_mini_batch = int(getattr(data, "meta_info", {}).get("num_mini_batch", mini_batch_ids.max().item() + 1))
    ordered_ids = list(range(num_mini_batch))
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(num_mini_batch, generator=generator).tolist()
        ordered_ids = [ordered_ids[idx] for idx in permutation]

    mini_batches = []
    for mini_batch_id in ordered_ids:
        indices = torch.nonzero(mini_batch_ids == mini_batch_id, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        mini_batches.append(data.select_idxs(indices))
    return mini_batches


def get_mini_batch_global_info(mini_batch: Any) -> dict[str, Any]:
    """Return loss normalization metadata for one planned mini-batch.

    The sequence denominator is at least one, including for fully masked batches.

    Args:
        mini_batch (Any): A DataProto-like object containing mini-batch metadata fields.

    Returns:
        dict[str, Any]: Global mini-batch size and token counts.
    """
    first_idx = 0
    global_size = int(mini_batch.batch["mini_batch_global_size"][first_idx].item())
    token_nums = mini_batch.batch["mini_batch_global_token_num"][first_idx]
    response_token_num = int(mini_batch.batch["mini_batch_global_response_token_num"][first_idx].item())
    return {
        "global_batch_size": global_size,
        "batch_num_tokens": response_token_num,
        "global_token_num": token_nums[token_nums > 0].tolist(),
    }


def _valid_indices(data: Any) -> list[int]:
    sample_mask = data.batch.get("sample_mask", None)
    if sample_mask is None:
        return list(range(len(data)))
    mask = sample_mask.detach().cpu().to(dtype=torch.bool).tolist()
    return [idx for idx, is_valid in enumerate(mask) if is_valid]


def _build_trajectory_batches(data: Any, valid_indices: list[int], mini_batch_size: int) -> list[list[list[int]]]:
    trajectory_uids = data.non_tensor_batch.get("trajectory_uids")
    if trajectory_uids is None:
        row_groups = [[idx] for idx in valid_indices]
        return _chunk_groups(row_groups, mini_batch_size)

    groups: OrderedDict[Any, list[int]] = OrderedDict()
    for idx in valid_indices:
        groups.setdefault(trajectory_uids[idx], []).append(idx)
    return _chunk_groups(list(groups.values()), mini_batch_size)


def _chunk_groups(groups: list[list[int]], chunk_size: int) -> list[list[list[int]]]:
    return [groups[idx : idx + chunk_size] for idx in range(0, len(groups), chunk_size)]


def _build_rank_ordered_entries(mini_batches: list[list[list[int]]], dp_size: int) -> list[_Entry]:
    per_rank_entries: list[list[_Entry]] = [[] for _ in range(dp_size)]

    for mini_batch_id, trajectory_groups in enumerate(mini_batches):
        per_rank_for_mini_batch: list[list[_Entry]] = [[] for _ in range(dp_size)]
        pad_source_idx = trajectory_groups[0][0]

        for group_idx, row_indices in enumerate(trajectory_groups):
            rank = group_idx % dp_size
            per_rank_for_mini_batch[rank].extend(
                _Entry(source_idx=row_idx, mini_batch_id=mini_batch_id, is_padding=False) for row_idx in row_indices
            )

        max_local_rows = max(1, *(len(entries) for entries in per_rank_for_mini_batch))
        for rank_entries in per_rank_for_mini_batch:
            while len(rank_entries) < max_local_rows:
                rank_entries.append(_Entry(source_idx=pad_source_idx, mini_batch_id=mini_batch_id, is_padding=True))

        for rank, rank_entries in enumerate(per_rank_for_mini_batch):
            per_rank_entries[rank].extend(rank_entries)

    return [entry for rank_entries in per_rank_entries for entry in rank_entries]


def _batch_device(batch: Any) -> torch.device:
    for value in batch.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _zero_padding_loss_masks(batch: Any, padding_mask: torch.Tensor) -> None:
    if not padding_mask.any():
        return
    for key in ("response_mask", "loss_mask"):
        if key in batch:
            batch[key][padding_mask] = 0


def _assign_global_mini_batch_info(
    prepared: Any,
    source_batch: Any,
    mini_batches: list[list[list[int]]],
    mini_batch_ids: torch.Tensor,
    device: torch.device,
) -> None:
    row_counts = [sum(len(group) for group in mini_batch) for mini_batch in mini_batches]
    max_rows = max(row_counts)
    global_sizes = torch.tensor(row_counts, dtype=torch.long, device=device)

    token_num_table = torch.zeros((len(mini_batches), max_rows), dtype=torch.long, device=device)
    response_token_nums = torch.zeros(len(mini_batches), dtype=torch.long, device=device)
    attention_mask = source_batch.get("attention_mask", None)
    response_mask = source_batch.get("response_mask", None)

    for mini_batch_id, mini_batch in enumerate(mini_batches):
        row_indices = [row_idx for group in mini_batch for row_idx in group]
        if attention_mask is not None:
            source_token_nums = attention_mask.new_tensor(
                [_sum_row_tokens(attention_mask, row_idx) for row_idx in row_indices], dtype=torch.long
            )
            token_num_table[mini_batch_id, : len(row_indices)] = source_token_nums.to(device)
        if response_mask is not None:
            global_sizes[mini_batch_id] = response_mask[row_indices].any(dim=-1).sum().clamp(min=1).to(device)
            response_token_nums[mini_batch_id] = int(
                sum(_sum_row_tokens(response_mask, row_idx) for row_idx in row_indices)
            )

    prepared.batch["mini_batch_global_size"] = global_sizes[mini_batch_ids]
    prepared.batch["mini_batch_global_token_num"] = token_num_table[mini_batch_ids]
    prepared.batch["mini_batch_global_response_token_num"] = response_token_nums[mini_batch_ids]


def _sum_row_tokens(tensor: torch.Tensor, row_idx: int) -> int:
    return int(tensor[row_idx].sum().detach().cpu().item())
