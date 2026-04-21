# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Trajectory-aware mini-batch packing utilities.

These helpers reorder step-level DataProto rows so each PPO mini-batch
contains whole trajectories. The core invariant is:

    Every step that belongs to trajectory T appears in the same mini-batch.

This keeps multi-step agent rollouts intact during forward and backward passes
and avoids splitting a single trajectory across separate PPO updates.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from verl import DataProto

# Tensor fields that must be zeroed for padding rows so that they never
# contribute to masked-loss computations in actor/critic workers.
# response_mask is the primary loss gate; attention_mask controls the
# transformer's attention - zeroing it prevents the model from attending
# to padding tokens and producing non-zero log-probs that could leak into loss.
_MASK_FIELDS_TO_ZERO = ("response_mask", "attention_mask")


@dataclass(frozen=True)
class _TrajectorySummary:
    uid: object
    indices: tuple[int, ...]
    step_lens: tuple[int, ...]
    num_steps: int
    sum_step_lens: int
    max_step_len: int
    min_step_len: int
    arrival_order: int


@dataclass
class _MiniBatchState:
    row_indices: list[int]
    used_steps: int
    sum_step_lens: int
    max_step_len: int
    min_step_len: int
    creation_order: int


def _compute_step_lengths(attention_mask: torch.Tensor, is_pad_arr: np.ndarray) -> np.ndarray:
    step_lens = attention_mask.sum(dim=1).detach().cpu().numpy().astype(np.int32)
    step_lens[is_pad_arr] = 0
    return step_lens


def _compute_pad_cost(used_steps: int, sum_step_lens: int, max_step_len: int) -> int:
    if used_steps == 0:
        return 0
    return used_steps * max_step_len - sum_step_lens


def _build_trajectory_summaries(
    traj_uids: np.ndarray,
    step_idxs: np.ndarray,
    step_lens: np.ndarray,
    valid_global_indices: np.ndarray,
    ppo_mini_batch_size: int,
) -> list[_TrajectorySummary]:
    traj_to_sorted_indices: OrderedDict[object, list[int]] = OrderedDict()
    for global_idx in valid_global_indices:
        uid = traj_uids[global_idx]
        if uid not in traj_to_sorted_indices:
            traj_to_sorted_indices[uid] = []
        traj_to_sorted_indices[uid].append(int(global_idx))

    summaries = []
    for arrival_order, (uid, indices) in enumerate(traj_to_sorted_indices.items()):
        ordered_indices = tuple(sorted(indices, key=lambda idx: int(step_idxs[idx])))
        ordered_lens = tuple(int(step_lens[idx]) for idx in ordered_indices)
        num_steps = len(ordered_indices)
        if num_steps > ppo_mini_batch_size:
            raise ValueError(
                f"Trajectory '{uid}' has {num_steps} steps, which exceeds "
                f"ppo_mini_batch_size={ppo_mini_batch_size}. "
                f"Please increase ppo_mini_batch_size to at least {num_steps}."
            )
        summaries.append(
            _TrajectorySummary(
                uid=uid,
                indices=ordered_indices,
                step_lens=ordered_lens,
                num_steps=num_steps,
                sum_step_lens=sum(ordered_lens),
                max_step_len=max(ordered_lens),
                min_step_len=min(ordered_lens),
                arrival_order=arrival_order,
            )
        )

    return sorted(
        summaries,
        key=lambda item: (
            -item.max_step_len,
            -item.sum_step_lens,
            -item.num_steps,
            item.arrival_order,
        ),
    )


def _candidate_key(
    minibatch: _MiniBatchState,
    trajectory: _TrajectorySummary,
    ppo_mini_batch_size: int,
) -> tuple[int, int, int, int]:
    used_after = minibatch.used_steps + trajectory.num_steps
    max_after = max(minibatch.max_step_len, trajectory.max_step_len)
    min_after = min(minibatch.min_step_len, trajectory.min_step_len)
    sum_after = minibatch.sum_step_lens + trajectory.sum_step_lens

    delta_pad = _compute_pad_cost(used_after, sum_after, max_after) - _compute_pad_cost(
        minibatch.used_steps,
        minibatch.sum_step_lens,
        minibatch.max_step_len,
    )
    range_after = max_after - min_after
    remaining_after = ppo_mini_batch_size - used_after

    return (delta_pad, range_after, remaining_after, minibatch.creation_order)


def _place_trajectory(
    minibatches: list[_MiniBatchState],
    trajectory: _TrajectorySummary,
    ppo_mini_batch_size: int,
) -> None:
    best_index = None
    best_key = None

    for idx, minibatch in enumerate(minibatches):
        if minibatch.used_steps + trajectory.num_steps > ppo_mini_batch_size:
            continue
        current_key = _candidate_key(minibatch, trajectory, ppo_mini_batch_size)
        if best_key is None or current_key < best_key:
            best_index = idx
            best_key = current_key

    if best_index is None:
        minibatches.append(
            _MiniBatchState(
                row_indices=list(trajectory.indices),
                used_steps=trajectory.num_steps,
                sum_step_lens=trajectory.sum_step_lens,
                max_step_len=trajectory.max_step_len,
                min_step_len=trajectory.min_step_len,
                creation_order=len(minibatches),
            )
        )
        return

    target = minibatches[best_index]
    target.row_indices.extend(trajectory.indices)
    target.used_steps += trajectory.num_steps
    target.sum_step_lens += trajectory.sum_step_lens
    target.max_step_len = max(target.max_step_len, trajectory.max_step_len)
    target.min_step_len = min(target.min_step_len, trajectory.min_step_len)


def pack_trajectories_into_minibatches(
    batch: DataProto,
    ppo_mini_batch_size: int,
) -> DataProto:
    """Reorder and pad a step-level batch so that split(ppo_mini_batch_size)
    yields mini-batches where every trajectory is fully contained.

    The function is the single point of truth for trajectory-level packing.
    It operates on the flattened step-row layout produced by AgentFlowWorker,
    where each row is one agent step and trajectories are identified by
    non_tensor_batch["trajectory_uids"].

    Algorithm
    ---------
    1. Separate valid rows (is_pad=False) from existing world-size padding rows.
    2. Compute per-step effective lengths from attention_mask and build
       trajectory summaries sorted by packing difficulty.
    3. Dynamically place each full trajectory into the candidate mini-batch
       that minimizes incremental padding cost; break ties by resulting
       step-length range, then remaining capacity, then creation order.
    4. Pad each mini-batch to exactly ppo_mini_batch_size rows by appending
       copies of an existing pad row (response_mask all-zeros).
    5. Rebuild via DataProto.select_idxs(reorder_indices) which propagates all
       tensor and non_tensor fields; overwrite is_pad with the new mask.

    Output guarantees
    -----------------
    - All original valid rows are present exactly once, with all fields intact
      (trajectory_uids, step_indices, response_mask, advantages, old_log_probs, ...).
    - Padding rows have response_mask == 0, so they contribute zero to any
      masked loss aggregation (seq-mean-token-sum, token-mean, seq-mean-token-mean).
    - Output batch size == num_mini_batches * ppo_mini_batch_size, which is
      trivially divisible by ppo_mini_batch_size for DataProto.split().

    Args:
        batch: Step-level DataProto with non_tensor_batch fields
               "trajectory_uids", "step_indices", and optionally "is_pad".
        ppo_mini_batch_size: Maximum number of rows per mini-batch. Must be
               >= the longest single trajectory's step count.

    Returns:
        A new DataProto with rows reordered and padded for trajectory-aligned
        splitting. meta_info is forwarded unchanged.

    Raises:
        ValueError: If "trajectory_uids" or "step_indices" are absent.
        ValueError: If ``ppo_mini_batch_size`` exceeds the number of valid rows
            in the current batch.
        ValueError: If any single trajectory has more steps than ppo_mini_batch_size.
    """
    if "trajectory_uids" not in batch.non_tensor_batch:
        raise ValueError("batch.non_tensor_batch must contain 'trajectory_uids'")
    if "step_indices" not in batch.non_tensor_batch:
        raise ValueError("batch.non_tensor_batch must contain 'step_indices'")
    if batch.batch is None or "attention_mask" not in batch.batch.keys():
        raise ValueError("batch.batch must contain 'attention_mask'")

    is_pad_arr = batch.non_tensor_batch.get("is_pad", np.zeros(len(batch), dtype=bool)).astype(bool)
    traj_uids = batch.non_tensor_batch["trajectory_uids"]
    step_idxs = batch.non_tensor_batch["step_indices"]
    attention_mask = batch.batch["attention_mask"]

    valid_global_indices = np.where(~is_pad_arr)[0]
    pad_global_indices = np.where(is_pad_arr)[0]

    if ppo_mini_batch_size > len(valid_global_indices):
        raise ValueError(
            f"ppo_mini_batch_size={ppo_mini_batch_size} exceeds the current "
            f"effective train batch size ({len(valid_global_indices)} valid rows)."
        )

    # Padding template: reuse an existing pad row (response_mask guaranteed all-zero
    # by _pad_dataproto_to_world_size). Fallback: last valid row (rare, edge case).
    if len(pad_global_indices) > 0:
        pad_template_idx = int(pad_global_indices[0])
    elif len(valid_global_indices) > 0:
        pad_template_idx = int(valid_global_indices[-1])
    else:
        raise ValueError("batch contains no rows to pack.")

    step_lens = _compute_step_lengths(attention_mask, is_pad_arr)
    summaries = _build_trajectory_summaries(
        traj_uids=traj_uids,
        step_idxs=step_idxs,
        step_lens=step_lens,
        valid_global_indices=valid_global_indices,
        ppo_mini_batch_size=ppo_mini_batch_size,
    )

    minibatches: list[_MiniBatchState] = []
    for trajectory in summaries:
        _place_trajectory(minibatches, trajectory, ppo_mini_batch_size)

    # --- Build final index array: real rows + padding up to ppo_mini_batch_size ---
    final_indices: list[int] = []
    final_is_pad: list[bool] = []

    for minibatch in minibatches:
        n_real = len(minibatch.row_indices)
        n_pad = ppo_mini_batch_size - n_real
        final_indices.extend(minibatch.row_indices)
        final_is_pad.extend([False] * n_real)
        final_indices.extend([pad_template_idx] * n_pad)
        final_is_pad.extend([True] * n_pad)

    # --- Rebuild DataProto with the new row order ---
    new_batch = batch.select_idxs(np.array(final_indices, dtype=np.int64))
    new_batch.non_tensor_batch["is_pad"] = np.array(final_is_pad, dtype=bool)

    # Zero out mask fields for all padding rows. pad_dataproto_to_divisor creates
    # pad rows by copying real rows verbatim, so their masks are not guaranteed
    # to be zero unless we fix them here.
    if new_batch.batch is not None:
        pad_row_mask = torch.from_numpy(np.array(final_is_pad, dtype=bool))
        for field in _MASK_FIELDS_TO_ZERO:
            if field in new_batch.batch.keys():
                new_batch.batch[field][pad_row_mask] = 0

    return new_batch
