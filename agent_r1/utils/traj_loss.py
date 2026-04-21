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
Loss weighting helpers for trajectory-aware and step-aware mini-batching.

For trajectory mode we pre-scale per-step advantages so verl's
``seq-mean-token-sum`` aggregation gives every trajectory equal weight:

    L = (1 / N_traj) * sum_T (1 / |T|) * sum_{i in T} token_sum(i)

instead of the default per-step mean:

    L = (1 / ppo_mini_batch_size) * sum_i token_sum(i)

The input batch must already be packed so each PPO mini-batch contains whole
trajectories.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from verl import DataProto

_STEP_POOL_SUPPORTED_MODES = frozenset(["uniform-valid"])


def compute_traj_step_weights(
    trajectory_uids: np.ndarray,
    is_pad: np.ndarray,
    ppo_mini_batch_size: int,
) -> torch.Tensor:
    """Compute per-step advantage scaling weights for trajectory-level loss.

    For each valid step i in trajectory T (with |T| steps) inside a mini-batch
    that contains N_traj distinct trajectories:

        w_i = ppo_mini_batch_size / (N_traj * |T_i|)

    Pad rows receive weight 0.

    The weights satisfy ``sum_i w_i = ppo_mini_batch_size`` within every
    mini-batch chunk, which is the same total weight that verl's unweighted
    ``seq-mean-token-sum`` would apply.  The gradient contribution of each
    trajectory is therefore ``ppo_mini_batch_size / N_traj`` regardless of
    how many steps it contains.

    Args:
        trajectory_uids: Array of trajectory uid strings/objects, shape (bsz,).
            Must come from ``batch.non_tensor_batch["trajectory_uids"]`` after
            trajectory-aware packing.
        is_pad: Boolean array, shape (bsz,).  True for padding rows.
        ppo_mini_batch_size: Actor's ``ppo_mini_batch_size`` config value.
            The packed batch length must be divisible by this.

    Returns:
        Float tensor of shape (bsz,) with per-step weights.

    Raises:
        ValueError: If the batch length is not divisible by
            ``ppo_mini_batch_size``.
        ValueError: If any valid mini-batch chunk contains zero valid rows
            (degenerate packing).
    """
    bsz = len(trajectory_uids)
    if bsz % ppo_mini_batch_size != 0:
        raise ValueError(
            f"Batch size {bsz} is not divisible by ppo_mini_batch_size "
            f"{ppo_mini_batch_size}. Trajectory packing must be applied "
            "before trajectory loss weighting."
        )

    weights = torch.zeros(bsz, dtype=torch.float32)
    n_chunks = bsz // ppo_mini_batch_size

    for k in range(n_chunks):
        start = k * ppo_mini_batch_size
        end = start + ppo_mini_batch_size

        chunk_is_pad = is_pad[start:end]
        chunk_uids = trajectory_uids[start:end]

        valid_mask = ~chunk_is_pad.astype(bool)
        valid_uids = chunk_uids[valid_mask]

        if len(valid_uids) == 0:
            # All-pad chunk (can occur if world-size padding exceeds one full
            # mini-batch).  Leave weights as 0 for the whole chunk.
            continue

        uid_counts = Counter(valid_uids)
        n_traj = len(uid_counts)

        for local_idx in range(ppo_mini_batch_size):
            global_idx = start + local_idx
            if chunk_is_pad[local_idx]:
                continue
            uid = chunk_uids[local_idx]
            traj_len = uid_counts[uid]
            weights[global_idx] = ppo_mini_batch_size / (n_traj * traj_len)

    return weights


def apply_traj_loss_weights(
    batch: DataProto,
    ppo_mini_batch_size: int,
) -> DataProto:
    """Return a copy of *batch* with ``advantages`` pre-scaled for
    trajectory-level loss aggregation.

    The returned batch is a shallow copy with only the ``advantages`` tensor
    replaced; all other fields (including the original ``response_mask``) are
    shared references and are not modified.

    Args:
        batch: Full packed batch.  Must have
            ``batch["advantages"]`` (shape ``(bsz, response_length)``) and
            ``non_tensor_batch["trajectory_uids"]`` / ``["is_pad"]``.
        ppo_mini_batch_size: Actor's ``ppo_mini_batch_size`` config value.

    Returns:
        DataProto with scaled ``advantages``.

    Raises:
        KeyError: If required batch fields are absent.
    """
    trajectory_uids = batch.non_tensor_batch["trajectory_uids"]
    is_pad = batch.non_tensor_batch.get("is_pad", np.zeros(len(batch), dtype=bool)).astype(bool)

    weights = compute_traj_step_weights(
        trajectory_uids=trajectory_uids,
        is_pad=is_pad,
        ppo_mini_batch_size=ppo_mini_batch_size,
    )

    # Move weights to the same device as advantages before broadcasting.
    advantages = batch.batch["advantages"]
    weights = weights.to(advantages.device)

    # advantages shape: (bsz, response_length); broadcast weight per row.
    scaled_advantages = advantages * weights.unsqueeze(1)

    # Shallow-copy the DataProto so the caller's original batch is untouched.
    new_batch = DataProto(
        batch=batch.batch.clone(recurse=False),
        non_tensor_batch=batch.non_tensor_batch,
        meta_info=batch.meta_info,
    )
    new_batch.batch["advantages"] = scaled_advantages

    return new_batch


# ---------------------------------------------------------------------------
# Step-pool-level loss weighting (used by step-based mini-batch mode)
# ---------------------------------------------------------------------------


def compute_step_pool_weights(
    is_placeholder: np.ndarray,
    ppo_mini_batch_size: int,
) -> torch.Tensor:
    """Compute per-row weights for a single step pool.

    In step-based mode every row in a step pool corresponds to the same step
    index across different trajectories.  Placeholder rows (for trajectories
    shorter than the current step) must contribute **zero** to the loss.
    Valid rows are weighted uniformly: ``w_i = ppo_mini_batch_size / N_valid``
    within each mini-batch chunk, so the total weight matches what verl's
    ``seq-mean-token-sum`` expects.

    Args:
        is_placeholder: Boolean array, shape ``(pool_size,)``.
            ``True`` for placeholder / padding rows.
        ppo_mini_batch_size: The mini-batch size used by the downstream worker
            split.  Pool size must be divisible by this.

    Returns:
        Float tensor of shape ``(pool_size,)`` with per-row weights.
    """
    pool_size = len(is_placeholder)
    if pool_size == 0:
        return torch.zeros(0, dtype=torch.float32)

    if pool_size % ppo_mini_batch_size != 0:
        raise ValueError(f"Step pool size {pool_size} is not divisible by ppo_mini_batch_size {ppo_mini_batch_size}.")

    weights = torch.zeros(pool_size, dtype=torch.float32)
    n_chunks = pool_size // ppo_mini_batch_size

    for k in range(n_chunks):
        start = k * ppo_mini_batch_size
        end = start + ppo_mini_batch_size

        chunk_ph = is_placeholder[start:end].astype(bool)
        n_valid = int((~chunk_ph).sum())
        if n_valid == 0:
            continue

        w = ppo_mini_batch_size / n_valid
        for local_idx in range(ppo_mini_batch_size):
            if not chunk_ph[local_idx]:
                weights[start + local_idx] = w

    return weights


def apply_step_pool_loss_weights(
    pool: DataProto,
    ppo_mini_batch_size: int,
    mode: str = "uniform-valid",
) -> DataProto:
    """Pre-scale advantages in a step pool so that placeholders produce zero
    gradient and valid rows are uniformly weighted.

    Args:
        pool: A single step-pool DataProto (output of
            :func:`group_batch_by_step_index`).  Must have
            ``batch["advantages"]`` and
            ``non_tensor_batch["is_placeholder"]``.
        ppo_mini_batch_size: Actor's ``ppo_mini_batch_size`` config value.
        mode: Weighting mode.  Currently only ``"uniform-valid"`` is supported.

    Returns:
        DataProto with scaled ``advantages``.  Placeholder rows will have
        ``advantages == 0`` regardless of their original value.
    """
    if not mode or mode == "disabled":
        return pool

    if mode not in _STEP_POOL_SUPPORTED_MODES:
        raise ValueError(f"Unsupported step pool loss mode: '{mode}'. Supported: {sorted(_STEP_POOL_SUPPORTED_MODES)}")

    is_placeholder = pool.non_tensor_batch.get("is_placeholder", np.zeros(len(pool), dtype=bool)).astype(bool)

    weights = compute_step_pool_weights(
        is_placeholder=is_placeholder,
        ppo_mini_batch_size=ppo_mini_batch_size,
    )

    advantages = pool.batch["advantages"]
    weights = weights.to(advantages.device)

    scaled_advantages = advantages * weights.unsqueeze(1)

    new_pool = DataProto(
        batch=pool.batch.clone(recurse=False),
        non_tensor_batch=pool.non_tensor_batch,
        meta_info=pool.meta_info,
    )
    new_pool.batch["advantages"] = scaled_advantages

    return new_pool
