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
Step-based mini-batch utilities.

Provides functions for reorganising a step-level DataProto batch (where each
row = one agent step) into per-step-index *step pools*, so that the PPO update
loop can iterate over step indices in reverse order (T -> T-1 -> ... -> 0).

Design constraints
------------------
- Each step pool ``D_t`` contains **all** rows whose ``step_indices == t``,
  plus placeholder rows for trajectories shorter than ``t + 1`` steps.
- Placeholder rows have ``response_mask == 0``, ``attention_mask == 0``, and
  ``advantages == 0`` so they never contribute to policy loss, advantage
  normalisation, or transformer attention.
- ``trajectory_uids`` is preserved on every row (including placeholders) so
  that downstream trajectory-level credit / metrics remain valid.
- Pure functions only: no trainer state, no global side effects.
"""

from __future__ import annotations

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor

_MASK_FIELDS_TO_ZERO = ("response_mask", "attention_mask")
_ADVANTAGE_FIELDS_TO_ZERO = ("advantages",)


def group_batch_by_step_index(
    batch: DataProto,
    max_steps: int | None = None,
    ppo_mini_batch_size: int | None = None,
) -> list[DataProto]:
    """Partition a step-level batch into per-step-index pools.

    Args:
        batch: Full rollout batch where each row is one agent step.
            Must contain ``non_tensor_batch["step_indices"]`` and
            ``non_tensor_batch["trajectory_uids"]``.
        max_steps: Number of step pools to create (0 .. max_steps-1).
            ``None`` auto-infers from the data (``max(step_indices) + 1``).
        ppo_mini_batch_size: If provided, each step pool is padded to be
            divisible by this value (needed by the downstream worker split).

    Returns:
        A list of ``max_steps`` :class:`DataProto` objects.  ``step_pools[t]``
        contains all rows with ``step_indices == t`` plus placeholder rows for
        trajectories that have fewer than ``t + 1`` steps.  Placeholder rows
        are marked with ``non_tensor_batch["is_placeholder"] == True``.

    Raises:
        ValueError: If required non-tensor fields are missing.
    """
    if "step_indices" not in batch.non_tensor_batch:
        raise ValueError("batch.non_tensor_batch must contain 'step_indices'")
    if "trajectory_uids" not in batch.non_tensor_batch:
        raise ValueError("batch.non_tensor_batch must contain 'trajectory_uids'")

    step_indices = batch.non_tensor_batch["step_indices"].astype(np.int32)
    traj_uids = batch.non_tensor_batch["trajectory_uids"]
    is_pad = batch.non_tensor_batch.get(
        "is_pad", np.zeros(len(batch), dtype=bool)
    ).astype(bool)

    if max_steps is None:
        valid_steps = step_indices[~is_pad]
        max_steps = int(valid_steps.max()) + 1 if len(valid_steps) > 0 else 1

    # Build trajectory -> max_step_index mapping (valid rows only).
    traj_max_step: dict[object, int] = {}
    for i in range(len(batch)):
        if is_pad[i]:
            continue
        uid = traj_uids[i]
        s = int(step_indices[i])
        if uid not in traj_max_step or s > traj_max_step[uid]:
            traj_max_step[uid] = s

    all_traj_uids = list(traj_max_step.keys())

    # Pick a template row for placeholder creation (prefer an existing pad row).
    pad_indices = np.where(is_pad)[0]
    if len(pad_indices) > 0:
        template_idx = int(pad_indices[0])
    else:
        template_idx = 0

    step_pools: list[DataProto] = []

    for t in range(max_steps):
        # Collect global indices of valid rows at step t.
        real_indices: list[int] = []
        present_uids: set[object] = set()
        for i in range(len(batch)):
            if is_pad[i]:
                continue
            if int(step_indices[i]) == t:
                real_indices.append(i)
                present_uids.add(traj_uids[i])

        # Trajectories that need a placeholder at step t: those with
        # max_step < t (they don't have a real row for this step index).
        need_placeholder_uids = [
            uid for uid in all_traj_uids
            if uid not in present_uids and traj_max_step[uid] < t
        ]

        n_real = len(real_indices)
        n_placeholder = len(need_placeholder_uids)
        pool_indices = real_indices + [template_idx] * n_placeholder
        is_placeholder_flags = [False] * n_real + [True] * n_placeholder

        pool = batch.select_idxs(np.array(pool_indices, dtype=np.int64))

        # Stamp metadata on the pool.
        pool.non_tensor_batch["is_placeholder"] = np.array(
            is_placeholder_flags, dtype=bool
        )
        # Overwrite trajectory_uids on placeholder rows so they still link
        # back to their originating trajectory.
        for k, uid in enumerate(need_placeholder_uids):
            pool.non_tensor_batch["trajectory_uids"][n_real + k] = uid
        # Overwrite step_indices on placeholder rows.
        for k in range(n_placeholder):
            pool.non_tensor_batch["step_indices"][n_real + k] = np.int32(t)

        # Zero out masks and advantages on placeholder rows.
        if pool.batch is not None and n_placeholder > 0:
            ph_mask = torch.zeros(len(pool), dtype=torch.bool)
            ph_mask[n_real:] = True
            for field in _MASK_FIELDS_TO_ZERO:
                if field in pool.batch.keys():
                    pool.batch[field][ph_mask] = 0
            for field in _ADVANTAGE_FIELDS_TO_ZERO:
                if field in pool.batch.keys():
                    pool.batch[field][ph_mask] = 0

        # Pad to be divisible by ppo_mini_batch_size if requested.
        if ppo_mini_batch_size is not None and len(pool) > 0:
            pool = _pad_step_pool(pool, ppo_mini_batch_size)

        step_pools.append(pool)

    return step_pools


def _pad_step_pool(pool: DataProto, ppo_mini_batch_size: int) -> DataProto:
    """Pad a step pool so its length is divisible by ``ppo_mini_batch_size``.

    Uses the same pattern as trajectory_batching: copy a template row and zero
    out masks.  Padding rows are marked ``is_placeholder = True``.
    """
    remainder = len(pool) % ppo_mini_batch_size
    if remainder == 0:
        return pool

    n_pad = ppo_mini_batch_size - remainder
    original_len = len(pool)

    pool, _ = pad_dataproto_to_divisor(pool, ppo_mini_batch_size)

    # Extend is_placeholder array.
    old_flags = pool.non_tensor_batch.get(
        "is_placeholder",
        np.zeros(original_len, dtype=bool),
    )
    new_flags = np.concatenate([
        old_flags[:original_len],
        np.ones(n_pad, dtype=bool),
    ])
    pool.non_tensor_batch["is_placeholder"] = new_flags

    # Zero out mask / advantage fields on the newly padded rows.
    if pool.batch is not None and n_pad > 0:
        pad_mask = torch.zeros(len(pool), dtype=torch.bool)
        pad_mask[original_len:] = True
        for field in _MASK_FIELDS_TO_ZERO:
            if field in pool.batch.keys():
                pool.batch[field][pad_mask] = 0
        for field in _ADVANTAGE_FIELDS_TO_ZERO:
            if field in pool.batch.keys():
                pool.batch[field][pad_mask] = 0

    return pool

