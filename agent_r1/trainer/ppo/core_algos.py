# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_uids: np.ndarray,
    step_indices: np.ndarray,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    device = token_level_rewards.device

    with torch.no_grad():
        # Step-level reward: sum of token rewards inside the step (only valid response tokens).
        rewards = (token_level_rewards * response_mask).sum(dim=1)

        # IMPORTANT: In our "sequence = action" setting, V_t should be the state value
        # BEFORE generating the first response token (i.e., after the last prompt token).
        # The critic (`dp_critic.py`) slices values as `values[:, -response_length-1:-1]`,
        # so `values[:, 0]` corresponds to the prompt-last position (action start).
        values = values[:, 0]

        # Map trajectories to contiguous ids for compact padding.
        # Use numpy's unique to handle both object and numeric types
        unique_traj_np, traj_inv_np = np.unique(trajectory_uids, return_inverse=True)
        num_traj = len(unique_traj_np)
        traj_inv = torch.as_tensor(traj_inv_np, dtype=torch.long, device=device)
        step_ids = torch.as_tensor(step_indices, device=device)
        max_step = int(step_ids.max().item()) + 1

        # reshape to (num_traj, max_step).
        # Use the same dtype as rewards and values to avoid type mismatch
        rewards_map = torch.zeros((num_traj, max_step), dtype=rewards.dtype, device=device)
        values_map = torch.zeros((num_traj, max_step), dtype=values.dtype, device=device)

        rewards_map[traj_inv, step_ids] = rewards
        values_map[traj_inv, step_ids] = values

        lastgaelam = 0
        advantages_reversed = []

        for t in reversed(range(max_step)):
            nextvalues = values_map[:, t + 1] if t < max_step - 1 else 0.0
            delta = rewards_map[:, t] + gamma * nextvalues - values_map[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages_map = torch.stack(advantages_reversed[::-1], dim=1)

        # Map back to batch rows and then to token level.
        advantages = advantages_map[traj_inv, step_ids]
        returns = advantages + values

        # Whiten at step-level (not token-level) to avoid counting duplicated values.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Broadcast to token level
        advantages = advantages.unsqueeze(1) * response_mask
        returns = returns.unsqueeze(1) * response_mask

    return advantages, returns


def compute_token_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_uids: np.ndarray,
    step_indices: np.ndarray,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """
    Token-level GAE for *multi-step* trajectories.

    Each row in the batch corresponds to one "agent step" (one generated sequence).
    A full trajectory is composed of multiple steps, identified by `trajectory_uids`,
    with within-trajectory ordering defined by `step_indices`.

    We compute GAE over the timeline of *LLM-generated tokens only* (where `response_mask == 1`),
    skipping tool/padding tokens (`response_mask == 0`) without advancing the GAE recursion.
    Critic values are expected to align with the "state before generating each response token",
    consistent with `verl/verl/workers/critic/dp_critic.py` slicing.

    Args:
        token_level_rewards: (bs, response_length)
        values: (bs, response_length)
        response_mask: (bs, response_length), 1 for LLM tokens (actions), 0 for tool/pad tokens
        trajectory_uids: (bs,) numpy array, same uid => same trajectory
        step_indices: (bs,) numpy array, the step index within the trajectory (0..T-1)
        gamma: discount factor
        lam: GAE lambda

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    device = token_level_rewards.device
    bsz, resp_len = token_level_rewards.shape

    with torch.no_grad():
        # Map trajectories to contiguous ids for compact padding.
        unique_traj_np, traj_inv_np = np.unique(trajectory_uids, return_inverse=True)
        num_traj = len(unique_traj_np)
        traj_inv = torch.as_tensor(traj_inv_np, dtype=torch.long, device=device)
        step_ids = torch.as_tensor(step_indices, dtype=torch.long, device=device)
        max_step = int(step_ids.max().item()) + 1 if bsz > 0 else 0

        # Build a (num_traj, max_step) table mapping (traj, step) -> batch row index.
        row_map = torch.full((num_traj, max_step), -1, dtype=torch.long, device=device)
        row_map[traj_inv, step_ids] = torch.arange(bsz, device=device, dtype=torch.long)

        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

        # Per-trajectory recursion state (the "next action token" in the future across steps).
        # IMPORTANT: keep recursion state in reward dtype (typically fp32).
        # Mixing fp32 rewards with bf16 values would otherwise promote the computation to fp32 and
        # cause dtype mismatch when writing back into bf16 tensors.
        gae_dtype = token_level_rewards.dtype
        bootstrap_value = torch.zeros((num_traj,), dtype=gae_dtype, device=device)
        lastgaelam = torch.zeros((num_traj,), dtype=gae_dtype, device=device)

        # Process steps in reverse chronological order.
        for t in reversed(range(max_step)):
            rows = row_map[:, t]  # (num_traj,)
            active = rows >= 0
            if not torch.any(active):
                continue

            idx = rows[active]  # (n_active,)
            r = token_level_rewards[idx]  # (n_active, resp_len)
            v = values[idx]  # (n_active, resp_len)  (may be bf16)
            m = response_mask[idx]  # (n_active, resp_len)
            m_bool = m.to(dtype=torch.bool)
            # Only action tokens (mask==1) participate in the token-level recursion.
            r = r * m

            # Initialize recursion for this step from the already-processed future.
            nextvalues = bootstrap_value[active].clone()  # (n_active,)
            lastgaelam_active = lastgaelam[active].clone()  # (n_active,)

            adv_step = torch.zeros_like(r)

            # Iterate tokens backwards; only update recursion on action tokens (m==1).
            for j in reversed(range(resp_len)):
                delta = r[:, j] + gamma * nextvalues - v[:, j]
                lastgaelam_ = delta + gamma * lam * lastgaelam_active

                mj = m[:, j].to(dtype=nextvalues.dtype)
                vj = v[:, j].to(dtype=nextvalues.dtype)
                nextvalues = vj * mj + (1 - mj) * nextvalues
                lastgaelam_active = lastgaelam_ * mj + (1 - mj) * lastgaelam_active
                adv_step[:, j] = lastgaelam_active

            adv_step = adv_step * m
            ret_step = (adv_step + v) * m

            advantages[idx] = adv_step
            returns[idx] = ret_step

            # Carry recursion state to the previous step (in time):
            # - lastgaelam continues across steps
            # - bootstrap_value becomes the first action token's value of this step (if any)
            has_action = m_bool.any(dim=-1)
            bootstrap_value_active = bootstrap_value[active]
            bootstrap_value_active = torch.where(has_action, nextvalues, bootstrap_value_active)
            bootstrap_value[active] = bootstrap_value_active
            lastgaelam[active] = lastgaelam_active

        # Normalize advantages over action tokens only.
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_uids: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    # NOTE:
    # - Input `token_level_rewards` are *step-level* immediate rewards distributed across tokens.
    # - GRPO needs *trajectory-level outcome* reward. For multi-step trajectories, we first sum
    #   rewards across all steps in the same trajectory, then compute GRPO groupwise advantage,
    #   and finally broadcast the advantage back to every step (and token) in that trajectory.

    # Step-level reward: sum of token rewards inside the step (only valid response tokens).
    step_scores = (token_level_rewards * response_mask).sum(dim=-1)

    # Accumulate trajectory-level outcome score.
    traj2total_score: dict[object, torch.Tensor] = {}
    traj2index: dict[object, object] = {}

    id2score = defaultdict(list)
    id2mean: dict[object, torch.Tensor] = {}
    id2std: dict[object, torch.Tensor] = {}

    with torch.no_grad():
        bsz = step_scores.shape[0]

        # 1) Sum rewards across steps for each trajectory.
        for i in range(bsz):
            traj_uid = trajectory_uids[i]
            if traj_uid in traj2total_score:
                traj2total_score[traj_uid] = traj2total_score[traj_uid] + step_scores[i]
            else:
                traj2total_score[traj_uid] = step_scores[i]
                traj2index[traj_uid] = index[i]

        # 2) Build per-group lists over trajectories (one score per trajectory).
        for traj_uid, total_score in traj2total_score.items():
            id2score[traj2index[traj_uid]].append(total_score)

        # 3) Compute per-group mean/std.
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        # 4) Normalize to GRPO advantage per trajectory, then broadcast to steps/tokens.
        traj2adv: dict[object, torch.Tensor] = {}
        for traj_uid, total_score in traj2total_score.items():
            idx = traj2index[traj_uid]
            if norm_adv_by_std_in_grpo:
                traj2adv[traj_uid] = (total_score - id2mean[idx]) / (id2std[idx] + epsilon)
            else:
                traj2adv[traj_uid] = total_score - id2mean[idx]

        scores = step_scores.clone()
        for i in range(bsz):
            scores[i] = traj2adv[trajectory_uids[i]]

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
