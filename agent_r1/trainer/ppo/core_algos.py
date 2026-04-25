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
from typing import Any, Optional

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


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """Aggregate loss with pad-aware sequence counting and zero-safe empty-batch handling."""

    def is_zero_denom(denom):
        if torch.is_tensor(denom):
            return denom.detach().item() == 0
        return denom == 0

    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            denom = loss_mask.sum()
        else:
            denom = batch_num_tokens
        if is_zero_denom(denom):
            return loss_mat.sum() * 0.0
        loss = verl_F.masked_sum(loss_mat, loss_mask) / denom * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            denom = seq_mask.sum()
        else:
            denom = global_batch_size
        if is_zero_denom(denom):
            return loss_mat.sum() * 0.0
        loss = verl_F.masked_sum(seq_losses, seq_mask) / denom * dp_size
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_token_count = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_token_count + 1e-8)
        seq_mask = (seq_token_count > 0).float()
        if global_batch_size is None:
            denom = seq_mask.sum()
        else:
            denom = global_batch_size
        if is_zero_denom(denom):
            return loss_mat.sum() * 0.0
        loss = verl_F.masked_sum(seq_losses, seq_mask) / denom * dp_size
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        if loss_scale_factor == 0:
            return loss_mat.sum() * 0.0
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """Local value loss that uses pad-aware `agg_loss`."""
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[Any] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **getattr(config, "global_batch_info", {}),
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[Any] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    if rollout_is_weights is not None:
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        pg_losses = -advantages * log_prob

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **getattr(config, "global_batch_info", {}),
    )

    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[Any] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"
    del rollout_is_weights

    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None
    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", True)

    rollout_log_prob = old_log_prob
    rollout_metrics, modified_response_mask, rollout_is_weights_proto = compute_rollout_correction_and_rejection_mask(
        old_log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
        rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        rollout_token_veto_threshold=rollout_token_veto_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
    )

    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None
    effective_mask = modified_response_mask

    if loss_type == "reinforce":
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )
    elif loss_type == "ppo_clip":
        pg_loss, pg_metrics = compute_policy_loss_vanilla(
            old_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,
        )
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    pg_metrics.update(rollout_metrics)
    return pg_loss, pg_metrics


def get_policy_loss_fn(name: str):
    local_policy_loss_fns = {
        "vanilla": compute_policy_loss_vanilla,
        "reinforce": compute_policy_loss_reinforce,
        "bypass_mode": compute_policy_loss_bypass_mode,
    }
    if name in local_policy_loss_fns:
        return local_policy_loss_fns[name]

    from verl.trainer.ppo.core_algos import get_policy_loss_fn as upstream_get_policy_loss_fn

    return upstream_get_policy_loss_fn(name)
