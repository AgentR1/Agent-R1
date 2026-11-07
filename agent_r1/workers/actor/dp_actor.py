# Copyright 2025 Agent-R1 Teams
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
Agent-R1 local PPO actor overrides.
"""

import logging
import os

import torch

from agent_r1.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn
from agent_r1.trainer.ppo.trajectory_batching import get_mini_batch_global_info, split_data_proto_by_mini_batch_id
from verl import DataProto
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor as VerlDataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(VerlDataParallelPPOActor):
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        has_planned_mini_batches = "mini_batch_id" in data.batch.keys()
        if has_planned_mini_batches:
            select_keys.extend(
                [
                    "mini_batch_id",
                    "mini_batch_global_size",
                    "mini_batch_global_token_num",
                    "mini_batch_global_response_token_num",
                    "sample_mask",
                ]
            )

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        if has_planned_mini_batches:
            mini_batches = split_data_proto_by_mini_batch_id(data)
        else:
            mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                use_global_mini_batch_info = "mini_batch_global_size" in mini_batch.batch.keys()
                if use_global_mini_batch_info:
                    global_info = get_mini_batch_global_info(mini_batch)
                    if not hasattr(self.config, "global_batch_info") or self.config.global_batch_info is None:
                        self.config.global_batch_info = {}
                    dp_size = (
                        torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
                        if torch.distributed.is_initialized()
                        else 1
                    )
                    self.config.global_batch_info.update(
                        {
                            "dp_size": dp_size,
                            "batch_num_tokens": global_info["batch_num_tokens"],
                            "global_batch_size": global_info["global_batch_size"],
                            "loss_scale_factor": self.config.loss_scale_factor,
                        }
                    )
                elif hasattr(self.config, "global_batch_info"):
                    self.config.global_batch_info.clear()

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if not has_planned_mini_batches:
                    micro_batches = [mb for mb in micro_batches if bool(mb.batch["response_mask"].any().item())]
                if not micro_batches:
                    append_to_dict(metrics, {"actor/grad_norm": 0.0})
                    continue

                if not self.config.use_dynamic_bsz:
                    self.gradient_accumulation = len(micro_batches)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if use_global_mini_batch_info:
                        loss_scale_factor = 1.0
                    elif self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    # With dp > 1, trajectory batching pads ranks with rows whose response_mask is all zero, so a
                    # micro-batch can contain only padding. The rollout-corr helper asserts at least one valid
                    # token; skip this diagnostic metric for such micro-batches (the loss path is unchanged).
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None and response_mask.any():
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            **getattr(self.config, "global_batch_info", {}),
                        )
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            **getattr(self.config, "global_batch_info", {}),
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
