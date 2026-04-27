# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Agent-R1 local PPO critic overrides.
"""

import logging
import os

from agent_r1.trainer.ppo.core_algos import compute_value_loss
from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.workers.critic.dp_critic import DataParallelPPOCritic as VerlDataParallelPPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(VerlDataParallelPPOCritic):
    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        self.critic_module.train()
        metrics = {
            "critic/vf_loss": 0.0,
        }

        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                micro_batches = [mb for mb in micro_batches if bool(mb.batch["response_mask"].any().item())]
                if not micro_batches:
                    append_to_dict(metrics, {"critic/grad_norm": 0.0})
                    continue

                if not self.config.use_dynamic_bsz:
                    self.gradient_accumulation = len(micro_batches)

                self.critic_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    vpreds = self._forward_micro_batch(model_inputs)
                    vf_loss, vf_clipfrac = compute_value_loss(
                        vpreds=vpreds,
                        values=values,
                        returns=returns,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    loss = vf_loss * loss_scale_factor
                    loss.backward()

                    append_to_dict(
                        metrics,
                        {
                            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                        },
                    )
                    metrics["critic/vf_loss"] += vf_loss.detach().item() * loss_scale_factor

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})

        self.critic_optimizer.zero_grad()
        return metrics
