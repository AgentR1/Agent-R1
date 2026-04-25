# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from agent_r1.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        loss_mask = data["loss_mask"]
        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def _slice_response_from_unpad_output(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        max_response_len = response_ids.offsets().max().item()
    else:
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]

    response_list = []
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (0, pad_size)))

    return torch.stack(response_list, dim=0)


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = _slice_response_from_unpad_output(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = _slice_response_from_unpad_output(entropy, data)

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}
    response_mask = data["response_mask"].to(bool)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)
    loss_agg_mode = config.loss_agg_mode
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = pg_loss.detach().item()
    policy_loss = pg_loss

    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        policy_loss -= config.entropy_coeff * entropy_loss

    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )
        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = kl_loss.detach().item()
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    vpreds = _slice_response_from_unpad_output(model_output["values"], data)
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {
        "critic/vf_loss": vf_loss.detach().item(),
        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics
