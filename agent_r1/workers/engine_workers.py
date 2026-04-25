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
Thin new-engine worker wrappers that swap in Agent-R1 local losses.
"""

from functools import partial
from itertools import chain

import torch
from codetiming import Timer
from tensordict import NonTensorData, TensorDict

from agent_r1.workers.utils.losses import ppo_loss
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.py_functional import append_to_dict
from verl.workers.config import ActorConfig
from verl.workers.engine import utils as engine_utils
from verl.workers.engine_workers import ActorRolloutRefWorker as VerlActorRolloutRefWorker
from verl.workers.engine_workers import TrainingWorker as VerlTrainingWorker


_ORIGINAL_PREPARE_MICRO_BATCHES = engine_utils.prepare_micro_batches


def _prepare_micro_batches(
    data: TensorDict,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    # Keep verl's dynamic path unchanged. For static micro-batching, allow a short
    # tail batch and sync the micro-batch count across DP ranks before slicing.
    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    if use_dynamic_bsz:
        return _ORIGINAL_PREPARE_MICRO_BATCHES(
            data=data,
            dp_group=dp_group,
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=same_micro_num_in_dp,
            min_num_micro_batch=min_num_micro_batch,
            use_dynamic_bsz_balance=use_dynamic_bsz_balance,
        )

    micro_batch_size_per_gpu = data["micro_batch_size_per_gpu"]
    assert micro_batch_size_per_gpu > 0, f"micro_batch_size_per_gpu must be positive, got {micro_batch_size_per_gpu}"
    num_micro_batches = (len(data) + micro_batch_size_per_gpu - 1) // micro_batch_size_per_gpu

    if torch.distributed.is_initialized() and same_micro_num_in_dp:
        num_micro_batches_tensor = torch.tensor([num_micro_batches], dtype=torch.long, device=data["input_ids"].device)
        torch.distributed.all_reduce(num_micro_batches_tensor, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        num_micro_batches = int(num_micro_batches_tensor.cpu().item())

    if num_batches_divided_by is not None:
        num_micro_batches = ((num_micro_batches + num_batches_divided_by - 1) // num_batches_divided_by) * (
            num_batches_divided_by
        )

    if num_micro_batches > len(data):
        raise ValueError(
            f"num_micro_batches ({num_micro_batches}) must be <= local batch size ({len(data)}) "
            "when use_dynamic_bsz is disabled"
        )

    micro_batches = [
        tu.index_select_tensor_dict(
            data,
            list(
                range(
                    micro_batch_id * len(data) // num_micro_batches,
                    (micro_batch_id + 1) * len(data) // num_micro_batches,
                )
            ),
        )
        for micro_batch_id in range(num_micro_batches)
    ]
    return micro_batches, None


def _install_prepare_micro_batches_patch() -> None:
    from verl.workers.engine.fsdp import transformer_impl as fsdp_transformer_impl

    engine_utils.prepare_micro_batches = _prepare_micro_batches
    fsdp_transformer_impl.prepare_micro_batches = _prepare_micro_batches


_install_prepare_micro_batches_patch()


class TrainingWorker(VerlTrainingWorker):
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        if "mini_batch_id" not in data.keys():
            return super().train_mini_batch(data)

        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})
        mini_batch_ids = data.pop("mini_batch_id").to(dtype=torch.long)
        mini_batch_global_sizes = data.pop("mini_batch_global_size").to(dtype=torch.long)
        mini_batch_global_token_nums = data.pop("mini_batch_global_token_num").to(dtype=torch.long)

        assert mini_batch_size is not None or num_mini_batch is not None
        assert dataloader_kwargs.keys() <= {"shuffle"}, f"Unsupported dataloader_kwargs: {dataloader_kwargs.keys()}"

        unique_mini_batch_ids = torch.unique(mini_batch_ids, sorted=True).cpu()
        total_num_iterations = len(unique_mini_batch_ids) * epochs
        shuffle = dataloader_kwargs.get("shuffle", False)

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            output_lst = []
            iteration_idx = 0
            for epoch in range(epochs):
                epoch_mini_batch_ids = unique_mini_batch_ids
                if shuffle:
                    generator = torch.Generator()
                    generator.manual_seed(seed + epoch)
                    permutation = torch.randperm(len(epoch_mini_batch_ids), generator=generator)
                    epoch_mini_batch_ids = epoch_mini_batch_ids[permutation]

                for mini_batch_id in epoch_mini_batch_ids:
                    indices = torch.nonzero(mini_batch_ids.cpu() == mini_batch_id, as_tuple=False).flatten()
                    mini_batch_td = tu.index_select_tensor_dict(data, indices)

                    global_token_num = mini_batch_global_token_nums[indices[0]]
                    global_token_num = global_token_num[global_token_num > 0].tolist()
                    global_batch_size = int(mini_batch_global_sizes[indices[0]].item())

                    tu.assign_non_tensor(
                        mini_batch_td,
                        global_token_num=NonTensorData(global_token_num),
                        global_batch_size=global_batch_size,
                        update_lr_scheduler=iteration_idx == total_num_iterations - 1,
                        disable_auto_offload=True,
                    )
                    output_lst.append(self.train_batch(mini_batch_td))
                    iteration_idx += 1

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        if isinstance(val, list):
                            output[key] = list(chain.from_iterable(val))
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output


class ActorRolloutRefWorker(VerlActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import verl.workers.engine_workers as upstream_engine_workers

        original_training_worker = upstream_engine_workers.TrainingWorker
        upstream_engine_workers.TrainingWorker = TrainingWorker
        try:
            super().init_model()
        finally:
            upstream_engine_workers.TrainingWorker = original_training_worker

        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = self.config.model
            self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor.set_loss_fn(self.loss_fn)
