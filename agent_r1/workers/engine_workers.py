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
from verl.workers.engine_workers import ActorRolloutRefWorker as VerlActorRolloutRefWorker
from verl.workers.engine_workers import TrainingWorker as VerlTrainingWorker


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

                    global_token_num = mini_batch_global_token_nums[indices[0]].tolist()
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
