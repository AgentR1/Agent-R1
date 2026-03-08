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
import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from agent_r1.reward_loop import RewardLoopWorker
from verl.experimental.agent_loop.agent_loop import (
    AsyncLLMServerManager,
    DictConfigWrap,
)
from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentFlowMetrics(BaseModel):
    """Agent flow performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentFlowStep(BaseModel):
    """Agent flow step."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    input_ids: Optional[list[int]] = None
    """Input token ids (prompt_ids + response_ids)."""
    position_ids: Optional[list[int]] = None
    """Position ids."""
    attention_mask: Optional[list[int]] = None
    """Attention mask."""
    response_mask: Optional[list[int]] = None
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the step."""
    num_turns: int = 2
    """Number of chat turns, including user, assistant, tool."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentFlowStep(AgentFlowStep):
    """Internal agent flow step with padded sequences and processed multi-modal data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class AgentFlowOutput(BaseModel):
    """Agent flow output."""

    steps: list[_InternalAgentFlowStep]
    """List of agent flow steps."""
    metrics: AgentFlowMetrics
    """Auxiliary performance metrics"""


class AgentFlowBase(ABC):
    """An agent flow takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfig,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (DictConfigWrap): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            reward_loop_worker (RewardLoopWorker): Reward loop worker.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
            dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
            dataset_config (DictConfig): Dataset config.
        """
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.reward_loop_worker = reward_loop_worker
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config
        self.apply_chat_template_kwargs = dataset_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Extract images and videos from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys "images" and "videos".
        """
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(
                messages, image_patch_size=self.processor.image_processor.patch_size, config=self.dataset_config
            )
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            # split the videos and according metadatas
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadatas=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentFlowOutput: Agent flow output.
        """
        raise NotImplementedError

    async def _postprocess(self, step: AgentFlowStep, **kwargs) -> _InternalAgentFlowStep:
        step.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": step.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": step.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        # Use provided response_mask or default to all 1s (all tokens are LLM generated)
        response_mask_ids = step.response_mask if step.response_mask is not None else [1] * len(step.response_ids)
        response_mask_output = self.tokenizer.pad(
            {"input_ids": response_mask_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if step.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(step.response_logprobs)
            response_logprobs = torch.tensor(step.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if step.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = step.routed_experts.shape
            experts_tensor = torch.from_numpy(step.routed_experts)
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(step.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(step, input_ids)
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)
        await self._compute_score(
            step,
            prompts=prompt_output["input_ids"],
            responses=response_output["input_ids"],
            attention_mask=attention_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            kwargs=kwargs,
        )

        assert step.reward_score is not None, "Reward score is required for agent flow"

        return _InternalAgentFlowStep(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            response_logprobs=response_logprobs,
            response_mask=response_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=step.multi_modal_data,
            reward_score=step.reward_score,
            num_turns=step.num_turns,
            extra_fields=step.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        images = output.multi_modal_data.get("images")
        videos = output.multi_modal_data.get("videos")
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = self.processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        """Compute reward score for single sample."""
        if output.reward_score is None:
            batch = TensorDict(
                {
                    "prompts": prompts,  # [1, prompt_length]
                    "responses": responses,  # [1, response_length]
                    "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                    "input_ids": input_ids,  # [1, prompt_length + response_length]
                    "position_ids": position_ids,
                },
                batch_size=1,
            )
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in kwargs.items()},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }

            data = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
            )
            result = await self.reward_loop_worker.compute_score.remote(data)
            output.reward_score = result["reward_score"]
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]


"""Agent flow registry: key is agent_name, value is a dict of agent flow config
used by hydra.utils.instantiate to initialize agent flow instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_flow_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent flow class."""

    def decorator(subclass: type[AgentFlowBase]) -> type[AgentFlowBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_flow_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentFlowWorkerBase:
    """Agent flow worker takes a batch of messages and run each message in an agent flow."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        """Initialize agent flow manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config

        # for recipe to change
        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_router_address = reward_router_address

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_flow_config_path = config.actor_rollout_ref.rollout.agent.agent_flow_config_path
        if agent_flow_config_path:
            resolved_path = resolve_config_path(agent_flow_config_path)
            agent_flow_configs = OmegaConf.load(resolved_path)
            for agent_flow_config in agent_flow_configs:
                _agent_flow_registry[agent_flow_config.name] = agent_flow_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_loop_worker = RewardLoopWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, self.reward_router_address)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_flow = config.agent.default_agent_flow
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_flow] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_flow(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)
        return output

    async def _run_agent_flow(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> AgentFlowOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_flow",
            trace=trace,
        ):
            assert agent_name in _agent_flow_registry, (
                f"Agent flow {agent_name} not registered, registered agent flows: {_agent_flow_registry.keys()}"
            )

            agent_flow_config = _agent_flow_registry[agent_name]
            agent_flow = hydra.utils.instantiate(
                config=agent_flow_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                reward_loop_worker=self.reward_loop_worker,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            output: AgentFlowOutput = await agent_flow.run(sampling_params, **kwargs)

            return output

    def _postprocess(self, inputs: list[AgentFlowOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_flow and combine them into a batch."""
        num_steps = []
        trajectory_uids = []
        step_indices = []
        prompt_ids = []
        response_ids = []
        response_mask = []
        attention_mask = []
        input_ids = []
        position_ids = []
        multi_modal_data = []
        multi_modal_inputs = []
        num_turns = []
        reward_tensors = []
        response_logprobs_list = []
        routed_experts_list = []
        for input in inputs:
            num_step = len(input.steps)
            num_steps.append(num_step)
            trajectory_uids.extend([uuid4().hex] * num_step)
            step_indices.extend(range(num_step))
            for step in input.steps:
                prompt_ids.append(step.prompt_ids)
                response_ids.append(step.response_ids)
                response_mask.append(step.response_mask)
                attention_mask.append(step.attention_mask)
                input_ids.append(step.input_ids)
                position_ids.append(step.position_ids)
                multi_modal_data.append(step.multi_modal_data)
                multi_modal_inputs.append(step.multi_modal_inputs)
                num_turns.append(step.num_turns)
                response_logprobs_list.append(step.response_logprobs)
                routed_experts_list.append(step.routed_experts)
                if step.reward_score is not None:
                    reward_tensor = torch.zeros_like(step.response_mask, dtype=torch.float32)
                    valid_length = step.response_mask.sum().item()
                    reward_tensor[0, valid_length - 1] = float(step.reward_score)
                    reward_tensors.append(reward_tensor)
                else:
                    reward_tensors.append(None)

        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat(prompt_ids, dim=0)
        response_ids = torch.cat(response_ids, dim=0)
        response_mask = torch.cat(response_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)

        # Handle optional outputs
        optional_outputs = {}
        if all(logprobs is not None for logprobs in response_logprobs_list):
            optional_outputs["rollout_log_probs"] = torch.cat(response_logprobs_list, dim=0)
        if all(routed_experts is not None for routed_experts in routed_experts_list):
            optional_outputs["routed_experts"] = torch.cat(routed_experts_list, dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=prompt_ids.size(0),
        )

        if all(reward_tensor is not None for reward_tensor in reward_tensors):
            reward_tensor = torch.cat(reward_tensors, dim=0)
            batch["rm_scores"] = reward_tensor

        non_tensor_batch = {
            "trajectory_uids": np.array(trajectory_uids, dtype=object),
            "step_indices": np.array(step_indices, dtype=np.int32),
            "__num_turns__": np.array(num_turns, dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = []
        for input in inputs:
            for step in input.steps:
                reward_extra_infos.append(step.extra_fields.get("reward_extra_info", {}))

        all_reward_keys = set()
        for info in reward_extra_infos:
            all_reward_keys.update(info.keys())
        reward_extra_keys = sorted(all_reward_keys)
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info.get(key) for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        if any(mmi is not None for mmi in multi_modal_inputs):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]

        # TODO: 验证 metrics 格式
        # Add num_steps to each metric dict for proper aggregation during concat
        for i, metric in enumerate(metrics):
            metric["num_steps"] = num_steps[i]

        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(
            key
            for input_item in inputs
            for step in input_item.steps
            for key in step.extra_fields
            if key != "reward_extra_info"  # already handled above
        )
        for key in all_keys:
            temp_list = []
            for input_item in inputs:
                for step in input_item.steps:
                    temp_list.append(step.extra_fields.get(key))
            extra_fields[key] = np.array(temp_list, dtype=object)

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    def create_transferqueue_client(
        self,
    ):
        """Create a client for data system (TransferQueue)."""
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)

        self.tq_client = create_transferqueue_client(
            client_id=f"AgentLoopWorker_{client_name}",
            config=self.config.transfer_queue,
        )


@ray.remote
class AgentFlowWorker(AgentFlowWorkerBase):
    """Agent flow worker takes a batch of messages and run each message in an agent flow."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        """Initialize agent flow manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        super().__init__(config, server_handles, reward_router_address)


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentFlowManager:
    """Agent flow manager that manages a group of agent flow workers."""

    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ):
        """Initialize agent flow manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rm_resource_pool (RayResourcePool): Resource pool for reward model (Standalone mode).
        """
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable:
            from verl.experimental.reward_loop import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_flow_workers_class"):
            self.agent_flow_workers_class = AgentFlowWorker

        self._initialize_llm_servers()
        self._init_agent_flow_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentFlowManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses)

    def _init_agent_flow_workers(self):
        self.agent_flow_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_flow_workers.append(
                self.agent_flow_workers_class.options(
                    name=f"agent_flow_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        split_size = (len(prompts) - 1) // len(self.agent_flow_workers) + 1
        chunks = prompts.split(split_size)
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_flow_workers, chunks, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]

        # Extract num_steps from metrics for each request
        num_steps = [metric["num_steps"] for chunk in metrics for metric in chunk]
        timing = self._performance_metrics(metrics, num_steps, output)

        output.meta_info = {"timing": timing, "num_steps": num_steps, **outputs[0].meta_info}
        return output

    def _performance_metrics(
        self, metrics: list[list[dict[str, str]]], num_steps: list[int], output: DataProto
    ) -> dict[str, float]:
        timing = {}

        # Extract step-level timing from metrics
        # Each metric dict corresponds to one trajectory, containing step-level timing data
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])

        # Step-level statistics (each number corresponds to one step)
        timing["agent_flow/step/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_flow/step/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_flow/step/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_flow/step/tool_calls/min"] = t_tool_calls.min()
        timing["agent_flow/step/tool_calls/max"] = t_tool_calls.max()
        timing["agent_flow/step/tool_calls/mean"] = t_tool_calls.mean()

        # Trajectory-level statistics - aggregate step times by trajectory
        # num_steps: [3, 2, 3] means 3 trajectories with 3, 2, 3 steps respectively
        trajectory_generate_times = []
        trajectory_tool_times = []
        trajectory_total_times = []
        idx = 0
        for n in num_steps:
            traj_gen_time = t_generate_sequences[idx : idx + n].sum()
            traj_tool_time = t_tool_calls[idx : idx + n].sum()
            trajectory_generate_times.append(traj_gen_time)
            trajectory_tool_times.append(traj_tool_time)
            trajectory_total_times.append(traj_gen_time + traj_tool_time)
            idx += n

        trajectory_generate_times = np.array(trajectory_generate_times)
        trajectory_tool_times = np.array(trajectory_tool_times)
        trajectory_total_times = np.array(trajectory_total_times)

        timing["agent_flow/trajectory/generate_sequences/min"] = trajectory_generate_times.min()
        timing["agent_flow/trajectory/generate_sequences/max"] = trajectory_generate_times.max()
        timing["agent_flow/trajectory/generate_sequences/mean"] = trajectory_generate_times.mean()
        timing["agent_flow/trajectory/tool_calls/min"] = trajectory_tool_times.min()
        timing["agent_flow/trajectory/tool_calls/max"] = trajectory_tool_times.max()
        timing["agent_flow/trajectory/tool_calls/mean"] = trajectory_tool_times.mean()
        timing["agent_flow/trajectory/total/min"] = trajectory_total_times.min()
        timing["agent_flow/trajectory/total/max"] = trajectory_total_times.max()
        timing["agent_flow/trajectory/total/mean"] = trajectory_total_times.mean()

        # Slowest trajectory (bounded by total trajectory time, not step time)
        slowest_traj_idx = np.argmax(trajectory_total_times)
        # Find the step index range of the slowest trajectory in the flattened step array
        slowest_step_start_idx = sum(num_steps[:slowest_traj_idx])
        slowest_step_end_idx = slowest_step_start_idx + num_steps[slowest_traj_idx]

        # Calculate total prompt and response length for the slowest trajectory
        prompt_length = output.batch["prompts"].shape[1]
        total_prompt_length = 0
        total_response_length = 0
        for step_idx in range(slowest_step_start_idx, slowest_step_end_idx):
            attention_mask = output.batch["attention_mask"][step_idx]
            total_prompt_length += attention_mask[:prompt_length].sum().item()
            total_response_length += attention_mask[prompt_length:].sum().item()

        timing["agent_flow/slowest/num_steps"] = num_steps[slowest_traj_idx]
        timing["agent_flow/slowest/total_prompt_length"] = total_prompt_length
        timing["agent_flow/slowest/total_response_length"] = total_response_length

        return timing

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
