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

import aiohttp
import ray
from omegaconf import DictConfig

from verl.experimental.reward_loop.reward_loop import get_reward_manager_cls
from verl.protocol import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class RewardLoopWorker:
    def __init__(self, config: DictConfig, reward_router_address: str = None):
        """
        RewardLoopWork can tackle reward computation:
        (1) rule-based reward computation
        (2) reward model-based reward computation (both disrm and genrm)
        (3) high-flexible user-customized reward function (can access rm by posting requests to reward_model_router)

        Reward Computation Logic:
        - if user-customized reward function is provided:
            -> directly use user-customized reward function
        - if user-customized reward function is not provided:
            -> rm is not enabled: use default rule-based reward function
            -> rm is disrm: compute reward score using disrm
            -> rm is genrm: raise error (user-costomized reward func must be provided)

        Args:
            config: DictConfig, the config for reward loop worker.
            reward_router_address: str, the address of reward router.
        """
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn()

    def _init_reward_fn(self):
        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)
        self.reward_fn = get_custom_reward_fn(self.config)

        # Load reward loop manager class
        # Support both registry and importlib loading methods
        reward_loop_source = self.config.reward_model.get("reward_loop_source", "register")

        if reward_loop_source == "register":
            # Load from registry (default behavior)
            reward_manager_cls = get_reward_manager_cls(self.config.reward_model.reward_manager)
        elif reward_loop_source == "importlib":
            # Load from external module using importlib
            from verl.utils.import_utils import load_extern_object

            reward_loop_module_path = self.config.reward_model.get("reward_loop_module_path", None)
            reward_loop_class_name = self.config.reward_model.get("reward_loop_class_name", None)

            assert reward_loop_module_path is not None, (
                "reward_loop_module_path must be set when reward_loop_source='importlib'"
            )
            assert reward_loop_class_name is not None, (
                "reward_loop_class_name must be set when reward_loop_source='importlib'"
            )

            reward_manager_cls = load_extern_object(
                module_path=reward_loop_module_path, object_name=reward_loop_class_name
            )
        else:
            raise ValueError(f"Unknown reward_loop_source: {reward_loop_source}. Must be 'register' or 'importlib'")

        self.reward_loop = reward_manager_cls(
            self.config, self.input_tokenizer, self.reward_fn, self.reward_router_address, self.reward_model_tokenizer
        )

    async def compute_score_batch(self, data: DataProto) -> list[dict]:
        tasks = []
        for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
        outputs = await asyncio.gather(*tasks)
        return outputs

    async def compute_score(self, input_data) -> dict:
        """Compute reward score.

        Supports verl-native `DataProto` inputs, and (for DisRM only) DataProto-less inputs:
        - `str`: already-prepared classifier prompt
        - `list[dict]`: chat messages to be templated by `reward_model_tokenizer`
        """
        # verl-native path.
        if isinstance(input_data, DataProto):
            data = input_data
            assert len(data) == 1, "RewardLoopWorker only support single data item"
            if self.config.custom_reward_function.path is not None:
                # directly use user-customized reward function
                return await self.reward_loop.run_single(data)
            else:
                if self.config.reward_model.enable:
                    # we assume the rm is disrm
                    # genrm must set custom_reward_function
                    return await self.compute_score_disrm(data)
                else:
                    return await self.reward_loop.run_single(data)

        # For now, only support DataProto-less inputs for DisRM.
        if getattr(self.config, "custom_reward_function", None) is not None and self.config.custom_reward_function.path:
            raise NotImplementedError(
                "RewardLoopWorker currently supports non-DataProto inputs only in the DisRM path. "
                "When custom_reward_function is configured, you must pass DataProto."
            )
        if not self.config.reward_model.enable:
            raise NotImplementedError(
                "RewardLoopWorker only supports non-DataProto inputs in DisRM mode. "
                "Set reward_model.enable=True (and do not use custom_reward_function), or pass DataProto."
            )

        return await self.compute_score_disrm(input_data)

    async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 16):
        url = f"http://{self.reward_router_address}/{endpoint}"
        last_exception = None
        for attempt in range(max_retries):
            try:
                # It's safer to have a timeout instead of None, which can hang indefinitely.
                timeout = aiohttp.ClientTimeout(total=None)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientResponseError as e:
                # Do not retry on 4xx client errors, but retry on 5xx server errors.
                if 400 <= e.status < 500:
                    logger.error(f"Request to {url} failed with client error HTTP {e.status}: {e}. Not retrying.")
                    raise
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with HTTP {e.status}: {e}. "
                    "Retrying..."
                )
            except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
                last_exception = e
                logger.warning(f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed: {e}. Retrying...")
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with unexpected error: {e}. "
                    "Retrying..."
                )

            if attempt < max_retries - 1:
                # Using exponential backoff is generally better than a fixed sleep.
                backoff_seconds = 2**attempt
                await asyncio.sleep(min(backoff_seconds, 30))

        logger.error(f"Max retries ({max_retries}) reached for request to {url}.")
        if last_exception:
            raise last_exception

    async def _preprocess_reward_inputs(self, data: DataProto) -> str:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        data_item = data[0]
        assert "raw_prompt" in data_item.non_tensor_batch

        # extract raw prompt
        chat: list = list(data_item.non_tensor_batch["raw_prompt"])

        # extract response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        rollout_response = self.input_tokenizer.decode(valid_response_ids)
        # remove bos and eos
        rollout_response = rollout_response.replace(self.input_tokenizer.eos_token, "")

        chat.append({"role": "assistant", "content": rollout_response})

        rm_prompt = self.reward_model_tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            tokenize=False,
        )

        # llama tokenizer will add bos token by default
        # will be removed in vllm >= 0.11.2, where we can add "add_special_tokens" = False
        if self.reward_model_tokenizer.bos_token is not None and rm_prompt.startswith(
            self.reward_model_tokenizer.bos_token
        ):
            rm_prompt = rm_prompt[len(self.reward_model_tokenizer.bos_token) :]

        return rm_prompt

    async def compute_score_disrm(self, input_data) -> dict:
        # verl-native path.
        if isinstance(input_data, DataProto):
            data = input_data
            disrm_prompt = await self._preprocess_reward_inputs(data)
        else:
            # Build DisRM prompt.
            disrm_prompt: str
            if isinstance(input_data, str):
                # Treat as already-prepared prompt for classifier.
                disrm_prompt = input_data
            elif isinstance(input_data, list):
                if self.reward_model_tokenizer is None:
                    raise RuntimeError(
                        "reward_model_tokenizer is not initialized; cannot accept messages input for DisRM."
                    )
                disrm_prompt = self.reward_model_tokenizer.apply_chat_template(
                    input_data,
                    add_generation_prompt=False,
                    tokenize=False,
                )

                bos = getattr(self.reward_model_tokenizer, "bos_token", None)
                if bos and isinstance(bos, str) and disrm_prompt.startswith(bos):
                    disrm_prompt = disrm_prompt[len(bos) :]
            else:
                raise TypeError(
                    f"Unsupported input type for DisRM: {type(input_data)}. "
                    "Supported: DataProto | str | messages(list[dict])"
                )

        engine_name = self.config.reward_model.rollout.name
        model_name = self.config.reward_model.model.path
        if engine_name == "vllm":
            # TODO (dyy): the "activation" has been changed to "use_activation" in vllm 0.11.2
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
                "activation": False,
                # "add_special_tokens": False,  # vllm >= 0.11.2
            }
            output = await self._post_request(payloads, "classify")
            rm_score = output["data"][-1]["probs"][-1]
        elif engine_name == "sglang":
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
            }
            output = await self._post_request(payloads, "v1/embeddings")
            rm_score = output["data"][-1]["embedding"][-1]
        else:
            raise NotImplementedError(f"RewardLoopWorker does not support reward engine {engine_name}")

        return {"reward_score": rm_score}
