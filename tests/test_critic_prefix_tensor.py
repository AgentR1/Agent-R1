"""Real CPU DataProto/critic tests. No Ray cluster, CUDA allocation or pretrained weights."""

import importlib.util
import unittest
from unittest.mock import patch

from test_critic_prefix import ROOT, Tokenizer, config

DEPENDENCIES = ("torch", "numpy", "ray", "hydra", "transformers", "verl", "tensordict")
HAS_STACK = all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES)


@unittest.skipUnless(HAS_STACK, "Requires installed verl/PyTorch stack for CPU tensor tests")
class CriticTensorTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch

        from agent_r1.trainer.ppo.critic_prefix import CriticPrefixBuilder
        from verl import DataProto

        self.torch, self.np = torch, np
        self.Builder, self.DataProto = CriticPrefixBuilder, DataProto

    def batch(self):
        tensor = self.torch.tensor
        prompts = tensor([[0, 0, 1, 2], [0, 3, 4, 5], [0, 0, 6, 7]])
        responses = tensor([[8, 9, 0], [10, 11, 12], [13, 14, 0]])
        attention = tensor([[0, 0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0]])
        return self.DataProto.from_dict(tensors={
            "prompts": prompts, "responses": responses,
            "input_ids": self.torch.cat((prompts, responses), dim=1), "attention_mask": attention,
            "position_ids": (attention.cumsum(-1) - 1).clamp_min(0).masked_fill(attention == 0, 0),
            "response_mask": tensor([[1, 1, 0], [1, 0, 1], [1, 1, 0]]),
            "sample_mask": tensor([True, True, True]),
        }, non_tensors={
            "multi_modal_inputs": self.np.array([{}, {}, {}], dtype=object),
            "source_uid": self.np.array(["a", "b", "a"], dtype=object),
            "trajectory_uids": self.np.array(["ta", "tb", "ta"], dtype=object),
            "step_indices": self.np.array([0, 0, 1]),
            "reward_model": self.np.array([{"ground_truth": "1"}, {"ground_truth": "22"}, {"ground_truth": "1"}],
                                          dtype=object),
        }, meta_info={"global_token_num": [4, 6, 4], "unchanged": "metadata"})

    def builder(self, batch, **options):
        builder = self.Builder(config(template="{ground_truth}", **options), Tokenizer())
        builder.freeze_batch(batch)
        return builder

    def critic(self):
        from types import SimpleNamespace

        from transformers import Qwen3Config, Qwen3ForTokenClassification

        from agent_r1.workers.critic.dp_critic import DataParallelPPOCritic

        self.torch.manual_seed(7)
        model_config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                  max_position_embeddings=512, num_labels=1, pad_token_id=0,
                                  attention_dropout=0.0, classifier_dropout=0.0)
        model_config._attn_implementation = "eager"
        critic = object.__new__(DataParallelPPOCritic)
        critic.critic_module = Qwen3ForTokenClassification(model_config).to("cpu")
        critic.critic_optimizer = self.torch.optim.AdamW(critic.critic_module.parameters(), lr=1e-3)
        critic.device_name = "cpu"
        critic.use_remove_padding = False
        critic.ulysses_sequence_parallel_size = 1
        critic.config = SimpleNamespace(ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=1, ppo_epochs=1,
                                        use_dynamic_bsz=False, grad_clip=1.0, cliprange_value=0.5,
                                        loss_agg_mode="seq-mean-token-mean")
        return critic

    def test_layout_masks_positions_and_actor_invariance(self):
        batch = self.batch()
        originals = {key: value.clone() for key, value in batch.batch.items()}
        builder = self.builder(batch)
        view = builder.build_batch(batch)
        self.assertEqual(view.batch["input_ids"].shape, (3, 8))
        self.assertEqual(view.batch["prompts"].shape, (3, 5))
        self.assertEqual(view.batch["prompts"][0].tolist(), [0, 0, *builder.prefix_ids["a"], 1, 2])
        self.assertEqual(view.batch["prompts"][1].tolist(), [*builder.prefix_ids["b"], 3, 4, 5])
        self.assertEqual(view.batch["position_ids"][0].tolist(), [0, 0, 0, 1, 2, 3, 4, 0])
        self.assertEqual(view.meta_info["global_token_num"], [5, 8, 5])
        self.assertEqual(view.meta_info["critic_prefix_token_counts"], [1, 2, 1])
        self.assertEqual(batch.meta_info["global_token_num"], [4, 6, 4])
        self.assertTrue(self.torch.equal(view.batch["responses"], originals["responses"]))
        self.assertTrue(self.torch.equal(view.batch["response_mask"], originals["response_mask"]))
        view.batch["response_mask"].zero_()
        for key, value in originals.items():
            self.assertTrue(self.torch.equal(batch.batch[key], value), key)

    def test_cached_prefix_survives_metadata_changes_and_reordering(self):
        batch = self.batch()
        builder = self.builder(batch)
        old_view = builder.build_batch(batch)
        batch.non_tensor_batch["reward_model"][0] = {"ground_truth": "changed during reward processing"}
        reordered = batch.select_idxs([2, 1, 0])
        new_view = builder.build_batch(reordered)
        self.assertTrue(self.torch.equal(new_view.batch["input_ids"], old_view.batch["input_ids"][[2, 1, 0]]))

    def test_update_without_frozen_prefix_fails(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.Builder(config(), Tokenizer()).build_batch(self.batch())

    def test_empty_prefix_preserves_valid_tokens_and_positions(self):
        batch = self.batch()
        builder = self.Builder(config(template=""), Tokenizer())
        builder.freeze_batch(batch)
        view = builder.build_batch(batch)
        for i in range(len(batch)):
            old_mask = batch.batch["attention_mask"][i].bool()
            new_mask = view.batch["attention_mask"][i].bool()
            self.assertTrue(self.torch.equal(
                batch.batch["input_ids"][i][old_mask], view.batch["input_ids"][i][new_mask]
            ))
            self.assertTrue(self.torch.equal(batch.batch["position_ids"][i][old_mask],
                                            view.batch["position_ids"][i][new_mask]))

    def test_text_rollout_empty_multimodal_inputs_is_safe(self):
        for values in ([{}, {}, {}], [None, None, None], [{}, None, {}]):
            batch = self.batch()
            column = self.np.array(values, dtype=object)
            batch.non_tensor_batch["multi_modal_inputs"] = column
            builder = self.builder(batch)
            view = builder.build_batch(batch)
            self.assertNotIn("multi_modal_inputs", view.non_tensor_batch)
            self.assertIs(batch.non_tensor_batch["multi_modal_inputs"], column)
            self.assertTrue(self.torch.equal(view.batch["responses"], batch.batch["responses"]))
            self.assertEqual(view.meta_info["critic_prefix_token_counts"], [1, 2, 1])

    def test_multimodal_and_bad_layout_fail(self):
        batch = self.batch()
        builder = self.builder(batch)
        batch.non_tensor_batch["multi_modal_inputs"] = self.np.array(
            [{}, {"pixel_values": self.torch.zeros(1)}, None], dtype=object,
        )
        with self.assertRaisesRegex(ValueError, "text-only"):
            builder.build_batch(batch)
        del batch.non_tensor_batch["multi_modal_inputs"]
        positions = batch.batch["position_ids"]
        batch.batch["position_ids"] = positions.unsqueeze(1)
        with self.assertRaisesRegex(ValueError, "text-only"):
            builder.build_batch(batch)
        batch.batch["position_ids"] = positions
        batch.batch["input_ids"][0, -1] = 99
        with self.assertRaisesRegex(ValueError, "exact prompt"):
            builder.build_batch(batch)

    def test_interior_prompt_padding_and_bad_response_attention_fail(self):
        for change in ("prompt", "response", "action"):
            batch = self.batch()
            builder = self.builder(batch)
            if change == "prompt":
                batch.batch["attention_mask"][0, :4] = self.torch.tensor([1, 0, 1, 1])
            elif change == "response":
                batch.batch["attention_mask"][0, -3:] = self.torch.tensor([1, 0, 1])
            else:
                batch.batch["response_mask"][0, -1] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                builder.build_batch(batch)

    def test_mini_batching_uses_critic_token_counts_and_keeps_actor_masks(self):
        from agent_r1.trainer.ppo.trajectory_batching import prepare_trajectory_mini_batch

        batch = self.batch()
        builder = self.builder(batch)
        view = builder.build_batch(batch)
        originals = batch.batch["response_mask"].clone()
        prepared = prepare_trajectory_mini_batch(view, mini_batch_size=2, dp_size=2)
        self.assertEqual(prepared.batch["mini_batch_global_token_num"][0].tolist(), [5, 5, 8])
        self.assertEqual(prepared.batch["sample_mask"].tolist(), [True, True, True, False])
        self.assertFalse(bool(prepared.batch["response_mask"][-1].any()))
        self.assertTrue(self.torch.equal(batch.batch["response_mask"], originals))

    def test_real_critic_state_value_alignment_and_causality(self):
        batch = self.batch()
        builder = self.builder(batch)
        view = builder.build_batch(batch)
        critic = self.critic()
        critic.critic_module.eval()
        inputs = dict(view.batch.items())
        with self.torch.no_grad():
            values = critic._forward_micro_batch(inputs)
            actor_only_values = critic._forward_micro_batch(dict(batch.batch.items()))
            output = critic.critic_module(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                          position_ids=inputs["position_ids"], use_cache=False)
        self.assertEqual(values.shape, batch.batch["responses"].shape)
        self.assertFalse(self.torch.allclose(values[:, 0], actor_only_values[:, 0]))
        self.assertTrue(self.torch.allclose(values[:, 0].float(), output.logits[:, -4, 0].float(), atol=1e-3))
        changed = {key: value.clone() for key, value in inputs.items()}
        changed["input_ids"][:, -3:] = self.torch.tensor([[31, 30, 0], [29, 28, 27], [26, 25, 0]])
        changed["responses"] = changed["input_ids"][:, -3:]
        with self.torch.no_grad():
            future_changed_values = critic._forward_micro_batch(changed)
        self.assertTrue(self.torch.equal(values[:, 0], future_changed_values[:, 0]))
        self.assertFalse(self.torch.allclose(values[:, 1], future_changed_values[:, 1]))

    def test_real_trainer_compute_and_critic_backward_do_not_change_actor(self):
        from omegaconf import OmegaConf

        from agent_r1.trainer.ppo.ray_trainer import RayAgentTrainer

        batch = self.batch()
        original_inputs = {key: value.clone() for key, value in batch.batch.items()}
        critic = self.critic()
        owner = self

        class Group:
            world_size = 1
            _dispatch_info = {"critic": [0]}

            def compute_values(self, data):
                data.meta_info.update(micro_batch_size=1, use_dynamic_bsz=False)
                with patch("verl.workers.critic.dp_critic.get_device_id", return_value=owner.torch.device("cpu")):
                    values = critic.compute_values(data)
                return owner.DataProto.from_dict(tensors={"values": values})

            def update_critic(self, data):
                self.update_input_ids = data.batch["input_ids"].clone()
                with patch("agent_r1.workers.critic.dp_critic.get_device_id", return_value=owner.torch.device("cpu")):
                    metrics = critic.update_critic(data)
                return owner.DataProto.from_dict(tensors={}, meta_info={"metrics": metrics})

        trainer = object.__new__(RayAgentTrainer)
        trainer.use_legacy_worker_impl = "enable"
        trainer.config = OmegaConf.create({"critic": {"ppo_mini_batch_size": 4},
                                           "actor_rollout_ref": {"rollout": {"n": 1}}})
        trainer.critic_prefix_builder = self.Builder(config(template="{ground_truth}"), Tokenizer())
        trainer.critic_wg = Group()
        values = trainer._compute_values(batch)
        batch = batch.union(values)
        batch.batch["returns"] = self.torch.ones_like(values.batch["values"])
        # Emulate the existing step-GAE critic loss mask without changing the actor object.
        critic_loss_batch = batch.select_idxs(list(range(len(batch))))
        critic_loss_batch.batch["response_mask"][:, 1:] = 0
        old_weights = [parameter.detach().clone() for parameter in critic.critic_module.parameters()]
        result = trainer._update_critic(critic_loss_batch)
        metrics = result.meta_info["metrics"]
        self.assertTrue(self.torch.isfinite(self.torch.tensor(metrics["critic/vf_loss"])))
        self.assertGreater(metrics["critic/grad_norm"][0], 0)
        self.assertTrue(any(not self.torch.equal(before, after) for before, after in
                            zip(old_weights, critic.critic_module.parameters(), strict=True)))
        for key, original in original_inputs.items():
            self.assertTrue(self.torch.equal(batch.batch[key], original), key)
        self.assertNotEqual(trainer.critic_wg.update_input_ids.shape[1], batch.batch["input_ids"].shape[1])

    def test_disabled_trainer_passes_original_inputs_to_value_worker(self):
        from agent_r1.trainer.ppo.ray_trainer import RayAgentTrainer

        batch = self.batch()
        owner = self

        class Group:
            def compute_values(self, data):
                self.received = data
                return owner.DataProto.from_dict(tensors={"values": owner.torch.zeros_like(data.batch["responses"])})

        trainer = object.__new__(RayAgentTrainer)
        trainer.use_legacy_worker_impl = "enable"
        trainer.critic_prefix_builder = self.Builder({"enabled": False}, Tokenizer())
        trainer.critic_wg = Group()
        trainer._compute_values(batch)
        self.assertIs(trainer.critic_wg.received, batch)

    def test_hydra_preset_does_not_pollute_verl_critic_dataclass(self):
        from hydra import compose, initialize_config_dir

        from agent_r1.trainer.ppo.critic_prefix import validate_asymmetric_critic
        from verl.utils.config import omega_conf_to_dataclass

        with initialize_config_dir(config_dir=str(ROOT / "agent_r1/config"), version_base=None):
            composed = compose(config_name="asymmetric_ppo_trainer", overrides=[
                "actor_rollout_ref.model.path=same", "critic.model.path=same",
                "critic.ppo_micro_batch_size_per_gpu=1",
            ])
        validate_asymmetric_critic(composed)
        self.assertEqual(composed.algorithm.adv_estimator, "gae")
        self.assertTrue(composed.asymmetric_critic.enabled)
        builder = self.Builder(composed.asymmetric_critic, Tokenizer())
        self.assertIn("Critic-only information", builder.template)
        # Check the actual critic schema/validators, without the unrelated HF
        # metadata loader interpreting the dummy "same" path as a Hub model.
        with patch("verl.workers.config.model.HFModelConfig.__post_init__", return_value=None):
            critic_config = omega_conf_to_dataclass(composed.critic)
        self.assertFalse(hasattr(critic_config, "asymmetric_critic"))


if __name__ == "__main__":
    unittest.main()
