"""Dependency-free parser/configuration tests for critic-only prefixes."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "critic_prefix_under_test", ROOT / "agent_r1/trainer/ppo/critic_prefix.py"
)
prefix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prefix
SPEC.loader.exec_module(prefix)


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Prefix must not add BOS/EOS tokens")
        return [20 + ord(char) % 32 for char in text]


def config(**options):
    return {"enabled": True, "prefix": {"source": "template", "template": "Answer: {ground_truth}\n",
                                         "max_tokens": 256, **options}}


def row(answer="72", solution="36 * 2 = 72"):
    return {"reward_model": {"ground_truth": answer}, "extra_info": {"answer": solution, "question": "How many?"},
            "data_source": "openai/gsm8k"}


class PrefixTests(unittest.TestCase):
    def test_inline_template_and_literal_braces(self):
        builder = prefix.CriticPrefixBuilder(
            config(template="{{info}} {question}: {ground_truth}\n{reference_solution}"), Tokenizer()
        )
        self.assertEqual(builder.render(row()), "{info} How many?: 72\n36 * 2 = 72")
        self.assertTrue(builder.encode(row()))

    def test_constant_and_empty_template_need_no_metadata(self):
        for template in ("constant", ""):
            builder = prefix.CriticPrefixBuilder(config(template=template), Tokenizer())
            self.assertEqual(builder.render({}), template)
        self.assertEqual(prefix.CriticPrefixBuilder(config(template="", max_tokens=0), Tokenizer()).encode({}), ())

    def test_file_is_project_relative_and_loaded_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.txt"
            path.write_text("Reference: {ground_truth}", encoding="utf-8")
            builder = prefix.CriticPrefixBuilder(config(template=None, template_path="prefix.txt"), Tokenizer(),
                                                 Path(directory))
            path.write_text("changed", encoding="utf-8")
            self.assertEqual(builder.render(row()), "Reference: 72")

    def test_field_mode(self):
        builder = prefix.CriticPrefixBuilder(config(source="field", template=None, field="extra_info.critic_prefix"),
                                             Tokenizer())
        for text in ("private facts", ""):
            self.assertEqual(builder.render({"extra_info": {"critic_prefix": text}}), text)
        for value in (None, 123, {}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                builder.render({"extra_info": {"critic_prefix": value}})

    def test_disabled_is_a_noop_without_reading_files_or_tensors(self):
        builder = prefix.CriticPrefixBuilder({"enabled": False, "prefix": {"template_path": "/missing"}}, Tokenizer())
        sentinel = object()
        self.assertIs(builder.build_batch(sentinel), sentinel)
        builder.freeze_batch(sentinel)
        builder.save_snapshot("/not-used")
        self.assertEqual(builder.render({}), "")

    def test_missing_and_bad_variables_fail(self):
        for template in ("{unknown}", "{ground_truth.__class__}", "{ground_truth[0]}", "{ground_truth!r}",
                         "{ground_truth:>10}", "{", "{}"):
            with self.subTest(template=template), self.assertRaises(ValueError):
                prefix.CriticPrefixBuilder(config(template=template), Tokenizer())
        builder = prefix.CriticPrefixBuilder(config(), Tokenizer())
        for bad_row in ({}, row(answer=None), row(answer=""), row(answer=[])):
            with self.subTest(row=bad_row), self.assertRaises(ValueError):
                builder.render(bad_row)

    def test_bad_source_combinations_fail(self):
        cases = (
            {"template": None}, {"template_path": "also.txt"}, {"field": "extra_info.critic_prefix"},
            {"source": "field", "field": "extra_info.critic_prefix"}, {"source": "provider"},
            {"source": "field", "template": None, "field": ""},
            {"source": "field", "template": None, "field": "extra_info[0]"},
            {"source": "template", "template": 123}, {"overflow": "truncate"},
            {"max_token": 10},
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                prefix.CriticPrefixBuilder(config(**options), Tokenizer())

    def test_token_limit_is_checked_without_truncation(self):
        for max_tokens in (-1, 1.5, True):
            with self.subTest(limit=max_tokens), self.assertRaises(ValueError):
                prefix.CriticPrefixBuilder(config(max_tokens=max_tokens), Tokenizer())
        builder = prefix.CriticPrefixBuilder(config(template="123", max_tokens=2), Tokenizer())
        with self.assertRaisesRegex(ValueError, "3 tokens"):
            builder.encode({})

    def test_snapshot_never_overwrites_different_config(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = prefix.CriticPrefixBuilder(config(), Tokenizer())
            builder.save_snapshot(directory)
            builder.save_snapshot(directory)
            path = Path(directory) / "asymmetric_critic_prefix.json"
            snapshot = json.loads(path.read_text())
            self.assertEqual(snapshot["template"], "Answer: {ground_truth}\n")
            with self.assertRaisesRegex(ValueError, "snapshot differs"):
                prefix.CriticPrefixBuilder(config(template="changed"), Tokenizer()).save_snapshot(directory)
            self.assertEqual(json.loads(path.read_text()), snapshot)

    def test_freeze_is_per_uid_and_new_batch_replaces_cache(self):
        class Batch:
            def __len__(self):
                return len(self.non_tensor_batch["source_uid"])

        batch = Batch()
        batch.non_tensor_batch = {"source_uid": ["a", "b", "a"],
                                  "reward_model": [{"ground_truth": "1"}, {"ground_truth": "2"}, None]}
        builder = prefix.CriticPrefixBuilder(config(), Tokenizer())
        builder.freeze_batch(batch)
        self.assertEqual(set(builder.prefix_ids), {"a", "b"})
        self.assertNotEqual(builder.prefix_ids["a"], builder.prefix_ids["b"])
        batch.non_tensor_batch = {"source_uid": ["c"], "reward_model": [{"ground_truth": "3"}]}
        builder.freeze_batch(batch)
        self.assertEqual(set(builder.prefix_ids), {"c"})

    def test_bad_uid_and_missing_metadata_fail_with_context(self):
        class Batch:
            def __len__(self):
                return 1

        batch = Batch()
        builder = prefix.CriticPrefixBuilder(config(), Tokenizer())
        for fields in ({}, {"source_uid": [None]}, {"source_uid": [""]}, {"source_uid": []}):
            batch.non_tensor_batch = fields
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                builder.freeze_batch(batch)
        batch.non_tensor_batch = {"uid": ["task-a"]}
        with self.assertRaisesRegex(ValueError, "task-a"):
            builder.freeze_batch(batch)

    def test_training_path_guard(self):
        class Section(dict):
            __getattr__ = dict.__getitem__

        def training_config():
            return Section(asymmetric_critic={"enabled": True}, algorithm=Section(adv_estimator="gae"),
                           critic=Section(enable=True, strategy="fsdp", model=Section(tokenizer_path="same")),
                           actor_rollout_ref=Section(model=Section(path="same")), trainer=Section())

        valid = training_config()
        prefix.validate_asymmetric_critic(valid)
        mutations = (
            ("algorithm", "adv_estimator", "grpo"), ("algorithm", "adv_estimator", "token_gae"),
            ("critic", "enable", False), ("critic", "strategy", "megatron"),
            ("trainer", "use_legacy_worker_impl", "disable"),
        )
        for section, key, value in mutations:
            bad = training_config()
            bad[section][key] = value
            with self.subTest(mutation=(section, key)), self.assertRaises(ValueError):
                prefix.validate_asymmetric_critic(bad)
        bad = training_config()
        bad.critic.model["tokenizer_path"] = "different"
        with self.assertRaisesRegex(ValueError, "tokenizer_path"):
            prefix.validate_asymmetric_critic(bad)
        prefix.validate_asymmetric_critic(Section(asymmetric_critic={"enabled": False}))
        with self.assertRaisesRegex(ValueError, "boolean"):
            prefix.validate_asymmetric_critic(Section(asymmetric_critic={"enabled": "false"}))


if __name__ == "__main__":
    unittest.main()
