"""CPU-only contract tests using simulated generation/environment backends.

Run: python3 -B -m unittest discover -s tests -v

The production module imports the GPU/Ray stack. To exercise its actual loop
and orchestration methods on a dependency-free machine, load those methods
from their AST with explicit test doubles. This is NOT a GPU integration test.
"""

import ast
import asyncio
import importlib.util
import math
import sys
import unittest
from contextlib import nullcontext
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "greedy_rollout_utils_under_test", ROOT / "agent_r1/agent_flow/rollout_utils.py"
)
utils = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = utils
spec.loader.exec_module(utils)


def load_methods(path, class_name, method_names, namespace):
    """Compile unmodified production method bodies, removing only decorators."""
    tree = ast.parse((ROOT / path).read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = []
    for method in definition.body:
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name in method_names:
            method = deepcopy(method)
            method.decorator_list = []
            nodes.append(method)
    if len(nodes) != len(method_names):
        raise AssertionError(f"Missing production methods in {class_name}")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = dict(namespace)
    exec(compile(module, str(ROOT / path), "exec"), scope)
    return {name: scope[name] for name in method_names}


def rollout_config():
    return SimpleNamespace(
        temperature=0.8,
        top_p=0.9,
        calculate_log_probs=True,
        response_length=8,
        n=2,
        val_kwargs=SimpleNamespace(temperature=0.4, top_p=0.7),
        agent=SimpleNamespace(default_agent_flow="fake"),
    )


def zip_with_strict(*iterables, strict=False):
    """Python 3.9 test-host shim for the production Python 3.10+ zip contract."""
    if not strict:
        yield from zip(*iterables)  # noqa: B905
        return
    sentinel = object()
    for row in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in row):
            raise ValueError("zip() arguments have different lengths")
        yield row


def baseline_row(uid="a", trajectory="g-a", step=0, reward=1.0, final=True, truncated=False):
    return {
        "source_uid": uid,
        "trajectory_uid": trajectory,
        "step_index": step,
        "reward": reward,
        "rollout_mode": "greedy",
        "terminated": final and not truncated,
        "truncated": final and truncated,
        "termination_reason": ("max_steps" if truncated else "env_done") if final else "ongoing",
    }


class SamplingAndPairingTests(unittest.TestCase):
    def test_sampling_and_validation_defaults(self):
        config = rollout_config()
        params = utils.build_sampling_params(config, {})
        self.assertEqual((params["temperature"], params["top_p"]), (0.8, 0.9))
        self.assertEqual(params["max_tokens"], 8)
        self.assertTrue(params["logprobs"])
        params = utils.build_sampling_params(config, {"validate": True})
        self.assertEqual((params["temperature"], params["top_p"]), (0.4, 0.7))

    def test_greedy_overrides_validation_without_mutation(self):
        config = rollout_config()
        for validate in (False, True):
            with self.subTest(validate=validate):
                meta = {"rollout_mode": "greedy", "validate": validate}
                params = utils.build_sampling_params(config, meta)
                self.assertEqual((params["temperature"], params["top_p"]), (0.0, 1.0))
                self.assertEqual(params["repetition_penalty"], 1.0)
                self.assertFalse(params["logprobs"])
                self.assertNotIn("do_sample", params)
                self.assertNotIn("n", params)
                params["temperature"] = 99
        self.assertEqual(config.temperature, 0.8)
        self.assertEqual(config.val_kwargs.temperature, 0.4)

    def test_bad_modes_and_budgets_fail(self):
        with self.assertRaises(ValueError):
            utils.build_sampling_params(rollout_config(), {"rollout_mode": "typo"})
        config = rollout_config()
        config.response_length = 0
        with self.assertRaises(ValueError):
            utils.build_sampling_params(config, {})

    def test_generation_empty_and_aborted_fail(self):
        for tokens, reason in (([], "completed"), ([1], "aborted"), ([1], "error")):
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                utils.generation_metadata(SimpleNamespace(token_ids=tokens, stop_reason=reason), 8)

    def test_budget_and_eos_classification(self):
        cases = [
            ([1], None, "completed", False),
            ([1, 2], None, "completed", True),
            ([1, 9], None, "completed", False),
            ([1, 9, 3], None, "stop", True),
            ([1, 2], "stop", "completed", False),
            ([1], "length", "completed", True),
        ]
        for tokens, finish, stop, truncated in cases:
            with self.subTest(tokens=tokens, finish=finish):
                output = SimpleNamespace(token_ids=tokens, finish_reason=finish, stop_reason=stop)
                meta = utils.generation_metadata(output, 2, eos_token_id=[9, 10])
                self.assertEqual(meta["response_truncated"], truncated)
                status = utils.terminal_status(meta)
                self.assertEqual(status["truncated"], truncated)
                self.assertEqual(status["terminated"], not truncated)

    def test_multistep_rewards_pair_by_uid_not_order(self):
        rows = [
            baseline_row(step=1, reward=0.75),
            baseline_row(uid="b", trajectory="g-b", reward=2, truncated=True),
            baseline_row(step=0, reward=-0.25, final=False),
        ]
        baselines = utils.summarize_greedy_baselines(rows, ["a", "b"])
        self.assertEqual(baselines["a"].reward, 0.5)
        self.assertEqual(baselines["a"].num_steps, 2)
        self.assertTrue(baselines["b"].truncated)
        self.assertEqual(utils.pair_baseline_rewards(["b", "a", "b", "a", "a"], baselines), [2, 0.5, 2, 0.5, 0.5])

    def test_missing_duplicate_mixed_and_invalid_baselines_fail(self):
        bad_rows = [
            [],
            [baseline_row(uid="other")],
            [baseline_row(), baseline_row(trajectory="second")],
            [baseline_row(step=1)],
            [baseline_row(), baseline_row()],
            [baseline_row(final=False)],
            [baseline_row(), baseline_row(step=1)],
            [baseline_row(final=False), baseline_row(uid="b", step=1)],
            [{**baseline_row(), "rollout_mode": "sample"}],
            [{**baseline_row(), "termination_reason": "unknown"}],
            [{**baseline_row(), "truncated": True}],
        ]
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                utils.summarize_greedy_baselines(rows, ["a"])

    def test_nonfinite_rewards_fail(self):
        for reward in (math.nan, math.inf, -math.inf):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                utils.summarize_greedy_baselines([baseline_row(reward=reward)], ["a"])

    def test_bad_source_uids_and_missing_pair_fail(self):
        for uid in (None, "", "  "):
            with self.subTest(uid=uid), self.assertRaises(ValueError):
                utils.normalize_source_uid(uid)
        self.assertEqual(utils.normalize_source_uid(42), "42")
        with self.assertRaises(ValueError):
            utils.summarize_greedy_baselines([baseline_row()], ["a", "a"])
        with self.assertRaises(ValueError):
            utils.pair_baseline_rewards(["a"], {})


class FakeArray(list):
    def tolist(self):
        return list(self)


class FakeTensor:
    def __init__(self, values, dtype="float32", device="cpu"):
        self.values = values
        self.dtype = dtype
        self.device = device

    def __mul__(self, other):
        return FakeTensor([[a * b for a, b in zip(x, y)] for x, y in zip(self.values, other.values)])

    def sum(self, dim):
        assert dim == -1
        return FakeTensor([sum(row) for row in self.values])

    def tolist(self):
        return deepcopy(self.values)


class FakeDataProto:
    def __init__(self, uids, payloads=None, meta_info=None, batch=None, non_tensor_batch=None):
        self.non_tensor_batch = non_tensor_batch or {"uid": FakeArray(uids)}
        if payloads is not None:
            self.non_tensor_batch["payload"] = FakeArray(payloads)
        self.meta_info = meta_info or {}
        self.batch = batch or {}

    def __len__(self):
        return len(next(iter(self.non_tensor_batch.values())))

    def repeat(self, repeat_times, interleave):
        assert interleave
        fields = {key: FakeArray(deepcopy(item) for item in values for _ in range(repeat_times))
                  for key, values in self.non_tensor_batch.items()}
        return FakeDataProto([], meta_info=deepcopy(self.meta_info), non_tensor_batch=fields)

    def split(self, split_size):
        return [
            FakeDataProto(
                [], meta_info=deepcopy(self.meta_info),
                non_tensor_batch={key: FakeArray(values[i:i + split_size])
                                  for key, values in self.non_tensor_batch.items()},
            )
            for i in range(0, len(self), split_size)
        ]

    @staticmethod
    def concat(outputs):
        fields = {key: FakeArray(value for output in outputs for value in output.non_tensor_batch[key])
                  for key in outputs[0].non_tensor_batch}
        return FakeDataProto([], non_tensor_batch=fields)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        namespace = {
            "deepcopy": deepcopy,
            "uuid4": uuid4,
            "np": SimpleNamespace(array=lambda values, dtype: FakeArray(values)),
            "torch": SimpleNamespace(tensor=lambda values, **kwargs: FakeTensor(values, **kwargs)),
            "normalize_source_uid": utils.normalize_source_uid,
            "pair_baseline_rewards": utils.pair_baseline_rewards,
            "summarize_greedy_baselines": utils.summarize_greedy_baselines,
            "ReMaxRolloutCollection": utils.ReMaxRolloutCollection,
        }
        methods = load_methods(
            "agent_r1/agent_flow/agent_flow.py", "AgentFlowManager",
            ["_prepare_original_tasks", "generate_greedy_sequences", "collect_remax_rollouts"], namespace,
        )
        methods["_prepare_original_tasks"] = staticmethod(methods["_prepare_original_tasks"])
        manager_cls = type("TestManager", (), methods)
        self.manager = manager_cls()
        self.manager.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout_config()))
        self.calls = []
        self.bad_greedy = False

        def generate(prompts):
            self.calls.append(deepcopy(prompts))
            mode = prompts.meta_info["rollout_mode"]
            rows = []
            for i, uid in enumerate(prompts.non_tensor_batch["uid"]):
                rewards = [0.2, 0.3, 0.5] if i == 0 and mode == "greedy" else [2.0]
                for step, reward in enumerate(rewards):
                    rows.append(baseline_row(uid, f"{mode}-{i}", step, reward, final=step == len(rewards) - 1))
            rows.reverse()  # Simulate arbitrary row order after distributed batching.
            if self.bad_greedy and mode == "greedy":
                rows.pop()
            fields = {
                key: FakeArray(row[row_key] for row in rows)
                for key, row_key in {
                    "source_uid": "source_uid", "trajectory_uids": "trajectory_uid",
                    "step_indices": "step_index", "terminated": "terminated",
                    "truncated": "truncated", "termination_reason": "termination_reason",
                }.items()
            }
            fields["rollout_mode"] = FakeArray([mode] * len(rows))
            batch = {
                "rm_scores": FakeTensor([[row["reward"], 999] for row in rows]),
                "response_mask": FakeTensor([[1, 0] for _ in rows]),
                "responses": FakeTensor([[11, 0] for _ in rows]),
            }
            prompts.meta_info["mutated_by_backend"] = True
            return FakeDataProto([], batch=batch, non_tensor_batch=fields)

        self.manager.generate_sequences = generate

    def test_collect_one_greedy_and_n_samples_with_masked_rewards(self):
        original = FakeDataProto(["a", "b"], payloads=[{"state": 0}, {"state": 0}])
        result = self.manager.collect_remax_rollouts(original, num_samples=3)
        self.assertEqual([len(call) for call in self.calls], [2, 6])
        self.assertEqual([call.meta_info["rollout_mode"] for call in self.calls], ["greedy", "sample"])
        self.assertEqual(result.baselines["a"].reward, 1.0)
        self.assertEqual(result.baselines["b"].reward, 2.0)
        self.assertEqual(result.sampled.batch["reward_baselines"].tolist(), [2, 2, 2, 1, 1, 1])
        self.assertNotIn("advantages", result.sampled.batch)
        self.assertNotIn("reward_baselines", result.greedy.batch)
        self.assertEqual(original.meta_info, {})
        self.assertEqual(original.non_tensor_batch["payload"], [{"state": 0}, {"state": 0}])

    def test_generated_uid_is_shared_across_paths_not_written_to_caller(self):
        original = FakeDataProto([], non_tensor_batch={"payload": FakeArray(["x", "y"])})
        result = self.manager.collect_remax_rollouts(original)
        greedy_uids = self.calls[0].non_tensor_batch["uid"]
        self.assertEqual(len(set(greedy_uids)), 2)
        self.assertEqual(self.calls[1].non_tensor_batch["uid"], [greedy_uids[0]] * 2 + [greedy_uids[1]] * 2)
        self.assertEqual(set(result.baselines), set(greedy_uids))
        self.assertNotIn("uid", original.non_tensor_batch)

    def test_greedy_does_not_repeat_or_mutate_caller(self):
        original = FakeDataProto(["a", "b"], meta_info={"rollout_mode": "sample", "global_steps": 4})
        self.manager.generate_greedy_sequences(original)
        self.assertEqual(len(self.calls[0]), 2)
        self.assertEqual(original.meta_info, {"rollout_mode": "sample", "global_steps": 4})

    def test_invalid_original_tasks_fail_before_generation(self):
        for uids in ([], ["a", "a"], [None], [1, "1"]):
            with self.subTest(uids=uids), self.assertRaises(ValueError):
                self.manager.collect_remax_rollouts(FakeDataProto(uids))
        self.assertEqual(self.calls, [])

    def test_bad_sample_count_and_validation_fail_before_generation(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.manager.collect_remax_rollouts(FakeDataProto(["a"]), count)
        with self.assertRaises(ValueError):
            self.manager.collect_remax_rollouts(FakeDataProto(["a"], meta_info={"validate": True}))
        self.assertEqual(self.calls, [])

    def test_incomplete_greedy_fails_without_sampling(self):
        self.bad_greedy = True
        with self.assertRaises(ValueError):
            self.manager.collect_remax_rollouts(FakeDataProto(["a", "b"]))
        self.assertEqual(len(self.calls), 1)


class FakeStep(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(extra_fields=kwargs.pop("extra_fields", {}), **kwargs)


class FakeFlowOutput(SimpleNamespace):
    def __init__(self, **kwargs):
        defaults = dict(source_uid=None, rollout_mode="sample", terminated=False, truncated=False,
                        termination_reason="unknown")
        defaults.update(kwargs)
        super().__init__(**defaults)


class LoopAndWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        namespace = {
            "uuid4": uuid4, "AgentFlowStep": FakeStep, "AgentFlowOutput": FakeFlowOutput,
            "Action": lambda **kwargs: SimpleNamespace(**kwargs),
            "simple_timer": lambda *args: nullcontext(),
            "terminal_status": utils.terminal_status,
            "logger": SimpleNamespace(warning=lambda *args: None),
        }
        methods = load_methods("agent_r1/agent_flow/agent_env_loop.py", "AgentEnvLoop", ["run"], namespace)
        self.flow_cls = type("ActualLoopHarness", (), methods)
        self.instances = []

    def make_flow(self, done_after=3, env_truncated=False, final_answer=False):
        env = SimpleNamespace(position=0, actions=[], reset_count=0)

        def reset(**kwargs):
            env.position = 0
            env.reset_count += 1
            return [0]

        async def step(action):
            env.actions.append(action.token_ids[0])
            env.position += 1
            info = {"truncated": env_truncated}
            if final_answer:
                info["termination_reason"] = "final_answer"
            return [env.position, action.token_ids[0]], float(env.position), env.position >= done_after, info

        env.reset, env.step = reset, step
        flow = self.flow_cls()
        flow.prompt_length, flow.response_length, flow.max_steps = 16, 8, 5
        flow.skip_special_tokens = True
        flow.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens))
        flow.loop = asyncio.get_running_loop()
        flow.requests = []
        flow._create_env = lambda **kwargs: env

        async def obs_to_prompt(obs, **kwargs):
            return obs

        async def generate(**kwargs):
            flow.requests.append(deepcopy(kwargs))
            chosen = 11 if kwargs["sampling_params"]["temperature"] == 0 else 21
            return SimpleNamespace(token_ids=[chosen], log_probs=None, routed_experts=None, stop_reason="completed")

        async def postprocess(step, **kwargs):
            return step

        flow._obs_to_prompt = obs_to_prompt
        flow.server_manager = SimpleNamespace(generate=generate)
        flow._postprocess = postprocess
        flow._generation_metadata = lambda output: utils.generation_metadata(output, flow.response_length)
        flow.env = env
        self.instances.append(flow)
        return flow

    async def test_actual_loop_generates_every_step_from_its_own_environment(self):
        greedy = self.make_flow()
        sampled = self.make_flow(done_after=2)
        greedy_result = await greedy.run(utils.build_sampling_params(rollout_config(), {"rollout_mode": "greedy"}))
        sample_result = await sampled.run(utils.build_sampling_params(rollout_config(), {}))
        self.assertEqual(greedy.env.actions, [11, 11, 11])
        self.assertEqual(sampled.env.actions, [21, 21])
        self.assertEqual([request["prompt_ids"] for request in greedy.requests], [[0], [1, 11], [2, 11]])
        self.assertEqual([step.reward_score for step in greedy_result.steps], [1, 2, 3])
        self.assertTrue(greedy_result.terminated)
        self.assertTrue(sample_result.terminated)
        self.assertEqual(greedy_result.termination_reason, "env_done")
        self.assertIsNot(greedy.env, sampled.env)

    async def test_actual_loop_max_steps_and_prompt_overflow(self):
        flow = self.make_flow(done_after=10)
        flow.max_steps = 2
        output = await flow.run(utils.build_sampling_params(rollout_config(), {"rollout_mode": "greedy"}))
        self.assertEqual(len(output.steps), 2)
        self.assertTrue(output.truncated)
        self.assertFalse(output.terminated)
        self.assertEqual(output.termination_reason, "max_steps")
        flow = self.make_flow()
        flow.prompt_length = 0
        output = await flow.run(utils.build_sampling_params(rollout_config(), {}))
        self.assertEqual(output.steps, [])
        self.assertEqual(output.termination_reason, "prompt_length")
        self.assertEqual(flow.requests, [])

    async def test_actual_loop_preserves_environment_time_limit(self):
        flow = self.make_flow(done_after=1, env_truncated=True)
        output = await flow.run(utils.build_sampling_params(rollout_config(), {}))
        self.assertTrue(output.truncated)
        self.assertEqual(output.termination_reason, "env_truncated")

    async def test_truncated_final_answer_not_claimed_as_natural_end(self):
        flow = self.make_flow(done_after=1, final_answer=True)
        flow.response_length = 1
        output = await flow.run(utils.build_sampling_params(rollout_config(), {}))
        self.assertTrue(output.truncated)
        self.assertEqual(output.termination_reason, "response_length")

    async def test_actual_worker_sets_uid_mode_and_creates_fresh_flows(self):
        async def trajectory_info(step, index, validate):
            return [dict(step=step, sample_index=i, rollout_n=0, validate=validate) for i in index]

        def instantiate(**kwargs):
            return self.make_flow(done_after=2)

        namespace = {
            "asyncio": asyncio, "uuid4": uuid4,
            "np": SimpleNamespace(
                array=lambda values, dtype: FakeArray(values), arange=lambda size: FakeArray(range(size))
            ),
            "build_sampling_params": utils.build_sampling_params,
            "normalize_source_uid": utils.normalize_source_uid,
            "RolloutTraceConfig": SimpleNamespace(
                get_instance=lambda: SimpleNamespace(max_samples_per_step_per_worker=None)
            ),
            "get_trajectory_info": trajectory_info,
            "rollout_trace_attr": lambda **kwargs: nullcontext(),
            "_agent_flow_registry": {"fake": {}},
            "hydra": SimpleNamespace(utils=SimpleNamespace(instantiate=instantiate)),
            "DictConfigWrap": lambda **kwargs: SimpleNamespace(**kwargs),
        }
        methods = load_methods(
            "agent_r1/agent_flow/agent_flow.py", "AgentFlowWorkerBase",
            ["generate_sequences", "_run_agent_flow"], namespace,
        )
        worker = type("ActualWorkerHarness", (), methods)()
        worker.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout_config()), data={})
        for name in ("server_manager", "reward_loop_worker", "tokenizer", "processor", "dataset_cls"):
            setattr(worker, name, None)
        worker._postprocess = lambda outputs: outputs
        batch = FakeDataProto(["a", "a", "b"], meta_info={"rollout_mode": "greedy", "validate": True})
        outputs = await worker.generate_sequences(batch)
        self.assertEqual([output.source_uid for output in outputs], ["a", "a", "b"])
        self.assertEqual([output.rollout_mode for output in outputs], ["greedy"] * 3)
        self.assertEqual(len({id(flow.env) for flow in self.instances}), 3)
        for flow in self.instances:
            self.assertEqual(flow.env.reset_count, 1)
            self.assertEqual(flow.env.actions, [11, 11])
            self.assertTrue(all(request["sampling_params"]["temperature"] == 0 for request in flow.requests))
        self.assertEqual(worker.config.actor_rollout_ref.rollout.temperature, 0.8)


class DispatchTests(unittest.TestCase):
    def make_manager(self, fail=False):
        namespace = {
            "ray": SimpleNamespace(get=lambda outputs: outputs),
            "DataProto": FakeDataProto,
            "zip": zip if sys.version_info >= (3, 10) else zip_with_strict,
        }
        methods = load_methods(
            "agent_r1/agent_flow/agent_flow.py", "AgentFlowManager", ["generate_sequences"], namespace
        )
        manager = type("ActualDispatchHarness", (), methods)()
        self.events = []

        def generate(chunk):
            self.events.append("generate")
            if fail:
                raise RuntimeError("simulated worker failure")
            chunk.meta_info["metrics"] = [dict(num_steps=1) for _ in range(len(chunk))]
            return chunk

        manager.agent_flow_workers = [
            SimpleNamespace(generate_sequences=SimpleNamespace(remote=generate)) for _ in range(8)
        ]
        manager.reward_model_manager = SimpleNamespace(
            wake_up=lambda: self.events.append("reward_wake"), sleep=lambda: self.events.append("reward_sleep")
        )
        manager.wake_up = lambda: self.events.append("wake")
        manager.sleep = lambda: self.events.append("sleep")
        manager._performance_metrics = lambda *args: {}
        return manager

    def test_task_count_smaller_than_worker_count(self):
        manager = self.make_manager()
        result = manager.generate_sequences(FakeDataProto(["a", "b"]))
        self.assertEqual(result.meta_info["num_steps"], [1, 1])
        self.assertEqual(self.events.count("generate"), 2)
        self.assertEqual(self.events[-2:], ["sleep", "reward_sleep"])

    def test_backend_failure_still_releases_rollout_and_reward_engines(self):
        manager = self.make_manager(fail=True)
        with self.assertRaises(RuntimeError):
            manager.generate_sequences(FakeDataProto(["a"]))
        self.assertEqual(self.events[-2:], ["sleep", "reward_sleep"])

    def test_rollout_sleep_failure_still_releases_reward_engine(self):
        manager = self.make_manager()

        def sleep():
            self.events.append("sleep")
            raise RuntimeError("simulated sleep failure")

        manager.sleep = sleep
        with self.assertRaises(RuntimeError):
            manager.generate_sequences(FakeDataProto(["a"]))
        self.assertEqual(self.events[-2:], ["sleep", "reward_sleep"])

    def test_empty_input_fails_without_waking_engines(self):
        manager = self.make_manager()
        with self.assertRaises(ValueError):
            manager.generate_sequences(FakeDataProto([]))
        self.assertEqual(self.events, [])


class EnvironmentAdapterTests(unittest.TestCase):
    def test_alfworld_preserves_gymnasium_end_flags_and_legacy_done(self):
        methods = load_methods(
            "recipes/alfworld/env/alfworld_wrapper.py", "AlfworldTextworldEnv",
            ["_unwrap_batch_item", "_normalize_step_output"], {},
        )
        methods["_unwrap_batch_item"] = staticmethod(methods["_unwrap_batch_item"])
        adapter = type("ActualAdapterHarness", (), methods)()
        for terminated, truncated in ((False, False), (True, False), (False, True)):
            with self.subTest(terminated=terminated, truncated=truncated):
                info = {"success": False}
                obs, reward, done, result_info = adapter._normalize_step_output(
                    ("obs", [0.5], [terminated], [truncated], [info])
                )
                self.assertEqual((obs, reward, done), ("obs", 0.5, terminated or truncated))
                self.assertEqual(result_info["terminated"], terminated)
                self.assertEqual(result_info["truncated"], truncated)
                self.assertEqual(info, {"success": False})
        self.assertEqual(adapter._normalize_step_output(("obs", [1], [True], [{}])), ("obs", 1, True, {}))
        self.assertEqual(adapter._normalize_step_output(("obs", [1], [True])), ("obs", 1, True, {}))


if __name__ == "__main__":
    unittest.main()
