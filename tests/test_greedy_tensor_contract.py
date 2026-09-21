"""Optional real DataProto/Pydantic/CPU-tensor tests; never start Ray or a GPU model."""

import importlib.util
import unittest

DEPENDENCIES = ("torch", "numpy", "pydantic", "ray", "hydra", "transformers", "verl", "tensordict")
HAS_STACK = all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES)


@unittest.skipUnless(HAS_STACK, "Requires installed verl/PyTorch stack for real tensor contract tests")
class TensorContractTests(unittest.TestCase):
    def setUp(self):
        import torch

        from agent_r1.agent_flow.agent_flow import AgentFlowOutput, AgentFlowWorkerBase, _InternalAgentFlowStep

        self.torch = torch
        self.output_cls = AgentFlowOutput
        self.step_cls = _InternalAgentFlowStep
        self.worker = object.__new__(AgentFlowWorkerBase)

    def make_step(self, reward=0.75, mask=None, extra_fields=None):
        tensor = self.torch.tensor
        return self.step_cls(
            prompt_ids=tensor([[1, 2]]),
            response_ids=tensor([[3, 77, 4]]),
            input_ids=tensor([[1, 2, 3, 77, 4]]),
            attention_mask=tensor([[1, 1, 1, 1, 1]]),
            position_ids=tensor([[0, 1, 2, 3, 4]]),
            response_mask=tensor([mask or [1, 0, 1]]),
            reward_score=reward,
            extra_fields=extra_fields or {},
        )

    def test_real_flattening_provenance_end_markers_and_last_action_reward(self):
        outputs = [
            self.output_cls(
                steps=[
                    self.make_step(0),
                    self.make_step(
                        0.75,
                        extra_fields={
                            "source_uid": "spoof",
                            "trajectory_uids": "spoof",
                            "step_indices": 99,
                        },
                    ),
                ],
                metrics={},
                source_uid="a",
                rollout_mode="greedy",
                terminated=True,
                termination_reason="env_done",
            ),
            self.output_cls(
                steps=[self.make_step(2)],
                metrics={},
                source_uid="b",
                rollout_mode="greedy",
                truncated=True,
                termination_reason="max_steps",
            ),
        ]
        result = self.worker._postprocess(outputs)
        fields = result.non_tensor_batch
        self.assertEqual(fields["source_uid"].tolist(), ["a", "a", "b"])
        self.assertEqual(fields["step_indices"].tolist(), [0, 1, 0])
        self.assertEqual(fields["terminated"].tolist(), [False, True, False])
        self.assertEqual(fields["truncated"].tolist(), [False, False, True])
        self.assertEqual(fields["termination_reason"].tolist(), ["ongoing", "env_done", "max_steps"])
        self.assertEqual(result.batch["rm_scores"].tolist(), [[0, 0, 0], [0, 0, 0.75], [0, 0, 2]])
        self.assertEqual(fields["trajectory_uids"][0], fields["trajectory_uids"][1])
        self.assertNotEqual(fields["trajectory_uids"][1], fields["trajectory_uids"][2])

    def test_real_flattening_rejects_empty_trajectory_and_response(self):
        for steps in ([], [self.make_step(mask=[0, 0, 0])]):
            with self.subTest(steps=steps), self.assertRaises(RuntimeError):
                self.worker._postprocess([self.output_cls(steps=steps, metrics={}, source_uid="a")])

    def test_real_distributed_concat_keeps_new_step_metadata(self):
        from verl import DataProto

        outputs = []
        for uid in ("a", "b"):
            outputs.append(
                self.worker._postprocess(
                    [
                        self.output_cls(
                            steps=[self.make_step()],
                            metrics={},
                            source_uid=uid,
                            rollout_mode="greedy",
                            terminated=True,
                            termination_reason="env_done",
                        )
                    ]
                )
            )
        combined = DataProto.concat(outputs)
        self.assertEqual(combined.non_tensor_batch["source_uid"].tolist(), ["a", "b"])
        self.assertEqual(combined.non_tensor_batch["rollout_mode"].tolist(), ["greedy", "greedy"])


if __name__ == "__main__":
    unittest.main()
