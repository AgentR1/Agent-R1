# ReMax: complete greedy rollout collection

This first stage provides collection APIs, not a complete ReMax training path.
The trainer still uses its existing sampler; setting `algorithm.adv_estimator=remax`
still raises `NotImplementedError`. Advantage computation and actor loss integration
are a separate next step.

## Using the collection APIs

Use the initialized `AgentFlowManager` with an **unrepeated** generation batch:

```python
# Inside an initialized trainer, before gen_batch.repeat(...).
# Use the normal _get_gen_batch preparation and task metadata.
manager = trainer.async_rollout_manager
greedy = manager.generate_greedy_sequences(gen_batch)

# Alternatively, collect both paths and pair their rewards in one call.
collection = manager.collect_remax_rollouts(gen_batch, num_samples=2)
sampled = collection.sampled
greedy = collection.greedy
baseline_per_sampled_step = sampled.batch["reward_baselines"]
baseline_by_task = collection.baselines
```

These are two alternative entry points, not two calls needed for one collection.
`num_samples` defaults to `actor_rollout_ref.rollout.n`. Collection runs one greedy
episode per original task, then `num_samples` independently reset sampled episodes.
Do not update actor weights or change environment/reward configuration between these
passes. No advantage, return, or actor update is computed here.

Each greedy step generates with request-local `temperature=0`, `top_p=1`, and
`logprobs=False`. Validation temperature cannot override greedy mode. Both sampling
and greedy requests now explicitly set `max_tokens=rollout.response_length`, so the
backend budget matches the response slice used by the flow. This replaces the previous
implicit backend budget; it can affect sampled episodes that formerly generated more
tokens than their locally retained response. Global sampling settings are unchanged.
Backend numerical nondeterminism and stochastic external environments are not removed
by greedy token selection.

## Inspecting a complete trajectory

Outputs remain flattened step-level `DataProto` objects. In `non_tensor_batch`:

- `source_uid` identifies the original task; duplicate original `uid` values are rejected.
- `trajectory_uids` identifies an episode, and `step_indices` orders its steps.
- `rollout_mode` distinguishes `greedy` from `sample`.
- `terminated`, `truncated`, and `termination_reason` are episode end markers on the
  last step only; previous steps have `False`, `False`, and `ongoing`.
- `generation_stop_reason`, `response_at_limit`, and `response_truncated` describe
  each model generation.

Group rows by `trajectory_uids`, then sort by `step_indices` to inspect that episode's
prompts, actions, and immediate rewards. The next prompt is built from feedback to
that episode's own action; greedy does not reuse sampled tool feedback.

The baseline is the undiscounted sum of all existing step rewards in that greedy
episode (`rm_scores` masked by `response_mask`), not just the last step. Pairing uses
`source_uid`, never row order or equal episode length. The resulting scalar is
broadcast to every sampled step for that task. It is a baseline, **not an advantage**.

## End states and failures

Built-in single-step, generic environment, HotpotQA, PaperSearch, ALFWorld, and WebShop
flows report end states. Natural final answers/environment `done` are distinguished
from `max_steps`, `prompt_length`, response budget, and environment time limits.
`terminated` means the episode ended naturally, not that the task succeeded.
Existing recipe reward rules, including ALFWorld's synthetic final reward row, are
preserved; baseline summation uses those existing rows.

An explicitly truncated episode keeps its existing finite-horizon reward. Some verl
backends collapse `stop` and `length` into `completed`; without a raw finish reason,
full-budget responses without a final EOS are conservatively flagged as truncated.
This fallback can also flag a natural stop at the exact budget.

Empty/aborted generations, missing or duplicate greedy episodes, noncontiguous steps,
unknown episode end status, and nonfinite rewards raise errors. They are not silently
replaced with a zero baseline. A custom flow must declare an explicit end status and
propagate per-generation metadata to participate in collection.

## Verification

```bash
python3 -B -m unittest discover -s tests -v
```

Dependency-light tests execute the production orchestration and environment-loop
method bodies with explicit test doubles. Optional real DataProto/Pydantic/CPU-tensor
tests run when the verl/PyTorch stack is installed. Neither suite is a GPU model or
live recipe environment integration test. Verify on a configured training machine
before connecting collection to advantage computation and actor updates.
