# Troubleshooting

This page collects the failures that are most likely to happen during the first Agent-R1 run.

## `ModuleNotFoundError` for `verl` or `agent_r1`

What it usually means:

- the Python environment does not contain `verl==0.7.0`
- the command is not being run from the repository root

What to check:

```bash
python3 -c "import verl, agent_r1; print('verl:', verl.__file__); print('agent_r1:', agent_r1.__file__)"
pwd
```

Run Agent-R1 commands from the repository root so Python can resolve the local `agent_r1/` package.

## The script cannot find the model

What it usually means:

- `actor_rollout_ref.model.path` still points to the example model name
- the runtime cannot access the remote model hub you expect

What to check:

- replace `actor_rollout_ref.model.path` in the example script with a model path you can actually access
- keep or remove `HF_ENDPOINT` depending on your environment

The two example scripts are:

- `examples/run_qwen2.5-3b.sh`
- `examples/run_qwen3-4b_gsm8k_tool.sh`

## Ray or trainer settings do not match the visible GPUs

What it usually means:

- `CUDA_VISIBLE_DEVICES` exposes fewer GPUs than `trainer.n_gpus_per_node`
- the per-GPU micro batch sizes are too aggressive for your hardware

What to check first:

- make `trainer.n_gpus_per_node` match the number of visible GPUs
- reduce `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- reduce `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
- reduce `actor_rollout_ref.rollout.n`

## Out-of-memory during rollout or training

The fastest levers are:

- reduce `data.max_prompt_length`
- reduce `data.max_response_length`
- reduce `actor_rollout_ref.rollout.prompt_length`
- reduce `actor_rollout_ref.rollout.response_length`
- reduce `actor_rollout_ref.rollout.n`
- reduce the per-GPU micro batch sizes

The agent example is intentionally larger than the single-step sanity check. If it OOMs, get the single-step script stable first.

## Prompt length exceeds the configured limit

`AgentEnvLoop` stops the trajectory early when the prompt becomes longer than `actor_rollout_ref.rollout.prompt_length`.

What to do:

- increase `actor_rollout_ref.rollout.prompt_length` if memory allows
- shorten the initial prompt or tool observations
- reduce `actor_rollout_ref.rollout.agent.max_steps` if your environment produces long histories quickly

This check happens inside `agent_r1/agent_flow/agent_env_loop.py`.

## Tool-related errors such as `tool not found` or missing `ground_truth`

What it usually means:

- the dataset row names a tool that was not registered
- `env_kwargs` does not contain the configuration expected by the tool

For the GSM8K tutorial, each row must provide tool configuration in `env_kwargs`, including:

```json
{
  "env_type": "tool",
  "tools": ["calc_gsm8k_reward"],
  "tool_format": "hermes",
  "tools_kwargs": {
    "ground_truth": "..."
  }
}
```

Relevant implementation files:

- `examples/data_preprocess/gsm8k_tool.py`
- `agent_r1/env/envs/tool.py`
- `agent_r1/tool/tools/gsm8k.py`

## The run finishes but no checkpoints are written

This is expected for the example scripts. They both set:

```bash
trainer.save_freq=-1
```

If you want periodic checkpoints, replace that value with a positive interval.

## You want to debug the docs locally

From the repository root:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Then open the local address printed by MkDocs.
