# Quick Start

This quick start is a **sanity check**, not the main Agent-R1 workflow. Its purpose is to verify that your environment, dataset path, model path, and training stack are wired correctly.

## Before You Run

Make sure all of the following are true:

- your Python environment already works for `verl==0.7.0`
- you are running commands from the Agent-R1 repository root
- you know which model checkpoint to use for `actor_rollout_ref.model.path`
- you have decided where the GSM8K parquet files will be written

## 1. Prepare a Minimal Dataset

Use the GSM8K preprocessing script:

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

This produces:

- `~/data/gsm8k/train.parquet`
- `~/data/gsm8k/test.parquet`

The generated rows contain the minimum fields needed by the single-step script:

- `prompt`: the chat-style input messages
- `reward_model.ground_truth`: the target answer used for evaluation
- `extra_info`: bookkeeping such as split, sample index, and original question text

## 2. Run the Sanity Check Script

Use the provided single-step script:

```bash
bash examples/run_qwen2.5-3b.sh
```

Before running, check these values in `examples/run_qwen2.5-3b.sh`:

- `CUDA_VISIBLE_DEVICES`
- `actor_rollout_ref.model.path`
- `trainer.n_gpus_per_node`
- dataset paths under `~/data/gsm8k`

The single-step script is intentionally small enough to use as a configuration template. If your hardware is smaller than the example assumes, reduce these first:

- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
- `actor_rollout_ref.rollout.n`

The script entrypoint is [`examples/run_qwen2.5-3b.sh`](https://github.com/AgentR1/Agent-R1/blob/main/examples/run_qwen2.5-3b.sh), which launches `python3 -m agent_r1.main_agent_ppo`.

## 3. What Success Looks Like

The point of this run is not benchmark quality. It is only a smoke test that confirms:

- Ray initializes successfully
- the model can be loaded
- the parquet dataset can be read
- the trainer can enter the normal training and validation loop

Two details that often confuse first-time users:

- the script uses `trainer.logger='["console"]'`, so the main signal is in terminal logs
- the script sets `trainer.save_freq=-1`, so no periodic checkpoint is written unless you change that setting

## 4. What to Do Next

- Read [`Step-level MDP`](../core-concepts/step-level-mdp.md) to understand the main training abstraction.
- Read [`Layered Abstractions`](../core-concepts/layered-abstractions.md) to see how `AgentFlowBase`, `AgentEnvLoop`, and `ToolEnv` fit together.
- Continue to the [`Agent Task Tutorial`](../tutorials/agent-task.md) for the main Agent-R1 workflow based on multi-step interaction.
- If the smoke test fails, start with [`Troubleshooting`](troubleshooting.md) before changing the core code.
