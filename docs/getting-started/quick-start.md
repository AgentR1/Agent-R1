# Quick Start

This quick start is a **sanity check**, not the main Agent-R1 workflow. Its purpose is to verify that your environment, dataset path, model path, and training stack are wired correctly.

## 1. Prepare a Minimal Dataset

Use the GSM8K preprocessing script:

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

This produces:

- `~/data/gsm8k/train.parquet`
- `~/data/gsm8k/test.parquet`

## 2. Run the Sanity Check Script

Use the provided single-step script:

```bash
bash examples/run_qwen2.5-3b.sh
```

If needed, adjust the following values before running:

- `CUDA_VISIBLE_DEVICES`
- `actor_rollout_ref.model.path`
- dataset paths under `~/data/gsm8k`

The script entrypoint is [`examples/run_qwen2.5-3b.sh`](https://github.com/AgentR1/Agent-R1/blob/main/examples/run_qwen2.5-3b.sh), which launches `python3 -m agent_r1.main_agent_ppo`.

## 3. What to Do Next

- Read [`Step-level MDP`](../core-concepts/step-level-mdp.md) to understand the main training abstraction.
- Read [`Layered Abstractions`](../core-concepts/layered-abstractions.md) to see how `AgentFlowBase`, `AgentEnvLoop`, and `ToolEnv` fit together.
- Continue to the [`Agent Task Tutorial`](../tutorials/agent-task.md) for the main Agent-R1 workflow based on multi-step interaction.
