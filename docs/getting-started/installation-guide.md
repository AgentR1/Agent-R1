# Installation Guide

Agent-R1 uses the same environment setup as `verl`, so the correct mental model is:

- install and validate the base `verl` runtime first
- clone Agent-R1 and run it directly from the repository root
- only then start changing datasets, models, and rollout settings

## 1. Prepare the Base Environment

Follow the official [`verl` installation guide](https://verl.readthedocs.io/en/latest/start/install.html), but make sure the environment ends up with `verl==0.7.0`.

If you want a broader overview of the base training workflow, the [`verl` quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html) is also useful.

## 2. Clone Agent-R1

```bash
git clone https://github.com/AgentR1/Agent-R1.git
cd Agent-R1
```

You do not need a separate `pip install agent-r1`. The example scripts run `python3 -m agent_r1.main_agent_ppo` directly from this source tree.

## 3. Confirm the Runtime Is Visible

From the repository root, run a lightweight import check:

```bash
python3 -c "import verl, agent_r1; print('verl ok:', verl.__file__); print('agent_r1 ok:', agent_r1.__file__)"
```

If this fails, fix the Python environment before trying the training scripts.

## 4. Know the First Parameters You Will Edit

Before the first run, you will almost always need to update these values in the example shell scripts:

- `CUDA_VISIBLE_DEVICES`: choose the GPUs you actually want to use
- `actor_rollout_ref.model.path`: point to a model checkpoint available in your environment
- `trainer.n_gpus_per_node`: make it match the number of visible GPUs
- dataset paths under `$HOME/data/...`: keep them aligned with where you saved the parquet files

Do this in:

- `examples/run_qwen2.5-3b.sh`
- `examples/run_qwen3-4b_gsm8k_tool.sh`

## 5. Optional: Preview This Documentation Locally

If you are editing the docs themselves:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

This serves the documentation locally with the same MkDocs configuration used for the published site.

## What This Means for Agent-R1

Once the `verl` environment is working, Agent-R1 should run in the same environment. The documentation here intentionally focuses on Agent-R1-specific decisions instead of duplicating a second infrastructure guide.
