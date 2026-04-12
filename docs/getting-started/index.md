# Getting Started

Agent-R1 shares the same runtime environment requirements as `verl`, but the first successful run still depends on a few Agent-R1-specific checks: using the right script, pointing to the right dataset, and understanding which knobs matter first.

## In This Section

- [`Installation Guide`](installation-guide.md): reuse the official `verl` environment and verify that Agent-R1 is runnable from the repository root.
- [`Quick Start`](quick-start.md): prepare a small GSM8K dataset and run the single-step sanity check script.
- [`Troubleshooting`](troubleshooting.md): fix the common first-run failures before digging into the codebase.

## Recommended Path

1. Set up the environment by following the `verl` installation guide.
2. Run the single-step sanity check to confirm that your model path, dependencies, and training stack are wired correctly.
3. Move to the Agent tutorial once the setup is stable.

## Before You Start

Make sure you already know the answer to these four questions:

- Which Python environment contains `verl==0.7.0`?
- Which GPUs will you use, and how many of them are visible through `CUDA_VISIBLE_DEVICES`?
- Which model checkpoint will you put into `actor_rollout_ref.model.path`?
- Where will you store the generated parquet datasets, for example under `$HOME/data/`?

If any of those are unclear, start with the installation guide instead of the tutorial pages.
