# Installation Guide

Agent-R1 uses the same environment setup as `verl`.

## Base Environment

Follow the official [`verl` installation guide](https://verl.readthedocs.io/en/latest/start/install.html), but make sure the environment ends up with `verl==0.7.0`.

If you want a broader overview of the base training workflow, the [`verl` quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html) is also useful.

## What This Means for Agent-R1

Once the `verl` environment is working, Agent-R1 should run in the same environment. In practice, that means you can:

- prepare a Python environment with `verl==0.7.0`
- clone this repository
- run Agent-R1 commands directly from the repository root

You do not need to install Agent-R1 as a separate package.

The documentation in this repository intentionally does not duplicate a separate environment guide, so that the infrastructure setup stays aligned with `verl`.
