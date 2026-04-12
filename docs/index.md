# Agent-R1

## Training Powerful LLM Agents with End-to-End Reinforcement Learning

Agent-R1 is an open-source framework for training powerful language agents with end-to-end reinforcement learning. With Agent-R1, you can build custom agent workflows, define interactive environments and tools, and train multi-step agents in a unified RL pipeline.

## Fast Path

If you want the shortest route from clone to a real Agent-R1 run, use this order:

1. Follow [`Installation Guide`](getting-started/installation-guide.md) to reuse the same environment as `verl==0.7.0`.
2. Run [`Quick Start`](getting-started/quick-start.md) to verify dataset paths, model paths, and the base training stack.
3. Move to [`Agent Task Tutorial`](tutorials/agent-task.md) to run the multi-step `AgentEnvLoop + ToolEnv` workflow.

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **Step-level MDP**

    ---

    A principled MDP formulation that enables flexible context management and per-step reward signals.

    [:octicons-arrow-down-24: Learn more](core-concepts/step-level-mdp.md)

-   :material-layers-outline:{ .lg .middle } **Layered Abstractions**

    ---

    From maximum flexibility to out-of-the-box, choose the right level of abstraction for your use case.

    [:octicons-arrow-down-24: Learn more](core-concepts/layered-abstractions.md)

-   :material-rocket-launch-outline:{ .lg .middle } **First Successful Run**

    ---

    Start from environment setup, then run the single-step smoke test before touching the agent loop.

    [:octicons-arrow-down-24: Open Getting Started](getting-started/index.md)

-   :material-robot-outline:{ .lg .middle } **Main Agent Workflow**

    ---

    Learn how a dataset row becomes a multi-step trajectory through `AgentEnvLoop`, `ToolEnv`, and tools.

    [:octicons-arrow-down-24: Open Tutorial](tutorials/agent-task.md)

</div>

---

## Reading Guide

- Start with [`Getting Started`](getting-started/index.md) if you want the minimal path: use the same environment as `verl`, run a sanity check, and confirm the repository is ready.
- Read [`Step-level MDP`](core-concepts/step-level-mdp.md) and [`Layered Abstractions`](core-concepts/layered-abstractions.md) if you want to understand the framework design before touching code.
- Follow [`Agent Task Tutorial`](tutorials/agent-task.md) if you want to see the main Agent-R1 workflow: multi-step interaction through `AgentEnvLoop` and `ToolEnv`.

## Scope of This Documentation

This documentation focuses on the current design center of Agent-R1:

- getting a clean first run on top of the `verl` environment
- understanding the step-level abstractions that make multi-step agent RL possible
- extending the framework through dataset-side `env_kwargs`, environments, and tools

It is still deliberately scoped, but the goal is now practical completeness for the main workflow rather than minimalism for its own sake.

---

<div style="text-align: center; color: #888; margin-top: 2em;" markdown>
Supported by the [State Key Laboratory of Cognitive Intelligence](https://cogskl.iflytek.com/){ target=_blank }, University of Science and Technology of China (USTC).
</div>
