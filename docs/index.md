# Agent-R1

## Training Powerful LLM Agents with End-to-End Reinforcement Learning

Agent-R1 is an open-source framework for training powerful language agents with end-to-end reinforcement learning. With Agent-R1, you can build custom agent workflows, define interactive environments and tools, and train multi-step agents in a unified RL pipeline.

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **Step-level MDP**

    ---

    A principled MDP formulation that enables flexible context management and per-step reward signals.

    [:octicons-arrow-down-24: Learn more](core-concepts/step-level-mdp.md)

-   :material-source-branch:{ .lg .middle } **Step-Level Trajectory Representation**

    ---

    See how Agent-R1 represents trajectories at the same semantic level as multi-step interaction.

    [:octicons-arrow-down-24: Learn more](core-concepts/step-level-trajectory-representation.md)

-   :material-chart-timeline-variant:{ .lg .middle } **Step-Level Credit Assignment**

    ---

    See why Agent-R1 propagates reward at the level of interaction steps rather than only tokens.

    [:octicons-arrow-down-24: Learn more](core-concepts/step-level-credit-assignment.md)

-   :material-layers-outline:{ .lg .middle } **Layered Abstractions**

    ---

    From maximum flexibility to out-of-the-box, choose the right level of abstraction for your use case.

    [:octicons-arrow-down-24: Learn more](core-concepts/layered-abstractions.md)

</div>

---

## Reading Guide

- Start with [`Getting Started`](getting-started/index.md) if you want the minimal path: use the same environment as `verl`, run a sanity check, and confirm the repository is ready.
- Read [`Step-Level Training Logic`](background/step-level-training-logic.md) if you want the full conceptual argument behind Agent-R1's step-level perspective.
- Read [`Step-level MDP`](core-concepts/step-level-mdp.md), [`Step-Level Trajectory Representation`](core-concepts/step-level-trajectory-representation.md), [`Step-Level Credit Assignment`](core-concepts/step-level-credit-assignment.md), and [`Layered Abstractions`](core-concepts/layered-abstractions.md) if you want the framework ideas broken into concrete pieces.
- Follow [`Agent Task Tutorial`](tutorials/agent-task.md) if you want to see the main Agent-R1 workflow: multi-step interaction through `AgentEnvLoop` and `ToolEnv`.

## Scope of This Documentation

This version of the documentation is intentionally compact. It focuses on the parts that are already central to Agent-R1 today while making the core design logic more explicit: step-level MDP, step-level trajectory representation, step-level credit assignment, and the layered abstractions used to build agent tasks.

---

<div style="text-align: center; color: #888; margin-top: 2em;" markdown>
Supported by the [State Key Laboratory of Cognitive Intelligence](https://cogskl.iflytek.com/){ target=_blank }, University of Science and Technology of China (USTC).
</div>
