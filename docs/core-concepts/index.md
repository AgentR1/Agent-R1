# Core Concepts

This section introduces the ideas that shape Agent-R1 as a framework for agent tasks.

## In This Section

- [`Step-level MDP`](step-level-mdp.md): why Agent-R1 models agent training as multi-step interaction instead of a single growing token stream.
- [`Layered Abstractions`](layered-abstractions.md): how `AgentFlowBase`, `AgentEnvLoop`, `AgentEnv`, `ToolEnv`, and `BaseTool` fit together.

## Why These Concepts Matter

Agent-R1 is designed for agent tasks where an LLM interacts with an environment, receives new observations, and improves through reinforcement learning over trajectories. These two pages explain the core formulation and the programming model that support that workflow.
