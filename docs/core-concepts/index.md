# Core Concepts

This section introduces the ideas that shape Agent-R1 as a framework for agent tasks.

## In This Section

- [`Step-level MDP`](step-level-mdp.md): why Agent-R1 models agent training as multi-step interaction instead of a single growing token stream.
- [`Step-Level Trajectory Representation`](step-level-trajectory-representation.md): how Agent-R1 stores and replays interaction history without collapsing everything into one growing token stream.
- [`Step-Level Credit Assignment`](step-level-credit-assignment.md): why reward propagation should follow interaction steps rather than only tokens or whole trajectories.
- [`Layered Abstractions`](layered-abstractions.md): how `AgentFlowBase`, `AgentEnvLoop`, `AgentEnv`, `ToolEnv`, and `BaseTool` fit together.

## Why These Concepts Matter

Agent-R1 is designed for agent tasks where an LLM interacts with an environment, receives new observations, and improves through reinforcement learning over trajectories. Together, these pages explain the framework's step-level MDP, trajectory representation, credit assignment, and programming model.
