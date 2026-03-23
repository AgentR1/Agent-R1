# Layered Abstractions

## From Maximum Flexibility to Out-of-the-Box

Agent-R1 provides a **layered abstraction** system. Each layer adds more structure and convention while lowering the barrier to entry. The key design choice is that all layers still fit the same step-level RL view.

```mermaid
graph TD
    agentFlowBase["AgentFlowBase"]
    singleStep["SingleStepAgentFlow"]
    customWorkflow["Your Custom Workflow"]
    agentEnvLoop["AgentEnvLoop"]
    agentEnv["AgentEnv"]
    toolEnv["ToolEnv"]
    baseTool["BaseTool"]
    customEnv["Your Custom Env"]

    agentFlowBase -->|"subclass"| singleStep
    agentFlowBase -->|"subclass"| customWorkflow
    agentFlowBase -->|"subclass"| agentEnvLoop
    agentEnvLoop -->|"uses"| agentEnv
    agentEnv -->|"built-in"| toolEnv
    agentEnv -->|"subclass"| customEnv
    toolEnv -->|"uses"| baseTool
```

## Layer 1: `AgentFlowBase`

Subclass `AgentFlowBase` when you want full control over how prompts are built, how the LLM is called, and how steps are assembled into an `AgentFlowOutput`.

This is the most flexible layer, but it is also the lowest-level one. It is useful for custom workflows and experiments where you do not want to model the task explicitly as an environment.

```python
from agent_r1.agent_flow import AgentFlowBase, AgentFlowOutput

class MyWorkflow(AgentFlowBase):
    async def run(self, sampling_params, **kwargs):
        ...
        return AgentFlowOutput(steps=[step1, step2], metrics=metrics)
```

## Layer 2: `AgentEnvLoop + AgentEnv`

This is the main abstraction for Agent-R1.

When your task can be written as an environment with `reset()` and `step()`, use `AgentEnvLoop`. The loop handles the LLM generation, while the environment controls the next observation and reward.

This aligns directly with the step-level MDP idea:

- the environment returns an `Observation`
- the LLM produces an `Action`
- the environment computes the next observation, reward, and termination condition

```python
from agent_r1.env import AgentEnv, Observation, Action

@AgentEnv.register("my_env")
class MyEnv(AgentEnv):
    def reset(self, **kwargs) -> Observation:
        return Observation(messages=[...])

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        ...
        return Observation(messages=[...]), reward, done, info
```

The relevant implementation lives in:

- `agent_r1/agent_flow/agent_env_loop.py`
- `agent_r1/env/base.py`

## Layer 3: `ToolEnv + BaseTool`

Many Agent-R1 tasks are naturally expressed as **tool-augmented multi-step interaction**. For this case, Agent-R1 provides `ToolEnv`, a built-in environment that:

- stores conversation history
- parses tool calls from model output
- executes registered tools
- feeds tool observations back into the next turn

Tools are defined independently through `BaseTool`.

```python
from agent_r1.tool import BaseTool, ToolResponse

@BaseTool.register("calculator")
class Calculator(BaseTool):
    name = "calculator"
    description = "Evaluate a math expression."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The math expression"}
        },
        "required": ["expression"],
    }

    async def execute(self, args, **kwargs) -> tuple[ToolResponse, float | None, dict]:
        ...
```

The relevant implementation lives in:

- `agent_r1/env/envs/tool.py`
- `agent_r1/tool/base.py`
- `agent_r1/tool/tools/gsm8k.py`

## What Matters in This Version

For the current lightweight documentation, the key takeaway is:

- `SingleStepAgentFlow` exists and is useful for sanity checks.
- `AgentEnvLoop` is the center of the framework design.
- `ToolEnv + BaseTool` is the most direct way to build multi-step agent tasks today.
