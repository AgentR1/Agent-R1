# Agent Task Tutorial

This tutorial shows the simplest multi-step tool-calling path in Agent-R1: **GSM8K + Tool** built with the generic `AgentEnvLoop`, a recipe-local `ToolEnv`, and a recipe-local `BaseTool`.

The example uses GSM8K, but the important part is not the benchmark itself. The goal is to show how Agent-R1 turns a dataset row into an environment-driven, multi-step rollout.

## What You Will Run

This tutorial uses two existing files:

- dataset preprocessing: [`recipes/gsm8k/data_preprocess/process_gsm8k_tool.py`](https://github.com/AgentR1/Agent-R1/blob/main/recipes/gsm8k/data_preprocess/process_gsm8k_tool.py)
- training script: [`examples/gsm8k/run_steppo_tool.sh`](https://github.com/AgentR1/Agent-R1/blob/main/examples/gsm8k/run_steppo_tool.sh)

## 1. Prepare the Tool Dataset

Generate the tool-augmented GSM8K dataset:

```bash
python3 -m recipes.gsm8k.data_preprocess.process_gsm8k_tool --local_save_dir ~/data/gsm8k_tool
```

Compared with the single-step sanity-check dataset, this preprocessing script keeps structured task fields for the tool path:

- `agent_name: "gsm8k_tool"`
- `question` and `ground_truth`, plus a tool-calling prompt stored in `prompt`
- `env_kwargs` for per-sample tool metadata such as the ground-truth answer

Conceptually, each sample says:

1. use the configured `gsm8k_tool` entry, which points to the generic `AgentEnvLoop`
2. instantiate the built-in tool environment
3. expose the `calc_gsm8k_reward` tool inside that environment

## 2. Launch the Agent Task Training Script

Run:

```bash
bash examples/gsm8k/run_steppo_tool.sh
```

This script switches the rollout from single-step generation to the generic agent-environment loop:

```bash
actor_rollout_ref.rollout.agent.default_agent_flow=gsm8k_tool \
actor_rollout_ref.rollout.agent.max_steps=5 \
```

It also points the trainer to the tool dataset:

```bash
data.train_files=$HOME/data/gsm8k_tool/train.parquet \
data.val_files=$HOME/data/gsm8k_tool/test.parquet \
```

## 3. What Happens During One Trajectory

At a high level, one sample follows this path:

```mermaid
graph TD
    datasetRow["Dataset row"] --> agentFlow["AgentEnvLoop"]
    agentFlow --> toolEnv["ToolEnv"]
    toolEnv --> llmStep["LLM response"]
    llmStep --> toolCall["Tool call parsing"]
    toolCall --> toolExec["BaseTool execution"]
    toolExec --> nextObs["Next observation"]
    nextObs --> agentFlow
```

More concretely:

1. `AgentEnvLoop` reads recipe defaults and per-sample `env_kwargs`.
2. `AgentEnv.from_config(env_type="tool", ...)` creates the built-in `ToolEnv`.
3. `ToolEnv.reset()` uses the chat prompt stored in the dataset row.
4. The LLM produces a response.
5. `ToolEnv.step()` parses tool calls from the response and executes the registered tool.
6. Tool output is appended to the conversation as the next observation.
7. The loop continues until the environment returns `done=True` or `max_steps` is reached.

## 4. Where the Reward Comes From

The GSM8K tool is registered as `calc_gsm8k_reward` in `recipes/gsm8k/tool.py`.

Its role in this example is to:

- receive the model's proposed answer
- compare it with the sample's ground truth
- return tool text back into the conversation

This is what makes the tutorial useful for Agent-R1: the model is not just generating one final answer, it is interacting with an environment that can evaluate and feed back information across multiple steps.

## 5. Why This Tutorial Is Separate From the Single-Step Script

The single-step GSM8K script is still useful as a setup check. This tutorial is different: it is the minimal example for the lowest abstraction layer, where users only define tools and let `ToolEnv + BaseTool` handle standard multi-turn tool interaction. It demonstrates:

- a step-level environment transition
- a multi-step agent loop
- tool-augmented interaction
- reward signals attached to environment-mediated behavior

## 6. Where to Look Next

- Read [`Step-level MDP`](../core-concepts/step-level-mdp.md) to connect this tutorial to the core RL formulation.
- Read [`Layered Abstractions`](../core-concepts/layered-abstractions.md) to see why this example maps naturally to `ToolEnv + BaseTool`.
- Read [`Recipes and Algorithms`](recipes-and-algorithms.md) to see the other task recipes and launch scripts.

## 7. Use Built-in Web Search Tools

`ToolEnv` includes web search and page fetching through [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp).
Anonymous search uses Fast mode, with no account or API key required. Install the optional dependencies from the repository root:

```bash
pip install -r requirements-search.txt
```

For a new task using `AgentEnvLoop`, set `env_type: tool` in its environment configuration. Omitting `tools` exposes
`parallel_search` and `parallel_fetch` by default. Existing explicit tool lists remain authoritative; `tools: []` disables
all tools. The default agent flow and existing recipe configurations stay unchanged.

The same environment can be used directly:

```python
from agent_r1.env import AgentEnv
from agent_r1.env.base import Action

env = AgentEnv.from_config(env_type="tool")
env.reset(raw_prompt=[{"role": "user", "content": "Find the current Agent-R1 documentation."}])
# In an async function, execute a model-style tool call and read its observation:
observation, reward, done, info = await env.step(Action(
    text='<tool_call>{"name":"parallel_search","arguments":{"query":"Agent R1 documentation"}}</tool_call>'
))
print(observation.messages[-1]["content"])
```

`parallel_search` takes a nonempty `query` string and returns source URLs and excerpts. `parallel_fetch` takes one HTTP or HTTPS
`url` and an optional `objective` (up to 200 characters). Both return text observations through the selected tool format.
An environment reset starts a new research session; related search and fetch calls within that environment reuse its session ID.
Concurrent environments have independent sessions.

Use `tools=["parallel_search", "parallel_fetch", "your_registered_tool"]` to combine web access with task-specific tools.
Keep task rewards and dataset preparation in your recipe. Existing retrieval recipes continue to use their configured services.

Optional settings live under `tools_kwargs.parallel`:

```yaml
env_type: tool
tools_kwargs:
  parallel:
    timeout: 30       # Total seconds allowed for each MCP request, including setup.
    enabled: true    # Set false to block web requests even if a web tool is selected.
```

For higher service limits, supply `PARALLEL_API_KEY` or `tools_kwargs.parallel.api_key`. An explicit `api_key` overrides the
environment variable, including an empty string to select anonymous access. Keep credentials out of datasets and prompts.
The anonymous endpoint has usage limits. Service errors become tool observations; the tools do not switch providers or retry
automatically. Cancellation propagates to the caller. Dependencies are loaded only when a web tool runs, so local-only recipes
can continue without installing the search extras.
