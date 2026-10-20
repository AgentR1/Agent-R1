# 智能体任务教程

本教程展示 Agent-R1 中最简单的多步工具调用路径：基于通用 `AgentEnvLoop`、recipe-local `ToolEnv` 和 recipe-local `BaseTool` 实现的 **GSM8K + Tool**。

示例使用 GSM8K，但重点不是这个 benchmark 本身，而是展示 Agent-R1 如何把一行数据变成环境驱动的多步 rollout。

## 你将运行什么

本教程使用两个已有文件：

- 数据预处理：[`recipes/gsm8k/data_preprocess/process_gsm8k_tool.py`](https://github.com/AgentR1/Agent-R1/blob/main/recipes/gsm8k/data_preprocess/process_gsm8k_tool.py)
- 训练脚本：[`examples/gsm8k/run_steppo_tool.sh`](https://github.com/AgentR1/Agent-R1/blob/main/examples/gsm8k/run_steppo_tool.sh)

## 1. 准备工具数据集

生成工具增强的 GSM8K 数据：

```bash
python3 -m recipes.gsm8k.data_preprocess.process_gsm8k_tool --local_save_dir ~/data/gsm8k_tool
```

相比单步 sanity-check 数据，这个预处理脚本会为 tool path 保留结构化任务字段：

- `agent_name: "gsm8k_tool"`
- `question` 和 `ground_truth`，以及保存在 `prompt` 中的工具调用提示
- `env_kwargs`，用于保存每个样本的工具元数据，例如 ground-truth answer

概念上，每个样本会表达：

1. 使用配置中的 `gsm8k_tool` 入口，这个入口指向通用 `AgentEnvLoop`
2. 实例化内置 tool environment
3. 在环境中暴露 `calc_gsm8k_reward` 工具

## 2. 启动智能体训练脚本

运行：

```bash
bash examples/gsm8k/run_steppo_tool.sh
```

这个脚本会把 rollout 从单步生成切换到通用 agent-environment loop：

```bash
actor_rollout_ref.rollout.agent.default_agent_flow=gsm8k_tool \
actor_rollout_ref.rollout.agent.max_steps=5 \
```

同时，它会把 trainer 指向 tool 数据集：

```bash
data.train_files=$HOME/data/gsm8k_tool/train.parquet \
data.val_files=$HOME/data/gsm8k_tool/test.parquet \
```

## 3. 一条轨迹中会发生什么

从高层看，一个样本会经过以下路径：

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

更具体地说：

1. `AgentEnvLoop` 读取 recipe defaults 和每个样本的 `env_kwargs`。
2. `AgentEnv.from_config(env_type="tool", ...)` 创建内置 `ToolEnv`。
3. `ToolEnv.reset()` 使用数据行中保存的 chat prompt。
4. LLM 生成回复。
5. `ToolEnv.step()` 从回复中解析工具调用，并执行注册工具。
6. 工具输出会作为下一步 observation 加入对话。
7. 循环持续到环境返回 `done=True` 或达到 `max_steps`。

## 4. 奖励来自哪里

GSM8K 工具在 `recipes/gsm8k/tool.py` 中注册为 `calc_gsm8k_reward`。

它在这个示例中的作用是：

- 接收模型提出的答案
- 与样本 ground truth 比较
- 将工具文本返回到对话中

这正是本教程对 Agent-R1 有价值的地方：模型不只是生成一个最终答案，而是在可以评估并反馈信息的环境中进行多步交互。

## 5. 为什么这个教程与单步脚本分开

单步 GSM8K 脚本仍然有用，但它主要是 setup check。本教程不同：它是最低抽象层的最小示例，用户只需要定义工具，标准多轮工具调用由 `ToolEnv + BaseTool` 承担。它展示了：

- step-level 环境转移
- 多步 agent loop
- 工具增强交互
- 与环境调解行为绑定的奖励信号

## 6. 接下来阅读

- 阅读 [`Step-level MDP`](../core-concepts/step-level-mdp.md)，将本教程与核心 RL formulation 对齐。
- 阅读 [`分层抽象`](../core-concepts/layered-abstractions.md)，理解这个示例为什么自然对应 `ToolEnv + BaseTool`。
- 阅读 [`Recipes 与算法`](recipes-and-algorithms.md)，查看其他任务 recipe 和启动脚本。

## 7. 使用内置网页搜索工具

`ToolEnv` 通过 [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) 提供网页搜索与页面内容提取。
匿名搜索使用 Fast 模式，无需账号或 API key。在仓库根目录安装可选依赖：

```bash
pip install -r requirements-search.txt
```

对于使用 `AgentEnvLoop` 的新任务，在环境配置中设置 `env_type: tool`。省略 `tools` 时，默认提供
`parallel_search` 和 `parallel_fetch`。显式配置的工具列表始终优先；`tools: []` 禁用所有工具。
默认 agent flow 和已有 recipe 配置保持不变。

也可以直接使用同一个环境：

```python
from agent_r1.env import AgentEnv
from agent_r1.env.base import Action

env = AgentEnv.from_config(env_type="tool")
env.reset(raw_prompt=[{"role": "user", "content": "查找当前 Agent-R1 文档。"}])
# 在异步函数中执行模型格式的工具调用，并读取 observation：
observation, reward, done, info = await env.step(Action(
    text='<tool_call>{"name":"parallel_search","arguments":{"query":"Agent R1 documentation"}}</tool_call>'
))
print(observation.messages[-1]["content"])
```

`parallel_search` 接收非空的 `query` 字符串，返回来源 URL 和相关摘录。`parallel_fetch` 接收一个 HTTP 或 HTTPS
`url`，以及可选的 `objective`（最多 200 个字符）。两个工具均通过所选工具格式返回文本 observation。
环境 reset 会开始新的研究会话；同一环境中的相关搜索和页面提取调用复用会话 ID，并发环境各自拥有独立会话。

使用 `tools=["parallel_search", "parallel_fetch", "your_registered_tool"]` 可以组合网页访问和任务专用工具。
奖励与数据集处理仍在任务 recipe 中实现。已有检索 recipe 继续使用其配置的服务。

可选设置放在 `tools_kwargs.parallel` 下：

```yaml
env_type: tool
tools_kwargs:
  parallel:
    timeout: 30       # 每次 MCP 请求（包含连接初始化）的总超时秒数。
    enabled: true    # 设为 false 时，即使选择了网页工具也不会发送请求。
```

如需更高服务限额，可设置 `PARALLEL_API_KEY` 或 `tools_kwargs.parallel.api_key`。
显式 `api_key` 优先于环境变量；空字符串表示使用匿名访问。请勿将凭据放入数据集或 prompt。
匿名端点有用量限制。服务错误会作为工具 observation 返回；工具不会切换提供商，也不会自动重试。
取消操作会传播到调用方。依赖仅在网页工具执行时加载，因此只使用本地工具的 recipe 无需安装搜索依赖。
