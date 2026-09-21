# 非对称 actor-critic：critic 专用前缀

本功能只改变 critic 输入：actor 的生成、log-prob、reference 输入和 PPO 更新
仍使用原始对话。Critic 的目标是预测当前 actor 的回报，不是假设 actor 也知道答案。

首版支持 `gae`（step-level）、纯文本、相同 tokenizer、现有 legacy FSDP/FSDP2
worker。动态 provider、多模态、token-level GAE、Megatron 和新 engine worker
暂不支持，启用时会明确报错。默认 `asymmetric_critic.enabled=false`，原行为不变。
纯文本 rollout 的空 `multi_modal_inputs` 占位值（逐行 `{}` / `None`）不算多模态；
只从独立 critic 输入中移除该空列，actor 数据不变。真实非空多模态输入及 mRoPE
位置编码仍明确拒绝。
不要与 critic-free ReMax/GRPO 混用。

## 最方便的使用方式：编辑文本文件

修改 `recipes/gsm8k/critic_prefix.txt`，然后在已有训练命令中选择配置预设：

```bash
python -m agent_r1.trainer.main_agent_ppo \
  --config-name asymmetric_ppo_trainer \
  actor_rollout_ref.model.path=/path/to/model \
  critic.model.path=/path/to/model \
  data.train_files=/path/to/gsm8k/train.parquet \
  data.val_files=/path/to/gsm8k/test.parquet \
  YOUR_USUAL_TRAINING_OVERRIDES
```

这里的 `YOUR_USUAL_TRAINING_OVERRIDES` 是占位说明，运行前必须替换为自己的
batch、GPU、rollout、offload 等完整配置；这个示例不是已经调好的服务器 smoke
命令。仅有预设不会限制训练步数、显存或可用 GPU。已有无 critic 单卡结果不能
证明加入 critic 后仍能单卡运行。

模板在 trainer 初始化时读取一次，相对路径按项目根目录解析，与启动目录无关。
训练期间修改模板文件不会改变本次运行。

## YAML 内联模板

在 `agent_ppo_trainer` 的训练配置中添加/覆盖：

```yaml
asymmetric_critic:
  enabled: true
  prefix:
    source: template
    template_path: null
    template: |
      [Critic-only information]
      标准答案：{ground_truth}
      参考解题过程：{reference_solution}
      请估计只看到原对话的 actor 的未来回报。
      [/Critic-only information]
    field: null
    max_tokens: 256
    overflow: error
```

变量使用普通 `{name}`，不是 Hydra `${...}`。支持：

| 变量 | 数据来源 |
| --- | --- |
| `{ground_truth}` | `reward_model.ground_truth` |
| `{reference_solution}` | `extra_info.answer` |
| `{question}` | `extra_info.question` |
| `{data_source}` | `data_source` |

只有模板实际使用的变量必须存在。未知变量、缺失值、非标量值、格式表达式、
属性访问和下标访问都会报错，不执行 `eval`。字面花括号写成 `{{` / `}}`。
`template` 和 `template_path` 必须二选一；使用文件预设切换到内联时，记得清空
`template_path`。`template: ""` 配合 `max_tokens: 0` 可用于无前缀对照。

## 逐样本字段

把预先生成的字符串存入 `extra_info.critic_prefix`，不要拼进 actor 的 `prompt`：

```yaml
asymmetric_critic:
  enabled: true
  prefix:
    source: field
    template: null
    template_path: null
    field: extra_info.critic_prefix
    max_tokens: 256
    overflow: error
```

支持元数据中的点分路径。字段必须是字符串；空字符串是合法的无前缀样本，
缺失/None 会报错。同一 source UID 的静态前缀在各采样轨迹、各步骤间复用。
前缀只能来自动作前已有的信息，不能包含本次采样的未来动作、最终得分或
事后正确性判断。

## CPU 预览：不启动 Ray/GPU 模型

在安装了 verl/Torch 的环境中，从项目根目录运行：

```bash
python -B -m recipes.gsm8k.preview_critic_prefix \
  --config recipes/gsm8k/asymmetric_critic.yaml \
  --data-file /path/to/gsm8k/train.parquet \
  --model-path /path/to/model \
  --rows 2
```

预览读取 tokenizer 和真实样本，构造含合成响应的 CPU DataProto，检查 actor
张量与响应 token 未变化，输出前缀长度及 critic 输入长度。它不生成轨迹、不
计算真实 value、不更新参数。默认不展示特权文本，显式加 `--show-text` 才展示。
预览默认 `enable_thinking=False`；如果训练使用其他 chat-template 参数，可在
预览 YAML 的 `data.apply_chat_template_kwargs` 中配置相同参数。

## 输入与复现约定

前缀使用同一 tokenizer、`add_special_tokens=False` 独立编码，然后按 token
拼接到未 padding 的原 actor prompt 前面。不会把整段文本重新编码，也不会
自动插入新的 system/chat message；需要的分隔符由模板自己提供。完整响应
token 块保留不变，包括工具 token、EOS 和右 padding。

Critic prompt 重新左 padding，重算 attention mask / position IDs。Value 推理
和更新共用按 UID 固定的前缀 token，输出仍为原来的 `[batch, response_length]`；
第 0 列对应生成首个响应 token 前的状态。Critic mini-batch 的长度统计使用
新增前缀后的 attention mask，value loss 仍只覆盖原响应位置。

前缀超过 `max_tokens` 会报错，不静默截断；还需保证原 prompt + 前缀 + response
不超过 critic 模型上下文限制。实际模板与哈希保存为运行目录中的
`asymmetric_critic_prefix.json`，不会覆盖内容不同的快照。字段模式保存的是来源
配置，不保存逐样本答案；严格复现还应固定数据集版本。

现有 GAE 的终止/截断 bootstrap 逻辑及 GSM8K 数值评分规则没有在本功能中修改。
正式多步或效果实验前，应单独验证这些问题。CPU 测试通过不代表真实 FSDP GPU
训练或效果提升已验证。

## 测试

```bash
python -B -m unittest discover -s tests -v
```

无 GPU 栈的机器运行解析/配置/回归测试，真实 CPU tensor 测试会明确 skip。
完整环境额外验证独立 DataProto、工具/损失 mask、padding、位置对齐、UID 重排、
前缀冻结，以及微型随机 Qwen3 critic 的因果性和真实反向传播/optimizer 更新。

针对实际训练，还应分别检查原始 actor 与带前缀 critic 的输入隔离、value
位置对齐，以及两者梯度均为有限且非零。单步 smoke 仅验证训练链路，不能证明
前缀带来效果提升；正式对照需要固定共同初始权重、随机种子和评估集。
