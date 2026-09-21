# ReMax：完整贪心轨迹采集

当前完成的是第一阶段的**采集接口**，不是完整 ReMax 训练链路。
现有 trainer 仍使用原来的采样入口；设置 `algorithm.adv_estimator=remax`
仍会抛出 `NotImplementedError`。advantage 计算和 actor loss 接入留到下一阶段。

## 使用入口

在已初始化的 trainer 中，把**尚未按 rollout.n repeat 的原始任务生成批次**传入：

```python
# gen_batch 来自现有 _get_gen_batch 流程，保留正常的任务元数据。
manager = trainer.async_rollout_manager

# 入口一：只获取每个原始任务的一条完整 greedy 轨迹。
greedy = manager.generate_greedy_sequences(gen_batch)

# 入口二：一次完成 greedy + sampled 采集和 baseline 配对。
collection = manager.collect_remax_rollouts(gen_batch, num_samples=2)
sampled = collection.sampled
greedy = collection.greedy
baseline_per_sampled_step = sampled.batch["reward_baselines"]
baseline_by_task = collection.baselines
```

两个入口任选其一，不需要为一次采集同时调用。`num_samples` 不指定时采用
`actor_rollout_ref.rollout.n`。组合入口先对 B 个任务各采集一条 greedy episode，
再采集 B × num_samples 条 sampled episode；每条都重新创建 flow、独立 reset 环境。
两次采集之间不能更新 actor 权重或更改环境、奖励配置。接口不计算 advantage/return，
也不触发 actor 更新。

greedy 的每一步请求使用 `temperature=0`、`top_p=1`、`logprobs=False`，不会更改
全局采样配置，验证集温度也不会覆盖 greedy 模式。greedy 和 sampled 都显式传入
`max_tokens=rollout.response_length`，确保后端生成预算和 flow 保留的响应长度一致。
这替代了此前的后端隐式预算，可能影响原来先超长生成、再在本地截取的 sampled 轨迹。
贪心 token 选择不保证消除后端数值不确定性或外部环境随机性。

## 查看完整轨迹与 baseline

输出仍是按 step 展平的 `DataProto`。`non_tensor_batch` 新增或保留：

- `source_uid`：原始任务标识，用于 sampled/greedy 配对，原始任务不允许重复 `uid`。
- `trajectory_uids`、`step_indices`：episode 标识及步序号。
- `rollout_mode`：`greedy` 或 `sample`。
- `terminated`、`truncated`、`termination_reason`：只在 episode 最后一行标记结束状态；
  前面的行统一为 `False`、`False`、`ongoing`。
- `generation_stop_reason`、`response_at_limit`、`response_truncated`：每一步的生成状态。

按 `trajectory_uids` 分组，再按 `step_indices` 排序，可还原该 episode 各步的 prompt、
action 和即时奖励。下一步 prompt 来自当前 episode 自己执行 action 后获得的反馈，
不会复用 sampled 的工具反馈。

baseline 使用 greedy **所有 step 的原有即时奖励之和**，即对 `rm_scores` 按
`response_mask` 掩码求和后，再对整条 episode 求和，当前采用不折扣的累计奖励。
它不是仅取最后一步，也不要求 sampled 和 greedy 步数相同。
通过 `source_uid` 将一个任务的 baseline 广播到它的每个 sampled step，
结果位于 `sampled.batch["reward_baselines"]`，此时仍是 baseline，**不是 advantage**。

## 结束状态与异常处理

单步 flow、通用环境循环、HotpotQA、PaperSearch、ALFWorld、WebShop 已补充结束状态。
自然 final answer/环境 done 与 `max_steps`、`prompt_length`、响应预算、环境时间限制
截断分开记录。`terminated` 表示 episode 自然结束，不代表任务成功。
保持各 recipe 的现有奖励规则，包括 ALFWorld 原有的合成最终奖励行；baseline 按这些
现有 step 行累计。

明确截断的 episode 继续使用现有有限步数下获得的奖励。部分 verl 后端把 stop/length
都映射成 completed：缺少原始 finish reason 时，恰好耗尽预算且末尾不是 EOS 的响应
会保守地标记为截断；恰好在预算边界自然停止的响应也可能被这样标记。

空响应、生成中止、缺失/重复 greedy episode、步序不连续、未知结束状态、非有限奖励
都会报错，不会偷偷用 0 baseline 代替。自定义 flow 需要显式设置结束状态，并记录
逐步生成元数据后才能参与这一采集流程。

## 验证范围

```bash
python3 -B -m unittest discover -s tests -v
```

轻量测试使用测试替身，执行生产代码的采集编排与环境循环方法。
安装 verl/PyTorch 依赖后还可执行真实 DataProto/Pydantic/CPU tensor 测试。
这些都不是 GPU 模型或真实任务环境的端到端验证；接入训练前需要在配置完整的训练机
上确认实际生成、环境隔离及奖励行为。
