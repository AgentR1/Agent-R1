# GSM8K Recipe

## Overview

This recipe keeps plain GSM8K as a single-turn sanity check and provides GSM8K + Tool as the minimal `ToolEnv + BaseTool` example. In the tool path, the model can call `calc_gsm8k_reward` before producing the final answer.

Official dataset reference: https://huggingface.co/datasets/openai/gsm8k. Processed Agent-R1 data for this recipe is available from the [Agent-R1-data ModelScope release](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data).

## Directory Layout

- `base.yaml`: AgentEnvLoop configuration for the GSM8K + Tool example.
- `prompts.py`: Prompt templates for plain and tool runs.
- `reward_fn.py`: Recipe-local GSM8K rule reward wrapper.
- `tool.py`: Recipe-local `calc_gsm8k_reward` tool.
- `data_preprocess/process_gsm8k.py`: Converts raw GSM8K examples into standard train/test parquet files.
- `data_preprocess/process_gsm8k_tool.py`: Converts raw GSM8K examples into tool train/test parquet files.
- `examples/gsm8k/run_steppo.sh`: Single-turn StepPO test script using the plain data.
- `examples/gsm8k/run_steppo_tool.sh`: Multi-turn StepPO test script using the tool data.

## Additional Requirements

Install recipe-specific extras after setting up the base Agent-R1 / verl environment:

```bash
pip install -r recipes/gsm8k/requirements.txt
```

## Data And Resources

Expected processed files:

- Plain PPO path: `$HOME/data/gsm8k/train.parquet` and `$HOME/data/gsm8k/test.parquet`.
- Tool path: `$HOME/data/gsm8k_tool/train.parquet` and `$HOME/data/gsm8k_tool/test.parquet`.

Each processed row follows the verl RLHFDataset style with `prompt`, `reward_model`, and `extra_info`. The tool version stores a tool-calling chat prompt and `env_kwargs.tools_kwargs.ground_truth` so the reward tool can check answers during rollout.

## Data Preparation

Download the processed release from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data), then place or symlink the GSM8K files to the paths above. To regenerate from the public GSM8K source for local testing:

```bash
python recipes/gsm8k/data_preprocess/process_gsm8k.py \
  --local_save_dir "$HOME/data/gsm8k"

python recipes/gsm8k/data_preprocess/process_gsm8k_tool.py \
  --local_save_dir "$HOME/data/gsm8k_tool"
```

Use `--local_dataset_path` if the raw dataset has already been downloaded locally.

## Environment Setup

No external environment server is required. The tool path uses the generic `AgentEnvLoop`, the built-in `ToolEnv`, tool format `hermes`, and the recipe-local `calc_gsm8k_reward` tool.

## Training Scripts

```bash
bash examples/gsm8k/run_steppo.sh
bash examples/gsm8k/run_steppo_tool.sh
```

Both scripts accept trailing Hydra overrides through `"$@"`, for example:

```bash
bash examples/gsm8k/run_steppo_tool.sh trainer.total_epochs=1
```

## Core Code Entry Points

- Plain data conversion: `recipes/gsm8k/data_preprocess/process_gsm8k.py`.
- Tool data conversion: `recipes/gsm8k/data_preprocess/process_gsm8k_tool.py`.
- Prompt templates: `recipes/gsm8k/prompts.py`.
- Recipe reward: `recipes/gsm8k/reward_fn.py`.
- Tool rollout configuration: `recipes/gsm8k/base.yaml`.
- Tool definition: `recipes/gsm8k/tool.py`.

## Asymmetric Critic Prefix

See [the critic-prefix guide](ASYMMETRIC_CRITIC.md) for inline YAML, text-file,
per-sample metadata prefixes, and a CPU-only preview. The convenient training
preset is `--config-name asymmetric_ppo_trainer`; edit `critic_prefix.txt` to
change the critic-only information. This requires step-level GAE and a critic,
and is independent of the ReMax rollout-collection feature.

## Outputs And Evaluation

Training outputs follow the common Agent-R1 trainer configuration in the script overrides. Validation uses the test parquet configured as `data.val_files`.

## References

- GSM8K dataset: https://huggingface.co/datasets/openai/gsm8k
