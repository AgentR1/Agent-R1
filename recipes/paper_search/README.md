# Paper Search Recipe

## Overview

This recipe trains and evaluates an academic paper-search agent. The agent uses a paper search service, expands citation/reference neighborhoods, scores discovered papers with a selector service, and optimizes for discovering relevant papers.

Official background references: https://github.com/bytedance/pasa and https://arxiv.org/abs/2601.10029. Processed Agent-R1 train, test, paper-corpus, and retrieval-service assets for this recipe are available from the [Agent-R1-data ModelScope release](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data).

## Directory Layout

- `base.yaml`: Paper Search training agent configuration.
- `paper_search_agent_flow.py`: Agent-R1 rollout loop for training.
- `runtime.py`: Shared agent runtime for training and inference.
- `utils.py`: Tool-call parsing helpers.
- `env/paper_client.py`: Paper search and selector clients.
- `env/http_retry.py`: Shared HTTP retry helper.
- `prompts.py`: System prompt, user prompt, selector prompt, and tool schema.
- `env/run_papersearch_selector_service.sh`: vLLM selector service launcher.
- `inference/default.yaml`: Hydra config for offline inference and evaluation.
- `inference/retrieval_client.py`: Inference retrieval adapter.
- `inference/run.py`: Inference entry point.
- `examples/paper_search/*.sh`: Training launch scripts.
- `examples/paper_search/inference/run_evaluation.sh`: Inference plus threshold evaluation script.

## Additional Requirements

Install recipe-specific extras after setting up the base Agent-R1 / verl environment:

```bash
pip install -r recipes/paper_search/requirements.txt
```

The selector service and inference path require GPU resources suitable for the selected vLLM model. vLLM is expected to come from the base Agent-R1 training environment rather than this recipe-specific requirements file.

## Data And Resources

Expected processed training files:

- `data/pasa/train.parquet`
- `data/pasa/test.parquet`

Each parquet row should include the Agent-R1 training columns used by the rollout code: `prompt`, `reward_model`, and `extra_info`. Paper Search examples also need the query and annotated paper identifiers in `extra_info` / reward metadata according to the processed ModelScope release.

Expected external services:

- Paper search service at `PAPER_SEARCH_BASE_URL`, default `http://localhost:4000`.
- Selector service at `PAPERSEARCH_SELECTOR_BASE_URL`, default `http://localhost:8000`.

The paper search service must expose the search/expand API expected by `PaperSearchClient`. The selector service must expose an OpenAI-compatible endpoint for relevance scoring.

## Data Preparation

Download the processed release from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data), then place or symlink the Paper Search files to the paths above. Public PaSa/PaperScout resources are background references.

After placing processed files in the expected path, training scripts can consume:

```bash
data/pasa/train.parquet
data/pasa/test.parquet
```

For inference, set `PAPERSEARCH_INFER_DATASET` to an external JSONL dataset path. The inference runner reads the path from `recipes/paper_search/inference/default.yaml`.

## Environment Setup

Start the selector service:

```bash
bash recipes/paper_search/env/run_papersearch_selector_service.sh
```

Set the service URLs before training or inference:

```bash
export PAPER_SEARCH_BASE_URL=http://localhost:4000
export PAPERSEARCH_SELECTOR_BASE_URL=http://localhost:8000
```

The paper search service itself is treated as an external resource and should be started with the processed paper corpus supplied by the project maintainers.

For offline inference plus automatic evaluation:

```bash
export PAPERSEARCH_INFER_DATASET=/path/to/paper_search_eval.jsonl
export PAPERSEARCH_INFER_MODEL_PATH=/path/to/model_or_checkpoint
bash examples/paper_search/inference/run_evaluation.sh
```

Inference searches the local paper corpus by default (`PAPERSEARCH_INFER_SEARCH_SOURCE=local_db`). To search arXiv on the live web instead:

- `PAPERSEARCH_INFER_SEARCH_SOURCE=google` uses Serper and needs `SERPER_API_KEY` (or a comma-separated `PAPER_SEARCH_SERPER_API_KEYS`).
- `PAPERSEARCH_INFER_SEARCH_SOURCE=you` uses You.com with a `site:arxiv.org` filter. It works without a key on the free tier (100 searches a day per IP); set `YDC_API_KEY` for full evaluation runs. `search.paper_from_month` / `search.paper_to_month` are applied to the arXiv ID's YYMM.

Both web sources still resolve paper metadata through `PAPER_SEARCH_BASE_URL`.

## Training Scripts

```bash
bash examples/paper_search/run_ppo.sh
bash examples/paper_search/run_steppo.sh
bash examples/paper_search/run_grpo.sh
bash examples/paper_search/run_rloo.sh
bash examples/paper_search/run_reinforce.sh
bash examples/paper_search/run_gspo.sh
bash examples/paper_search/run_gigpo.sh
```

Scripts accept trailing Hydra overrides through `"$@"`.

## Core Code Entry Points

- Training rollout loop: `recipes/paper_search/paper_search_agent_flow.py`.
- Shared runtime and online step reward: `recipes/paper_search/runtime.py`.
- Search and selector clients: `recipes/paper_search/env/paper_client.py`.
- Inference runner: `recipes/paper_search/inference/run.py`.
- Evaluation: `recipes/paper_search/inference/evaluation.py`.

## Outputs And Evaluation

Training uses selector-scored discovered papers as reward evidence. Paper Search does not use a separate `reward_fn.py`; `PaperSearchRuntime.search` and `PaperSearchRuntime.expand` compute online step rewards, and `PaperSearchAgentFlow` writes those values directly into `AgentFlowStep.reward_score`. Inference writes sample outputs under `PAPERSEARCH_INFER_OUTPUT_DIR`, default `results/paper_search/inference`.

When `evaluation.enabled=true`, inference also writes `evaluation.json` with threshold results for `0.0` through `0.9` and the configured `top_k_values`.

## References

- PaSa: https://github.com/bytedance/pasa
- PaperScout: https://arxiv.org/abs/2601.10029
