# ALSO Paper Artifact

This directory contains the cleaned experiment code for ALSO: Adversarial Online Strategy Optimization for Social Agents. It keeps the main paper reproduction path: runner, bandit/env/evaluator code, strategy definitions, main tmuxinator configs, cache generation, evaluation, and focused tests.

Generated runs, caches, embeddings, figures, spreadsheets, rebuttal-only analysis, and one-off maintenance scripts are intentionally not kept here. Historical data was backed up externally before this cleanup.

## Layout

```text
experiments/also/
├── core/                         # Bandit algorithms, dynamic environments, evaluators
├── conf/main_experiments/         # Main paper and smoke-test tmuxinator configs
├── generated_strategies/          # Small generated strategy pools used by strategy_loader
├── scripts/generate_strategy_cache.py
├── tests/                        # Focused regression tests for this artifact
├── calculate_cost.py
├── evaluate_by_tag.py
└── run_bandit_simulation_context.py
```

Main entrypoints:

- `run_bandit_simulation_context.py`: run single-scenario or batch simulations.
- `evaluate_by_tag.py`: summarize experiments from the Sotopia database or local output directories.
- `scripts/generate_strategy_cache.py`: precompute strategy embeddings for paper-scale strategy runs.

## Setup

Install from the repository root:

```bash
uv sync --extra api --extra test --extra paper
uv run sotopia install
```

Configure model provider credentials in `.env` or your shell:

```bash
export OPENROUTER_API_KEY=your_key
export OPENAI_API_KEY=your_key_if_using_openai_models
```

For tmux-based configs, install `tmuxinator` separately:

```bash
sudo apt install tmuxinator
```

If proxy variables should not be used for API calls:

```bash
unset ALL_PROXY all_proxy
```

## Smoke Test

From the repository root:

```bash
tmuxinator start \
  -p experiments/also/conf/main_experiments/smoke_test.yml \
  project_root=$(pwd)
```

Equivalent direct command:

```bash
cd experiments/also

uv run python run_bandit_simulation_context.py \
  --batch \
  --subset hard \
  --max-episodes 1 \
  --batch-size 1 \
  --selection-mode strategy \
  --strategy-version v3 \
  --model openrouter/openai/gpt-4o-mini \
  --env-model openrouter/openai/gpt-4o-mini \
  --reward-eval-model openrouter/openai/gpt-4o-mini \
  --bandit-type adversarial \
  --optimize both \
  --max-turns 2 \
  --tag smoke_test \
  --output outputs/smoke_test.json
```

The command creates `outputs/smoke_test.json`. If `--push-to-db` is added, completed episodes can also be evaluated with `evaluate_by_tag.py`.

## Main Experiments

Paper-scale strategy experiments use the V3 strategy space, hard subset, 20-turn episodes, and batch execution. The primary configs are:

| Method | Config |
| --- | --- |
| Baseline | `conf/main_experiments/baseline_v3.yml` |
| ALSO / adversarial bandit | `conf/main_experiments/adversarial_v3_hard.yml` |
| OPRO | `conf/main_experiments/opro_v3.yml` |
| EvoPrompt | `conf/main_experiments/evoprompt_v3.yml` |
| PromptBreeder | `conf/main_experiments/promptbreeder_v3.yml` |
| Neural UCB | `conf/main_experiments/neural_ucb_no_ctx_v3.yml` |

Precompute strategy embeddings before full runs:

```bash
cd experiments/also

uv run python scripts/generate_strategy_cache.py \
  --subset hard \
  --strategy-version v3 \
  --cache-dir cache/strategy_embeddings_v3_slim \
  --skip-existing
```

Run a main config:

```bash
tmuxinator start \
  -p experiments/also/conf/main_experiments/adversarial_v3_hard.yml \
  project_root=$(pwd) \
  batch=40 \
  eta=0.5
```

Smaller paper-aligned direct run:

```bash
cd experiments/also

uv run python run_bandit_simulation_context.py \
  --selection-mode strategy \
  --strategy-version v3 \
  --context-embedding \
  --embedding-model qwen/qwen3-embedding-8b \
  --context-embedding-dim 4096 \
  --batch \
  --subset hard_small \
  --batch-size 14 \
  --no-mask-unselected-scores \
  --model openrouter/deepseek/deepseek-v3.2 \
  --reward-eval-model openrouter/deepseek/deepseek-v3.2 \
  --bandit-type adversarial \
  --optimize both \
  --eta 10 \
  --depth 2 \
  --max-turns 20 \
  --push-to-db \
  --strategy-cache-dir cache/strategy_embeddings_v3_slim \
  --tag-prefix reproduction
```

## Evaluation

List experiment tags:

```bash
cd experiments/also
uv run python evaluate_by_tag.py --list-tags
```

Evaluate one run:

```bash
uv run python evaluate_by_tag.py \
  --tag reproduction_bandit_adversarial_both_hard_small \
  --eval-set hard
```

Compare multiple runs:

```bash
uv run python evaluate_by_tag.py \
  --tags tag_a tag_b tag_c \
  --output results/comparison.csv \
  --output-xlsx results/comparison.xlsx \
  --export-csv results/tables \
  --save-all
```

## Generated Artifacts

The cleaned artifact does not include generated data. These paths are created or regenerated as needed and should remain uncommitted:

- `outputs/`: local experiment outputs and logs.
- `cache/`: strategy embedding caches.
- `results/`: evaluation exports.
- `embeddings*/`, `paraphrased*/`, and other historical intermediate datasets.
