<div align="center">

# ALSO: Adversarial Online Strategy Optimization for Social Agents

**ICML 2026**

Xiang Li<sup>1</sup>, Liping Yi<sup>1</sup>, Mingze Kong<sup>2</sup>, Ming Zhang<sup>3</sup>, Zhongxiang Dai<sup>2</sup>, Qinghua Hu<sup>1</sup>

<sup>1</sup>Tianjin University &nbsp;&nbsp;
<sup>2</sup>The Chinese University of Hong Kong, Shenzhen &nbsp;&nbsp;
<sup>3</sup>East China Normal University

[[Project Page](https://babylonehy.github.io/ALSO/)]
[[Paper](https://babylonehy.github.io/ALSO/static/pdfs/ALSO_paper.pdf)]
[[arXiv](https://arxiv.org/)]
[[Code](https://github.com/Babylonehy/ALSO)]

</div>

## News

- **2026**: ALSO is accepted to ICML 2026.
- The repository has been cleaned as a lightweight paper artifact. Generated outputs, large caches, figures, and historical analysis files are not committed.

## Overview

ALSO studies online strategy optimization for LLM-based social agents in multi-turn social simulation. In environments such as Sotopia, agents face evolving dialogue contexts and non-stationary opponents, so a static persona or fixed behavioral instruction can lead to repeated deadlocks and poor goal completion.

ALSO formulates turn-level strategy adaptation as an adversarial bandit problem. At each dialogue turn, the system selects a persona-strategy arm, injects the selected social strategy into the agent prompt, observes reward feedback from the interaction, and updates a lightweight neural surrogate for sample-efficient online adaptation. No model weights are fine-tuned.

This repository contains the Sotopia-based implementation used for the paper experiments, including the main ALSO runner, strategy spaces, bandit baselines, evaluation scripts, and focused regression tests.

<div align="center">
  <img src="docs/static/images/framework.png" alt="ALSO framework overview" width="900">
</div>

## Highlights

- **Online adaptation**: adapts within a single multi-turn interaction instead of relying on offline retraining.
- **Adversarial bandit formulation**: does not assume a stationary or cooperative opponent.
- **Strategy injection**: optimizes high-level behavioral strategies at prompt time without fine-tuning LLMs.
- **Neural reward surrogate**: predicts per-arm rewards from interaction context to reduce exploration cost.
- **Sotopia evaluation**: supports all/hard scenario splits, bilateral optimization, static baselines, and evolutionary prompt-optimization baselines.

## Results Snapshot

The project page reports that ALSO is best or near-best across Sotopia-All and Sotopia-Hard in the bilateral optimization setting. The table below shows the Overall score summary.

| Model | Split | Original | Instinct | OPRO | EvoPrompt | ALSO |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V3.2 | Sotopia-All | 3.619 | 3.851 | 3.787 | 3.737 | **3.889** |
| DeepSeek-V3.2 | Sotopia-Hard | 3.025 | 3.427 | 3.344 | 3.292 | **3.527** |
| Qwen2.5-72B | Sotopia-All | 3.676 | 3.848 | 3.689 | 3.825 | **3.882** |
| Qwen2.5-72B | Sotopia-Hard | 3.347 | **3.666** | 3.242 | 3.491 | 3.648 |

See the [project page](https://babylonehy.github.io/ALSO/) for additional ablations, strategy drift analysis, heterogeneous model pairing, and case studies.

## Repository Layout

```text
.
├── sotopia/                                  # Sotopia package code
├── tests/                                    # Upstream Sotopia tests
└── experiments/also/                       # ALSO paper artifact
    ├── core/                                 # Bandits, strategy spaces, dynamic envs, evaluators
    ├── conf/main_experiments/                # Tmuxinator configs for smoke and paper runs
    ├── generated_strategies/                 # Small strategy pools used by the strategy loader
    ├── scripts/generate_strategy_cache.py    # Strategy embedding cache generation
    ├── tests/                                # Focused artifact tests
    ├── calculate_cost.py
    ├── evaluate_by_tag.py
    └── run_bandit_simulation_context.py
```

Generated runtime artifacts are intentionally excluded from git:

- `experiments/also/outputs/`
- `experiments/also/cache/`
- `experiments/also/results/`
- embedding caches, figures, spreadsheets, and historical intermediate datasets

## Installation

### 1. Create the Python environment

Use Python 3.10-3.12 and `uv`.

```bash
git clone https://github.com/Babylonehy/ALSO.git
cd ALSO

uv sync --extra api --extra test --extra paper
```

### 2. Install Sotopia data

```bash
uv run sotopia install
```

This initializes the Sotopia runtime data needed by the experiment runner. The full paper runs also require access to model APIs and, when using database-backed evaluation, the Sotopia/Redis setup expected by the base Sotopia package.

### 3. Create your local `.env`

Copy the template and edit the values for your machine:

```bash
cp .env.example .env
```

At minimum, set the model-provider keys you plan to use:

```bash
OPENROUTER_API_KEY=replace_with_your_openrouter_key
OPENAI_API_KEY=replace_with_your_openai_key_if_using_openai_models
```

If you use a remote or password-protected Redis service, also set `REDIS_OM_URL` in `.env`. Keep `.env` private; it is ignored by git.

### 4. Verify the database service

ALSO needs the Sotopia Redis database for scenario, agent, and episode records. If `uv run sotopia install` created the Docker service successfully, `redis-stack` should be running on port `6379`.

```bash
docker ps | grep redis-stack
docker exec redis-stack redis-cli ping
```

Expected output:

```text
PONG
```

If the container exists but is stopped, restart it:

```bash
docker start redis-stack
```

If the container does not exist yet, run a non-interactive Sotopia install with Docker and published data:

```bash
uv run sotopia install \
  --use-docker \
  --load-database \
  --redis-data-path "$(pwd)" \
  --overwrite-existing-data
```

Then verify that Sotopia can query the loaded data:

```bash
uv run python - <<'PY'
from sotopia.database import AgentProfile, EnvironmentProfile
from sotopia.database.env_agent_combo_storage import EnvAgentComboStorage

print("agents:", len(list(AgentProfile.all_pks())))
print("environments:", len(list(EnvironmentProfile.all_pks())))
print("env_agent_combos:", len(list(EnvAgentComboStorage.all_pks())))
PY
```

If Redis is remote or password-protected, set `REDIS_OM_URL` in `.env` before running Python scripts:

```bash
REDIS_OM_URL=redis://default:password@host:6379
```

If local proxy variables should not be used for API calls:

```bash
unset ALL_PROXY all_proxy
```

### 5. Optional: install tmuxinator

Paper-scale configs are written as tmuxinator files.

```bash
sudo apt install tmuxinator
```

## Quick Start

Run a one-scenario, two-turn smoke test from the repository root:

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

Expected output:

```text
experiments/also/outputs/smoke_test.json
```

## Experiments

### 1. Precompute strategy embeddings

Full paper runs use strategy mode with the V3 strategy space. Generate the cache once before launching batch experiments:

```bash
cd experiments/also

uv run python scripts/generate_strategy_cache.py \
  --subset hard \
  --strategy-version v3 \
  --cache-dir cache/strategy_embeddings_v3_slim \
  --skip-existing
```

### 2. Run ALSO

From the repository root:

```bash
tmuxinator start \
  -p experiments/also/conf/main_experiments/adversarial_v3_hard.yml \
  project_root=$(pwd) \
  batch=40 \
  eta=0.5
```

The main config launches P1-only, P2-only, and bilateral optimization panes for the hard split.

### 3. Run baselines

| Method | Config |
| --- | --- |
| Original / no optimization | `experiments/also/conf/main_experiments/baseline_v3.yml` |
| ALSO / adversarial bandit | `experiments/also/conf/main_experiments/adversarial_v3_hard.yml` |
| OPRO | `experiments/also/conf/main_experiments/opro_v3.yml` |
| EvoPrompt | `experiments/also/conf/main_experiments/evoprompt_v3.yml` |
| PromptBreeder | `experiments/also/conf/main_experiments/promptbreeder_v3.yml` |
| Neural UCB | `experiments/also/conf/main_experiments/neural_ucb_no_ctx_v3.yml` |

Example:

```bash
tmuxinator start \
  -p experiments/also/conf/main_experiments/opro_v3.yml \
  project_root=$(pwd) \
  batch=40
```

### 4. Smaller direct run

For a smaller command-line run without tmuxinator:

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

List available experiment tags:

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

Compare multiple runs and export tables:

```bash
uv run python evaluate_by_tag.py \
  --tags tag_a tag_b tag_c \
  --output results/comparison.csv \
  --output-xlsx results/comparison.xlsx \
  --export-csv results/tables \
  --save-all
```

## Testing

Run focused artifact tests:

```bash
uv run pytest experiments/also/tests -q
```

Check retained entrypoints:

```bash
uv run python -m py_compile \
  experiments/also/run_bandit_simulation_context.py \
  experiments/also/evaluate_by_tag.py \
  experiments/also/calculate_cost.py \
  experiments/also/scripts/generate_strategy_cache.py
```

## Citation

If this artifact is useful for your research, please cite:

```bibtex
@inproceedings{li2026also,
  title = {ALSO: Adversarial Online Strategy Optimization for Social Agents},
  author = {Li, Xiang and Yi, Liping and Kong, Mingze and Zhang, Ming and Dai, Zhongxiang and Hu, Qinghua},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year = {2026}
}
```

This repository builds on Sotopia:

```bibtex
@inproceedings{zhou2024sotopia,
  title = {SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents},
  author = {Zhou, Xuhui and Zhu, Hao and Mathur, Leena and Zhang, Ruohong and Qi, Zhengyang and Yu, Haofei and Morency, Louis-Philippe and Bisk, Yonatan and Fried, Daniel and Neubig, Graham and Sap, Maarten},
  booktitle = {International Conference on Learning Representations},
  year = {2024}
}
```

## Acknowledgements

The experiment environment is built on the Sotopia social simulation framework. The project-page style follows common academic project-page conventions and links to the public ALSO page for figures, ablations, and qualitative examples.
