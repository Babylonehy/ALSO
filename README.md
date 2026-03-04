## Overview

We propose an adversarial bandit algorithm that dynamically optimizes agent social strategies in multi-agent conversation scenarios. The agent selects from a strategy space using contextual embeddings and updates its policy based on reward feedback from LLM evaluators.

## Project Structure

```
paper_code/
├── sotopia/                          # Sotopia base framework (agents, envs, database)
├── core/
│   ├── bandits/                      # Bandit algorithm implementations
│   │   ├── base_bandit.py            # Abstract base class
│   │   ├── adversarial_bandit.py     # Adversarial bandit (proposed)
│   │   ├── neural_adversarial_bandit.py  # Neural adversarial bandit
│   │   ├── neural_ucb_bandit.py      # Neural UCB baseline
│   │   ├── linucb_bandit.py          # LinUCB baseline
│   │   ├── exp3_bandit.py            # Exp3 baseline
│   │   ├── opro_bandit.py            # OPRO baseline
│   │   ├── evoprompt_bandit.py       # EvoPrompt baseline
│   │   ├── prompt_breeder_bandit.py  # PromptBreeder baseline
│   │   ├── strategy_space.py         # Strategy space definition
│   │   └── social_strategies.py      # Social strategy library
│   ├── envs_dynamic_parallel.py      # Parallel simulation environment
│   ├── evaluator_reward_in_trun.py   # In-turn LLM reward evaluator
│   └── message_dynamic_observation.py
├── lib/
│   ├── simulation_runner.py          # Core simulation loop
│   ├── execution_modes.py            # Experiment execution modes
│   ├── config.py                     # Config parsing
│   └── ...
├── experiments/dynamic_observation/
│   ├── run_bandit_simulation_context.py   # Main experiment entry point
│   ├── run_generalization_experiment.py   # Generalization experiments
│   ├── evaluate_episodes_multi_run.py     # Multi-run evaluation
│   ├── conf/
│   │   ├── main_experiments/         # Main experiment configs
│   │   └── ablation/                 # Ablation study configs
│   └── README.md                     # Detailed usage instructions
├── pyproject.toml
└── .env.example
```

## Setup

### Requirements

- Python 3.10–3.12
- Redis (for episode storage)
- `uv` package manager (recommended)

### Installation

```bash
# Clone and enter the project
cd /path/to/project

# Install with uv
uv sync

# Or with pip
pip install -e .
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env and fill in your API keys:
#   OPENROUTER_API_KEY=...
#   REDIS_OM_URL=redis://:password@localhost:6379
```

## Running Experiments

All experiments are run from `experiments/dynamic_observation/`. See [`experiments/dynamic_observation/README.md`](experiments/dynamic_observation/README.md) for detailed instructions.

### Quick Start

```bash
cd /path/to/project
source .venv/bin/activate

# Run the main adversarial bandit experiment
cd experiments/dynamic_observation
python run_bandit_simulation_context.py \
    --bandit-type adversarial \
    --subset hard \
    --model openrouter/deepseek/deepseek-v3.2 \
    --reward-eval-model openrouter/deepseek/deepseek-v3.2 \
    --epochs 50 --eta 10.0 --optimize both \
    --push-to-db
```

### Using Tmuxinator Configs

Experiment configs in `conf/main_experiments/` and `conf/ablation/` are [tmuxinator](https://github.com/tmuxinator/tmuxinator) configs for running multi-GPU parallel experiments:

```bash
# Example: run the main adversarial experiment
tmuxinator start -p conf/main_experiments/adversarial_v3.yml \
    project_root=/path/to/project

# Example: run ablation on eta
tmuxinator start -p conf/ablation/adversarial_eta_ablation.yml \
    project_root=/path/to/project
```

### Using a Custom Local Model

To use a locally-hosted model, pass its OpenAI-compatible endpoint:

```bash
python run_bandit_simulation_context.py \
    --model "custom/your-model@http://YOUR_MODEL_ENDPOINT/v1" \
    ...
```

