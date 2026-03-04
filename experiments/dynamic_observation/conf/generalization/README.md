# Cross-Scenario Generalization Experiments

This directory contains tmuxinator configurations for cross-scenario generalization experiments.

## Experimental Design

The experiments evaluate whether strategies learned in one set of scenarios can transfer to unseen scenarios.

### Scenario Splits

Using SOTOPIA-HARD's 14 environments:
- **Split A**: Train on first 7 envs, test on last 7 envs
- **Split B**: Train on last 7 envs, test on first 7 envs

### Test Modes

| Mode     | Pretrain | Test Learning | Description                    |
|----------|----------|---------------|--------------------------------|
| baseline | ❌       | ✅            | Cold start (from scratch)      |
| both     | ✅       | ❌            | Train + frozen test            |
| finetune | ✅       | ✅            | Warm start (pretrain + adapt)  |

### Comparison Analysis

- **both (test) vs baseline**: Generalization benefit of pretraining
- **finetune vs baseline**: Warm start benefit
- **finetune vs both (test)**: Benefit of continued learning

## Usage

### 1. Run Baseline + Both in Parallel (Two Panes)

```bash
# Adversarial bandit, Split A (runs baseline and both in parallel)
tmuxinator start -p conf/generalization/generalization_adversarial_splitA.yml

# Neural UCB bandit, Split A
tmuxinator start -p conf/generalization/generalization_neural_ucb_splitA.yml

# Split B
tmuxinator start -p conf/generalization/generalization_adversarial_splitB.yml
```

This will open a tmux window with 2 panes:
- **Pane 1 (baseline)**: Online learning on test set without pretraining
- **Pane 2 (both)**: Train on train set, then test with frozen policy

### 2. Run Finetune (After Both Completes)

```bash
# After 'both' completes, run finetune with the pretrained model
tmuxinator start -p conf/generalization/generalization_adversarial_splitA.yml \
    mode=finetune \
    model_dir=results/generalization/adversarial_splitA_20260127_xxx/train
```

### 3. Custom Model

```bash
# Use custom Qwen model
tmuxinator start -p conf/generalization/generalization_adversarial_splitA.yml \
    model="custom/qwen/qwen-2.5-72b-instruct@http://127.0.0.1:8888/v1"

# Use DeepSeek
tmuxinator start -p conf/generalization/generalization_adversarial_splitA.yml \
    model="openrouter/deepseek/deepseek-v3.2"
```

## Configuration Files

| File | Bandit | Split |
|------|--------|-------|
| `generalization_adversarial_splitA.yml` | Adversarial (EXP3) | A |
| `generalization_adversarial_splitB.yml` | Adversarial (EXP3) | B |
| `generalization_neural_ucb_splitA.yml` | Neural UCB | A |

## Output Structure

```
results/generalization/
├── adversarial_splitA_20260127_123456/      # from 'both' mode
│   ├── config.json
│   ├── train/
│   │   ├── config.json
│   │   ├── summary.json
│   │   ├── scenarios/
│   │   │   ├── {scenario_id_1}.json
│   │   │   └── ...
│   │   ├── p1_bandit/
│   │   │   ├── model_weights.pt
│   │   │   ├── optimizer_state.pt
│   │   │   ├── state_data.json
│   │   │   └── selection_history.json
│   │   └── p2_bandit/
│   ├── test/
│   │   ├── config.json
│   │   ├── summary.json
│   │   └── scenarios/
│   └── final_summary.json
├── adversarial_baseline_20260127_123457/    # from 'baseline' mode
│   ├── baseline/
│   │   └── ...
│   └── final_summary.json
└── adversarial_finetune_20260127_123458/    # from 'finetune' mode
    ├── finetune/
    │   └── ...
    └── final_summary.json
```

