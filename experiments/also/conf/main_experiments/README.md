# Main Experiment Configs

These tmuxinator configs run the ALSO paper experiments with the V3 strategy space.

Run configs from the repository root and pass `project_root=$(pwd)` so the commands do not depend on machine-specific paths:

```bash
tmuxinator start \
  -p experiments/also/conf/main_experiments/smoke_test.yml \
  project_root=$(pwd)
```

## Configs

| Purpose | Config | Notes |
| --- | --- | --- |
| Smoke test | `smoke_test.yml` | One hard scenario, two turns, no database push by default. |
| Baseline | `baseline_v3.yml` | No optimization, strategy V3. |
| ALSO / adversarial | `adversarial_v3_hard.yml` | Main adversarial strategy-selection run. |
| OPRO | `opro_v3.yml` | OPRO prompt optimization baseline. |
| EvoPrompt | `evoprompt_v3.yml` | Evolutionary prompt baseline. |
| PromptBreeder | `promptbreeder_v3.yml` | PromptBreeder baseline. |
| Neural UCB | `neural_ucb_no_ctx_v3.yml` | Neural UCB baseline without context embedding. |

## Common Parameters

Most configs accept tmuxinator settings:

```bash
tmuxinator start \
  -p experiments/also/conf/main_experiments/adversarial_v3_hard.yml \
  project_root=$(pwd) \
  gpu=0 \
  batch=40 \
  eta=0.5
```

Defaults used by the paper-scale configs:

- `subset`: `hard`
- `max-turns`: `20`
- `strategy-version`: `v3`
- `selection-mode`: `strategy`
- `batch`: `40` unless overridden
- `eta`: `0.5` for adversarial configs unless overridden
- `beta`: `0.5` for Neural UCB configs unless overridden

Generated outputs, caches, and figures are intentionally ignored by git. See `experiments/also/README.md` for the full reproducibility workflow.
