# Main Experiments Configuration

主实验配置文件夹，使用策略 V3，测试两个模型：
- DeepSeek V3.2: `openrouter/deepseek/deepseek-v3.2`
- Qwen 2.5-72B: `openrouter/qwen/qwen-2.5-72b-instruct`

## 方法列表

| 方法 | 配置文件 | Bandit Type |
|------|----------|-------------|
| Baseline | `baseline_v3.yml` | - |
| OPRO | `opro_v3.yml` | `opro` |
| EvoPrompt | `evoprompt_v3.yml` | `evoprompt_ga` |
| PromptBreeder | `promptbreeder_v3.yml` | `prompt_breeder` |
| Neural UCB | `neural_ucb_v3.yml` | `neural_ucb` |
| Adversarial | `adversarial_v3.yml` | `adversarial` |
| All Methods | `all_methods_v3.yml` | 一次运行所有方法 |

## 使用方法

```bash
# 单独运行某个方法
tmuxinator start -p conf/main_experiments/adversarial_v3.yml

# 自定义参数
tmuxinator start -p conf/main_experiments/adversarial_v3.yml gpu=0 batch=40

# 运行所有方法对比
tmuxinator start -p conf/main_experiments/all_methods_v3.yml gpu=0
```

## 默认参数

- `batch`: 40 (批量大小)
- `max-turns`: 20
- `strategy-version`: v3
- `selection-mode`: strategy
- `subset`: hard
- `evolution-interval`: 5 (进化方法专用)
- `eta`: 0.5 (Adversarial 专用)
- `beta`: 0.5 (Neural UCB 专用)

