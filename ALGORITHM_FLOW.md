# Dynamic Observation Bandit 算法流程详解

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         run_bandit_simulation_context.py                │
│                              (主入口脚本)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BanditSimulationRunner                           │
│                        (simulation_runner.py)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Bandit    │  │    Env      │  │  Evaluator  │  │  Strategy   │    │
│  │  (P1/P2)    │  │  (动态环境)  │  │  (奖励评估)  │  │   Space     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. 核心组件

### 2.1 策略空间 (StrategySpace)

**文件**: `strategy_space.py`, `social_strategies.py`

策略空间定义了所有可选的 "arms"，每个 arm 对应一种社交策略：

```python
# Arm 0: 无优化 (baseline)
strategies[0] = "No optimization - use original bio"

# Arm 1-N: 不同的社交策略
strategies[1] = "Exchange: Propose trading resources or favors"
strategies[2] = "Persuasion: Use logical arguments to convince"
strategies[3] = "Collaboration: Suggest working together"
# ... 更多策略
```

每个策略包含：
- **name**: 策略名称
- **description**: 策略描述
- **embedding**: 策略的向量表示 (用于 NN 预测)

### 2.2 Neural Adversarial Bandit

**文件**: `neural_adversarial_bandit.py`

核心思想：用神经网络预测每个 arm 的累积分数，然后用 EXP3 风格的 softmax 选择。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neural Adversarial Bandit                    │
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Context    │ ──▶ │   Value     │ ──▶ │  Cumulative │       │
│  │  Embedding  │     │   Network   │     │   Scores    │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│         │                                       │               │
│         │            ┌─────────────┐            │               │
│         └──────────▶ │   EXP3      │ ◀──────────┘               │
│                      │  Selection  │                            │
│                      └─────────────┘                            │
│                             │                                   │
│                             ▼                                   │
│                      Selected Arm                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 奖励评估器 (RewardInTurnEvaluator)

**文件**: `evaluator_reward_in_trun.py`

每轮对话后，LLM 评估当前轮次的表现：

```python
# 评估维度
dimensions = {
    "believability": (0, 10),      # 可信度
    "relationship": (-5, 5),       # 关系变化
    "knowledge": (0, 10),          # 知识获取
    "secret": (-10, 0),            # 秘密保护
    "social_rules": (-10, 0),      # 社交规则
    "financial_and_material_benefits": (-5, 5),  # 物质利益
    "goal": (0, 10),               # 目标达成
}

# 归一化到 [0, 1]
normalized_reward = (raw_score - min) / (max - min)
```

## 3. Episode 执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        Episode 开始                              │
│  1. 加载场景 (环境 + 两个 Agent)                                  │
│  2. 初始化 Bandit (P1 和/或 P2)                                  │
│  3. 初始化累积分数 cumulative_scores = [0, 0, ..., 0]           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Turn 循环 (t = 1, 2, ..., max_turns)        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 1: Agent 生成对话                                    │   │
│  │   response = agent.act(observation, bio=current_bio)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 2: 评估 Turn Reward                                  │   │
│  │   reward = RewardInTurnEvaluator(messages)               │   │
│  │   # 返回 P1 和 P2 的归一化奖励 [0, 1]                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 3: 更新 Bandit (bandit.update)                       │   │
│  │   # 更新累积分数和训练 NN                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 4: 选择新策略 (bandit.select)                        │   │
│  │   arm_idx, new_bio = bandit.select(agent, turn)          │   │
│  │   env.update_agent_context(new_bio)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                    检查是否终止 (terminated?)                    │
│                              │                                  │
│              ┌───────────────┴───────────────┐                  │
│              │ No                            │ Yes              │
│              ▼                               ▼                  │
│         继续下一轮                      Episode 结束             │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Neural Adversarial Bandit 算法细节

### 4.1 累积分数更新 (update)

```python
def update(agent, arm_index, reward, turn):
    """
    更新累积分数并训练神经网络
    
    Args:
        agent: "p1" 或 "p2"
        arm_index: 被选中的 arm 索引
        reward: 当前轮次的奖励 [0, 1]
        turn: 当前轮次
    """
    # 1. 衰减历史累积分数 (遗忘旧信息)
    cumulative_scores *= score_decay  # 默认 0.7
    
    # 2. 更新被选中 arm 的累积分数
    cumulative_scores[arm_index] += reward
    
    # 3. 记录训练数据
    training_data.append({
        "context": current_context_embedding,
        "arm": arm_index,
        "cumulative_score": cumulative_scores[arm_index]
    })
    
    # 4. 训练神经网络 (每隔一定轮次)
    if len(training_data) >= min_samples:
        train_value_network(epochs=100)
```

### 4.2 Arm 选择 (select)

```python
def select(agent, turn):
    """
    使用 NN 预测 + EXP3 选择下一个 arm
    
    Returns:
        (arm_index, strategy_bio, probability)
    """
    # 1. 获取当前 context embedding
    context = get_context_embedding()
    
    # 2. NN 预测所有 arm 的累积分数
    predictions = value_network(context)  # shape: [n_arms]
    
    # 3. EXP3 风格的 softmax 选择
    eta = config.eta  # 默认 10-15
    
    # Softmax 计算概率
    exp_scores = exp(eta * (predictions - max(predictions)))
    probs = exp_scores / sum(exp_scores)
    
    # Gamma 混合保证最小探索
    gamma = config.gamma  # 默认 0.1-0.2
    probs = (1 - gamma) * probs + gamma / n_arms
    
    # 4. 按概率采样
    selected_arm = random.choice(n_arms, p=probs)
    
    # 5. 获取对应的策略 bio
    strategy_bio = strategy_space.get_bio(selected_arm)
    
    return selected_arm, strategy_bio, probs[selected_arm]
```

### 4.3 Value Network 结构

```python
class ValueNetwork(nn.Module):
    """
    预测每个 arm 的累积分数
    
    输入: context_embedding (对话历史的向量表示)
    输出: 每个 arm 的预测累积分数
    """
    def __init__(self, context_dim, n_arms, hidden_dim=128):
        self.layers = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_arms)
        )
    
    def forward(self, context):
        return self.layers(context)
```

## 5. 关键参数说明

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `eta` | 10-15 | 控制探索/利用平衡。越大越倾向于选择高分 arm |
| `gamma` | 0.1-0.2 | 最小探索概率。保证每个 arm 至少有 γ/K 的概率被选中 |
| `score_decay` | 0.7-0.9 | 累积分数衰减。越小越快遗忘历史 |
| `epochs` | 100-1000 | NN 训练轮数。越多 NN 预测越准确 |
| `alpha` | 0.1-1.0 | NN 正则化强度。防止过拟合 |
| `max_turns` | 20 | 每个 episode 的最大轮次 |

## 6. 文件依赖关系

```
run_bandit_simulation_context.py
    │
    ├── lib/simulation_runner.py (BanditSimulationRunner)
    │       │
    │       ├── core/bandits/__init__.py (create_bandit)
    │       │       │
    │       │       ├── neural_adversarial_bandit.py (NeuralAdversarialBandit)
    │       │       │       │
    │       │       │       ├── base_bandit.py (BaseBandit, BanditConfig)
    │       │       │       └── exp3_bandit.py (EXP3 算法逻辑)
    │       │       │
    │       │       └── strategy_space.py (StrategySpace)
    │       │               │
    │       │               └── social_strategies.py (策略定义)
    │       │
    │       ├── core/envs_dynamic_parallel.py (DynamicPromptParallelSotopiaEnv)
    │       │
    │       └── core/evaluator_reward_in_trun.py (RewardInTurnEvaluator)
    │
    ├── lib/config.py (配置管理)
    │
    └── lib/execution_modes.py (batch/resume 模式)
```

