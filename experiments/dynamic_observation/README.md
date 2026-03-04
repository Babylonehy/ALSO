# Dynamic Observation Experiments

本项目实现了基于 Bandit 算法的动态 prompt 优化实验，用于多智能体社交场景仿真。

## 目录结构

```
experiments/dynamic_observation/
├── paraphrase_backgrounds/      # 生成改写的核心模块
├── embeddings_backgrounds/      # 存储计算好的 embeddings
├── outputs/                     # 实验输出目录
├── core/                        # 核心模块 (bandits, envs, evaluators)
├── lib/                         # 工具库
└── analysis/                    # 分析脚本
```

---

## 第一部分：生成统一改写与计算 Embedding

### 1.1 生成 Agent Background 改写

使用 LLM 对 agent backgrounds 进行同义改写，保持语义不变但表述多样化。

```bash
# 进入项目目录并激活环境
cd /path/to/project && source .venv/bin/activate

# 测试模式（仅处理 2 个 scenarios）
python -m experiments.dynamic_observation.paraphrase_backgrounds.run_paraphrase --test

# 处理 hard 子集
python -m experiments.dynamic_observation.paraphrase_backgrounds.run_paraphrase --hard-only

# 处理全部数据
python -m experiments.dynamic_observation.paraphrase_backgrounds.run_paraphrase

# 指定模型和生成数量
python -m experiments.dynamic_observation.paraphrase_backgrounds.run_paraphrase \
    --num-paraphrases 20 \
    --model "openrouter/openai/gpt-4o" \
    --temperature 1.2 \
    --max-concurrency 30
```

**参数说明：**
- `--test`: 测试模式，仅处理 2 个 scenarios
- `--hard-only`: 仅处理 hard 子集（70 个 scenarios）
- `--num-paraphrases`: 每个 background 生成的改写数量（默认 10）
- `--model`: 使用的 LLM 模型
- `--temperature`: 生成温度（默认 1.0）
- `--max-concurrency`: 最大并发请求数（默认 20）

**输出目录：** `paraphrased_backgrounds/`

### 1.2 计算 Embedding 和相似度

对改写后的 backgrounds 计算 embedding 向量并分析相似度。

```bash
cd /path/to/project && source .venv/bin/activate

# 计算 hard 子集的 embeddings
python experiments/dynamic_observation/compute_embeddings_similarity.py \
    --mode background \
    --hard-only

# 计算全部数据的 embeddings
python experiments/dynamic_observation/compute_embeddings_similarity.py \
    --mode background

# 自定义输入输出目录
python experiments/dynamic_observation/compute_embeddings_similarity.py \
    --mode background \
    --input-dir /path/to/paraphrased_backgrounds \
    --output-dir /path/to/embeddings_output \
    --max-concurrency 50
```

**参数说明：**
- `--mode`: 模式选择 (`profile` | `background` | `background_strategic`)
- `--hard-only`: 仅处理 hard 子集
- `--max-items`: 限制处理的 scenario 数量
- `--max-concurrency`: 最大并发 API 请求数（默认 70）

**输出目录：** `embeddings_backgrounds/`

**输出内容：**
- `texts.json`: 原始和改写后的文本
- `p1_embeddings.npy`, `p2_embeddings.npy`: Embedding 向量
- `p1_cosine_similarity.csv`, `p2_cosine_similarity.csv`: 相似度矩阵
- `statistics.json`: 统计信息
- `similarity_heatmaps.png`: 相似度热力图

### 1.3 扩展改写数量（可选）

在已有 embeddings 基础上继续生成更多改写。

```bash
python experiments/dynamic_observation/extend_paraphrases.py \
    --input-dir experiments/dynamic_observation/embeddings_backgrounds \
    --output-dir experiments/dynamic_observation/embeddings_backgrounds_extended \
    --target-count 30 \
    --difficulty hard \
    --max-concurrency 20
```

---

## 第二部分：运行仿真

支持多种运行模式和 Bandit 算法类型。

### 2.1 单场景模式

运行单个 scenario 的仿真。

```bash
cd /path/to/project && source .venv/bin/activate

# 基础运行（优化双方）
python experiments/dynamic_observation/run_bandit_simulation_context.py \
    --scenario-id 01H7VKQHT745XAP1A4DDV8H419 \
    --optimize both

# 使用 adversarial bandit 优化 P1
python experiments/dynamic_observation/run_bandit_simulation_context.py \
    --scenario-id 01H7VKQHT745XAP1A4DDV8H419 \
    --bandit-type adversarial \
    --optimize p1 \
    --max-turns 10 \
    --push-to-db

# 基线模式（无优化）
python experiments/dynamic_observation/run_bandit_simulation_context.py \
    --scenario-id 01H7VKQHT745XAP1A4DDV8H419 \
    --optimize none
```

### 2.2 批量模式

并行运行多个 scenarios。

```bash
# 运行 hard 子集全部 scenarios
python experiments/dynamic_observation/run_bandit_simulation_context.py \
    --batch \
    --subset hard \
    --batch-size 5 \
    --optimize both \
    --bandit-type adversarial \
    --push-to-db

# 限制运行数量
python experiments/dynamic_observation/run_bandit_simulation_context.py \
    --batch \
    --subset hard \
    --max-episodes 10 \
    --batch-size 3 \
    --optimize both

# 指定不同模型给 P1 和 P2
python experiments/dynamic_observation/run_bandit_simulation_context.py \
    --batch \
    --subset hard \
    --p1-model "openrouter/openai/gpt-4o-mini" \
    --p2-model "openrouter/anthropic/claude-3-haiku" \
    --env-model "openrouter/openai/gpt-4o" \
    --push-to-db
```

### 2.3 恢复模式

从之前失败的实验继续运行。

```bash
# 恢复失败的 scenarios
python experiments/dynamic_observation/run_bandit_simulation_context.py \
    --resume outputs/bandit_adversarial_both_hard_20250110_120000 \
    --batch-size 5

# 带自动重试
python experiments/dynamic_observation/run_bandit_simulation_context.py \
    --resume outputs/experiment_tag \
    --max-retries 3
```

### 2.4 Bandit 类型

支持多种 Bandit 算法：

| 类型 | 说明 | 关键参数 |
|------|------|----------|
| `adversarial` | Adversarial Bandit（默认） | `--gamma`, `--score-decay` |
| `exp3` | EXP3 算法 | `--eta` |
| `linucb` | Linear UCB | `--alpha` |
| `neural_ucb` | Neural UCB | `--beta` |
| `prompt_breeder` | 进化式 Prompt 优化 | `--evolution-interval` |
| `neural_evolution` | 神经进化 Bandit | `--evolution-interval` |
| `none` | 无优化（基线） | - |

### 2.5 完整参数列表

```bash
python experiments/dynamic_observation/run_bandit_simulation_context.py --help
```

**核心参数：**
- `--scenario-id`: 单场景 ID
- `--batch`: 启用批量模式
- `--subset`: 数据集子集 (`hard` | `all`)
- `--optimize`: 优化目标 (`p1` | `p2` | `both` | `none`)
- `--bandit-type`: Bandit 算法类型
- `--max-turns`: 每个 episode 最大轮次（默认 10）
- `--push-to-db`: 将结果保存到数据库
- `--tag`: 自定义实验标签
- `--calculate-cost`: 计算 API 成本

**模型参数：**
- `--model`: 默认 agent 模型
- `--p1-model`: P1 专用模型
- `--p2-model`: P2 专用模型
- `--env-model`: 环境/评估器模型
- `--max-tokens`: LLM 输出最大 token 数

**Bandit 参数：**
- `--eta`: EXP3 探索参数（默认 5.0）
- `--alpha`: LinUCB 探索参数（默认 1.0）
- `--beta`: NeuralUCB 探索参数（默认 1.0）
- `--gamma`: 探索混合参数（默认 0.1）
- `--score-decay`: 累积分数衰减因子（默认 0.9）
- `--update-interval`: 模型更新间隔（默认 1）
- `--evolution-interval`: 进化间隔（默认 5）

---

## 第三部分：评估

使用 `evaluate_by_tag.py` 分析实验结果。

### 3.1 列出所有实验

```bash
python experiments/dynamic_observation/evaluate_by_tag.py --list-tags
```

### 3.2 评估单个实验

```bash
# 基础评估
python experiments/dynamic_observation/evaluate_by_tag.py \
    --tag bandit_adversarial_both_hard_20250110_120000

# 保存结果到文件
python experiments/dynamic_observation/evaluate_by_tag.py \
    --tag experiment_tag \
    --output results.json
```

### 3.3 对比多个实验

```bash
# 对比指定的多个实验
python experiments/dynamic_observation/evaluate_by_tag.py \
    --tags exp1 exp2 exp3

# 使用模式匹配
python experiments/dynamic_observation/evaluate_by_tag.py \
    --pattern "bandit_adversarial"

# 从 CSV 文件读取实验列表
python experiments/dynamic_observation/evaluate_by_tag.py \
    --csv experiments_list.csv

# 显示 95% 置信区间
python experiments/dynamic_observation/evaluate_by_tag.py \
    --pattern "bandit" \
    --use-ci

# 生成误差条形图
python experiments/dynamic_observation/evaluate_by_tag.py \
    --pattern "bandit" \
    --plot comparison_plot.png \
    --plot-metric goal

# 导出所有表格到 CSV
python experiments/dynamic_observation/evaluate_by_tag.py \
    --pattern "bandit" \
    --export-csv ./export_dir

# 保存完整分析结果
python experiments/dynamic_observation/evaluate_by_tag.py \
    --pattern "bandit" \
    --save-all
```

### 3.4 评估指标

**Overall 指标：**
- `p1_final_mean` / `p2_final_mean`: 最终奖励均值
- `p1_final_se` / `p2_final_se`: 标准误
- `avg_final`: 平均最终奖励

**维度分解 (Sotopia Dimensions)：**
- `believability`: 可信度 (0-10)
- `relationship`: 关系 (-5 to 5)
- `knowledge`: 知识 (0-10)
- `secret`: 秘密 (-10 to 0)
- `social_rules`: 社会规则 (-10 to 0)
- `financial_and_material_benefits`: 物质利益 (-5 to 5)
- `goal`: 目标达成 (0-10)

### 3.5 输出文件

使用 `--save-all` 时生成的文件：
- `metadata.json`: 实验元数据
- `results.csv`: 主要对比结果
- `dimensions.csv`: 维度分解数据
- `overlap_analysis.txt`: 误差重叠分析
- `errorbar_*.png`: 各指标误差条形图
- `errorbar_summary_all.png`: 综合图表

---

## 快速开始示例

```bash
# 1. 激活环境
cd /path/to/project && source .venv/bin/activate

# 2. 生成改写（如果尚未生成）
python -m experiments.dynamic_observation.paraphrase_backgrounds.run_paraphrase --hard-only

# 3. 计算 embeddings
python experiments/dynamic_observation/compute_embeddings_similarity.py --mode background --hard-only

# 4. 运行批量实验
python experiments/dynamic_observation/run_bandit_simulation_context.py \
    --batch --subset hard --batch-size 5 \
    --bandit-type adversarial --optimize both \
    --push-to-db --tag my_experiment

# 5. 评估结果
python experiments/dynamic_observation/evaluate_by_tag.py --tag my_experiment
```

---

## 环境要求

- Python 3.10+
- 使用 `uv` 管理虚拟环境
- 需要 `.env` 文件配置 `OPENROUTER_API_KEY`
- 代理问题：运行前执行 `unset ALL_PROXY all_proxy`
