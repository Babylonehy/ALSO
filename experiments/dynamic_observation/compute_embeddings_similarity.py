"""
Embedding 相似度计算与质量检查脚本

支持模式:
1. --mode profile: 读取改写后的 agent profiles (paraphrased_profiles) 并计算 embedding
2. --mode background: 读取改写后的 agent backgrounds (paraphrased_backgrounds) 并计算 embedding
3. --mode background_strategic: 读取策略化改写后的 agent backgrounds 并计算 embedding
4. --mode check: 检查已有 embeddings 目录中的文本质量（非英语/乱码检测）
5. --mode fix: 修复有问题的 paraphrases 并更新 embeddings
6. --mode recompute: 重新计算指定 embeddings 目录的所有 embeddings

运行方式:
# 首次计算 embedding
cd /path/to/project && source .venv/bin/activate && unset ALL_PROXY all_proxy && \\
    python experiments/dynamic_observation/compute_embeddings_similarity.py --mode background --hard-only

# 检查文本质量
python experiments/dynamic_observation/compute_embeddings_similarity.py --mode check \\
    --embeddings-dir experiments/dynamic_observation/embeddings_backgrounds/hard

# 修复有问题的 paraphrases
python experiments/dynamic_observation/compute_embeddings_similarity.py --mode fix \\
    --embeddings-dir experiments/dynamic_observation/embeddings_backgrounds/hard

# 重新计算 embeddings
python experiments/dynamic_observation/compute_embeddings_similarity.py --mode recompute \\
    --embeddings-dir experiments/dynamic_observation/embeddings_backgrounds/hard
"""
import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AsyncOpenAI

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
load_dotenv(project_root / ".env")

console = Console()

# ========== 配置区域 ==========
# 使用 OpenRouter 的 embedding 模型
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/embeddings"
# Profile 模式配置
PROFILE_INPUT_DIR = Path("/path/to/project/experiments/dynamic_observation/paraphrased_profiles/openrouter/openai/gpt-5/t1.2")
PROFILE_OUTPUT_DIR = Path("experiments/dynamic_observation/embeddings-gpt5")
# Background 模式配置
BACKGROUND_INPUT_DIR = Path("experiments/dynamic_observation/paraphrased_backgrounds")
BACKGROUND_OUTPUT_DIR = Path("experiments/dynamic_observation/embeddings_backgrounds")
# Strategic background 模式配置
STRATEGIC_BACKGROUND_INPUT_DIR = Path("experiments/dynamic_observation/paraphrased_backgrounds_strategic")
STRATEGIC_BACKGROUND_OUTPUT_DIR = Path("experiments/dynamic_observation/embeddings_backgrounds_strategic")
MAX_SCENARIOS = None  # 限制处理的 scenario 数量（None 表示处理全部）
MAX_CONCURRENCY = 70  # 最大并发请求数
# ==============================


async def get_embedding(
    text: str, semaphore: asyncio.Semaphore, session: aiohttp.ClientSession
) -> list[float]:
    """获取单个文本的 embedding 向量（直接调用 OpenRouter API）"""
    max_retries = 5
    base_delay = 2.0

    async with semaphore:
        api_key = os.getenv("OPENROUTER_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text,
        }

        for attempt in range(max_retries):
            async with session.post(OPENROUTER_API_BASE, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "data" not in data:
                        raise ValueError(f"OpenRouter API response missing 'data' field. Response: {data}")
                    embedding = data["data"][0]["embedding"]
                    if embedding is None:
                        raise ValueError(f"Embedding returned None for text: {text[:100]}...")
                    return embedding
                elif resp.status in (500, 502, 503, 504, 429):
                    # 服务器错误或限流，重试
                    error_text = await resp.text()
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"OpenRouter API error {resp.status}, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries}): {error_text[:100]}"
                    )
                    await asyncio.sleep(delay)
                else:
                    error_text = await resp.text()
                    raise ValueError(f"OpenRouter API error: {resp.status} - {error_text}")

        raise ValueError(f"Max retries ({max_retries}) exceeded for embedding request")


async def get_embeddings_batch(
    texts: list[str], semaphore: asyncio.Semaphore, session: aiohttp.ClientSession
) -> np.ndarray:
    """批量获取多个文本的 embedding 向量"""
    tasks = [get_embedding(text, semaphore, session) for text in texts]
    embeddings = await asyncio.gather(*tasks)
    return np.array(embeddings)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """计算 cosine similarity 矩阵"""
    return cosine_similarity(embeddings)


def compute_statistics(similarity_matrix: np.ndarray) -> dict[str, Any]:
    """计算相似度矩阵的统计信息（排除对角线）"""
    # 获取上三角矩阵（不包括对角线）的值
    n = similarity_matrix.shape[0]
    upper_triangle_indices = np.triu_indices(n, k=1)
    upper_values = similarity_matrix[upper_triangle_indices]

    return {
        "mean": float(np.mean(upper_values)) if upper_values.size > 0 else 0.0,
        "min": float(np.min(upper_values)) if upper_values.size > 0 else 0.0,
        "max": float(np.max(upper_values)) if upper_values.size > 0 else 0.0,
        "std": float(np.std(upper_values)) if upper_values.size > 0 else 0.0,
        # 元数据：生成日期和模型名称
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
        },
    }


def save_agent_results(
    agent_pk: str,
    original: str,
    paraphrases: list[str],
    embeddings: np.ndarray,
    similarity_matrix: np.ndarray,
    statistics: dict[str, Any],
    output_dir: Path,
    paraphrase_metadata: dict[str, Any] | None = None,
) -> None:
    """保存单个 agent 的所有结果"""
    agent_dir = output_dir / agent_pk
    agent_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存改写版本文本（包含原始 paraphrase 的 metadata）
    paraphrases_data: dict[str, Any] = {"original": original, "paraphrases": paraphrases}
    if paraphrase_metadata:
        paraphrases_data["paraphrase_metadata"] = paraphrase_metadata
    with open(agent_dir / "paraphrases.json", "w", encoding="utf-8") as f:
        json.dump(paraphrases_data, f, ensure_ascii=False, indent=2)

    # 2. 保存 embeddings
    np.save(agent_dir / "embeddings.npy", embeddings)

    # 3. 保存 cosine similarity 矩阵（含标签）
    labels = ["origin"] + [f"v{i}" for i in range(len(paraphrases))]
    df_sim = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    df_sim.to_csv(agent_dir / "cosine_similarity.csv")

    # 4. 保存统计信息
    with open(agent_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)

    # 5. 绘制 cosine similarity 热力图
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        df_sim,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.7,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title(f"Cosine Similarity Matrix\nAgent: {agent_pk}\nMean: {statistics['mean']:.4f}, Std: {statistics['std']:.4f}", fontsize=12)
    ax.set_xlabel("Versions")
    ax.set_ylabel("Versions")
    plt.tight_layout()
    plt.savefig(agent_dir / "cosine_similarity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved results for agent {agent_pk} to {agent_dir}")


async def process_single_profile(
    json_path: Path,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    output_dir: Path,
) -> dict[str, Any]:
    """处理单个 agent profile 的 JSON 文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    agent_pk = data["agent_pk"]
    original = data["original_background"]
    paraphrases = data["paraphrased_versions"]
    # 读取 paraphrase 脚本生成的 metadata（如果存在）
    paraphrase_metadata = data.get("metadata", None)

    logger.info(f"Processing agent {agent_pk}, {len(paraphrases)} paraphrases")

    # 获取所有文本的 embeddings（原始 + 改写版本）
    all_texts = [original] + paraphrases
    embeddings = await get_embeddings_batch(all_texts, semaphore, session)

    # 计算相似度矩阵
    similarity_matrix = compute_similarity_matrix(embeddings)

    # 计算统计信息
    statistics = compute_statistics(similarity_matrix)
    statistics["agent_pk"] = agent_pk

    # 保存结果（包含 paraphrase metadata）
    save_agent_results(
        agent_pk, original, paraphrases, embeddings, similarity_matrix, statistics, output_dir,
        paraphrase_metadata=paraphrase_metadata,
    )

    return statistics


async def process_single_background_scenario(
    scenario_dir: Path,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    output_dir: Path,
) -> dict[str, Any]:
    """
    处理单个 background scenario。
    计算 p1_background 和 p2_background 各自的 paraphrase 与原始描述的相似度。
    """
    # 找到该 scenario 下的 JSON 文件（取第一个即可，两个文件的 paraphrase 相同）
    json_files = sorted(scenario_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {scenario_dir}")

    json_path = json_files[0]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenario_id = scenario_dir.name
    original_data = data["original"]
    paraphrased_data = data["paraphrased"]
    metadata = data.get("metadata", {})

    # 提取原始背景和改写版本
    p1_original = original_data["p1_background"]
    p2_original = original_data["p2_background"]
    p1_paraphrases = paraphrased_data["p1_background"]
    p2_paraphrases = paraphrased_data["p2_background"]

    logger.info(
        f"Processing scenario {scenario_id}: "
        f"p1={len(p1_paraphrases)} paraphrases, p2={len(p2_paraphrases)} paraphrases"
    )

    # 获取所有文本的 embeddings
    # p1: [original, paraphrase_0, paraphrase_1, ...]
    # p2: [original, paraphrase_0, paraphrase_1, ...]
    p1_texts = [p1_original] + p1_paraphrases
    p2_texts = [p2_original] + p2_paraphrases

    p1_embeddings = await get_embeddings_batch(p1_texts, semaphore, session)
    p2_embeddings = await get_embeddings_batch(p2_texts, semaphore, session)

    # 计算 p1 和 p2 各自的相似度矩阵
    p1_sim_matrix = compute_similarity_matrix(p1_embeddings)
    p2_sim_matrix = compute_similarity_matrix(p2_embeddings)

    # 计算 p1 和 p2 之间的相似度（原始 vs 原始, paraphrase_i vs paraphrase_i）
    # cross_similarities[i] = cosine_sim(p1_texts[i], p2_texts[i])
    cross_similarities = []
    for i in range(min(len(p1_texts), len(p2_texts))):
        sim = cosine_similarity([p1_embeddings[i]], [p2_embeddings[i]])[0, 0]
        cross_similarities.append(float(sim))

    # 计算统计信息
    p1_stats = compute_statistics(p1_sim_matrix)
    p2_stats = compute_statistics(p2_sim_matrix)

    # 计算与原始描述的相似度（每个 paraphrase 与原始的相似度）
    p1_to_origin = [float(p1_sim_matrix[0, i]) for i in range(1, len(p1_texts))]
    p2_to_origin = [float(p2_sim_matrix[0, i]) for i in range(1, len(p2_texts))]

    statistics = {
        "scenario_id": scenario_id,
        "p1_name": original_data.get("p1_name", ""),
        "p2_name": original_data.get("p2_name", ""),
        "p1_stats": {
            "mean": p1_stats["mean"],
            "min": p1_stats["min"],
            "max": p1_stats["max"],
            "std": p1_stats["std"],
            "to_origin_mean": float(np.mean(p1_to_origin)) if p1_to_origin else 0.0,
            "to_origin_min": float(np.min(p1_to_origin)) if p1_to_origin else 0.0,
            "to_origin_max": float(np.max(p1_to_origin)) if p1_to_origin else 0.0,
            "to_origin_all": p1_to_origin,
        },
        "p2_stats": {
            "mean": p2_stats["mean"],
            "min": p2_stats["min"],
            "max": p2_stats["max"],
            "std": p2_stats["std"],
            "to_origin_mean": float(np.mean(p2_to_origin)) if p2_to_origin else 0.0,
            "to_origin_min": float(np.min(p2_to_origin)) if p2_to_origin else 0.0,
            "to_origin_max": float(np.max(p2_to_origin)) if p2_to_origin else 0.0,
            "to_origin_all": p2_to_origin,
        },
        "cross_agent_similarity": {
            "original_p1_vs_p2": cross_similarities[0] if cross_similarities else None,
            "paraphrase_mean": float(np.mean(cross_similarities[1:])) if len(cross_similarities) > 1 else None,
            "all": cross_similarities,
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "paraphrase_metadata": metadata,
        },
    }

    # 保存结果
    save_background_results(
        scenario_id,
        original_data,
        paraphrased_data,
        p1_embeddings,
        p2_embeddings,
        p1_sim_matrix,
        p2_sim_matrix,
        statistics,
        output_dir,
    )

    return statistics


def save_background_results(
    scenario_id: str,
    original_data: dict[str, Any],
    paraphrased_data: dict[str, Any],
    p1_embeddings: np.ndarray,
    p2_embeddings: np.ndarray,
    p1_sim_matrix: np.ndarray,
    p2_sim_matrix: np.ndarray,
    statistics: dict[str, Any],
    output_dir: Path,
) -> None:
    """保存 background scenario 的所有结果"""
    scenario_output_dir = output_dir / scenario_id
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存原始和改写文本
    texts_data = {
        "original": {
            "p1_background": original_data["p1_background"],
            "p2_background": original_data["p2_background"],
        },
        "paraphrased": paraphrased_data,
    }
    with open(scenario_output_dir / "texts.json", "w", encoding="utf-8") as f:
        json.dump(texts_data, f, ensure_ascii=False, indent=2)

    # 2. 保存 embeddings
    np.save(scenario_output_dir / "p1_embeddings.npy", p1_embeddings)
    np.save(scenario_output_dir / "p2_embeddings.npy", p2_embeddings)

    # 3. 保存相似度矩阵
    p1_labels = ["origin"] + [f"v{i}" for i in range(len(paraphrased_data["p1_background"]))]
    p2_labels = ["origin"] + [f"v{i}" for i in range(len(paraphrased_data["p2_background"]))]

    df_p1 = pd.DataFrame(p1_sim_matrix, index=p1_labels, columns=p1_labels)
    df_p2 = pd.DataFrame(p2_sim_matrix, index=p2_labels, columns=p2_labels)
    df_p1.to_csv(scenario_output_dir / "p1_cosine_similarity.csv")
    df_p2.to_csv(scenario_output_dir / "p2_cosine_similarity.csv")

    # 4. 保存统计信息
    with open(scenario_output_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)

    # 5. 绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # P1 热力图
    sns.heatmap(
        df_p1, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.7, vmax=1.0,
        square=True, linewidths=0.5, annot_kws={"size": 7}, ax=axes[0],
    )
    p1_stats = statistics["p1_stats"]
    axes[0].set_title(
        f"P1 ({original_data.get('p1_name', 'Agent1')}) Similarity\n"
        f"Mean: {p1_stats['mean']:.4f}, To-Origin Mean: {p1_stats['to_origin_mean']:.4f}",
        fontsize=11,
    )

    # P2 热力图
    sns.heatmap(
        df_p2, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.7, vmax=1.0,
        square=True, linewidths=0.5, annot_kws={"size": 7}, ax=axes[1],
    )
    p2_stats = statistics["p2_stats"]
    axes[1].set_title(
        f"P2 ({original_data.get('p2_name', 'Agent2')}) Similarity\n"
        f"Mean: {p2_stats['mean']:.4f}, To-Origin Mean: {p2_stats['to_origin_mean']:.4f}",
        fontsize=11,
    )

    plt.suptitle(f"Scenario: {scenario_id}", fontsize=13)
    plt.tight_layout()
    plt.savefig(scenario_output_dir / "similarity_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved results for scenario {scenario_id}")


def save_background_summary_statistics(
    all_statistics: list[dict[str, Any]], output_dir: Path
) -> None:
    """保存 background 模式的汇总统计"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取数据
    rows = []
    for stat in all_statistics:
        rows.append({
            "scenario_id": stat["scenario_id"],
            "p1_name": stat["p1_name"],
            "p2_name": stat["p2_name"],
            "p1_mean": stat["p1_stats"]["mean"],
            "p1_to_origin_mean": stat["p1_stats"]["to_origin_mean"],
            "p2_mean": stat["p2_stats"]["mean"],
            "p2_to_origin_mean": stat["p2_stats"]["to_origin_mean"],
            "cross_original": stat["cross_agent_similarity"]["original_p1_vs_p2"],
            "cross_paraphrase_mean": stat["cross_agent_similarity"]["paraphrase_mean"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("p1_to_origin_mean", ascending=False).reset_index(drop=True)

    # 保存 CSV
    csv_path = output_dir / "summary_statistics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary statistics to {csv_path}")

    # 生成可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Background Paraphrase Similarity Statistics", fontsize=14)

    # 1. P1 和 P2 的 to-origin 相似度分布
    ax1 = axes[0, 0]
    ax1.hist(df["p1_to_origin_mean"], bins=15, alpha=0.6, label="P1", color="blue")
    ax1.hist(df["p2_to_origin_mean"], bins=15, alpha=0.6, label="P2", color="orange")
    ax1.axvline(df["p1_to_origin_mean"].mean(), color="blue", linestyle="--",
                label=f"P1 Mean: {df['p1_to_origin_mean'].mean():.4f}")
    ax1.axvline(df["p2_to_origin_mean"].mean(), color="orange", linestyle="--",
                label=f"P2 Mean: {df['p2_to_origin_mean'].mean():.4f}")
    ax1.set_xlabel("Mean Similarity to Origin")
    ax1.set_ylabel("Count")
    ax1.set_title("Paraphrase-to-Origin Similarity Distribution")
    ax1.legend()

    # 2. Cross-agent similarity
    ax2 = axes[0, 1]
    ax2.scatter(df["cross_original"], df["cross_paraphrase_mean"], alpha=0.6)
    ax2.plot([0.7, 1.0], [0.7, 1.0], "r--", label="y=x")
    ax2.set_xlabel("Original P1 vs P2 Similarity")
    ax2.set_ylabel("Paraphrase P1 vs P2 Mean Similarity")
    ax2.set_title("Cross-Agent Similarity: Original vs Paraphrased")
    ax2.legend()

    # 3. P1 vs P2 to-origin 相似度对比
    ax3 = axes[1, 0]
    ax3.scatter(df["p1_to_origin_mean"], df["p2_to_origin_mean"], alpha=0.6)
    ax3.plot([0.7, 1.0], [0.7, 1.0], "r--", label="y=x")
    ax3.set_xlabel("P1 To-Origin Mean Similarity")
    ax3.set_ylabel("P2 To-Origin Mean Similarity")
    ax3.set_title("P1 vs P2 Paraphrase Quality")
    ax3.legend()

    # 4. Overall statistics bar chart
    ax4 = axes[1, 1]
    metrics = ["p1_mean", "p1_to_origin_mean", "p2_mean", "p2_to_origin_mean"]
    means = [df[m].mean() for m in metrics]
    colors = ["blue", "lightblue", "orange", "lightyellow"]
    ax4.bar(range(len(metrics)), means, color=colors, edgecolor="black")
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels(["P1 All", "P1 to Origin", "P2 All", "P2 to Origin"], rotation=15)
    ax4.set_ylabel("Mean Cosine Similarity")
    ax4.set_title("Overall Mean Similarities")
    for i, v in enumerate(means):
        ax4.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig_path = output_dir / "summary_statistics.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved summary plot to {fig_path}")

    # 打印统计摘要
    console.print("\n[bold cyan]===== Background Summary Statistics =====[/]")
    console.print(f"[green]Total scenarios processed:[/] {len(df)}")
    console.print(f"[green]P1 to-origin mean similarity:[/] {df['p1_to_origin_mean'].mean():.4f}")
    console.print(f"[green]P2 to-origin mean similarity:[/] {df['p2_to_origin_mean'].mean():.4f}")
    console.print(f"[green]Cross-agent original similarity mean:[/] {df['cross_original'].mean():.4f}")
    console.print(f"[green]Cross-agent paraphrase similarity mean:[/] {df['cross_paraphrase_mean'].mean():.4f}")


def save_summary_statistics(
    all_statistics: list[dict[str, Any]], output_dir: Path
) -> None:
    """保存汇总统计信息并生成可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建 DataFrame
    df = pd.DataFrame(all_statistics)
    df = df[["agent_pk", "mean", "min", "max", "std"]]  # 重排列顺序
    df = df.sort_values("mean", ascending=False).reset_index(drop=True)

    # 保存 CSV
    csv_path = output_dir / "summary_statistics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary statistics to {csv_path}")

    # 生成可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Paraphrase Similarity Statistics Across All Agents", fontsize=14)

    # 1. Mean similarity 柱状图
    ax1 = axes[0, 0]
    ax1.bar(range(len(df)), df["mean"], color="steelblue", alpha=0.7)
    ax1.axhline(y=df["mean"].mean(), color="red", linestyle="--", label=f"Overall Mean: {df['mean'].mean():.4f}")
    ax1.set_xlabel("Agent Index")
    ax1.set_ylabel("Mean Cosine Similarity")
    ax1.set_title("Mean Similarity per Agent")
    ax1.legend()

    # 2. Min/Max similarity 对比
    ax2 = axes[0, 1]
    x = range(len(df))
    ax2.scatter(x, df["min"], color="blue", alpha=0.6, label="Min", s=30)
    ax2.scatter(x, df["max"], color="red", alpha=0.6, label="Max", s=30)
    ax2.set_xlabel("Agent Index")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Min/Max Similarity per Agent")
    ax2.legend()

    # 3. Standard deviation 柱状图
    ax3 = axes[1, 0]
    ax3.bar(range(len(df)), df["std"], color="orange", alpha=0.7)
    ax3.axhline(y=df["std"].mean(), color="red", linestyle="--", label=f"Overall Mean Std: {df['std'].mean():.4f}")
    ax3.set_xlabel("Agent Index")
    ax3.set_ylabel("Standard Deviation")
    ax3.set_title("Similarity Std Dev per Agent")
    ax3.legend()

    # 4. 分布直方图
    ax4 = axes[1, 1]
    ax4.hist(df["mean"], bins=15, color="green", alpha=0.7, edgecolor="black")
    ax4.axvline(x=df["mean"].mean(), color="red", linestyle="--", label=f"Mean: {df['mean'].mean():.4f}")
    ax4.set_xlabel("Mean Cosine Similarity")
    ax4.set_ylabel("Count")
    ax4.set_title("Distribution of Mean Similarities")
    ax4.legend()

    plt.tight_layout()

    # 保存图片
    fig_path = output_dir / "summary_statistics.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved summary plot to {fig_path}")

    # 打印统计摘要
    console.print("\n[bold cyan]===== Summary Statistics =====[/]")
    console.print(f"[green]Total agents processed:[/] {len(df)}")
    console.print(f"[green]Overall mean similarity:[/] {df['mean'].mean():.4f}")
    console.print(f"[green]Overall min similarity:[/] {df['min'].min():.4f}")
    console.print(f"[green]Overall max similarity:[/] {df['max'].max():.4f}")
    console.print(f"[green]Overall mean std:[/] {df['std'].mean():.4f}")


# ========== 质量检查相关函数 ==========

def contains_non_english(text: str, threshold: float = 0.02) -> tuple[bool, float, str]:
    """
    检查文本是否包含大量非英语字符或乱码模式。

    Args:
        text: 要检查的文本
        threshold: 非 ASCII 字符占比阈值，超过则视为问题文本 (default: 0.02)

    Returns:
        (is_problematic, non_ascii_ratio, reason)
    """
    if not text:
        return False, 0.0, ""

    # 允许的特殊字符（常见英文标点和符号）
    allowed_special = set("''""—–…•·°±×÷©®™€£¥¢")

    # 统计非 ASCII 字符
    non_ascii_chars = []
    for char in text:
        if ord(char) > 127 and char not in allowed_special:
            non_ascii_chars.append(char)

    ratio = len(non_ascii_chars) / len(text) if text else 0.0

    # 检查1: 非 ASCII 比例超过阈值
    if ratio > threshold:
        sample = "".join(set(non_ascii_chars[:30]))
        return True, ratio, f"High non-ASCII ratio: {sample}"

    # 检查2: 检测多语言混合（乱码的典型特征）
    has_cyrillic = bool(re.search(r'[\u0400-\u04FF]', text))  # 俄语
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))     # 阿拉伯语
    has_cjk = bool(re.search(r'[\u4E00-\u9FFF]', text))        # 中日韩
    has_korean = bool(re.search(r'[\uAC00-\uD7AF]', text))     # 韩语
    has_hebrew = bool(re.search(r'[\u0590-\u05FF]', text))     # 希伯来语
    has_thai = bool(re.search(r'[\u0E00-\u0E7F]', text))       # 泰语
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text)) # 天城文(印地语)
    has_armenian = bool(re.search(r'[\u0530-\u058F]', text))   # 亚美尼亚语
    has_georgian = bool(re.search(r'[\u10A0-\u10FF]', text))   # 格鲁吉亚语
    has_greek = bool(re.search(r'[\u0370-\u03FF]', text))      # 希腊语

    language_count = sum([
        has_cyrillic, has_arabic, has_cjk, has_korean, has_hebrew,
        has_thai, has_devanagari, has_armenian, has_georgian, has_greek
    ])

    if language_count >= 2:
        return True, ratio, f"Multi-language mixing detected ({language_count} scripts)"

    # 检查3: 检测代码片段模式
    code_patterns = [
        r'\)\s*\{', r'\}\s*;', r'===', r'\[\s*\]', r'function\s*\(',
        r'return\s+\w+;', r'import\s+\w+', r'class\s+\w+', r'\$\{', r'=>',
        r'\.then\(', r'console\.', r'#\s*include', r'def\s+\w+\s*\(',
    ]
    code_matches = sum(1 for p in code_patterns if re.search(p, text))
    if code_matches >= 3:
        return True, ratio, f"Code-like patterns detected ({code_matches} patterns)"

    # 检查4: 过多的特殊符号
    special_chars = re.findall(r'[█▓▒░■□●○◆◇★☆►◄▲△▼▽⬚═║╔╗╚╝╠╣╬┌┐└┘├┤┬┴┼─│]', text)
    if len(special_chars) > 5:
        return True, ratio, f"Excessive special symbols ({len(special_chars)} found)"

    # 检查5: 连续的非英语字符序列
    consecutive_non_ascii = re.findall(r'[^\x00-\x7F]{3,}', text)
    if len(consecutive_non_ascii) > 3:
        return True, ratio, f"Multiple non-ASCII sequences ({len(consecutive_non_ascii)} found)"

    sample = "".join(set(non_ascii_chars[:30])) if non_ascii_chars else ""
    return False, ratio, sample


def check_scenario_quality(scenario_dir: Path) -> dict:
    """检查单个 scenario 目录的文本质量。"""
    texts_file = scenario_dir / "texts.json"
    if not texts_file.exists():
        return {"error": f"texts.json not found in {scenario_dir}"}

    with open(texts_file, "r", encoding="utf-8") as f:
        texts = json.load(f)

    results = {
        "scenario_id": scenario_dir.name,
        "scenario_dir": scenario_dir,
        "p1_issues": [],
        "p2_issues": [],
    }

    # 检查 p1 paraphrases
    for idx, text in enumerate(texts.get("paraphrased", {}).get("p1_background", [])):
        is_bad, ratio, sample = contains_non_english(text)
        if is_bad:
            results["p1_issues"].append({
                "index": idx,
                "ratio": ratio,
                "sample_chars": sample,
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
            })

    # 检查 p2 paraphrases
    for idx, text in enumerate(texts.get("paraphrased", {}).get("p2_background", [])):
        is_bad, ratio, sample = contains_non_english(text)
        if is_bad:
            results["p2_issues"].append({
                "index": idx,
                "ratio": ratio,
                "sample_chars": sample,
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
            })

    return results


# ========== 命令行参数解析 ==========

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Compute embedding similarity and check quality for paraphrased agent data"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["profile", "background", "background_strategic", "check", "fix", "recompute"],
        default="profile",
        help=(
            "Mode: 'profile'/'background'/'background_strategic' for computing embeddings, "
            "'check' for quality check, 'fix' for fixing issues, 'recompute' for recomputing embeddings"
        ),
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Embeddings directory for check/fix/recompute modes",
    )
    parser.add_argument(
        "--hard-only",
        action="store_true",
        help="Only process the 'hard' subset (for background mode)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to process",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=f"Maximum concurrent API requests (default: {MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Non-ASCII character ratio threshold for quality check (default: 0.02)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        help="Only process specified scenario IDs (space-separated)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"Embedding model to use (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Custom input directory (overrides default based on mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (overrides default based on mode)",
    )
    return parser.parse_args()

def _is_scenario_dir(path: Path) -> bool:
    """Heuristic: a scenario directory contains at least one JSON file."""
    return path.is_dir() and any(path.glob("*.json"))


def _resolve_background_roots(
    input_dir: Path,
    output_dir: Path,
    *,
    hard_only: bool,
) -> list[tuple[str, Path, Path]]:
    """
    Resolve background scenario roots.

    Supports:
    - input_dir is a subset dir (contains scenario dirs directly)
    - input_dir is a root dir that contains subset dirs like hard/easy
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # If input_dir looks like it contains scenario dirs directly, treat it as a single root
    if any(_is_scenario_dir(child) for child in input_dir.iterdir() if child.is_dir()):
        return [("custom", input_dir, output_dir)]

    # Otherwise, try subset structure under input_dir
    candidates: list[tuple[str, Path]] = []
    hard_dir = input_dir / "hard"
    easy_dir = input_dir / "easy"

    if hard_dir.exists():
        candidates.append(("hard", hard_dir))
    if not hard_only and easy_dir.exists():
        candidates.append(("easy", easy_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No scenario directories found in {input_dir} (expected scenario folders or hard/easy subsets)"
        )

    roots: list[tuple[str, Path, Path]] = []
    for subset_name, subset_dir in candidates:
        roots.append((subset_name, subset_dir, output_dir / subset_name))
    return roots


async def main_profile_mode(args: argparse.Namespace) -> None:
    """Profile 模式的主函数"""
    input_dir = args.input_dir if args.input_dir else PROFILE_INPUT_DIR
    output_dir = args.output_dir if args.output_dir else PROFILE_OUTPUT_DIR

    console.print(f"[cyan]Mode: profile[/]")
    console.print(f"[cyan]Input: {input_dir}[/]")
    console.print(f"[cyan]Output: {output_dir}[/]")

    # 获取所有 JSON 文件
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    console.print(f"[cyan]Found {len(json_files)} JSON files[/]")

    # 限制处理数量
    if args.max_items is not None:
        json_files = json_files[:args.max_items]
        console.print(f"[yellow]Limited to first {args.max_items} profiles[/]")

    # 过滤已处理的文件
    files_to_process = []
    skipped_files = []
    for json_path in json_files:
        agent_pk = json_path.stem
        stats_file = output_dir / agent_pk / "statistics.json"
        if stats_file.exists():
            skipped_files.append(json_path)
        else:
            files_to_process.append(json_path)

    if skipped_files:
        console.print(f"[yellow]Skipping {len(skipped_files)} already processed profiles[/]")
    console.print(f"[cyan]Processing {len(files_to_process)} remaining profiles[/]")

    # 创建信号量
    semaphore = asyncio.Semaphore(args.max_concurrency)

    # 收集统计信息
    all_statistics: list[dict[str, Any]] = []

    # 加载已处理的统计
    for json_path in skipped_files:
        agent_pk = json_path.stem
        stats_file = output_dir / agent_pk / "statistics.json"
        with open(stats_file, "r", encoding="utf-8") as f:
            all_statistics.append(json.load(f))

    # 处理剩余文件
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if files_to_process:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Computing embeddings...", total=len(files_to_process))

                tasks = [
                    process_single_profile(json_path, semaphore, session, output_dir)
                    for json_path in files_to_process
                ]
                for future in asyncio.as_completed(tasks):
                    try:
                        stats = await future
                        all_statistics.append(stats)
                        progress.update(task, advance=1, description=f"[cyan]Processed one profile")
                    except Exception as e:
                        logger.error(f"Error processing profile: {e}")

    # 保存汇总统计
    save_summary_statistics(all_statistics, output_dir)
    console.print(f"\n[bold green]Done! Results saved to {output_dir}[/]")

async def _main_background_roots(
    args: argparse.Namespace,
    *,
    mode_name: str,
    default_input_dir: Path,
    default_output_dir: Path,
) -> None:
    """Background/Background-Strategic 模式的主函数（支持 hard/easy 子目录或直接 scenario 目录）"""
    input_dir = args.input_dir if args.input_dir else default_input_dir
    output_dir = args.output_dir if args.output_dir else default_output_dir

    roots = _resolve_background_roots(input_dir, output_dir, hard_only=args.hard_only)

    semaphore = asyncio.Semaphore(args.max_concurrency)
    timeout = aiohttp.ClientTimeout(total=180)

    for subset_name, subset_input_dir, subset_output_dir in roots:
        console.print(f"[cyan]Mode: {mode_name} ({subset_name})[/]")
        console.print(f"[cyan]Input: {subset_input_dir}[/]")
        console.print(f"[cyan]Output: {subset_output_dir}[/]")

        # 获取所有 scenario 目录
        scenario_dirs = sorted([d for d in subset_input_dir.iterdir() if _is_scenario_dir(d)])
        if not scenario_dirs:
            raise FileNotFoundError(f"No scenario directories found in {subset_input_dir}")

        console.print(f"[cyan]Found {len(scenario_dirs)} scenarios[/]")

        # 限制处理数量
        if args.max_items is not None:
            scenario_dirs = scenario_dirs[:args.max_items]
            console.print(f"[yellow]Limited to first {args.max_items} scenarios[/]")

        # 过滤已处理的 scenarios
        dirs_to_process: list[Path] = []
        skipped_dirs: list[Path] = []
        for scenario_dir in scenario_dirs:
            stats_file = subset_output_dir / scenario_dir.name / "statistics.json"
            if stats_file.exists():
                skipped_dirs.append(scenario_dir)
            else:
                dirs_to_process.append(scenario_dir)

        if skipped_dirs:
            console.print(f"[yellow]Skipping {len(skipped_dirs)} already processed scenarios[/]")
        console.print(f"[cyan]Processing {len(dirs_to_process)} remaining scenarios[/]")

        # 收集统计信息
        all_statistics: list[dict[str, Any]] = []

        # 加载已处理的统计
        for scenario_dir in skipped_dirs:
            stats_file = subset_output_dir / scenario_dir.name / "statistics.json"
            with open(stats_file, "r", encoding="utf-8") as f:
                all_statistics.append(json.load(f))

        # 处理剩余 scenarios
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if dirs_to_process:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "[cyan]Computing embeddings...", total=len(dirs_to_process)
                    )

                    tasks = [
                        process_single_background_scenario(
                            scenario_dir, semaphore, session, subset_output_dir
                        )
                        for scenario_dir in dirs_to_process
                    ]
                    for future in asyncio.as_completed(tasks):
                        try:
                            stats = await future
                            all_statistics.append(stats)
                            progress.update(
                                task, advance=1, description=f"[cyan]Processed one scenario"
                            )
                        except Exception as e:
                            logger.error(f"Error processing scenario: {e}")

        # 保存汇总统计
        save_background_summary_statistics(all_statistics, subset_output_dir)
        console.print(f"\n[bold green]Done! Results saved to {subset_output_dir}[/]")


async def main_background_mode(args: argparse.Namespace) -> None:
    """Background 模式的主函数"""
    await _main_background_roots(
        args,
        mode_name="background",
        default_input_dir=BACKGROUND_INPUT_DIR,
        default_output_dir=BACKGROUND_OUTPUT_DIR,
    )


async def main_background_strategic_mode(args: argparse.Namespace) -> None:
    """Background Strategic 模式的主函数"""
    await _main_background_roots(
        args,
        mode_name="background_strategic",
        default_input_dir=STRATEGIC_BACKGROUND_INPUT_DIR,
        default_output_dir=STRATEGIC_BACKGROUND_OUTPUT_DIR,
    )


async def main_check_mode(args: argparse.Namespace) -> None:
    """质量检查模式的主函数"""
    embeddings_dir = args.embeddings_dir
    if not embeddings_dir:
        raise ValueError("--embeddings-dir is required for check mode")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Directory not found: {embeddings_dir}")

    # 递归获取所有包含 texts.json 的目录
    scenario_dirs = sorted([f.parent for f in embeddings_dir.rglob("texts.json")])

    console.print(f"\n[bold]Checking directory: {embeddings_dir}[/bold]")
    console.print(f"[dim]Found {len(scenario_dirs)} scenarios[/dim]\n")

    all_issues = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("[cyan]Checking...", total=len(scenario_dirs))
        for scenario_dir in scenario_dirs:
            result = check_scenario_quality(scenario_dir)
            if result.get("p1_issues") or result.get("p2_issues"):
                all_issues.append(result)
            progress.advance(task)

    if not all_issues:
        console.print("[bold green]✓ All texts passed quality check![/bold green]")
        return

    console.print(f"[bold red]Found {len(all_issues)} problematic scenarios:[/bold red]\n")

    for issue in all_issues:
        console.print(f"[bold yellow]Scenario: {issue['scenario_id']}[/bold yellow]")
        if issue["p1_issues"]:
            console.print(f"  [red]P1 issues ({len(issue['p1_issues'])}):[/red]")
            for p in issue["p1_issues"][:3]:
                console.print(f"    - Index {p['index']}: {p['ratio']:.1%} non-ASCII")
                console.print(f"      Sample: {p['sample_chars'][:30]}")
        if issue["p2_issues"]:
            console.print(f"  [red]P2 issues ({len(issue['p2_issues'])}):[/red]")
            for p in issue["p2_issues"][:3]:
                console.print(f"    - Index {p['index']}: {p['ratio']:.1%} non-ASCII")
                console.print(f"      Sample: {p['sample_chars'][:30]}")
        console.print()


async def main_fix_mode(args: argparse.Namespace) -> None:
    """修复模式的主函数"""
    from experiments.dynamic_observation.paraphrase_backgrounds import (
        paraphrase_single_text, setup_litellm,
    )
    from openai import OpenAI

    embeddings_dir = args.embeddings_dir
    if not embeddings_dir:
        raise ValueError("--embeddings-dir is required for fix mode")

    # 先执行质量检查
    scenario_dirs = sorted([f.parent for f in embeddings_dir.rglob("texts.json")])

    all_issues = []
    for scenario_dir in scenario_dirs:
        result = check_scenario_quality(scenario_dir)
        if result.get("p1_issues") or result.get("p2_issues"):
            all_issues.append(result)

    if not all_issues:
        console.print("[bold green]✓ No issues to fix![/bold green]")
        return

    console.print(f"[bold cyan]Fixing {len(all_issues)} problematic scenarios...[/bold cyan]")

    setup_litellm()
    semaphore = asyncio.Semaphore(5)

    # Setup embedding client
    api_key = os.getenv("OPENROUTER_API_KEY")
    embed_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def get_embedding_sync(text: str) -> np.ndarray:
        text = text.replace("\n", " ")
        response = embed_client.embeddings.create(input=[text], model=args.embedding_model)
        return np.array(response.data[0].embedding)

    for issue in all_issues:
        scenario_dir = issue.get("scenario_dir")
        if scenario_dir is None:
            continue

        texts_file = scenario_dir / "texts.json"
        p1_emb_file = scenario_dir / "p1_embeddings.npy"
        p2_emb_file = scenario_dir / "p2_embeddings.npy"

        with open(texts_file, "r", encoding="utf-8") as f:
            texts = json.load(f)

        original_p1 = texts.get("original", {}).get("p1_background", "")
        original_p2 = texts.get("original", {}).get("p2_background", "")
        p1_embeddings = np.load(str(p1_emb_file)) if p1_emb_file.exists() else None
        p2_embeddings = np.load(str(p2_emb_file)) if p2_emb_file.exists() else None

        modified = False

        # Fix P1 issues
        if issue["p1_issues"] and original_p1:
            bad_indices = {p["index"] for p in issue["p1_issues"]}
            p1_list = texts.get("paraphrased", {}).get("p1_background", [])
            console.print(f"  [cyan]Fixing {issue['scenario_id']} P1 ({len(bad_indices)})...[/cyan]")

            for idx in bad_indices:
                if idx < len(p1_list):
                    for attempt in range(5):
                        new_text, _ = await paraphrase_single_text(original_p1, semaphore)
                        is_bad, _, reason = contains_non_english(new_text, args.threshold)
                        if not is_bad:
                            p1_list[idx] = new_text
                            if p1_embeddings is not None:
                                p1_embeddings[idx + 1] = get_embedding_sync(new_text)
                            modified = True
                            console.print(f"    [green]✓ P1[{idx}] fixed (attempt {attempt + 1})[/green]")
                            break
            texts["paraphrased"]["p1_background"] = p1_list

        # Fix P2 issues
        if issue["p2_issues"] and original_p2:
            bad_indices = {p["index"] for p in issue["p2_issues"]}
            p2_list = texts.get("paraphrased", {}).get("p2_background", [])
            console.print(f"  [cyan]Fixing {issue['scenario_id']} P2 ({len(bad_indices)})...[/cyan]")

            for idx in bad_indices:
                if idx < len(p2_list):
                    for attempt in range(5):
                        new_text, _ = await paraphrase_single_text(original_p2, semaphore)
                        is_bad, _, reason = contains_non_english(new_text, args.threshold)
                        if not is_bad:
                            p2_list[idx] = new_text
                            if p2_embeddings is not None:
                                p2_embeddings[idx + 1] = get_embedding_sync(new_text)
                            modified = True
                            console.print(f"    [green]✓ P2[{idx}] fixed (attempt {attempt + 1})[/green]")
                            break
            texts["paraphrased"]["p2_background"] = p2_list

        if modified:
            with open(texts_file, "w", encoding="utf-8") as f:
                json.dump(texts, f, indent=2, ensure_ascii=False)
            if p1_embeddings is not None:
                np.save(str(p1_emb_file), p1_embeddings)
            if p2_embeddings is not None:
                np.save(str(p2_emb_file), p2_embeddings)
            console.print(f"  [green]✓ Saved {issue['scenario_id']}[/green]")


async def main_recompute_mode(args: argparse.Namespace) -> None:
    """重新计算 embeddings 模式的主函数"""
    embeddings_dir = args.embeddings_dir
    if not embeddings_dir:
        raise ValueError("--embeddings-dir is required for recompute mode")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    embed_client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    embedding_model = args.embedding_model
    semaphore = asyncio.Semaphore(args.max_concurrency)

    console.print(f"[bold]Using embedding model: {embedding_model}[/bold]")

    async def get_embedding_async(text: str) -> np.ndarray:
        async with semaphore:
            text = text.replace("\n", " ")
            for attempt in range(5):
                try:
                    response = await embed_client.embeddings.create(input=[text], model=embedding_model)
                    return np.array(response.data[0].embedding)
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise

    async def get_embeddings_batch_async(texts: list[str]) -> np.ndarray:
        tasks = [get_embedding_async(t) for t in texts]
        return np.array(await asyncio.gather(*tasks))

    # Get all scenario dirs
    all_scenario_dirs = sorted([d for d in embeddings_dir.iterdir() if d.is_dir()])
    if args.scenarios:
        scenario_dirs = [d for d in all_scenario_dirs if d.name in args.scenarios]
    else:
        scenario_dirs = all_scenario_dirs

    console.print(f"\n[bold]Recomputing embeddings for {len(scenario_dirs)} scenarios[/bold]")

    async def process_scenario(scenario_dir: Path) -> dict | None:
        texts_file = scenario_dir / "texts.json"
        if not texts_file.exists():
            return None

        with open(texts_file, "r", encoding="utf-8") as f:
            texts = json.load(f)

        original_p1 = texts.get("original", {}).get("p1_background", "")
        original_p2 = texts.get("original", {}).get("p2_background", "")
        p1_paraphrases = texts.get("paraphrased", {}).get("p1_background", [])
        p2_paraphrases = texts.get("paraphrased", {}).get("p2_background", [])

        p1_texts = [original_p1] + p1_paraphrases
        p2_texts = [original_p2] + p2_paraphrases

        p1_embeddings = await get_embeddings_batch_async(p1_texts)
        p2_embeddings = await get_embeddings_batch_async(p2_texts)

        np.save(str(scenario_dir / "p1_embeddings.npy"), p1_embeddings)
        np.save(str(scenario_dir / "p2_embeddings.npy"), p2_embeddings)

        # Compute similarity matrices
        p1_sim = cosine_similarity(p1_embeddings)
        p2_sim = cosine_similarity(p2_embeddings)

        p1_labels = ["origin"] + [f"v{i}" for i in range(len(p1_paraphrases))]
        p2_labels = ["origin"] + [f"v{i}" for i in range(len(p2_paraphrases))]

        pd.DataFrame(p1_sim, index=p1_labels, columns=p1_labels).to_csv(scenario_dir / "p1_cosine_similarity.csv")
        pd.DataFrame(p2_sim, index=p2_labels, columns=p2_labels).to_csv(scenario_dir / "p2_cosine_similarity.csv")

        # Compute statistics
        p1_to_origin = [float(p1_sim[0, i]) for i in range(1, len(p1_texts))]
        p2_to_origin = [float(p2_sim[0, i]) for i in range(1, len(p2_texts))]

        stats = {
            "scenario_id": scenario_dir.name,
            "p1_stats": {"to_origin_mean": float(np.mean(p1_to_origin)) if p1_to_origin else 0.0},
            "p2_stats": {"to_origin_mean": float(np.mean(p2_to_origin)) if p2_to_origin else 0.0},
        }

        with open(scenario_dir / "statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return stats

    all_stats = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Recomputing...", total=len(scenario_dirs))

        # Process in batches
        for i in range(0, len(scenario_dirs), 5):
            batch = scenario_dirs[i:i+5]
            results = await asyncio.gather(*[process_scenario(sd) for sd in batch])
            all_stats.extend([r for r in results if r])
            progress.advance(task, len(batch))

    console.print(f"\n[bold green]✓ Recomputed embeddings for {len(all_stats)} scenarios[/bold green]")


async def main() -> None:
    """主函数"""
    args = parse_args()

    console.print("[bold green]Starting Embedding Similarity Computation Script[/]")
    console.print(f"[cyan]Using embedding model: {args.embedding_model}[/]")

    # 检查 API key (not needed for check mode)
    if args.mode not in ("check",):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")

    if args.mode == "profile":
        await main_profile_mode(args)
    elif args.mode == "background":
        await main_background_mode(args)
    elif args.mode == "background_strategic":
        await main_background_strategic_mode(args)
    elif args.mode == "check":
        await main_check_mode(args)
    elif args.mode == "fix":
        await main_fix_mode(args)
    elif args.mode == "recompute":
        await main_recompute_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    # 配置 logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    asyncio.run(main())
