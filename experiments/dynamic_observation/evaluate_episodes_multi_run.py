#!/usr/bin/env python3
"""
Evaluate episodes by tag using terminal evaluator multiple times.

This script reads episode data from database by tag, runs the terminal evaluator
multiple times, and displays individual scores with averages.

Usage:
    python evaluate_episodes_multi_run.py --tag <tag_name> --num-runs 3
"""

import asyncio
import json
import math
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# Suppress Pydantic serialization warnings from litellm
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import typer
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Configure loguru to show code location
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Load environment variables
load_dotenv()

# Unset proxy if causing issues
if os.environ.get("ALL_PROXY") or os.environ.get("all_proxy"):
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)

from sotopia.database.logs import EpisodeLog
from experiments.dynamic_observation.core.evaluator_reward_in_trun import TerminalEvaluator
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    SotopiaDimensions,
    unweighted_aggregate_evaluate,
)
from sotopia.generation_utils.generate import enable_llm_call_logging, disable_llm_call_logging

console = Console()
app = typer.Typer()

# Default outputs directory
OUTPUTS_DIR = Path(__file__).parent / "outputs"

# Evaluation dimensions
DIMENSIONS = [
    "believability",
    "relationship",
    "knowledge",
    "secret",
    "social_rules",
    "financial_and_material_benefits",
    "goal",
]


async def calculate_cost_from_log(log_file: Path) -> dict[str, Any]:
    """
    Calculate cost from LLM call log file by querying OpenRouter API.
    
    Returns dict with total_cost, total_tokens, etc.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not found, skipping cost calculation")
        return {}
    
    if not log_file.exists():
        logger.warning(f"Log file not found: {log_file}")
        return {}
    
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    if not lines:
        return {}
    
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    processed_count = 0
    
    async def fetch_usage(session: aiohttp.ClientSession, gen_id: str) -> dict | None:
        url = f"https://openrouter.ai/api/v1/generation?id={gen_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {})
        except Exception:
            pass
        return None
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for line in lines:
            try:
                entry = json.loads(line)
                gen_id = entry.get("id")
                if gen_id:
                    tasks.append(fetch_usage(session, gen_id))
            except json.JSONDecodeError:
                pass
        
        # Process in batches
        batch_size = 50
        results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
    
    for data in results:
        if data:
            cost = data.get("total_cost", 0) or 0
            p_tokens = data.get("native_tokens_prompt", 0) or data.get("tokens_prompt", 0) or 0
            c_tokens = data.get("native_tokens_completion", 0) or data.get("tokens_completion", 0) or 0
            
            total_cost += cost
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            processed_count += 1
    
    return {
        "total_cost": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "processed_count": processed_count,
    }


def read_tags_from_csv(csv_path: Path) -> list[str]:
    """
    从CSV文件读取tag列表（每行一个tag）

    参考 evaluate_by_tag.py 第 2694 行的实现
    """
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    csv_df = pd.read_csv(csv_path, header=None)
    tags = csv_df.iloc[:, 0].dropna().tolist()

    # 过滤注释和空字符串
    tags = [t.strip() for t in tags if isinstance(t, str) and t.strip() and not t.strip().startswith('#')]

    return tags


def extract_evaluation_from_rewards(
    episode: EpisodeLog,
) -> dict[str, Any] | None:
    """
    从EpisodeLog.rewards提取单次评估结果

    参考：sotopia/database/logs.py 第 31 行
    EpisodeLog.rewards 格式：list[tuple[float, dict[str, float]] | float]
    长度为 2，分别对应 P1 和 P2
    """
    if not episode.rewards or len(episode.rewards) < 2:
        return None

    p1_reward = episode.rewards[0]
    p2_reward = episode.rewards[1]

    result = {
        "source": "database",
        "turns_evaluated": -1,  # 数据库中没有记录评估轮次
    }

    # 处理 P1
    if isinstance(p1_reward, tuple) and len(p1_reward) == 2:
        result["p1_overall"] = float(p1_reward[0])
        result["p1_dimensions"] = p1_reward[1]
    elif isinstance(p1_reward, (int, float)):
        result["p1_overall"] = float(p1_reward)
        result["p1_dimensions"] = {}
    else:
        return None

    # 处理 P2
    if isinstance(p2_reward, tuple) and len(p2_reward) == 2:
        result["p2_overall"] = float(p2_reward[0])
        result["p2_dimensions"] = p2_reward[1]
    elif isinstance(p2_reward, (int, float)):
        result["p2_overall"] = float(p2_reward)
        result["p2_dimensions"] = {}
    else:
        return None

    return result


def get_existing_evaluation_count(
    episode_pk: str,
    checkpoint_results: list[dict[str, Any]],
    episode: EpisodeLog,
) -> tuple[int, list[dict[str, Any]]]:
    """
    计算 episode 已有的评估次数

    Args:
        episode_pk: Episode 的主键
        checkpoint_results: 从 checkpoint.jsonl 加载的所有结果
        episode: EpisodeLog 对象（用于提取数据库评估）

    Returns:
        (已有评估次数, 已有评估结果列表)

    来源优先级：checkpoint > database
    """
    existing_runs: list[dict[str, Any]] = []

    # 1. 从 checkpoint 中查找该 episode 的评估结果
    for checkpoint_result in checkpoint_results:
        if checkpoint_result.get("episode_pk") == episode_pk:
            runs = checkpoint_result.get("runs", [])
            existing_runs.extend(runs)
            break  # 找到后退出

    # 2. 从数据库的 rewards 字段提取
    db_eval = extract_evaluation_from_rewards(episode)

    # 3. 去重逻辑：检查 checkpoint 中是否已包含数据库评估
    has_db_in_checkpoint = any(
        run.get("source") == "database"
        for run in existing_runs
    )

    if db_eval and not has_db_in_checkpoint:
        # 数据库有评估，但 checkpoint 中没有，添加到开头
        existing_runs.insert(0, db_eval)

    return len(existing_runs), existing_runs


async def evaluate_episode_with_llm_evaluator(
    episode: EpisodeLog,
    model_name: str,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """
    使用 EpisodeLLMEvaluator 评估单个 episode

    参考 fix_reward_quality.py 第 292-380 行的实现

    Args:
        episode: EpisodeLog 对象
        model_name: 评估器使用的模型
        temperature: 采样温度
        max_retries: 最大重试次数

    Returns:
        评估结果字典，包含 p1_overall, p2_overall, p1_dimensions, p2_dimensions
    """
    from sotopia.database import EnvironmentProfile, AgentProfile

    # 1. 重建对话历史（参考 fix_reward_quality.py 第 159-290 行）
    history = ""

    # 方法 1：从 rewards_prompt 提取（主要方法）
    if episode.rewards_prompt:
        if "Terminal evaluation for history:" in episode.rewards_prompt:
            history = episode.rewards_prompt.replace("Terminal evaluation for history:", "").strip()
        else:
            # 尝试从 rewards_prompt 提取对话部分
            # 假设格式为 "Prompt after formatting:\n[history]\n,\nBased on previous interactions"
            parts = episode.rewards_prompt.split(",\nBased on previous interactions")
            if len(parts) > 0:
                history = parts[0].replace("Prompt after formatting:", "").strip()

    # 方法 2：从数据库重建（回退方法，参考 fix_reward_quality.py 第 220-290 行）
    if not history and episode.environment and episode.agents:
        try:
            env_profile = EnvironmentProfile.get(pk=episode.environment)
            agent_profiles = [AgentProfile.get(pk=agent_id) for agent_id in episode.agents]

            # 从 messages 构建历史（与 evaluators.py:287-302 保持一致）
            # 原始逻辑仅过滤 "did nothing"，不过滤 receiver
            if episode.messages:
                history_lines = []
                for turn in episode.messages:
                    for sender, receiver, message in turn:
                        if "did nothing" not in message:
                            if sender != "Environment":
                                history_lines.append(f"{sender} {message}")
                            else:
                                history_lines.append(message)

                history = "\n".join(history_lines)
        except Exception as e:
            logger.warning(f"Failed to reconstruct history from database: {e}")

    if not history:
        logger.warning(f"No history available for episode {episode.pk}")
        return None
    
    # logger.info(f"History for episode {episode.pk}: {history}")
    
    # 2. 创建评估器
    evaluator = TerminalEvaluator(
        model_name=model_name,
        response_format_class=EvaluationForTwoAgents[SotopiaDimensions],
    )

    # 3. 执行评估（带重试逻辑，参考 fix_reward_quality.py 第 344-380 行）
    for attempt in range(max_retries):
        try:
            response_list = await evaluator.__acall__(
                turn_number=-1,
                history=history,
                messages=None,
                temperature=temperature,
            )

            if not response_list:
                logger.warning(f"Empty response for episode {episode.pk} (attempt {attempt + 1})")
                continue

            # 聚合评估结果
            response = unweighted_aggregate_evaluate(response_list)

            # 提取分数
            result: dict[str, Any] = {
                "p1_overall": response.p1_rate[0] if response.p1_rate else 0.0,
                "p2_overall": response.p2_rate[0] if response.p2_rate else 0.0,
                "p1_dimensions": response.p1_rate[1] if response.p1_rate else {},
                "p2_dimensions": response.p2_rate[1] if response.p2_rate else {},
            }

            return result

        except Exception as e:
            logger.warning(f"Evaluation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                traceback.print_exc()
                return None

    return None


def _aggregate_evaluation_results(
    episode: EpisodeLog,
    all_runs: list[dict[str, Any]],
    num_new_runs: int,
) -> dict[str, Any]:
    """
    聚合所有评估结果（已有 + 新增）

    保持与现有 evaluate_episode_multi_run 相同的返回格式（第 360-393 行）
    """
    # 基础信息
    aggregated: dict[str, Any] = {
        "episode_pk": episode.pk,
        "environment": episode.environment,
        "num_successful_runs": len(all_runs),
        "num_existing_runs": len(all_runs) - num_new_runs,
        "num_new_runs": num_new_runs,
        "runs": all_runs,  # 包含完整的 runs 数组（带 source 标注）
    }

    # 计算均值和标准差
    p1_overalls = [r["p1_overall"] for r in all_runs]
    p2_overalls = [r["p2_overall"] for r in all_runs]

    aggregated["p1_overall_mean"] = float(np.mean(p1_overalls))
    aggregated["p1_overall_std"] = float(np.std(p1_overalls))
    aggregated["p2_overall_mean"] = float(np.mean(p2_overalls))
    aggregated["p2_overall_std"] = float(np.std(p2_overalls))

    # 维度聚合
    aggregated["p1_dimensions_mean"] = {}
    aggregated["p1_dimensions_std"] = {}
    aggregated["p2_dimensions_mean"] = {}
    aggregated["p2_dimensions_std"] = {}

    for dim in DIMENSIONS:
        p1_vals = [r["p1_dimensions"].get(dim, 0) for r in all_runs if r.get("p1_dimensions")]
        p2_vals = [r["p2_dimensions"].get(dim, 0) for r in all_runs if r.get("p2_dimensions")]

        if p1_vals:
            aggregated["p1_dimensions_mean"][dim] = float(np.mean(p1_vals))
            aggregated["p1_dimensions_std"][dim] = float(np.std(p1_vals))
        if p2_vals:
            aggregated["p2_dimensions_mean"][dim] = float(np.mean(p2_vals))
            aggregated["p2_dimensions_std"][dim] = float(np.std(p2_vals))

    return aggregated


def calculate_confidence_interval(values: list[float]) -> tuple[float, float]:
    """
    Calculate 95% confidence interval for a list of values.

    Implementation follows evaluate_by_tag.py approach.

    Returns:
        Tuple of (mean, margin_of_error)
    """
    if not values:
        return (0.0, 0.0)

    if len(values) == 1:
        return (float(values[0]), 0.0)

    # Calculate mean
    mean = float(np.mean(values))

    # Calculate variance
    variance = float(np.var(values, ddof=1))  # Using n-1 for sample variance

    # Calculate standard error of the mean (SEM)
    sem = math.sqrt(variance / len(values))

    # Use t-distribution to calculate margin of error
    confidence_level = 0.95
    df = len(values) - 1

    # Sample from t-distribution to get critical value
    t_samples = np.random.standard_t(df=df, size=1000000)
    t_value = np.percentile(t_samples, 100 * (1 - (1 - confidence_level) / 2))

    # Calculate margin of error
    margin = t_value * sem

    return (mean, float(margin))


async def evaluate_with_incremental_support(
    episode: EpisodeLog,
    checkpoint_results: list[dict[str, Any]],
    model_name: str,
    num_runs: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    对单个 episode 进行增量评估

    步骤：
    1. 检查已有评估（checkpoint + database）
    2. 计算还需要评估多少次
    3. 只评估缺少的次数（使用 evaluate_episode_with_llm_evaluator）
    4. 合并所有结果并返回聚合结果
    """
    # 1. 检查已有评估
    existing_count, existing_runs = get_existing_evaluation_count(
        episode_pk=episode.pk,
        checkpoint_results=checkpoint_results,
        episode=episode,
    )

    # 2. 计算还需要评估多少次
    remaining = max(0, num_runs - existing_count)

    logger.info(
        f"Episode {episode.pk[:16]}: "
        f"existing={existing_count}, target={num_runs}, remaining={remaining}"
    )

    # 3. 如果已满足要求，直接返回聚合结果
    if remaining == 0:
        logger.info(f"Episode {episode.pk[:16]} already has enough evaluations, skipping")
        return _aggregate_evaluation_results(
            episode=episode,
            all_runs=existing_runs,
            num_new_runs=0,
        )

    # 4. 执行增量评估（使用 EpisodeLLMEvaluator）
    new_runs = []
    tasks = [
        evaluate_episode_with_llm_evaluator(
            episode=episode,
            model_name=model_name,
            temperature=temperature,
            max_retries=3,
        )
        for _ in range(remaining)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Evaluation failed: {result}")
            traceback.print_exc()
        elif result is not None:
            result["source"] = "incremental"
            new_runs.append(result)

    # 5. 合并所有评估结果
    all_runs = existing_runs + new_runs

    if not all_runs:
        raise ValueError(f"All evaluations failed for episode {episode.pk}")

    return _aggregate_evaluation_results(
        episode=episode,
        all_runs=all_runs,
        num_new_runs=len(new_runs),
    )


async def evaluate_episode_once(
    episode: EpisodeLog,
    model_name: str,
    temperature: float = 0.0,
    max_turns: int | None = None,
) -> dict[str, Any] | None:
    """
    Run terminal evaluator once for an episode.
    
    Args:
        episode: The episode to evaluate
        model_name: LLM model name
        temperature: Sampling temperature
        max_turns: If set, only evaluate up to this many turns of the conversation
    
    Returns:
        Dict with p1 and p2 evaluation scores, or None if failed.
    """
    # Generate history from messages (with optional turn truncation)
    history_lines = []
    total_turns = len(episode.messages) if episode.messages else 0
    turns_to_process = min(max_turns, total_turns) if max_turns else total_turns
    
    if episode.messages:
        for idx, turn in enumerate(episode.messages[:turns_to_process]):
            for sender, receiver, message in turn:
                if receiver == "Environment" and sender != "Environment":
                    if "did nothing" not in message:
                        if "said:" in message:
                            history_lines.append(f"{sender} {message}")
                        else:
                            history_lines.append(f"{sender}: {message}")
                elif sender == "Environment" and idx == 0:
                    # Include initial environment context for first turn
                    if "context of this interaction" in message:
                        history_lines.append(message)  # Truncate long context
    
    history = "\n".join(history_lines)
    logger.debug(f"History for episode {episode.pk}: {history}")
    # Fallback to rewards_prompt if no history generated
    if not history and episode.rewards_prompt and not max_turns:
        history = episode.rewards_prompt.replace("Prompt after formatting:", "").split(
            ",\nBased on previous interactions"
        )[0].strip()
    
    if not history:
        logger.warning(f"No history available for episode {episode.pk}")
        return None
    

    evaluator = TerminalEvaluator(
        model_name=model_name,
        response_format_class=EvaluationForTwoAgents[SotopiaDimensions],
    )
    
    response_list = await evaluator.__acall__(
        turn_number=-1,
        history=history,
        messages=None,
        temperature=temperature,
    )
    
    if not response_list:
        logger.warning(f"Empty response for episode {episode.pk}")
        return None
    
    # Aggregate using the existing function
    response = unweighted_aggregate_evaluate(response_list)
    
    # Extract scores
    result: dict[str, Any] = {
        "p1_overall": response.p1_rate[0] if response.p1_rate else 0.0,
        "p2_overall": response.p2_rate[0] if response.p2_rate else 0.0,
        "p1_dimensions": response.p1_rate[1] if response.p1_rate else {},
        "p2_dimensions": response.p2_rate[1] if response.p2_rate else {},
        "turns_evaluated": turns_to_process,
    }
    
    return result


async def evaluate_episode_multi_run(
    episode: EpisodeLog,
    model_name: str,
    num_runs: int = 3,
    temperature: float = 0.0,
    turn_interval: int | None = None,
) -> dict[str, Any]:
    """
    Run terminal evaluator multiple times for an episode and aggregate results.
    Uses asyncio.gather to run all evaluations concurrently.
    
    Args:
        episode: Episode to evaluate
        model_name: LLM model name
        num_runs: Number of evaluation runs
        temperature: Sampling temperature
        turn_interval: If set, evaluate at every N turns (e.g., 2 = evaluate at turn 2, 4, 6, ...)
    """
    total_turns = len(episode.messages) if episode.messages else 0
    
    if turn_interval and turn_interval > 0:
        # Evaluate at multiple turn points
        turn_points = list(range(turn_interval, total_turns + 1, turn_interval))
        if total_turns not in turn_points:
            turn_points.append(total_turns)  # Always include final turn
        
        # For each turn point, run evaluation num_runs times
        all_turn_results: list[dict[str, Any]] = []
        for max_turns in turn_points:
            tasks = [
                evaluate_episode_once(
                    episode=episode,
                    model_name=model_name,
                    temperature=temperature,
                    max_turns=max_turns,
                )
                for _ in range(num_runs)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            run_results: list[dict[str, Any]] = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Run failed with exception: {result}")
                elif result is not None:
                    run_results.append(result)
            
            if run_results:
                # Calculate mean for this turn point
                p1_overalls = [r["p1_overall"] for r in run_results]
                p2_overalls = [r["p2_overall"] for r in run_results]
                
                turn_result = {
                    "turn": max_turns,
                    "num_runs": len(run_results),
                    "p1_overall_mean": float(np.mean(p1_overalls)),
                    "p2_overall_mean": float(np.mean(p2_overalls)),
                    "runs": run_results,
                }
                
                # Add dimension means
                for dim in DIMENSIONS:
                    p1_vals = [r["p1_dimensions"].get(dim, 0) for r in run_results]
                    p2_vals = [r["p2_dimensions"].get(dim, 0) for r in run_results]
                    turn_result[f"p1_{dim}_mean"] = float(np.mean(p1_vals))
                    turn_result[f"p2_{dim}_mean"] = float(np.mean(p2_vals))
                
                all_turn_results.append(turn_result)
        
        # Return aggregated result with turn progression
        aggregated: dict[str, Any] = {
            "episode_pk": episode.pk,
            "environment": episode.environment,
            "total_turns": total_turns,
            "turn_interval": turn_interval,
            "turn_results": all_turn_results,
            # Use final turn results for overall stats
            "runs": all_turn_results[-1]["runs"] if all_turn_results else [],
            "num_successful_runs": all_turn_results[-1]["num_runs"] if all_turn_results else 0,
        }
        
        if all_turn_results:
            final = all_turn_results[-1]
            aggregated["p1_overall_mean"] = final["p1_overall_mean"]
            aggregated["p2_overall_mean"] = final["p2_overall_mean"]
            aggregated["p1_overall_std"] = 0.0
            aggregated["p2_overall_std"] = 0.0
            aggregated["p1_dimensions_mean"] = {dim: final[f"p1_{dim}_mean"] for dim in DIMENSIONS}
            aggregated["p2_dimensions_mean"] = {dim: final[f"p2_{dim}_mean"] for dim in DIMENSIONS}
            aggregated["p1_dimensions_std"] = {dim: 0.0 for dim in DIMENSIONS}
            aggregated["p2_dimensions_std"] = {dim: 0.0 for dim in DIMENSIONS}
        
        return aggregated
    
    # Standard evaluation (no turn interval)
    tasks = [
        evaluate_episode_once(
            episode=episode,
            model_name=model_name,
            temperature=temperature,
        )
        for _ in range(num_runs)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    run_results: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Run failed with exception: {result}")
        elif result is not None:
            run_results.append(result)
    
    if not run_results:
        raise ValueError(f"All {num_runs} runs failed for episode {episode.pk}")
    
    # Aggregate results
    aggregated: dict[str, Any] = {
        "episode_pk": episode.pk,
        "environment": episode.environment,
        "num_successful_runs": len(run_results),
        "runs": run_results,
    }
    
    # Calculate mean and std for overall scores
    p1_overalls = [r["p1_overall"] for r in run_results]
    p2_overalls = [r["p2_overall"] for r in run_results]
    
    aggregated["p1_overall_mean"] = float(np.mean(p1_overalls))
    aggregated["p1_overall_std"] = float(np.std(p1_overalls))
    aggregated["p2_overall_mean"] = float(np.mean(p2_overalls))
    aggregated["p2_overall_std"] = float(np.std(p2_overalls))
    
    # Calculate mean and std for each dimension
    aggregated["p1_dimensions_mean"] = {}
    aggregated["p1_dimensions_std"] = {}
    aggregated["p2_dimensions_mean"] = {}
    aggregated["p2_dimensions_std"] = {}
    
    for dim in DIMENSIONS:
        p1_dim_values = [r["p1_dimensions"].get(dim, 0) for r in run_results if r["p1_dimensions"]]
        p2_dim_values = [r["p2_dimensions"].get(dim, 0) for r in run_results if r["p2_dimensions"]]
        
        if p1_dim_values:
            aggregated["p1_dimensions_mean"][dim] = float(np.mean(p1_dim_values))
            aggregated["p1_dimensions_std"][dim] = float(np.std(p1_dim_values))
        if p2_dim_values:
            aggregated["p2_dimensions_mean"][dim] = float(np.mean(p2_dim_values))
            aggregated["p2_dimensions_std"][dim] = float(np.std(p2_dim_values))
    
    return aggregated


def display_episode_results(result: dict[str, Any], episode_idx: int) -> None:
    """
    Display results for a single episode with individual runs and 95% CI.

    References evaluate_by_tag.py's "Final Episode Rewards with 95% CI" table format.
    """
    console.print(f"\n[bold cyan]═══ Episode {episode_idx + 1} ═══[/]")
    console.print(f"[dim]PK: {result['episode_pk'][:16]}...[/]")

    # Display evaluation source statistics
    num_existing = result.get("num_existing_runs", 0)
    num_new = result.get("num_new_runs", 0)
    num_total = result.get("num_successful_runs", 0)

    console.print(
        f"[dim]Evaluations: {num_total} total "
        f"({num_existing} existing, {num_new} new)[/]"
    )

    # Get all runs
    runs = result.get("runs", [])

    if not runs:
        console.print("[yellow]No evaluation runs available[/]")
        return

    # ===== Table 1: Individual Evaluation Runs =====
    runs_table = Table(title=f"Episode {episode_idx + 1}: Individual Evaluation Runs")
    runs_table.add_column("Run #", justify="center", style="cyan")
    runs_table.add_column("Source", justify="center", style="dim")
    runs_table.add_column("P1 Overall", justify="right", style="yellow")
    runs_table.add_column("P2 Overall", justify="right", style="yellow")
    runs_table.add_column("P1 Goal", justify="right", style="green")
    runs_table.add_column("P2 Goal", justify="right", style="green")

    # Add rows for each run
    for idx, run in enumerate(runs, 1):
        source = run.get("source", "unknown")
        p1_overall = run.get("p1_overall", 0.0)
        p2_overall = run.get("p2_overall", 0.0)
        p1_goal = run.get("p1_dimensions", {}).get("goal", 0.0)
        p2_goal = run.get("p2_dimensions", {}).get("goal", 0.0)

        # 根据来源使用不同的颜色标识
        if source == "database":
            source_display = f"[bold blue]{source}[/]"
        elif source == "checkpoint":
            source_display = f"[yellow]{source}[/]"
        elif source == "incremental":
            source_display = f"[green]{source}[/]"
        else:
            source_display = f"[dim]{source}[/]"

        runs_table.add_row(
            str(idx),
            source_display,
            f"{p1_overall:.4f}",
            f"{p2_overall:.4f}",
            f"{p1_goal:.4f}",
            f"{p2_goal:.4f}",
        )

    console.print(runs_table)

    # ===== Table 2: Final Averages with 95% CI =====
    # Calculate CI for overall scores
    p1_overalls = [r["p1_overall"] for r in runs]
    p2_overalls = [r["p2_overall"] for r in runs]
    p1_mean, p1_ci = calculate_confidence_interval(p1_overalls)
    p2_mean, p2_ci = calculate_confidence_interval(p2_overalls)

    # Calculate CI for goal scores
    p1_goals = [r.get("p1_dimensions", {}).get("goal", 0.0) for r in runs]
    p2_goals = [r.get("p2_dimensions", {}).get("goal", 0.0) for r in runs]
    p1_goal_mean, p1_goal_ci = calculate_confidence_interval(p1_goals)
    p2_goal_mean, p2_goal_ci = calculate_confidence_interval(p2_goals)

    # Calculate SE (standard error)
    p1_se = np.std(p1_overalls, ddof=1) / np.sqrt(len(p1_overalls)) if len(p1_overalls) > 1 else 0.0
    p2_se = np.std(p2_overalls, ddof=1) / np.sqrt(len(p2_overalls)) if len(p2_overalls) > 1 else 0.0
    p1_goal_se = np.std(p1_goals, ddof=1) / np.sqrt(len(p1_goals)) if len(p1_goals) > 1 else 0.0
    p2_goal_se = np.std(p2_goals, ddof=1) / np.sqrt(len(p2_goals)) if len(p2_goals) > 1 else 0.0

    avg_table = Table(title="Final Averages with 95% Confidence Intervals")
    avg_table.add_column("Metric", style="cyan")
    avg_table.add_column("P1 Mean", justify="right", style="yellow")
    avg_table.add_column("P1 CI (±)", justify="right", style="dim")
    avg_table.add_column("P1 SE", justify="right", style="dim magenta")
    avg_table.add_column("P2 Mean", justify="right", style="yellow")
    avg_table.add_column("P2 CI (±)", justify="right", style="dim")
    avg_table.add_column("P2 SE", justify="right", style="dim magenta")
    avg_table.add_column("Avg", justify="right", style="bold green")

    avg_table.add_row(
        "Overall Score",
        f"{p1_mean:.4f}",
        f"{p1_ci:.4f}",
        f"{p1_se:.4f}",
        f"{p2_mean:.4f}",
        f"{p2_ci:.4f}",
        f"{p2_se:.4f}",
        f"{(p1_mean + p2_mean) / 2:.4f}",
    )

    avg_table.add_row(
        "Goal Score",
        f"{p1_goal_mean:.4f}",
        f"{p1_goal_ci:.4f}",
        f"{p1_goal_se:.4f}",
        f"{p2_goal_mean:.4f}",
        f"{p2_goal_ci:.4f}",
        f"{p2_goal_se:.4f}",
        f"{(p1_goal_mean + p2_goal_mean) / 2:.4f}",
    )

    console.print(avg_table)

    # ===== Table 3: Dimension Breakdown with CI =====
    dim_table = Table(title="Dimension Breakdown with 95% CI")
    dim_table.add_column("Dimension", style="cyan")
    dim_table.add_column("P1 Mean", justify="right")
    dim_table.add_column("P1 CI (±)", justify="right", style="dim")
    dim_table.add_column("P2 Mean", justify="right")
    dim_table.add_column("P2 CI (±)", justify="right", style="dim")

    for dim in DIMENSIONS:
        p1_dim_values = [r.get("p1_dimensions", {}).get(dim, 0.0) for r in runs]
        p2_dim_values = [r.get("p2_dimensions", {}).get(dim, 0.0) for r in runs]

        p1_dim_mean, p1_dim_ci = calculate_confidence_interval(p1_dim_values)
        p2_dim_mean, p2_dim_ci = calculate_confidence_interval(p2_dim_values)

        dim_table.add_row(
            dim,
            f"{p1_dim_mean:.4f}",
            f"{p1_dim_ci:.4f}",
            f"{p2_dim_mean:.4f}",
            f"{p2_dim_ci:.4f}",
        )

    console.print(dim_table)

    # Check if we have turn-level results
    if "turn_results" in result and result["turn_results"]:
        turn_interval = result.get("turn_interval")
        console.print(Panel(f"[bold cyan]Turn Progression Analysis (Interval: {turn_interval})[/]"))

        # Create a table showing progression
        prog_table = Table(title="Turn-by-Turn Scores")
        prog_table.add_column("Turn", justify="center", style="yellow")
        prog_table.add_column("P1 Overall", justify="center", style="green")
        prog_table.add_column("P2 Overall", justify="center", style="blue")

        for dim in DIMENSIONS:
             prog_table.add_column(dim[:4], justify="center", style="dim")

        for turn_res in result["turn_results"]:
            row_items = [
                str(turn_res["turn"]),
                f"{turn_res['p1_overall_mean']:.2f}",
                f"{turn_res['p2_overall_mean']:.2f}",
            ]
            
            # Simply use P1 mean for dimension columns to save space, or combined
            for dim in DIMENSIONS:
                p1_val = turn_res.get(f"p1_{dim}_mean", 0)
                p2_val = turn_res.get(f"p2_{dim}_mean", 0)
                row_items.append(f"{p1_val:.1f}/{p2_val:.1f}")
                
            prog_table.add_row(*row_items)
            
        console.print(prog_table)
        
        # Also show the final detailed run table as before
        console.print("[dim]Final Turn Details:[/]")

    # Individual runs table - show all dimensions
    runs_table = Table(title=f"Individual Runs ({result['num_successful_runs']} runs)")
    runs_table.add_column("Metric", justify="left", style="cyan")
    
    # Add columns for each run
    for run_idx in range(result['num_successful_runs']):
        runs_table.add_column(f"Run {run_idx + 1}", justify="center")
    runs_table.add_column("Mean", justify="center", style="bold")
    runs_table.add_column("Std", justify="center", style="dim")
    
    # P1 Overall
    p1_vals = [f"{r['p1_overall']:.2f}" for r in result["runs"]]
    runs_table.add_row(
        "[bold]P1 Overall[/]",
        *p1_vals,
        f"{result['p1_overall_mean']:.2f}",
        f"±{result['p1_overall_std']:.2f}",
    )
    
    # P2 Overall
    p2_vals = [f"{r['p2_overall']:.2f}" for r in result["runs"]]
    runs_table.add_row(
        "[bold]P2 Overall[/]",
        *p2_vals,
        f"{result['p2_overall_mean']:.2f}",
        f"±{result['p2_overall_std']:.2f}",
    )
    
    # Add separator
    runs_table.add_row("─" * 12, *["─" * 8] * (result['num_successful_runs'] + 2))
    
    # Add each dimension for P1 and P2
    for dim in DIMENSIONS:
        # P1 dimension
        p1_dim_vals = [
            f"{r['p1_dimensions'].get(dim, 0):.0f}" if r['p1_dimensions'] else "0"
            for r in result["runs"]
        ]
        p1_mean = result["p1_dimensions_mean"].get(dim, 0)
        p1_std = result["p1_dimensions_std"].get(dim, 0)
        runs_table.add_row(
            f"P1 {dim[:12]}",
            *p1_dim_vals,
            f"{p1_mean:.1f}",
            f"±{p1_std:.1f}",
        )
        
        # P2 dimension
        p2_dim_vals = [
            f"{r['p2_dimensions'].get(dim, 0):.0f}" if r['p2_dimensions'] else "0"
            for r in result["runs"]
        ]
        p2_mean = result["p2_dimensions_mean"].get(dim, 0)
        p2_std = result["p2_dimensions_std"].get(dim, 0)
        runs_table.add_row(
            f"P2 {dim[:12]}",
            *p2_dim_vals,
            f"{p2_mean:.1f}",
            f"±{p2_std:.1f}",
        )
    
    console.print(runs_table)


def display_multi_tag_comparison(
    tag_summaries: dict[str, dict[str, Any]]
) -> None:
    """
    显示多个 tag 的对比表格，with 95% CI

    参考 evaluate_by_tag.py 的 display_comparison 函数

    Args:
        tag_summaries: {tag_name: results.json 的内容}
    """
    console.print("\n[bold cyan]═══ Multi-Tag Comparison with 95% CI ═══[/]\n")

    # ===== Table 1: Main Comparison with CI =====
    table = Table(title="Final Evaluation Scores Comparison with 95% CI")
    table.add_column("Tag", style="cyan", no_wrap=False)
    table.add_column("Eps", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Exist", justify="right", style="dim")
    table.add_column("New", justify="right", style="green")
    table.add_column("P1 Mean", justify="right", style="yellow")
    table.add_column("P1 CI (±)", justify="right", style="dim")
    table.add_column("P1 SE", justify="right", style="dim magenta")
    table.add_column("P2 Mean", justify="right", style="yellow")
    table.add_column("P2 CI (±)", justify="right", style="dim")
    table.add_column("P2 SE", justify="right", style="dim magenta")
    table.add_column("P1 Goal", justify="right", style="cyan")
    table.add_column("P1 Goal CI", justify="right", style="dim cyan")
    table.add_column("P2 Goal", justify="right", style="cyan")
    table.add_column("P2 Goal CI", justify="right", style="dim cyan")
    table.add_column("Avg Final", justify="right", style="bold green")
    table.add_column("Avg Goal", justify="right", style="bold")

    for tag, summary in tag_summaries.items():
        eval_summary = summary.get("evaluation_summary", {})
        overall = summary.get("overall", {})
        dimensions = summary.get("dimensions", {})

        # Get goal dimension info
        goal_dim = dimensions.get("goal", {})
        p1_goal = goal_dim.get("p1_mean", 0)
        p2_goal = goal_dim.get("p2_mean", 0)

        # Calculate CI for goal if we have raw data (not yet implemented in save_results_to_file)
        # For now, use std as approximation (will be enhanced when raw data is available)
        p1_goal_ci = goal_dim.get("p1_ci", goal_dim.get("p1_std", 0))
        p2_goal_ci = goal_dim.get("p2_ci", goal_dim.get("p2_std", 0))

        # Get CI from overall (if available, otherwise use std)
        p1_ci = overall.get("p1_ci", overall.get("p1_std", 0))
        p2_ci = overall.get("p2_ci", overall.get("p2_std", 0))
        p1_se = overall.get("p1_se", overall.get("p1_std", 0))
        p2_se = overall.get("p2_se", overall.get("p2_std", 0))

        table.add_row(
            tag,
            str(summary.get("num_episodes", 0)),
            str(summary.get("num_runs", 0)),
            f"{eval_summary.get('avg_existing_per_episode', 0):.1f}",
            f"{eval_summary.get('avg_new_per_episode', 0):.1f}",
            f"{overall.get('p1_mean', 0):.4f}",
            f"{p1_ci:.4f}",
            f"{p1_se:.4f}",
            f"{overall.get('p2_mean', 0):.4f}",
            f"{p2_ci:.4f}",
            f"{p2_se:.4f}",
            f"{p1_goal:.4f}",
            f"{p1_goal_ci:.4f}",
            f"{p2_goal:.4f}",
            f"{p2_goal_ci:.4f}",
            f"{overall.get('combined_mean', 0):.4f}",
            f"{(p1_goal + p2_goal) / 2:.4f}",
        )

    console.print(table)

    # ===== Table 2: Dimension Breakdown Comparison =====
    console.print("\n[bold cyan]Dimension Breakdown Comparison[/]\n")

    dim_table = Table(title="Average Dimension Scores (P1+P2 Combined)")
    dim_table.add_column("Tag", style="cyan")
    for dim in DIMENSIONS:
        # Show dimension name and add CI column
        dim_table.add_column(f"{dim[:8]} Mean", justify="right")
    dim_table.add_column("Overall", justify="right", style="bold")

    for tag, summary in tag_summaries.items():
        row = [tag]
        dimensions = summary.get("dimensions", {})
        for dim in DIMENSIONS:
            dim_data = dimensions.get(dim, {})
            p1_val = dim_data.get("p1_mean", 0)
            p2_val = dim_data.get("p2_mean", 0)
            avg_val = (p1_val + p2_val) / 2
            row.append(f"{avg_val:.3f}")

        overall = summary.get("overall", {})
        row.append(f"{overall.get('combined_mean', 0):.4f}")

        dim_table.add_row(*row)

    console.print(dim_table)

    # ===== Table 3: Goal Dimension Detailed Comparison =====
    console.print("\n[bold cyan]Goal Dimension Detailed Comparison[/]\n")

    goal_table = Table(title="Goal Dimension with 95% CI")
    goal_table.add_column("Tag", style="cyan")
    goal_table.add_column("P1 Goal Mean", justify="right", style="green")
    goal_table.add_column("P1 Goal CI (±)", justify="right", style="dim")
    goal_table.add_column("P2 Goal Mean", justify="right", style="blue")
    goal_table.add_column("P2 Goal CI (±)", justify="right", style="dim")
    goal_table.add_column("Combined Goal", justify="right", style="bold yellow")

    for tag, summary in tag_summaries.items():
        dimensions = summary.get("dimensions", {})
        goal_dim = dimensions.get("goal", {})

        p1_goal = goal_dim.get("p1_mean", 0)
        p2_goal = goal_dim.get("p2_mean", 0)
        p1_goal_ci = goal_dim.get("p1_ci", goal_dim.get("p1_std", 0))
        p2_goal_ci = goal_dim.get("p2_ci", goal_dim.get("p2_std", 0))

        goal_table.add_row(
            tag,
            f"{p1_goal:.4f}",
            f"{p1_goal_ci:.4f}",
            f"{p2_goal:.4f}",
            f"{p2_goal_ci:.4f}",
            f"{(p1_goal + p2_goal) / 2:.4f}",
        )

    console.print(goal_table)

    # ===== NEW: Complete Summary Table with All Metrics =====
    console.print("\n[bold cyan]═══ Complete Summary Table ═══[/]\n")

    summary_full_table = Table(title="Complete Tag Summary with All Metrics")
    summary_full_table.add_column("Tag", style="cyan", no_wrap=False)
    summary_full_table.add_column("P1 Mean", justify="right", style="yellow")
    summary_full_table.add_column("P1 CI (±)", justify="right", style="dim")
    summary_full_table.add_column("P1 SE", justify="right", style="dim magenta")
    summary_full_table.add_column("P2 Mean", justify="right", style="yellow")
    summary_full_table.add_column("P2 CI (±)", justify="right", style="dim")
    summary_full_table.add_column("P2 SE", justify="right", style="dim magenta")
    summary_full_table.add_column("P1 Goal", justify="right", style="cyan")
    summary_full_table.add_column("P1 Goal CI", justify="right", style="dim cyan")
    summary_full_table.add_column("P1 Goal SE", justify="right", style="dim magenta")
    summary_full_table.add_column("P2 Goal", justify="right", style="cyan")
    summary_full_table.add_column("P2 Goal CI", justify="right", style="dim cyan")
    summary_full_table.add_column("P2 Goal SE", justify="right", style="dim magenta")
    summary_full_table.add_column("Avg Final", justify="right", style="bold green")
    summary_full_table.add_column("Avg Goal", justify="right", style="bold yellow")

    # Prepare data for sorting tables
    tag_data = []
    for tag, summary in tag_summaries.items():
        overall = summary.get("overall", {})
        dimensions = summary.get("dimensions", {})
        goal_dim = dimensions.get("goal", {})

        # Extract all metrics
        p1_mean = overall.get("p1_mean", 0)
        p1_ci = overall.get("p1_ci", overall.get("p1_std", 0))
        p1_se = overall.get("p1_se", overall.get("p1_std", 0))
        p2_mean = overall.get("p2_mean", 0)
        p2_ci = overall.get("p2_ci", overall.get("p2_std", 0))
        p2_se = overall.get("p2_se", overall.get("p2_std", 0))
        p1_goal = goal_dim.get("p1_mean", 0)
        p1_goal_ci = goal_dim.get("p1_ci", goal_dim.get("p1_std", 0))
        p1_goal_se = goal_dim.get("p1_se", goal_dim.get("p1_std", 0))
        p2_goal = goal_dim.get("p2_mean", 0)
        p2_goal_ci = goal_dim.get("p2_ci", goal_dim.get("p2_std", 0))
        p2_goal_se = goal_dim.get("p2_se", goal_dim.get("p2_std", 0))
        avg_final = overall.get("combined_mean", 0)
        avg_goal = (p1_goal + p2_goal) / 2

        # Add row to table
        summary_full_table.add_row(
            tag,
            f"{p1_mean:.4f}",
            f"{p1_ci:.4f}",
            f"{p1_se:.4f}",
            f"{p2_mean:.4f}",
            f"{p2_ci:.4f}",
            f"{p2_se:.4f}",
            f"{p1_goal:.4f}",
            f"{p1_goal_ci:.4f}",
            f"{p1_goal_se:.4f}",
            f"{p2_goal:.4f}",
            f"{p2_goal_ci:.4f}",
            f"{p2_goal_se:.4f}",
            f"{avg_final:.4f}",
            f"{avg_goal:.4f}",
        )

        # Store data for sorting
        tag_data.append({
            "tag": tag,
            "p1_mean": p1_mean,
            "p1_se": p1_se,
            "p2_mean": p2_mean,
            "p2_se": p2_se,
            "p1_goal": p1_goal,
            "p1_goal_se": p1_goal_se,
            "p2_goal": p2_goal,
            "p2_goal_se": p2_goal_se,
            "avg_final": avg_final,
            "avg_goal": avg_goal,
        })

    console.print(summary_full_table)

    # ===== NEW: Sorted Ranking Tables =====
    console.print("\n[bold cyan]═══ Sorted Rankings ═══[/]\n")

    # 1. Ranked by Average Final Reward
    sorted_by_avg_final = sorted(tag_data, key=lambda x: x["avg_final"], reverse=True)
    table_avg_final = Table(title="Ranked by Average Final Reward (Descending)")
    table_avg_final.add_column("Rank", justify="right", style="dim")
    table_avg_final.add_column("Tag", style="cyan")
    table_avg_final.add_column("Avg Final", justify="right", style="bold yellow")
    table_avg_final.add_column("P1 Mean", justify="right")
    table_avg_final.add_column("P2 Mean", justify="right")
    table_avg_final.add_column("Avg Goal", justify="right")

    for i, data in enumerate(sorted_by_avg_final, 1):
        table_avg_final.add_row(
            str(i),
            data["tag"],
            f"{data['avg_final']:.4f}",
            f"{data['p1_mean']:.4f}",
            f"{data['p2_mean']:.4f}",
            f"{data['avg_goal']:.4f}",
        )
    console.print(table_avg_final)

    # 2. Ranked by Average Goal Score
    sorted_by_avg_goal = sorted(tag_data, key=lambda x: x["avg_goal"], reverse=True)
    table_avg_goal = Table(title="Ranked by Average Goal Score (Descending)")
    table_avg_goal.add_column("Rank", justify="right", style="dim")
    table_avg_goal.add_column("Tag", style="cyan")
    table_avg_goal.add_column("Avg Goal", justify="right", style="bold green")
    table_avg_goal.add_column("P1 Goal", justify="right")
    table_avg_goal.add_column("P2 Goal", justify="right")
    table_avg_goal.add_column("Avg Final", justify="right")

    for i, data in enumerate(sorted_by_avg_goal, 1):
        table_avg_goal.add_row(
            str(i),
            data["tag"],
            f"{data['avg_goal']:.4f}",
            f"{data['p1_goal']:.4f}",
            f"{data['p2_goal']:.4f}",
            f"{data['avg_final']:.4f}",
        )
    console.print(table_avg_goal)

    # 3. Ranked by P1 Final Reward
    sorted_by_p1 = sorted(tag_data, key=lambda x: x["p1_mean"], reverse=True)
    table_p1 = Table(title="Ranked by P1 Final Reward (Descending)")
    table_p1.add_column("Rank", justify="right", style="dim")
    table_p1.add_column("Tag", style="cyan")
    table_p1.add_column("P1 Mean", justify="right", style="bold blue")
    table_p1.add_column("P1 SE", justify="right", style="dim magenta")
    table_p1.add_column("P1 Goal", justify="right")
    table_p1.add_column("Avg Final", justify="right")

    for i, data in enumerate(sorted_by_p1, 1):
        table_p1.add_row(
            str(i),
            data["tag"],
            f"{data['p1_mean']:.4f}",
            f"{data['p1_se']:.4f}",
            f"{data['p1_goal']:.4f}",
            f"{data['avg_final']:.4f}",
        )
    console.print(table_p1)

    # 4. Ranked by P2 Final Reward
    sorted_by_p2 = sorted(tag_data, key=lambda x: x["p2_mean"], reverse=True)
    table_p2 = Table(title="Ranked by P2 Final Reward (Descending)")
    table_p2.add_column("Rank", justify="right", style="dim")
    table_p2.add_column("Tag", style="cyan")
    table_p2.add_column("P2 Mean", justify="right", style="bold magenta")
    table_p2.add_column("P2 SE", justify="right", style="dim magenta")
    table_p2.add_column("P2 Goal", justify="right")
    table_p2.add_column("Avg Final", justify="right")

    for i, data in enumerate(sorted_by_p2, 1):
        table_p2.add_row(
            str(i),
            data["tag"],
            f"{data['p2_mean']:.4f}",
            f"{data['p2_se']:.4f}",
            f"{data['p2_goal']:.4f}",
            f"{data['avg_final']:.4f}",
        )
    console.print(table_p2)

    # 5. Ranked by P1 Goal Score
    sorted_by_p1_goal = sorted(tag_data, key=lambda x: x["p1_goal"], reverse=True)
    table_p1_goal = Table(title="Ranked by P1 Goal Score (Descending)")
    table_p1_goal.add_column("Rank", justify="right", style="dim")
    table_p1_goal.add_column("Tag", style="cyan")
    table_p1_goal.add_column("P1 Goal", justify="right", style="bold blue")
    table_p1_goal.add_column("P1 Goal SE", justify="right", style="dim magenta")
    table_p1_goal.add_column("P1 Mean", justify="right")
    table_p1_goal.add_column("Avg Goal", justify="right")

    for i, data in enumerate(sorted_by_p1_goal, 1):
        table_p1_goal.add_row(
            str(i),
            data["tag"],
            f"{data['p1_goal']:.4f}",
            f"{data['p1_goal_se']:.4f}",
            f"{data['p1_mean']:.4f}",
            f"{data['avg_goal']:.4f}",
        )
    console.print(table_p1_goal)

    # 6. Ranked by P2 Goal Score
    sorted_by_p2_goal = sorted(tag_data, key=lambda x: x["p2_goal"], reverse=True)
    table_p2_goal = Table(title="Ranked by P2 Goal Score (Descending)")
    table_p2_goal.add_column("Rank", justify="right", style="dim")
    table_p2_goal.add_column("Tag", style="cyan")
    table_p2_goal.add_column("P2 Goal", justify="right", style="bold magenta")
    table_p2_goal.add_column("P2 Goal SE", justify="right", style="dim magenta")
    table_p2_goal.add_column("P2 Mean", justify="right")
    table_p2_goal.add_column("Avg Goal", justify="right")

    for i, data in enumerate(sorted_by_p2_goal, 1):
        table_p2_goal.add_row(
            str(i),
            data["tag"],
            f"{data['p2_goal']:.4f}",
            f"{data['p2_goal_se']:.4f}",
            f"{data['p2_mean']:.4f}",
            f"{data['avg_goal']:.4f}",
        )
    console.print(table_p2_goal)


def check_evaluation_quality(
    all_results: list[dict[str, Any]],
    target_num_runs: int,
    tag: str,
) -> None:
    """
    检查评估质量，确保每个场景的评估次数都达标

    Args:
        all_results: 所有 episode 的评估结果
        target_num_runs: 目标评估次数
        tag: 实验标签
    """
    console.print("\n[bold cyan]═══ Evaluation Quality Check ═══[/]\n")

    total_episodes = len(all_results)

    # 分类统计
    fully_evaluated = []  # 达到目标评估次数的 episodes
    partially_evaluated = []  # 部分评估（1 <= runs < target）
    failed_episodes = []  # 完全失败（0 次评估）

    for result in all_results:
        episode_pk = result.get("episode_pk", "unknown")
        num_runs = result.get("num_successful_runs", 0)

        if num_runs >= target_num_runs:
            fully_evaluated.append((episode_pk, num_runs))
        elif num_runs > 0:
            partially_evaluated.append((episode_pk, num_runs))
        else:
            failed_episodes.append(episode_pk)

    # 创建质量报告表格
    quality_table = Table(title=f"Quality Report for Tag: {tag}")
    quality_table.add_column("Metric", style="cyan", width=40)
    quality_table.add_column("Count", justify="right", style="yellow")
    quality_table.add_column("Percentage", justify="right", style="green")

    quality_table.add_row(
        "Total Episodes",
        str(total_episodes),
        "100.0%",
    )
    quality_table.add_row(
        f"Fully Evaluated (>= {target_num_runs} runs)",
        str(len(fully_evaluated)),
        f"{len(fully_evaluated) / total_episodes * 100:.1f}%" if total_episodes > 0 else "0.0%",
        style="bold green" if len(fully_evaluated) == total_episodes else "yellow"
    )
    quality_table.add_row(
        "Partially Evaluated (1-{} runs)".format(target_num_runs - 1),
        str(len(partially_evaluated)),
        f"{len(partially_evaluated) / total_episodes * 100:.1f}%" if total_episodes > 0 else "0.0%",
        style="yellow" if len(partially_evaluated) > 0 else "dim"
    )
    quality_table.add_row(
        "Failed (0 runs)",
        str(len(failed_episodes)),
        f"{len(failed_episodes) / total_episodes * 100:.1f}%" if total_episodes > 0 else "0.0%",
        style="red" if len(failed_episodes) > 0 else "dim"
    )

    console.print(quality_table)

    # 显示部分评估的 episodes（如果有）
    if partially_evaluated:
        console.print("\n[bold yellow]⚠️  Partially Evaluated Episodes:[/]")
        partial_table = Table()
        partial_table.add_column("Episode PK (first 16 chars)", style="cyan")
        partial_table.add_column("Actual Runs", justify="right", style="yellow")
        partial_table.add_column("Missing Runs", justify="right", style="red")

        # 只显示前 10 个，避免输出过多
        for episode_pk, num_runs in partially_evaluated[:10]:
            partial_table.add_row(
                episode_pk[:16],
                str(num_runs),
                str(target_num_runs - num_runs),
            )

        if len(partially_evaluated) > 10:
            partial_table.add_row(
                f"... and {len(partially_evaluated) - 10} more",
                "",
                "",
                style="dim"
            )

        console.print(partial_table)

    # 显示完全失败的 episodes（如果有）
    if failed_episodes:
        console.print("\n[bold red]✗ Failed Episodes (0 successful runs):[/]")
        failed_table = Table()
        failed_table.add_column("Episode PK (first 16 chars)", style="cyan")

        # 只显示前 10 个
        for episode_pk in failed_episodes[:10]:
            failed_table.add_row(episode_pk[:16])

        if len(failed_episodes) > 10:
            failed_table.add_row(
                f"... and {len(failed_episodes) - 10} more",
                style="dim"
            )

        console.print(failed_table)

    # 最终判断
    if len(fully_evaluated) == total_episodes:
        console.print("\n[bold green]✓ All episodes successfully evaluated to target ({} runs each)![/]".format(target_num_runs))
    elif len(failed_episodes) == 0 and len(partially_evaluated) > 0:
        console.print(f"\n[yellow]⚠️  Some episodes did not reach target evaluations, but all have at least partial results[/]")
    elif len(failed_episodes) > 0:
        console.print(f"\n[red]✗ {len(failed_episodes)} episodes completely failed evaluation[/]")


def display_summary_table(all_results: list[dict[str, Any]], num_runs: int) -> None:
    """
    Display summary table for all episodes with 95% CI.

    References evaluate_by_tag.py's "Final Episode Rewards with 95% CI" table format.
    """
    console.print("\n")
    console.print(Panel("[bold cyan]═══ Summary Across All Episodes ═══[/]"))

    # ===== NEW: Table showing aggregated scores by run number =====
    # Organize runs by run number (R1, R2, R3, etc.) across all episodes
    if all_results and all_results[0].get("runs"):
        # Get number of runs per episode (should be same for all)
        runs_per_episode = len(all_results[0].get("runs", []))

        # Create table showing scores aggregated by run number
        runs_detail_table = Table(title=f"Aggregated Scores by Run Number ({len(all_results)} episodes × {num_runs} runs)")
        runs_detail_table.add_column("Metric", justify="left", style="cyan")

        # Add columns for each run number
        for run_idx in range(num_runs):
            runs_detail_table.add_column(f"R{run_idx + 1}", justify="center", style="dim yellow")

        runs_detail_table.add_column("Mean", justify="center", style="bold yellow")
        runs_detail_table.add_column("Std", justify="center", style="dim")
        runs_detail_table.add_column("CI (±)", justify="center", style="dim magenta")

        # Collect scores by run number
        p1_overalls_by_run = [[] for _ in range(num_runs)]
        p2_overalls_by_run = [[] for _ in range(num_runs)]
        p1_goals_by_run = [[] for _ in range(num_runs)]
        p2_goals_by_run = [[] for _ in range(num_runs)]

        for result in all_results:
            runs = result.get("runs", [])
            for run_idx, run in enumerate(runs[:num_runs]):  # Only take first num_runs
                p1_overalls_by_run[run_idx].append(run.get("p1_overall", 0.0))
                p2_overalls_by_run[run_idx].append(run.get("p2_overall", 0.0))
                p1_goals_by_run[run_idx].append(run.get("p1_dimensions", {}).get("goal", 0.0))
                p2_goals_by_run[run_idx].append(run.get("p2_dimensions", {}).get("goal", 0.0))

        # P1 Overall row
        p1_overall_means = [np.mean(p1_overalls_by_run[i]) if p1_overalls_by_run[i] else 0 for i in range(num_runs)]
        all_p1_overalls = [score for run_scores in p1_overalls_by_run for score in run_scores]
        p1_overall_mean, p1_overall_ci = calculate_confidence_interval(all_p1_overalls)
        p1_overall_std = float(np.std(all_p1_overalls))

        row_p1_overall = ["[bold]P1 Overall[/]"]
        row_p1_overall.extend([f"{mean:.2f}" for mean in p1_overall_means])
        row_p1_overall.extend([
            f"{p1_overall_mean:.4f}",
            f"{p1_overall_std:.4f}",
            f"{p1_overall_ci:.4f}"
        ])
        runs_detail_table.add_row(*row_p1_overall)

        # P2 Overall row
        p2_overall_means = [np.mean(p2_overalls_by_run[i]) if p2_overalls_by_run[i] else 0 for i in range(num_runs)]
        all_p2_overalls = [score for run_scores in p2_overalls_by_run for score in run_scores]
        p2_overall_mean, p2_overall_ci = calculate_confidence_interval(all_p2_overalls)
        p2_overall_std = float(np.std(all_p2_overalls))

        row_p2_overall = ["[bold]P2 Overall[/]"]
        row_p2_overall.extend([f"{mean:.2f}" for mean in p2_overall_means])
        row_p2_overall.extend([
            f"{p2_overall_mean:.4f}",
            f"{p2_overall_std:.4f}",
            f"{p2_overall_ci:.4f}"
        ])
        runs_detail_table.add_row(*row_p2_overall)

        # Add separator
        separator_row = ["─" * 12]
        separator_row.extend(["─" * 6] * num_runs)
        separator_row.extend(["─" * 8, "─" * 8, "─" * 8])
        runs_detail_table.add_row(*separator_row)

        # P1 Goal row
        p1_goal_means = [np.mean(p1_goals_by_run[i]) if p1_goals_by_run[i] else 0 for i in range(num_runs)]
        all_p1_goals = [score for run_scores in p1_goals_by_run for score in run_scores]
        p1_goal_mean, p1_goal_ci = calculate_confidence_interval(all_p1_goals)
        p1_goal_std = float(np.std(all_p1_goals))

        row_p1_goal = ["[bold]P1 Goal[/]"]
        row_p1_goal.extend([f"{mean:.1f}" for mean in p1_goal_means])
        row_p1_goal.extend([
            f"{p1_goal_mean:.4f}",
            f"{p1_goal_std:.4f}",
            f"{p1_goal_ci:.4f}"
        ])
        runs_detail_table.add_row(*row_p1_goal)

        # P2 Goal row
        p2_goal_means = [np.mean(p2_goals_by_run[i]) if p2_goals_by_run[i] else 0 for i in range(num_runs)]
        all_p2_goals = [score for run_scores in p2_goals_by_run for score in run_scores]
        p2_goal_mean, p2_goal_ci = calculate_confidence_interval(all_p2_goals)
        p2_goal_std = float(np.std(all_p2_goals))

        row_p2_goal = ["[bold]P2 Goal[/]"]
        row_p2_goal.extend([f"{mean:.1f}" for mean in p2_goal_means])
        row_p2_goal.extend([
            f"{p2_goal_mean:.4f}",
            f"{p2_goal_std:.4f}",
            f"{p2_goal_ci:.4f}"
        ])
        runs_detail_table.add_row(*row_p2_goal)

        console.print(runs_detail_table)
        console.print("\n")

    # ===== Table 1: Overall Summary with 95% CI (episode-level aggregation) =====
    summary_table = Table(title=f"Episode-Level Aggregated Results with 95% CI ({len(all_results)} episodes)")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("P1 Mean", justify="right", style="yellow")
    summary_table.add_column("P1 CI (±)", justify="right", style="dim")
    summary_table.add_column("P1 SE", justify="right", style="dim magenta")
    summary_table.add_column("P2 Mean", justify="right", style="yellow")
    summary_table.add_column("P2 CI (±)", justify="right", style="dim")
    summary_table.add_column("P2 SE", justify="right", style="dim magenta")
    summary_table.add_column("Combined Mean", justify="right", style="bold green")

    # Collect all values across episodes (mean per episode)
    p1_overalls = [r["p1_overall_mean"] for r in all_results]
    p2_overalls = [r["p2_overall_mean"] for r in all_results]

    # Calculate CI and SE for overall scores
    p1_mean, p1_ci = calculate_confidence_interval(p1_overalls)
    p2_mean, p2_ci = calculate_confidence_interval(p2_overalls)
    p1_se = np.std(p1_overalls, ddof=1) / np.sqrt(len(p1_overalls)) if len(p1_overalls) > 1 else 0.0
    p2_se = np.std(p2_overalls, ddof=1) / np.sqrt(len(p2_overalls)) if len(p2_overalls) > 1 else 0.0

    summary_table.add_row(
        "[bold]Overall Score[/]",
        f"[bold]{p1_mean:.4f}[/]",
        f"{p1_ci:.4f}",
        f"{p1_se:.4f}",
        f"[bold]{p2_mean:.4f}[/]",
        f"{p2_ci:.4f}",
        f"{p2_se:.4f}",
        f"[bold]{(p1_mean + p2_mean) / 2:.4f}[/]",
    )

    # Add dimension breakdown with CI
    for dim in DIMENSIONS:
        p1_dim_values = [r["p1_dimensions_mean"].get(dim, 0) for r in all_results]
        p2_dim_values = [r["p2_dimensions_mean"].get(dim, 0) for r in all_results]

        p1_dim_mean, p1_dim_ci = calculate_confidence_interval(p1_dim_values)
        p2_dim_mean, p2_dim_ci = calculate_confidence_interval(p2_dim_values)
        p1_dim_se = np.std(p1_dim_values, ddof=1) / np.sqrt(len(p1_dim_values)) if len(p1_dim_values) > 1 else 0.0
        p2_dim_se = np.std(p2_dim_values, ddof=1) / np.sqrt(len(p2_dim_values)) if len(p2_dim_values) > 1 else 0.0

        summary_table.add_row(
            dim,
            f"{p1_dim_mean:.4f}",
            f"{p1_dim_ci:.4f}",
            f"{p1_dim_se:.4f}",
            f"{p2_dim_mean:.4f}",
            f"{p2_dim_ci:.4f}",
            f"{p2_dim_se:.4f}",
            f"{(p1_dim_mean + p2_dim_mean) / 2:.4f}",
        )

    console.print(summary_table)

    # ===== Table 2: Evaluation Source Statistics =====
    console.print("\n")
    source_table = Table(title="Evaluation Source Statistics")
    source_table.add_column("Metric", style="cyan")
    source_table.add_column("Value", justify="right", style="yellow")

    total_existing = sum(r.get("num_existing_runs", 0) for r in all_results)
    total_new = sum(r.get("num_new_runs", 0) for r in all_results)
    total_evals = sum(r.get("num_successful_runs", 0) for r in all_results)

    source_table.add_row("Total Evaluations", str(total_evals))
    source_table.add_row("Existing (from checkpoint/DB)", f"{total_existing} ({total_existing / total_evals * 100:.1f}%)" if total_evals > 0 else "0")
    source_table.add_row("New (incremental)", f"{total_new} ({total_new / total_evals * 100:.1f}%)" if total_evals > 0 else "0")
    source_table.add_row("Avg Existing per Episode", f"{total_existing / len(all_results):.2f}" if all_results else "0")
    source_table.add_row("Avg New per Episode", f"{total_new / len(all_results):.2f}" if all_results else "0")

    console.print(source_table)

    # ===== Table 3: Final Grand Summary =====
    console.print("\n")
    grand_table = Table(title="[bold]Final Mean of Means (All Episodes)[/]")
    grand_table.add_column("Dimension", style="cyan")
    grand_table.add_column("P1 Mean", justify="right", style="green")
    grand_table.add_column("P2 Mean", justify="right", style="blue")
    grand_table.add_column("Combined", justify="right", style="yellow")

    # Overall
    grand_table.add_row(
        "[bold]Overall Score[/]",
        f"[bold]{np.mean(p1_overalls):.4f}[/]",
        f"[bold]{np.mean(p2_overalls):.4f}[/]",
        f"[bold]{(np.mean(p1_overalls) + np.mean(p2_overalls)) / 2:.4f}[/]",
    )
    
    # Add separator
    grand_table.add_row("─" * 15, "─" * 10, "─" * 10, "─" * 10)
    
    # Each dimension
    for dim in DIMENSIONS:
        p1_dim_values = [r["p1_dimensions_mean"].get(dim, 0) for r in all_results]
        p2_dim_values = [r["p2_dimensions_mean"].get(dim, 0) for r in all_results]
        p1_mean = np.mean(p1_dim_values)
        p2_mean = np.mean(p2_dim_values)
        combined = (p1_mean + p2_mean) / 2
        
        grand_table.add_row(
            dim,
            f"{p1_mean:.4f}",
            f"{p2_mean:.4f}",
            f"{combined:.4f}",
        )
    
    console.print(grand_table)
    
    # Combined average
    combined_overall = (np.mean(p1_overalls) + np.mean(p2_overalls)) / 2
    console.print(f"\n[bold green]★ Combined Average (P1+P2)/2: {combined_overall:.4f}[/]")


def generate_trend_analysis(
    all_results: list[dict[str, Any]],
    output_dir: Path,
    tag: str,
) -> None:
    """
    Generate trend analysis plots for evaluation scores over turns.
    
    Only effective if turn-level results are available (when --turn-interval is used).
    Plots:
    - Line chart for Overall Score (P1 vs P2) with error bands (std dev)
    - Line charts for each dimension
    """
    # Check if we have turn data
    if not all_results or "turn_results" not in all_results[0]:
        return
        
    console.print("\n")
    console.print(Panel("[bold cyan]═══ Turn Trend Analysis ═══[/]"))
    
    # Collect turn data across all episodes
    # Structure: {turn: {p1_overall: [], p2_overall: [], p1_dim: [], p2_dim: ...}}
    turn_data_map: dict[int, dict[str, list[float]]] = {}
    
    for res in all_results:
        if "turn_results" not in res:
            continue
            
        for turn_res in res["turn_results"]:
            turn = turn_res["turn"]
            if turn not in turn_data_map:
                turn_data_map[turn] = {
                    "p1_overall": [], "p2_overall": [],
                }
                for dim in DIMENSIONS:
                    turn_data_map[turn][f"p1_{dim}"] = []
                    turn_data_map[turn][f"p2_{dim}"] = []
            
            turn_data_map[turn]["p1_overall"].append(turn_res["p1_overall_mean"])
            turn_data_map[turn]["p2_overall"].append(turn_res["p2_overall_mean"])
            
            for dim in DIMENSIONS:
                turn_data_map[turn][f"p1_{dim}"].append(turn_res.get(f"p1_{dim}_mean", 0))
                turn_data_map[turn][f"p2_{dim}"].append(turn_res.get(f"p2_{dim}_mean", 0))
    
    if not turn_data_map:
        console.print("[yellow]No turn data available for trend analysis.[/]")
        return
        
    # Create trends directory
    trends_dir = output_dir / "trends"
    trends_dir.mkdir(parents=True, exist_ok=True)

    turns = sorted(turn_data_map.keys())
    
    # --- Helper function for plotting ---
    def plot_trend_data(
        data_map: dict[str, dict[str, Any]], 
        turn_keys: list[int],
        title_suffix: str, 
        filename: str
    ) -> None:
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics = ["overall"] + DIMENSIONS
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            d = data_map[metric]
            
            # Plot P1
            ax.plot(turn_keys, d["p1_mean"], 'o-', color="#3498db", label="P1", linewidth=2, markersize=5)
            ax.fill_between(turn_keys, 
                            d["p1_mean"] - d["p1_std"], 
                            d["p1_mean"] + d["p1_std"], 
                            color="#3498db", alpha=0.15)
            
            # Plot P2
            ax.plot(turn_keys, d["p2_mean"], 'o-', color="#e74c3c", label="P2", linewidth=2, markersize=5)
            ax.fill_between(turn_keys, 
                            d["p2_mean"] - d["p2_std"], 
                            d["p2_mean"] + d["p2_std"], 
                            color="#e74c3c", alpha=0.15)
            
            metric_name = "Overall Score" if metric == "overall" else metric.replace("_", " ").title()
            ax.set_title(metric_name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Turns")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
            
            if idx == 0:  # Only show legend on the first plot
                ax.legend(loc="best")
                
            # Ensure x-axis ticks are integers
            ax.set_xticks(turn_keys)
        
        # Hide unused subplot
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
            
        plt.suptitle(f"Score Trend Analysis over Turns - {title_suffix}\n{tag}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        # Save plot
        plt.savefig(trends_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # --- 1. Global Trend Plot (Aggregated) ---
    global_plot_data: dict[str, dict[str, Any]] = {}
    metrics = ["overall"] + DIMENSIONS
    
    for metric in metrics:
        p1_means, p1_stds = [], []
        p2_means, p2_stds = [], []
        
        for turn in turns:
            key_suffix = metric if metric != "overall" else "overall"
            p1_vals = turn_data_map[turn][f"p1_{key_suffix}"]
            p2_vals = turn_data_map[turn][f"p2_{key_suffix}"]
            
            p1_means.append(np.mean(p1_vals))
            p1_stds.append(np.std(p1_vals))
            p2_means.append(np.mean(p2_vals))
            p2_stds.append(np.std(p2_vals))
            
        global_plot_data[metric] = {
            "p1_mean": np.array(p1_means),
            "p1_std": np.array(p1_stds),
            "p2_mean": np.array(p2_means),
            "p2_std": np.array(p2_stds),
        }
    
    plot_trend_data(global_plot_data, turns, "All Scenarios", "global_trend.png")
    console.print(f"[bold green]Global trend analysis saved to:[/] {trends_dir / 'global_trend.png'}")
    
    # --- 2. Per-Scenario Trend Plots ---
    # Group results by environment
    env_results: dict[str, list[dict[str, Any]]] = {}
    for res in all_results:
        env = res.get("environment", "unknown")
        if env not in env_results:
            env_results[env] = []
        env_results[env].append(res)
    
    console.print(f"Generating trend plots for {len(env_results)} scenarios...")
    
    for env_name, res_list in tqdm(env_results.items(), desc="Plotting scenarios"):
        # Collect turn data for this environment
        env_turn_map: dict[int, dict[str, list[float]]] = {}
        for res in res_list:
            if "turn_results" not in res: continue
            for turn_res in res["turn_results"]:
                turn = turn_res["turn"]
                if turn not in env_turn_map:
                    env_turn_map[turn] = {
                        "p1_overall": [], "p2_overall": [],
                    }
                    for dim in DIMENSIONS:
                        env_turn_map[turn][f"p1_{dim}"] = []
                        env_turn_map[turn][f"p2_{dim}"] = []
                
                # If we have multiple runs, we can use the runs directly if available, 
                # but turn_results only stores aggregated means for now unless we dig deeper.
                # Actually, turn_res['runs'] contains individual run data!
                # Let's extract raw values from all runs to get better variance estimation
                if "runs" in turn_res:
                    for r in turn_res["runs"]:
                        env_turn_map[turn]["p1_overall"].append(r["p1_overall"])
                        env_turn_map[turn]["p2_overall"].append(r["p2_overall"])
                        for dim in DIMENSIONS:
                            env_turn_map[turn][f"p1_{dim}"].append(r["p1_dimensions"].get(dim, 0) if r["p1_dimensions"] else 0)
                            env_turn_map[turn][f"p2_{dim}"].append(r["p2_dimensions"].get(dim, 0) if r["p2_dimensions"] else 0)
                else:
                    # Fallback to means if runs not available
                    env_turn_map[turn]["p1_overall"].append(turn_res["p1_overall_mean"])
                    env_turn_map[turn]["p2_overall"].append(turn_res["p2_overall_mean"])
                    for dim in DIMENSIONS:
                        env_turn_map[turn][f"p1_{dim}"].append(turn_res.get(f"p1_{dim}_mean", 0))
                        env_turn_map[turn][f"p2_{dim}"].append(turn_res.get(f"p2_{dim}_mean", 0))

        if not env_turn_map:
            continue
            
        env_turns = sorted(env_turn_map.keys())
        env_plot_data = {}
        
        for metric in metrics:
            p1_means, p1_stds = [], []
            p2_means, p2_stds = [], []
            
            for turn in env_turns:
                 key_suffix = metric if metric != "overall" else "overall"
                 p1_vals = env_turn_map[turn][f"p1_{key_suffix}"]
                 p2_vals = env_turn_map[turn][f"p2_{key_suffix}"]
                 
                 p1_means.append(np.mean(p1_vals) if p1_vals else 0)
                 p1_stds.append(np.std(p1_vals) if len(p1_vals) > 1 else 0)
                 p2_means.append(np.mean(p2_vals) if p2_vals else 0)
                 p2_stds.append(np.std(p2_vals) if len(p2_vals) > 1 else 0)
            
            env_plot_data[metric] = {
                "p1_mean": np.array(p1_means),
                "p1_std": np.array(p1_stds),
                "p2_mean": np.array(p2_means),
                "p2_std": np.array(p2_stds),
            }
            
        safe_env_name = "".join(c if c.isalnum() else "_" for c in env_name)[:50]
        plot_trend_data(env_plot_data, env_turns, f"Scenario: {env_name[:30]}...", f"trend_{safe_env_name}.png")

    # --- 3. Per-Episode Trend Plots ---
    # Create episodes directory
    episodes_dir = trends_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"Generating trend plots for {len(all_results)} episodes...")
    
    for res in tqdm(all_results, desc="Plotting episodes"):
        if "turn_results" not in res or not res["turn_results"]:
            continue
            
        episode_pk = res["episode_pk"]
        
        # Collect turn data for this episode
        ep_turn_map: dict[int, dict[str, list[float]]] = {}
        for turn_res in res["turn_results"]:
            turn = turn_res["turn"]
            if turn not in ep_turn_map:
                ep_turn_map[turn] = {
                    "p1_overall": [], "p2_overall": [],
                }
                for dim in DIMENSIONS:
                    ep_turn_map[turn][f"p1_{dim}"] = []
                    ep_turn_map[turn][f"p2_{dim}"] = []
            
            # Use runs data if available, otherwise mean
            if "runs" in turn_res:
                for r in turn_res["runs"]:
                    ep_turn_map[turn]["p1_overall"].append(r["p1_overall"])
                    ep_turn_map[turn]["p2_overall"].append(r["p2_overall"])
                    for dim in DIMENSIONS:
                        ep_turn_map[turn][f"p1_{dim}"].append(r["p1_dimensions"].get(dim, 0) if r["p1_dimensions"] else 0)
                        ep_turn_map[turn][f"p2_{dim}"].append(r["p2_dimensions"].get(dim, 0) if r["p2_dimensions"] else 0)
            else:
                ep_turn_map[turn]["p1_overall"].append(turn_res["p1_overall_mean"])
                ep_turn_map[turn]["p2_overall"].append(turn_res["p2_overall_mean"])
                for dim in DIMENSIONS:
                    ep_turn_map[turn][f"p1_{dim}"].append(turn_res.get(f"p1_{dim}_mean", 0))
                    ep_turn_map[turn][f"p2_{dim}"].append(turn_res.get(f"p2_{dim}_mean", 0))
        
        if not ep_turn_map:
            continue

        ep_turns = sorted(ep_turn_map.keys())
        ep_plot_data = {}
        
        for metric in metrics:
            p1_means, p1_stds = [], []
            p2_means, p2_stds = [], []
            
            for turn in ep_turns:
                 key_suffix = metric if metric != "overall" else "overall"
                 p1_vals = ep_turn_map[turn][f"p1_{key_suffix}"]
                 p2_vals = ep_turn_map[turn][f"p2_{key_suffix}"]
                 
                 p1_means.append(np.mean(p1_vals) if p1_vals else 0)
                 p1_stds.append(np.std(p1_vals) if len(p1_vals) > 1 else 0)
                 p2_means.append(np.mean(p2_vals) if p2_vals else 0)
                 p2_stds.append(np.std(p2_vals) if len(p2_vals) > 1 else 0)
            
            ep_plot_data[metric] = {
                "p1_mean": np.array(p1_means),
                "p1_std": np.array(p1_stds),
                "p2_mean": np.array(p2_means),
                "p2_std": np.array(p2_stds),
            }
        
        plot_trend_data(ep_plot_data, ep_turns, f"Episode: {episode_pk[:8]}...", f"episodes/{episode_pk}.png")


def generate_dimension_analysis(
    all_results: list[dict[str, Any]],
    output_dir: Path,
    tag: str,
) -> None:
    """
    Generate variance analysis and box plots for evaluation dimensions.

    Includes:
    - Variance analysis table with F-statistic and p-value (separately for P1 and P2)
    - Box plots for each dimension comparing P1 vs P2
    """
    from scipy import stats

    console.print("\n")
    console.print(Panel("[bold cyan]═══ Dimension Variance Analysis ═══[/]"))

    # Collect dimension data
    dimension_data: dict[str, dict[str, list[float]]] = {}
    for dim in DIMENSIONS:
        dimension_data[dim] = {
            "p1": [r["p1_dimensions_mean"].get(dim, 0) for r in all_results],
            "p2": [r["p2_dimensions_mean"].get(dim, 0) for r in all_results],
        }

    # Also add overall scores
    dimension_data["_overall"] = {
        "p1": [r["p1_overall_mean"] for r in all_results],
        "p2": [r["p2_overall_mean"] for r in all_results],
    }

    # Determine number of runs from first result
    num_runs = len(all_results[0].get("runs", [])) if all_results else 0

    # ===== Cross-Run ANOVA Analysis (Separately for P1 and P2) =====
    # Test if there's significant difference between different runs (R1 vs R2 vs R3...)
    console.print(f"\n[bold cyan]Cross-Run ANOVA Analysis (Testing consistency across {num_runs} runs)[/]\n")

    # P1 ANOVA Table
    p1_anova_table = Table(title="P1: ANOVA Across Runs (H0: All runs have equal means)")
    p1_anova_table.add_column("Dimension", style="cyan")
    for i in range(num_runs):
        p1_anova_table.add_column(f"R{i+1} Mean", justify="right", style="dim")
    p1_anova_table.add_column("F-stat", justify="right", style="yellow")
    p1_anova_table.add_column("p-value", justify="right", style="yellow")
    p1_anova_table.add_column("Significant", justify="center", style="bold")

    # P2 ANOVA Table
    p2_anova_table = Table(title="P2: ANOVA Across Runs (H0: All runs have equal means)")
    p2_anova_table.add_column("Dimension", style="cyan")
    for i in range(num_runs):
        p2_anova_table.add_column(f"R{i+1} Mean", justify="right", style="dim")
    p2_anova_table.add_column("F-stat", justify="right", style="yellow")
    p2_anova_table.add_column("p-value", justify="right", style="yellow")
    p2_anova_table.add_column("Significant", justify="center", style="bold")

    dimensions_to_analyze = ["_overall"] + DIMENSIONS
    analysis_results = []

    for dim_name in dimensions_to_analyze:
        display_name = "Overall" if dim_name == "_overall" else dim_name

        # Collect scores by run number for P1 and P2
        p1_by_run: list[list[float]] = [[] for _ in range(num_runs)]
        p2_by_run: list[list[float]] = [[] for _ in range(num_runs)]

        for result in all_results:
            runs = result.get("runs", [])
            for run_idx, run in enumerate(runs[:num_runs]):
                if dim_name == "_overall":
                    p1_by_run[run_idx].append(run.get("p1_overall", 0.0))
                    p2_by_run[run_idx].append(run.get("p2_overall", 0.0))
                else:
                    p1_by_run[run_idx].append(run.get("p1_dimensions", {}).get(dim_name, 0.0))
                    p2_by_run[run_idx].append(run.get("p2_dimensions", {}).get(dim_name, 0.0))

        # Calculate means per run
        p1_run_means = [np.mean(scores) if scores else 0 for scores in p1_by_run]
        p2_run_means = [np.mean(scores) if scores else 0 for scores in p2_by_run]

        # Perform one-way ANOVA for P1
        p1_groups = [scores for scores in p1_by_run if len(scores) > 0]
        if len(p1_groups) >= 2 and all(len(g) >= 2 for g in p1_groups):
            p1_f_stat, p1_p_value = stats.f_oneway(*p1_groups)
        else:
            p1_f_stat, p1_p_value = float('nan'), float('nan')

        # Perform one-way ANOVA for P2
        p2_groups = [scores for scores in p2_by_run if len(scores) > 0]
        if len(p2_groups) >= 2 and all(len(g) >= 2 for g in p2_groups):
            p2_f_stat, p2_p_value = stats.f_oneway(*p2_groups)
        else:
            p2_f_stat, p2_p_value = float('nan'), float('nan')

        # Build P1 row
        p1_row = [f"[bold]{display_name}[/]" if dim_name == "_overall" else display_name]
        p1_row.extend([f"{m:.4f}" for m in p1_run_means])
        p1_row.append(f"{p1_f_stat:.4f}" if not np.isnan(p1_f_stat) else "N/A")
        p1_row.append(f"{p1_p_value:.4f}" if not np.isnan(p1_p_value) else "N/A")
        p1_significant = "Yes" if (not np.isnan(p1_p_value) and p1_p_value < 0.05) else "No"
        p1_row.append(f"[red]{p1_significant}[/]" if p1_significant == "Yes" else f"[green]{p1_significant}[/]")
        p1_anova_table.add_row(*p1_row)

        # Build P2 row
        p2_row = [f"[bold]{display_name}[/]" if dim_name == "_overall" else display_name]
        p2_row.extend([f"{m:.4f}" for m in p2_run_means])
        p2_row.append(f"{p2_f_stat:.4f}" if not np.isnan(p2_f_stat) else "N/A")
        p2_row.append(f"{p2_p_value:.4f}" if not np.isnan(p2_p_value) else "N/A")
        p2_significant = "Yes" if (not np.isnan(p2_p_value) and p2_p_value < 0.05) else "No"
        p2_row.append(f"[red]{p2_significant}[/]" if p2_significant == "Yes" else f"[green]{p2_significant}[/]")
        p2_anova_table.add_row(*p2_row)

        # Store analysis results
        analysis_results.append({
            "dimension": display_name,
            "p1_run_means": [float(m) for m in p1_run_means],
            "p1_anova_f_stat": float(p1_f_stat) if not np.isnan(p1_f_stat) else None,
            "p1_anova_p_value": float(p1_p_value) if not np.isnan(p1_p_value) else None,
            "p1_significant": p1_significant == "Yes",
            "p2_run_means": [float(m) for m in p2_run_means],
            "p2_anova_f_stat": float(p2_f_stat) if not np.isnan(p2_f_stat) else None,
            "p2_anova_p_value": float(p2_p_value) if not np.isnan(p2_p_value) else None,
            "p2_significant": p2_significant == "Yes",
        })

    console.print(p1_anova_table)
    console.print()
    console.print(p2_anova_table)

    # ===== Within-Episode Variance Analysis =====
    console.print(f"\n[bold cyan]Within-Episode Consistency Analysis[/]\n")

    var_table = Table(title="Average Within-Episode Variance Across Runs")
    var_table.add_column("Dimension", style="cyan")
    var_table.add_column("P1 Avg Var", justify="right", style="blue")
    var_table.add_column("P1 Avg Std", justify="right", style="dim blue")
    var_table.add_column("P2 Avg Var", justify="right", style="magenta")
    var_table.add_column("P2 Avg Std", justify="right", style="dim magenta")
    var_table.add_column("P1 CV (%)", justify="right", style="dim cyan")
    var_table.add_column("P2 CV (%)", justify="right", style="dim cyan")

    for idx, dim_name in enumerate(dimensions_to_analyze):
        # Collect within-episode variances
        p1_within_episode_vars = []
        p2_within_episode_vars = []
        p1_within_episode_means = []
        p2_within_episode_means = []

        for result in all_results:
            runs = result.get("runs", [])
            if len(runs) < 2:
                continue  # Need at least 2 runs to compute variance

            # Extract scores for this episode's runs
            if dim_name == "_overall":
                p1_scores = [r.get("p1_overall", 0.0) for r in runs]
                p2_scores = [r.get("p2_overall", 0.0) for r in runs]
            else:
                p1_scores = [r.get("p1_dimensions", {}).get(dim_name, 0.0) for r in runs]
                p2_scores = [r.get("p2_dimensions", {}).get(dim_name, 0.0) for r in runs]

            # Compute within-episode variance
            if len(p1_scores) > 1:
                p1_within_episode_vars.append(np.var(p1_scores, ddof=1))
                p1_within_episode_means.append(np.mean(p1_scores))
            if len(p2_scores) > 1:
                p2_within_episode_vars.append(np.var(p2_scores, ddof=1))
                p2_within_episode_means.append(np.mean(p2_scores))

        # Calculate average within-episode variance
        p1_avg_var = np.mean(p1_within_episode_vars) if p1_within_episode_vars else 0
        p2_avg_var = np.mean(p2_within_episode_vars) if p2_within_episode_vars else 0

        # Calculate average standard deviation
        p1_avg_std = np.sqrt(p1_avg_var) if p1_avg_var > 0 else 0
        p2_avg_std = np.sqrt(p2_avg_var) if p2_avg_var > 0 else 0

        # Calculate coefficient of variation (CV = std / mean * 100)
        # Lower CV means more consistent evaluations
        p1_overall_mean = np.mean(p1_within_episode_means) if p1_within_episode_means else 0
        p2_overall_mean = np.mean(p2_within_episode_means) if p2_within_episode_means else 0

        p1_cv = (p1_avg_std / abs(p1_overall_mean) * 100) if p1_overall_mean != 0 else 0
        p2_cv = (p2_avg_std / abs(p2_overall_mean) * 100) if p2_overall_mean != 0 else 0

        display_name = "Overall" if dim_name == "_overall" else dim_name

        var_table.add_row(
            f"[bold]{display_name}[/]" if dim_name == "_overall" else display_name,
            f"{p1_avg_var:.4f}",
            f"{p1_avg_std:.4f}",
            f"{p2_avg_var:.4f}",
            f"{p2_avg_std:.4f}",
            f"{p1_cv:.1f}",
            f"{p2_cv:.1f}",
        )

        # Update analysis results with within-episode variance info
        analysis_results[idx].update({
            "p1_avg_within_variance": float(p1_avg_var),
            "p1_avg_within_std": float(p1_avg_std),
            "p2_avg_within_variance": float(p2_avg_var),
            "p2_avg_within_std": float(p2_avg_std),
            "p1_coefficient_of_variation": float(p1_cv),
            "p2_coefficient_of_variation": float(p2_cv),
        })

    console.print(var_table)
    
    # Generate box plots with statistics table
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid: 2 rows for box plots, 1 row for table
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.6], hspace=0.35)
    
    plot_dims = ["_overall"] + DIMENSIONS
    
    # Statistics for table
    table_data = []
    col_labels = ["Dimension", "P1 Mean", "P1 Std", "P2 Mean", "P2 Std", "P1 Min", "P1 Max", "P2 Min", "P2 Max"]
    
    for idx, dim in enumerate(plot_dims):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        data = dimension_data.get(dim, {"p1": [], "p2": []})
        
        p1_data = np.array(data["p1"])
        p2_data = np.array(data["p2"])
        
        bp = ax.boxplot(
            [p1_data, p2_data],
            labels=["P1", "P2"],
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=8),
        )
        
        # Color the boxes
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][1].set_facecolor("#e74c3c")
        
        title = "Overall Score" if dim == "_overall" else dim.replace("_", " ").title()
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        
        # Add mean labels
        p1_mean = np.mean(p1_data) if len(p1_data) > 0 else 0
        p2_mean = np.mean(p2_data) if len(p2_data) > 0 else 0
        ax.annotate(f"μ={p1_mean:.2f}", xy=(1, p1_mean), xytext=(0.7, p1_mean),
                    fontsize=8, ha="right", va="center", color="#2980b9", fontweight="bold")
        ax.annotate(f"μ={p2_mean:.2f}", xy=(2, p2_mean), xytext=(2.3, p2_mean),
                    fontsize=8, ha="left", va="center", color="#c0392b", fontweight="bold")
        
        # Add median labels
        p1_median = np.median(p1_data) if len(p1_data) > 0 else 0
        p2_median = np.median(p2_data) if len(p2_data) > 0 else 0
        if abs(p1_median - p1_mean) > 0.3:
            ax.annotate(f"M={p1_median:.2f}", xy=(1, p1_median), xytext=(0.7, p1_median - 0.3),
                        fontsize=7, ha="right", va="top", color="#2980b9")
        if abs(p2_median - p2_mean) > 0.3:
            ax.annotate(f"M={p2_median:.2f}", xy=(2, p2_median), xytext=(2.3, p2_median - 0.3),
                        fontsize=7, ha="left", va="top", color="#c0392b")
        
        # Collect data for table
        table_data.append([
            title[:15],  # Truncate long names
            f"{p1_mean:.2f}",
            f"{np.std(p1_data):.2f}" if len(p1_data) > 1 else "0.00",
            f"{p2_mean:.2f}",
            f"{np.std(p2_data):.2f}" if len(p2_data) > 1 else "0.00",
            f"{np.min(p1_data):.1f}" if len(p1_data) > 0 else "-",
            f"{np.max(p1_data):.1f}" if len(p1_data) > 0 else "-",
            f"{np.min(p2_data):.1f}" if len(p2_data) > 0 else "-",
            f"{np.max(p2_data):.1f}" if len(p2_data) > 0 else "-",
        ])
    
    # Add statistics table at the bottom
    table_ax = fig.add_subplot(gs[2, :])
    table_ax.axis("off")
    
    table = table_ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colColours=["#ecf0f1"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header row
    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]
        if key[0] == 0:  # Header row
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#3498db")
            cell.set_text_props(color="white", weight="bold")
    
    plt.suptitle(f"Dimension Score Distribution\n{tag}", fontsize=14, fontweight="bold", y=0.98)
    
    # Save plot
    plot_file = output_dir / "boxplots.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print(f"\n[bold green]Box plots saved to:[/] {plot_file}")
    
    # Save analysis results
    analysis_file = output_dir / "variance_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis_results, f, indent=2)
    console.print(f"[bold green]Variance analysis saved to:[/] {analysis_file}")


def save_results_to_file(
    all_results: list[dict[str, Any]],
    tag: str,
    num_runs: int,
    model: str,
    temperature: float,
) -> Path:
    """
    Save evaluation results to a JSON file in the tag's output directory.

    Returns:
        Path to the saved file.
    """
    output_dir = OUTPUTS_DIR / tag / "evaluation"
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate aggregated statistics
    p1_overalls = [r["p1_overall_mean"] for r in all_results]
    p2_overalls = [r["p2_overall_mean"] for r in all_results]

    # Build dimension statistics
    dimension_stats: dict[str, Any] = {}
    for dim in DIMENSIONS:
        p1_dim_values = [r["p1_dimensions_mean"].get(dim, 0) for r in all_results]
        p2_dim_values = [r["p2_dimensions_mean"].get(dim, 0) for r in all_results]
        dimension_stats[dim] = {
            "p1_mean": float(np.mean(p1_dim_values)),
            "p1_std": float(np.std(p1_dim_values)),
            "p2_mean": float(np.mean(p2_dim_values)),
            "p2_std": float(np.std(p2_dim_values)),
        }

    # 新增：统计评估来源
    total_existing = sum(r.get("num_existing_runs", 0) for r in all_results)
    total_new = sum(r.get("num_new_runs", 0) for r in all_results)

    # Build summary
    summary = {
        "tag": tag,
        "model": model,
        "num_runs": num_runs,
        "temperature": temperature,
        "num_episodes": len(all_results),
        "timestamp": datetime.now().isoformat(),
        "evaluation_summary": {  # 新增字段
            "total_existing_evals": total_existing,
            "total_new_evals": total_new,
            "avg_existing_per_episode": total_existing / len(all_results) if all_results else 0,
            "avg_new_per_episode": total_new / len(all_results) if all_results else 0,
        },
        "overall": {
            "p1_mean": float(np.mean(p1_overalls)),
            "p1_std": float(np.std(p1_overalls)),
            "p2_mean": float(np.mean(p2_overalls)),
            "p2_std": float(np.std(p2_overalls)),
            "combined_mean": float((np.mean(p1_overalls) + np.mean(p2_overalls)) / 2),
        },
        "dimensions": dimension_stats,
        "episodes": all_results,
    }

    # Save to file
    output_file = output_dir / "results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_file}")
    console.print(f"\n[bold green]Results saved to:[/] {output_file}")

    return output_file

@app.command()
def main(
    tag: str = typer.Option(None, "--tag", "-t", help="Single tag to evaluate"),
    csv: str = typer.Option(None, "--csv", help="CSV file with tag list (one per line)"),
    model: str = typer.Option(
        "openrouter/openai/gpt-4o", "--model", "-m",
        help="Model to use for evaluation"
    ),
    num_runs: int = typer.Option(3, "--num-runs", "-n", help="Target number of evaluations per episode"),
    temperature: float = typer.Option(0.0, "--temperature", "-T", help="Sampling temperature"),
    max_episodes: int = typer.Option(None, "--max-episodes", help="Max episodes to evaluate"),
    concurrency: int = typer.Option(40, "--concurrency", "-c", help="Concurrent evaluations"),
    turn_interval: int = typer.Option(None, "--turn-interval", "-i", help="Evaluate every N turns"),
    stats_only: bool = typer.Option(False, "--stats-only", "-s", help="Skip evaluation, only compute stats from existing checkpoint"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-evaluation: delete non-database results and regenerate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """
    Evaluate episodes with incremental support and CSV input.

    Examples:
        # Single tag
        python evaluate_episodes_multi_run.py --tag experiment1 --num-runs 5

        # Multiple tags from CSV
        python evaluate_episodes_multi_run.py --csv tags.csv --num-runs 5

        # Resume from checkpoint
        python evaluate_episodes_multi_run.py --tag experiment1 --num-runs 10
        (automatically detects existing checkpoint and only runs missing evaluations)

        # Stats only (skip evaluation, compute from checkpoint)
        python evaluate_episodes_multi_run.py --tag experiment1 --stats-only

        # Force re-evaluation (delete non-database results)
        python evaluate_episodes_multi_run.py --tag experiment1 --num-runs 5 --force
    """
    # Run the async main function
    asyncio.run(main_async(
        tag=tag,
        csv=csv,
        model=model,
        num_runs=num_runs,
        temperature=temperature,
        max_episodes=max_episodes,
        concurrency=concurrency,
        turn_interval=turn_interval,
        stats_only=stats_only,
        force=force,
        verbose=verbose,
    ))

async def main_async(
    tag: str | None,
    csv: str | None,
    model: str,
    num_runs: int,
    temperature: float,
    max_episodes: int | None,
    concurrency: int,
    turn_interval: int | None,
    stats_only: bool,
    force: bool,
    verbose: bool,
) -> None:
    """Async implementation of main function."""
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )

    # 1. 读取 tags
    if csv:
        tags = read_tags_from_csv(Path(csv))
        console.print(f"[green]Loaded {len(tags)} tags from CSV: {csv}[/]")
    elif tag:
        tags = [tag]
    else:
        console.print("[red]Error: Must specify --tag or --csv[/]")
        raise typer.Exit(code=1)

    # 用于保存所有 tags 的汇总信息
    all_tag_summaries: dict[str, dict[str, Any]] = {}

    # 2. 遍历每个 tag 进行增量评估
    for current_tag in tags:
        console.print(f"\n[bold cyan]═══ Processing Tag: {current_tag} ═══[/]\n")
        turn_str = f"Every {turn_interval} turns" if turn_interval else "Full episode"
        console.print(Panel(f"""[bold cyan]Terminal Evaluator Multi-Run[/]
Tag: [yellow]{current_tag}[/]
Model: [green]{model}[/]
Runs per episode: [magenta]{num_runs}[/]
Concurrency: [cyan]{concurrency}[/]
Turn Interval: [yellow]{turn_str}[/]
Temperature: [blue]{temperature}[/]"""))

        # Load episodes from database
        logger.info(f"Loading episodes with tag: {current_tag}")
        episodes = list(EpisodeLog.find(EpisodeLog.tag == current_tag).all())

        if not episodes:
            logger.warning(f"No episodes found with tag: {current_tag}")
            continue

        if max_episodes:
            episodes = episodes[:max_episodes]

        console.print(f"[green]Found {len(episodes)} episodes[/]")

        # Setup output directory and LLM call logging
        output_dir = OUTPUTS_DIR / current_tag / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_log_file = output_dir / "llm_calls.jsonl"
        checkpoint_file = output_dir / "checkpoint.jsonl"

        # 加载现有 checkpoint（按 episode_pk 合并 runs）
        checkpoint_results_map: dict[str, dict[str, Any]] = {}  # episode_pk -> merged result
        raw_line_count = 0
        total_runs_count = 0

        if checkpoint_file.exists():
            console.print(f"[yellow]Found existing checkpoint, loading...[/]")
            with open(checkpoint_file, "r") as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        episode_pk = result.get("episode_pk")
                        if episode_pk:
                            raw_line_count += 1

                            # 标注来源
                            runs = result.get("runs", [])
                            for run in runs:
                                if "source" not in run:
                                    run["source"] = "checkpoint"
                            total_runs_count += len(runs)

                            # 合并同一个 episode 的 runs
                            if episode_pk in checkpoint_results_map:
                                existing = checkpoint_results_map[episode_pk]
                                existing_runs = existing.get("runs", [])
                                # 合并 runs，避免重复（基于 run 内容的简单去重）
                                existing_keys = set()
                                for r in existing_runs:
                                    key = (r.get("source"), r.get("turns_evaluated"), 
                                           r.get("p1_overall"), r.get("p2_overall"))
                                    existing_keys.add(key)
                                
                                for r in runs:
                                    key = (r.get("source"), r.get("turns_evaluated"),
                                           r.get("p1_overall"), r.get("p2_overall"))
                                    if key not in existing_keys:
                                        existing_runs.append(r)
                                        existing_keys.add(key)
                                
                                existing["runs"] = existing_runs
                                existing["num_successful_runs"] = len(existing_runs)
                            else:
                                checkpoint_results_map[episode_pk] = result
                    except json.JSONDecodeError:
                        pass

            # 转换为列表
            checkpoint_results = list(checkpoint_results_map.values())
            evaluated_pks = set(checkpoint_results_map.keys())
            
            # 统计合并后的总运行次数
            merged_runs_count = sum(len(r.get("runs", [])) for r in checkpoint_results)

            if raw_line_count != len(checkpoint_results):
                console.print(f"[yellow]Deduplicated: {raw_line_count} lines -> {len(checkpoint_results)} unique episodes[/]")
            
            # 截断超过 num_runs 的 runs
            truncated_count = 0
            for result in checkpoint_results:
                runs = result.get("runs", [])
                if len(runs) > num_runs:
                    result["runs"] = runs[:num_runs]
                    truncated_count += 1
            
            # 重新统计
            merged_runs_count = sum(len(r.get("runs", [])) for r in checkpoint_results)
            runs_per_episode = merged_runs_count / len(checkpoint_results) if checkpoint_results else 0
            
            console.print(f"[green]Checkpoint: {len(checkpoint_results)} episodes × {runs_per_episode:.1f} runs/ep = {merged_runs_count} total runs[/]")
            if truncated_count > 0:
                console.print(f"[yellow]Truncated {truncated_count} episodes to {num_runs} runs (as per --num-runs setting)[/]")
        else:
            checkpoint_results = []
            evaluated_pks = set()

        # Force mode: remove non-database evaluation results and rewrite checkpoint
        if force and checkpoint_results:
            console.print(f"[yellow]Force mode: removing non-database evaluation results...[/]")
            original_count = sum(len(r.get("runs", [])) for r in checkpoint_results)

            # Filter runs to keep only database-sourced ones
            filtered_results = []
            for result in checkpoint_results:
                runs = result.get("runs", [])
                # Keep only runs with source="database"
                db_runs = [r for r in runs if r.get("source") == "database"]

                if db_runs:
                    # Update result with filtered runs and recalculate aggregates
                    result["runs"] = db_runs
                    result["num_existing_runs"] = len(db_runs)
                    result["num_new_runs"] = 0

                    # Recalculate mean values
                    p1_overalls = [r.get("p1_overall", 0) for r in db_runs]
                    p2_overalls = [r.get("p2_overall", 0) for r in db_runs]
                    result["p1_overall_mean"] = float(np.mean(p1_overalls))
                    result["p2_overall_mean"] = float(np.mean(p2_overalls))
                    result["p1_overall_std"] = float(np.std(p1_overalls))
                    result["p2_overall_std"] = float(np.std(p2_overalls))

                    # Recalculate dimension means
                    p1_dims: dict[str, list[float]] = {dim: [] for dim in DIMENSIONS}
                    p2_dims: dict[str, list[float]] = {dim: [] for dim in DIMENSIONS}
                    for run in db_runs:
                        for dim in DIMENSIONS:
                            p1_dims[dim].append(run.get("p1_dimensions", {}).get(dim, 0))
                            p2_dims[dim].append(run.get("p2_dimensions", {}).get(dim, 0))

                    result["p1_dimensions_mean"] = {dim: float(np.mean(vals)) for dim, vals in p1_dims.items()}
                    result["p2_dimensions_mean"] = {dim: float(np.mean(vals)) for dim, vals in p2_dims.items()}

                    filtered_results.append(result)
                # If no database runs, the episode will be re-evaluated from scratch

            filtered_count = sum(len(r.get("runs", [])) for r in filtered_results)
            removed_count = original_count - filtered_count

            console.print(f"[yellow]Removed {removed_count} non-database runs, kept {filtered_count} database runs[/]")
            console.print(f"[yellow]Episodes with database results: {len(filtered_results)}/{len(checkpoint_results)}[/]")

            # Rewrite checkpoint file with filtered results
            if filtered_results:
                with open(checkpoint_file, "w") as f:
                    for result in filtered_results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                console.print(f"[green]Checkpoint file rewritten with {len(filtered_results)} episodes[/]")
            else:
                # No database results, remove checkpoint file
                checkpoint_file.unlink()
                console.print(f"[yellow]No database results found, checkpoint file removed[/]")

            # Update checkpoint_results for subsequent processing
            checkpoint_results = filtered_results

        # Stats-only mode: skip evaluation, use checkpoint directly
        if stats_only:
            if not checkpoint_results:
                console.print(f"[red]Error: No checkpoint found for tag {current_tag}. Cannot compute stats.[/]")
                continue

            console.print(f"[yellow]Stats-only mode: skipping evaluation, using {len(checkpoint_results)} results from checkpoint[/]")
            all_results = checkpoint_results

            # Determine num_runs from checkpoint data
            if all_results:
                num_runs = len(all_results[0].get("runs", []))
                console.print(f"[dim]Detected {num_runs} runs per episode from checkpoint[/]")

            # No cost info in stats-only mode
            cost_info = None
        else:
            # Normal evaluation mode
            # 过滤出需要评估的 episodes（不再完全过滤，在 evaluate_with_incremental_support 中处理）
            episodes_to_evaluate = episodes  # 所有episodes都需要检查

            console.print(f"[cyan]Episodes to process: {len(episodes_to_evaluate)}[/]")

            # Enable LLM call logging
            enable_llm_call_logging(llm_log_file, experiment_tag=current_tag)

            # 3. 增量评估所有 episodes（使用 evaluate_with_incremental_support）
            async def evaluate_single_episode(idx: int, episode: EpisodeLog) -> dict[str, Any] | None:
                """评估单个 episode（带增量支持）"""
                try:
                    result = await evaluate_with_incremental_support(
                        episode=episode,
                        checkpoint_results=checkpoint_results,  # 传入 checkpoint 结果
                        model_name=model,  # 传入模型名称
                        num_runs=num_runs,  # 传入目标评估次数
                        temperature=temperature,  # 传入温度参数
                    )
                    return result
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Failed to evaluate episode {episode.pk}: {e}")
                    return None

            async def run_all_evaluations() -> list[dict[str, Any]]:
                """使用并发控制运行所有评估"""
                semaphore = asyncio.Semaphore(concurrency)

                async def evaluate_with_semaphore(idx: int, episode: EpisodeLog) -> dict[str, Any] | None:
                    async with semaphore:
                        return await evaluate_single_episode(idx, episode)

                # 创建所有任务
                tasks = [
                    evaluate_with_semaphore(idx, episode)
                    for idx, episode in enumerate(episodes_to_evaluate)
                ]

                # 使用 tqdm 进度条运行
                all_results: list[dict[str, Any]] = []
                for coro in tqdm_asyncio.as_completed(tasks, desc="Evaluating episodes", total=len(tasks)):
                    result = await coro
                    if result:
                        all_results.append(result)

                        # 增量保存到 checkpoint
                        with open(checkpoint_file, "a") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

                        # 中间表格不打印（用户要求）- 仅在最后显示汇总
                        # display_episode_results(result, len(all_results) - 1)

                return all_results

            all_results = await run_all_evaluations()

            # Disable LLM call logging
            disable_llm_call_logging()

            # 4. 计算 API 成本
            console.print("\n[bold cyan]Calculating API costs...[/]")
            cost_info = await calculate_cost_from_log(llm_log_file)
            if cost_info:
                console.print(Panel(f"""[bold green]Cost Summary[/]
Total Cost: [yellow]${cost_info['total_cost']:.6f}[/]
Prompt Tokens: {cost_info['total_prompt_tokens']:,}
Completion Tokens: {cost_info['total_completion_tokens']:,}
Total Tokens: {cost_info['total_tokens']:,}
API Calls: {cost_info['processed_count']}"""))

        # 5. 保存结果并生成分析
        if all_results:
            # 首先检查评估质量（确保每个场景的评估都满足要求）
            check_evaluation_quality(all_results, num_runs, current_tag)

            # 显示汇总表格
            display_summary_table(all_results, num_runs)

            # 保存结果文件
            save_results_to_file(
                all_results=all_results,
                tag=current_tag,
                num_runs=num_runs,
                model=model,
                temperature=temperature,
            )

            # 加载保存的 summary 用于多 tag 对比
            results_file = output_dir / "results.json"
            with open(results_file, "r") as f:
                summary = json.load(f)
            all_tag_summaries[current_tag] = summary

            # 生成分析图表
            generate_dimension_analysis(
                all_results=all_results,
                output_dir=output_dir,
                tag=current_tag,
            )

            # 生成趋势分析（如果启用 turn_interval）
            if turn_interval:
                generate_trend_analysis(
                    all_results=all_results,
                    output_dir=output_dir,
                    tag=current_tag,
                )

            # 保存成本信息 (only when cost_info available)
            if cost_info:
                cost_file = output_dir / "cost_info.json"
                with open(cost_file, "w") as f:
                    json.dump(cost_info, f, indent=2)
                console.print(f"[bold green]Cost info saved to:[/] {cost_file}")
        else:
            logger.error(f"No successful evaluations for tag: {current_tag}")

    # 6. 如果有多个 tags，显示对比表格
    if len(all_tag_summaries) > 1:
        display_multi_tag_comparison(all_tag_summaries)

        # 保存对比结果到汇总文件
        comparison_dir = OUTPUTS_DIR / "multi_tag_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        comparison_file = comparison_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, "w") as f:
            json.dump(all_tag_summaries, f, indent=2)

        console.print(f"\n[green]Multi-tag comparison saved to: {comparison_file}[/]")

    console.print("\n[bold green]✓ All evaluations completed[/]")

    # Cleanup: close LiteLLM async clients to prevent "Event loop is closed" errors
    console.print("\n[dim]Cleaning up LiteLLM async clients...[/]")
    try:
        import litellm
        await litellm.close_litellm_async_clients()
        console.print("[dim]✓ LiteLLM async clients closed successfully[/]")
    except Exception as e:
        console.print(f"[yellow]Warning: Error closing LiteLLM clients: {e}[/]")


if __name__ == "__main__":
    app()

