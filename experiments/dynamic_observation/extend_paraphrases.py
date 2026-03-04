#!/usr/bin/env python3
"""
扩展改写脚本：在已有的 embeddings_backgrounds 基础上，继续改写到指定数量。

功能：
1. 读取现有的 texts.json，获取 original 和 paraphrased 背景
2. 如果 paraphrase 数量不足，继续生成直到达到指定数量
3. 验证生成的文本（检查非法字符）
4. 重新计算 embeddings
5. 输出到新文件夹，保持目录结构

用法：
    python extend_paraphrases.py --input-dir /path/to/embeddings_backgrounds \
        --output-dir /path/to/output --target-count 20
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
load_dotenv(project_root / ".env")

import litellm

# 导入 cost 计算函数
from experiments.dynamic_observation.calculate_cost import calculate_cost_async

console = Console()

# ========== 配置区域 ==========
MODEL_NAME = "openrouter/openai/gpt-4o"
TEMPERATURE = 1.2
EMBEDDING_MODEL = "qwen/qwen3-embedding-4b"
MAX_CONCURRENCY = 50
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
# ==============================

# 用于存储所有 LLM 调用的 generation ID
generation_ids: list[str] = []
generation_ids_lock = asyncio.Lock()


PARAPHRASE_PROMPT = """You are a skilled rewriter. Your task is to paraphrase the following agent background description while preserving ALL the original information (name, age, gender, occupation, personality, secrets, etc.).

Requirements:
1. Keep ALL factual information intact (names, ages, occupations, traits, secrets)
2. Change the sentence structure and word choices
3. Use different expressions and phrasings
4. Maintain the same meaning
5. Output ONLY the paraphrased text, nothing else

Original text:
{original_text}

Paraphrased version:"""


def contains_non_english(text: str, threshold: float = 0.05) -> tuple[bool, float, str]:
    """
    检查文本是否包含大量非英语字符。
    
    Returns:
        (is_problematic, non_ascii_ratio, sample_bad_chars)
    """
    if not text:
        return False, 0.0, ""
    
    non_ascii_chars = []
    for char in text:
        if ord(char) > 127 and char not in "''""—–…":
            non_ascii_chars.append(char)
    
    ratio = len(non_ascii_chars) / len(text)
    sample = "".join(set(non_ascii_chars[:50]))
    
    return ratio > threshold, ratio, sample


async def paraphrase_single(
    original_text: str, 
    semaphore: asyncio.Semaphore,
    max_retries_for_quality: int = 3,
) -> str:
    """使用 LLM 对单个文本进行同义改写，带重试和质量验证"""
    for quality_attempt in range(max_retries_for_quality):
        async with semaphore:
            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = await litellm.acompletion(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(original_text=original_text)}],
                        temperature=TEMPERATURE,
                    )
                    result = response.choices[0].message.content
                    if result is None:
                        raise ValueError(f"LLM returned None for paraphrase")
                    result = result.strip()
                    
                    # 验证文本质量
                    is_bad, ratio, sample = contains_non_english(result)
                    
                    # 记录 generation ID 用于后续 cost 计算
                    gen_id = getattr(response, "id", None)
                    if gen_id:
                        async with generation_ids_lock:
                            generation_ids.append(gen_id)
                    
                    if not is_bad:
                        return result
                    else:
                        logger.warning(f"生成的文本包含非法字符 ({ratio:.1%})，重试中... (质量尝试 {quality_attempt + 1}/{max_retries_for_quality})")
                        break  # 跳出 API 重试循环，重新生成
                        
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    should_retry = any(str(code) in error_str for code in RETRY_STATUS_CODES)
                    if not should_retry:
                        raise

                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(f"API 请求失败 (attempt {attempt + 1}/{MAX_RETRIES}): {error_str[:100]}... 等待 {delay:.1f}s 重试")
                    await asyncio.sleep(delay)
            else:
                raise RuntimeError(f"API 请求失败 after {MAX_RETRIES} retries. Last error: {last_error}")
    
    # 如果多次质量检查都失败，返回最后生成的结果（带警告）
    logger.warning(f"多次质量检查失败，使用最后生成的结果")
    return result


async def get_embedding(
    text: str, 
    embed_client: AsyncOpenAI, 
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> np.ndarray:
    """获取单个文本的 embedding"""
    async with semaphore:
        text = text.replace("\n", " ")
        for attempt in range(max_retries):
            try:
                response = await embed_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
                if response.data is None:
                    raise ValueError("API returned None data")
                return np.array(response.data[0].embedding)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Embedding 请求失败 (attempt {attempt + 1}/{max_retries}): {e}, 等待 {wait_time}s 重试")
                    await asyncio.sleep(wait_time)
                else:
                    raise


async def get_embeddings_batch(
    texts: list[str], 
    embed_client: AsyncOpenAI, 
    semaphore: asyncio.Semaphore,
) -> np.ndarray:
    """批量获取 embeddings"""
    tasks = [get_embedding(text, embed_client, semaphore) for text in texts]
    embeddings = await asyncio.gather(*tasks)
    return np.array(embeddings)


def compute_statistics(sim_matrix: np.ndarray, to_origin: list[float]) -> dict:
    """计算相似度统计信息"""
    n = sim_matrix.shape[0]
    upper_triangle_indices = np.triu_indices(n, k=1)
    upper_values = sim_matrix[upper_triangle_indices]
    return {
        "mean": float(np.mean(upper_values)) if upper_values.size > 0 else 0.0,
        "min": float(np.min(upper_values)) if upper_values.size > 0 else 0.0,
        "max": float(np.max(upper_values)) if upper_values.size > 0 else 0.0,
        "std": float(np.std(upper_values)) if upper_values.size > 0 else 0.0,
        "to_origin_mean": float(np.mean(to_origin)) if to_origin else 0.0,
        "to_origin_min": float(np.min(to_origin)) if to_origin else 0.0,
        "to_origin_max": float(np.max(to_origin)) if to_origin else 0.0,
    }


async def process_scenario(
    scenario_dir: Path,
    output_dir: Path,
    target_count: int,
    paraphrase_semaphore: asyncio.Semaphore,
    embed_client: AsyncOpenAI,
    embed_semaphore: asyncio.Semaphore,
) -> dict:
    """处理单个 scenario：扩展 paraphrases 并重新计算 embeddings"""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    texts_file = scenario_dir / "texts.json"
    if not texts_file.exists():
        return {"error": f"texts.json not found in {scenario_dir}"}
    
    with open(texts_file, "r", encoding="utf-8") as f:
        texts = json.load(f)
    
    original_p1 = texts.get("original", {}).get("p1_background", "")
    original_p2 = texts.get("original", {}).get("p2_background", "")
    p1_paraphrases = texts.get("paraphrased", {}).get("p1_background", [])
    p2_paraphrases = texts.get("paraphrased", {}).get("p2_background", [])
    
    result = {
        "scenario_id": scenario_dir.name,
        "p1_original_count": len(p1_paraphrases),
        "p2_original_count": len(p2_paraphrases),
        "p1_new_count": 0,
        "p2_new_count": 0,
    }
    
    # 计算需要新增的数量
    p1_needed = max(0, target_count - len(p1_paraphrases))
    p2_needed = max(0, target_count - len(p2_paraphrases))
    
    # 创建输出目录
    out_scenario_dir = output_dir / scenario_dir.name
    out_scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果都不需要新增，直接复制现有文件
    if p1_needed == 0 and p2_needed == 0:
        result["skipped"] = True
        result["skip_reason"] = "already_at_target"
        # 复制现有文件
        for file in scenario_dir.iterdir():
            if file.is_file():
                shutil.copy2(file, out_scenario_dir / file.name)
        logger.info(f"{scenario_dir.name}: 已达目标数量，直接复制")
        return result
    
    # 生成新的 paraphrases
    if p1_needed > 0 and original_p1:
        logger.info(f"{scenario_dir.name}: 为 P1 生成 {p1_needed} 个新的 paraphrase")
        tasks = [paraphrase_single(original_p1, paraphrase_semaphore) for _ in range(p1_needed)]
        new_paraphrases = await asyncio.gather(*tasks)
        p1_paraphrases.extend(new_paraphrases)
        result["p1_new_count"] = p1_needed
    
    if p2_needed > 0 and original_p2:
        logger.info(f"{scenario_dir.name}: 为 P2 生成 {p2_needed} 个新的 paraphrase")
        tasks = [paraphrase_single(original_p2, paraphrase_semaphore) for _ in range(p2_needed)]
        new_paraphrases = await asyncio.gather(*tasks)
        p2_paraphrases.extend(new_paraphrases)
        result["p2_new_count"] = p2_needed
    
    # 保存更新后的 texts.json
    updated_texts = {
        "original": {
            "p1_background": original_p1,
            "p2_background": original_p2,
        },
        "paraphrased": {
            "p1_background": p1_paraphrases,
            "p2_background": p2_paraphrases,
        },
    }
    with open(out_scenario_dir / "texts.json", "w", encoding="utf-8") as f:
        json.dump(updated_texts, f, indent=2, ensure_ascii=False)
    
    # 计算 embeddings (只有新增了才需要重新计算)
    p1_texts = [original_p1] + p1_paraphrases
    p2_texts = [original_p2] + p2_paraphrases
    
    p1_embeddings = await get_embeddings_batch(p1_texts, embed_client, embed_semaphore)
    p2_embeddings = await get_embeddings_batch(p2_texts, embed_client, embed_semaphore)
    
    np.save(str(out_scenario_dir / "p1_embeddings.npy"), p1_embeddings)
    np.save(str(out_scenario_dir / "p2_embeddings.npy"), p2_embeddings)
    
    # 计算 similarity matrix
    p1_sim_matrix = cosine_similarity(p1_embeddings)
    p2_sim_matrix = cosine_similarity(p2_embeddings)
    
    # 保存 similarity matrix (CSV)
    p1_labels = ["origin"] + [f"v{i}" for i in range(len(p1_paraphrases))]
    p2_labels = ["origin"] + [f"v{i}" for i in range(len(p2_paraphrases))]
    
    df_p1 = pd.DataFrame(p1_sim_matrix, index=p1_labels, columns=p1_labels)
    df_p2 = pd.DataFrame(p2_sim_matrix, index=p2_labels, columns=p2_labels)
    df_p1.to_csv(out_scenario_dir / "p1_cosine_similarity.csv")
    df_p2.to_csv(out_scenario_dir / "p2_cosine_similarity.csv")
    
    # 计算统计信息
    p1_to_origin = [float(p1_sim_matrix[0, i]) for i in range(1, len(p1_texts))]
    p2_to_origin = [float(p2_sim_matrix[0, i]) for i in range(1, len(p2_texts))]
    p1_stats = compute_statistics(p1_sim_matrix, p1_to_origin)
    p2_stats = compute_statistics(p2_sim_matrix, p2_to_origin)
    
    # Cross similarity
    cross_similarities = []
    for i in range(min(len(p1_texts), len(p2_texts))):
        sim = cosine_similarity([p1_embeddings[i]], [p2_embeddings[i]])[0, 0]
        cross_similarities.append(float(sim))
    
    statistics = {
        "scenario_id": scenario_dir.name,
        "p1_stats": p1_stats,
        "p2_stats": p2_stats,
        "cross_agent_similarity": {
            "original_p1_vs_p2": cross_similarities[0] if cross_similarities else 0.0,
            "paraphrase_mean": float(np.mean(cross_similarities[1:])) if len(cross_similarities) > 1 else 0.0,
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "p1_count": len(p1_paraphrases),
            "p2_count": len(p2_paraphrases),
        },
    }
    
    with open(out_scenario_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    
    # 绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(
        df_p1, annot=len(p1_labels) <= 12, fmt=".2f", cmap="RdYlGn", vmin=0.7, vmax=1.0,
        square=True, linewidths=0.5, annot_kws={"size": 7}, ax=axes[0],
    )
    axes[0].set_title(f"P1 Similarity\nMean: {p1_stats['mean']:.4f}, To-Origin Mean: {p1_stats['to_origin_mean']:.4f}")
    
    sns.heatmap(
        df_p2, annot=len(p2_labels) <= 12, fmt=".2f", cmap="RdYlGn", vmin=0.7, vmax=1.0,
        square=True, linewidths=0.5, annot_kws={"size": 7}, ax=axes[1],
    )
    axes[1].set_title(f"P2 Similarity\nMean: {p2_stats['mean']:.4f}, To-Origin Mean: {p2_stats['to_origin_mean']:.4f}")
    
    plt.suptitle(f"Scenario: {scenario_dir.name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_scenario_dir / "similarity_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="扩展改写脚本：在已有的 embeddings_backgrounds 基础上继续改写",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/dynamic_observation/embeddings_backgrounds"),
        help="输入目录（包含 difficulty 子目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
        help="目标 paraphrase 数量（默认：20）",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        help="难度级别子目录（默认：hard）",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=20,
        help="最大并发请求数（默认：20）",
    )
    
    args = parser.parse_args()
    
    # 设置 API
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    litellm.api_key = api_key
    
    embed_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    
    # 确定输入目录
    input_dir = args.input_dir / args.difficulty
    output_dir = args.output_dir / args.difficulty
    
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 获取所有 scenario 目录
    scenario_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    console.print(f"[bold cyan]扩展改写脚本[/]")
    console.print(f"[dim]输入: {input_dir}[/]")
    console.print(f"[dim]输出: {output_dir}[/]")
    console.print(f"[dim]目标数量: {args.target_count}[/]")
    console.print(f"[dim]Scenario 数量: {len(scenario_dirs)}[/]")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paraphrase_semaphore = asyncio.Semaphore(args.max_concurrency)
    embed_semaphore = asyncio.Semaphore(args.max_concurrency)
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]处理中...", total=len(scenario_dirs))
        
        # 批量处理
        batch_size = 3
        for i in range(0, len(scenario_dirs), batch_size):
            batch = scenario_dirs[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                process_scenario(
                    sd, output_dir, args.target_count,
                    paraphrase_semaphore, embed_client, embed_semaphore
                )
                for sd in batch
            ], return_exceptions=True)
            
            for j, res in enumerate(batch_results):
                if isinstance(res, Exception):
                    traceback.print_exc()
                    logger.error(f"处理 {batch[j].name} 失败: {res}")
                else:
                    results.append(res)
            
            progress.advance(task, len(batch))
    
    # 汇总结果
    total_p1_new = sum(r.get("p1_new_count", 0) for r in results if isinstance(r, dict))
    total_p2_new = sum(r.get("p2_new_count", 0) for r in results if isinstance(r, dict))
    
    console.print(f"\n[bold green]完成！[/]")
    console.print(f"[green]处理了 {len(results)} 个 scenario[/]")
    console.print(f"[green]P1 新增 paraphrases: {total_p1_new}[/]")
    console.print(f"[green]P2 新增 paraphrases: {total_p2_new}[/]")
    
    # 统计 skipped
    skipped_count = sum(1 for r in results if isinstance(r, dict) and r.get("skipped"))
    if skipped_count > 0:
        console.print(f"[yellow]跳过 {skipped_count} 个已达目标数量的 scenario[/]")
    
    # 计算并保存 cost 统计
    cost_info = {
        "completed_at": datetime.now().isoformat(),
        "target_count": args.target_count,
        "total_scenarios": len(results),
        "skipped_scenarios": skipped_count,
        "new_paraphrases": {"p1": total_p1_new, "p2": total_p2_new},
        "generation_count": len(generation_ids),
    }
    
    if generation_ids:
        console.print(f"\n[cyan]正在计算 {len(generation_ids)} 次 LLM 调用的 cost...[/]")
        
        # 保存 generation IDs 到日志文件
        log_file = output_dir / "generation_ids.jsonl"
        with open(log_file, "w", encoding="utf-8") as f:
            for gen_id in generation_ids:
                f.write(json.dumps({"id": gen_id}) + "\n")
        logger.info(f"已保存 {len(generation_ids)} 个 generation IDs 到 {log_file}")
        
        # 计算 cost
        cost_result = await calculate_cost_async(log_file)
        
        # 合并 cost 结果
        cost_info.update(cost_result)
        
        console.print(f"\n[cyan]Cost 信息:[/]")
        console.print(f"  Total Cost: ${cost_info.get('total_cost', 0):.6f}")
        console.print(f"  Total Tokens: {cost_info.get('total_tokens', 0):,}")
        console.print(f"  LLM 调用次数: {len(generation_ids)}")
    else:
        console.print("[yellow]无新的 LLM 调用，跳过 cost 计算[/]")
    
    cost_file = output_dir / "cost_info.json"
    with open(cost_file, "w", encoding="utf-8") as f:
        json.dump(cost_info, f, indent=2)
    console.print(f"[green]Cost 信息已保存至: {cost_file}[/]")
    
    # 生成并显示汇总统计信息
    console.print(f"\n[bold cyan]===== 改写统计信息 =====[/]")
    
    # 收集所有 statistics.json
    all_statistics = []
    for scenario_dir in (output_dir).iterdir():
        if scenario_dir.is_dir():
            stats_file = scenario_dir / "statistics.json"
            if stats_file.exists():
                with open(stats_file, "r", encoding="utf-8") as f:
                    all_statistics.append(json.load(f))
    
    if all_statistics:
        # 计算汇总数据
        p1_to_origin_means = [s["p1_stats"]["to_origin_mean"] for s in all_statistics]
        p2_to_origin_means = [s["p2_stats"]["to_origin_mean"] for s in all_statistics]
        cross_original = [s["cross_agent_similarity"]["original_p1_vs_p2"] for s in all_statistics]
        
        from rich.table import Table
        
        stats_table = Table(title="Paraphrase Similarity Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right")
        
        stats_table.add_row("Total Scenarios", str(len(all_statistics)))
        stats_table.add_row("P1 To-Origin Mean", f"{np.mean(p1_to_origin_means):.4f}")
        stats_table.add_row("P2 To-Origin Mean", f"{np.mean(p2_to_origin_means):.4f}")
        stats_table.add_row("Cross-Agent Original", f"{np.mean(cross_original):.4f}")
        console.print(stats_table)
        
        # 保存汇总 CSV
        import pandas as pd
        rows = []
        for stat in all_statistics:
            rows.append({
                "scenario_id": stat["scenario_id"],
                "p1_to_origin_mean": stat["p1_stats"]["to_origin_mean"],
                "p2_to_origin_mean": stat["p2_stats"]["to_origin_mean"],
                "p1_mean": stat["p1_stats"]["mean"],
                "p2_mean": stat["p2_stats"]["mean"],
                "cross_original": stat["cross_agent_similarity"]["original_p1_vs_p2"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "summary_statistics.csv", index=False)
        console.print(f"[green]汇总统计已保存至: {output_dir / 'summary_statistics.csv'}[/]")
    
    console.print(f"[green]输出目录: {output_dir}[/]")


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    
    asyncio.run(main())
