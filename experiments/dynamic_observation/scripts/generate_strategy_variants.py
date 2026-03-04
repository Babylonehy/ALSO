"""
Generate strategy space variants with different sizes for ablation study.

This script generates paraphrased versions of V3 strategies using LLM,
creating strategy spaces of sizes: 5, 10, 20, 50, 100.

Usage:
    python scripts/generate_strategy_variants.py --target-sizes 5 10 20 50 100
    python scripts/generate_strategy_variants.py --target-sizes 50 100 --model openrouter/deepseek/deepseek-v3.2
    python scripts/generate_strategy_variants.py --concurrency 50  # Limit concurrent LLM calls
"""

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TaskID
import litellm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bandits.social_strategies import SOCIAL_STRATEGIES_V3, get_strategies

load_dotenv()
console = Console()

# Prompt template for generating paraphrases
PARAPHRASE_PROMPT = """You are an expert in social psychology and negotiation theory. Your task is to generate semantically equivalent paraphrases of social strategies.

**Original Strategy:**
{strategy_name}: {strategy_description}

**Theoretical Basis:** {theory}

**Instructions:**
1. Generate {n} paraphrased versions of this strategy
2. Each paraphrase must:
   - Preserve the core behavioral intent and theoretical grounding
   - Use different wording, sentence structures, and examples
   - Maintain the same level of actionability and specificity
   - Be directly usable as an agent prompt
   - Start with "In your response," to maintain consistency
3. Vary the linguistic style: some formal, some conversational
4. Do NOT change the underlying negotiation tactic

**Output Format (JSON only, no markdown):**
{{
  "original_id": "{strategy_id}",
  "paraphrases": [
    {{"id": "{strategy_id}_v1", "description": "..."}},
    {{"id": "{strategy_id}_v2", "description": "..."}}
  ]
}}
"""

# Quality check thresholds
QUALITY_THRESHOLD: float = 0.05  # Max ratio of non-ASCII characters
QUALITY_MAX_RETRIES: int = 3  # Max retries for quality check


def contains_non_english(text: str, threshold: float = 0.02) -> tuple[bool, float, str]:
    """
    检查文本是否包含大量非英语字符或乱码模式。
    
    参考 experiments/dynamic_observation/paraphrase_agent_profiles_chain.py

    Returns:
        (is_problematic, non_ascii_ratio, reason)
    """
    import re

    if not text:
        return False, 0.0, ""

    # 允许的特殊字符（常见英文标点和符号）
    allowed_special = set("''\"\"—–…•·°±×÷©®™€£¥¢")

    # 统计非 ASCII 字符
    non_ascii_chars: list[str] = []
    for char in text:
        if ord(char) > 127 and char not in allowed_special:
            non_ascii_chars.append(char)

    ratio = len(non_ascii_chars) / len(text) if text else 0.0

    # 检查1: 非 ASCII 比例超过阈值
    if ratio > threshold:
        sample = "".join(set(non_ascii_chars[:30]))
        return True, ratio, f"High non-ASCII ratio: {sample}"

    # 检查2: 检测多语言混合（乱码的典型特征）
    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", text))  # 俄语
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))  # 阿拉伯语
    has_cjk = bool(re.search(r"[\u4E00-\u9FFF]", text))  # 中日韩
    has_korean = bool(re.search(r"[\uAC00-\uD7AF]", text))  # 韩语
    has_hebrew = bool(re.search(r"[\u0590-\u05FF]", text))  # 希伯来语
    has_thai = bool(re.search(r"[\u0E00-\u0E7F]", text))  # 泰语
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))  # 天城文(印地语)
    has_armenian = bool(re.search(r"[\u0530-\u058F]", text))  # 亚美尼亚语
    has_georgian = bool(re.search(r"[\u10A0-\u10FF]", text))  # 格鲁吉亚语
    has_greek = bool(re.search(r"[\u0370-\u03FF]", text))  # 希腊语

    language_count = sum([
        has_cyrillic, has_arabic, has_cjk, has_korean, has_hebrew,
        has_thai, has_devanagari, has_armenian, has_georgian, has_greek,
    ])

    if language_count >= 2:
        return True, ratio, f"Multi-language mixing detected ({language_count} scripts)"

    # 检查3: 检测代码片段模式（乱码中常见）
    code_patterns = [
        r"\)\s*\{", r"\}\s*;", r"===", r"\[\s*\]", r"function\s*\(",
        r"return\s+\w+;", r"import\s+\w+", r"class\s+\w+", r"\$\{",
        r"=>", r"\.then\(", r"console\.", r"#\s*include", r"def\s+\w+\s*\(",
    ]

    code_matches = sum(1 for p in code_patterns if re.search(p, text))
    if code_matches >= 3:
        return True, ratio, f"Code-like patterns detected ({code_matches} patterns)"

    # 检查4: 过多的特殊符号（乱码特征）
    special_chars = re.findall(r"[█▓▒░■□●○◆◇★☆►◄▲△▼▽⬚═║╔╗╚╝╠╣╬┌┐└┘├┤┬┴┼─│]", text)
    if len(special_chars) > 5:
        return True, ratio, f"Excessive special symbols ({len(special_chars)} found)"

    # 检查5: 连续的非英语字符序列（正常英文不应该有）
    consecutive_non_ascii = re.findall(r"[^\x00-\x7F]{3,}", text)
    if len(consecutive_non_ascii) > 3:
        return True, ratio, f"Multiple non-ASCII sequences ({len(consecutive_non_ascii)} found)"

    sample = "".join(set(non_ascii_chars[:30])) if non_ascii_chars else ""
    return False, ratio, sample


# Max paraphrases per LLM query (to avoid quality degradation)
MAX_PARAPHRASES_PER_QUERY: int = 5


async def _single_llm_query(
    strategy: dict,
    n_paraphrases: int,
    model: str,
    max_retries: int = 3,
) -> list[dict]:
    """Make a single LLM query for paraphrases with retry and quality check."""
    prompt = PARAPHRASE_PROMPT.format(
        strategy_name=strategy.get("name", strategy["id"]),
        strategy_description=strategy["description"],
        theory=strategy.get("theory", "Social Psychology"),
        n=n_paraphrases,
        strategy_id=strategy["id"],
    )
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            result = json.loads(content)
            paraphrases = result.get("paraphrases", [])
            
            # Quality check: filter out garbled/non-English paraphrases
            valid_paraphrases = []
            for p in paraphrases:
                desc = p.get("description", "")
                is_bad, ratio, reason = contains_non_english(desc, threshold=QUALITY_THRESHOLD)
                if is_bad:
                    logger.warning(f"Filtered bad paraphrase for {strategy['id']}: {reason}")
                else:
                    valid_paraphrases.append(p)
            
            if valid_paraphrases:
                return valid_paraphrases
            elif paraphrases:
                # All paraphrases were bad, retry
                raise ValueError(f"All {len(paraphrases)} paraphrases failed quality check")
            else:
                return []
                
        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {strategy['id']}: {e}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
    
    # All retries failed
    logger.error(f"All {max_retries} attempts failed for {strategy['id']}: {last_error}")
    return []


async def generate_paraphrases(
    strategy: dict,
    n_paraphrases: int,
    model: str = "openrouter/deepseek/deepseek-v3.2",
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = 3,
) -> tuple[dict, list[dict]]:
    """Generate n paraphrases for a single strategy using LLM.
    
    If n_paraphrases > MAX_PARAPHRASES_PER_QUERY, makes multiple queries.
    
    Returns:
        Tuple of (original_strategy, list of paraphrases)
    """
    if strategy["id"] == "no_strategy":
        # Don't paraphrase the baseline
        return strategy, []
    
    all_paraphrases: list[dict] = []
    remaining = n_paraphrases
    query_idx = 0
    
    async def _do_queries():
        nonlocal all_paraphrases, remaining, query_idx
        while remaining > 0:
            batch_size = min(remaining, MAX_PARAPHRASES_PER_QUERY)
            # Adjust strategy_id suffix to avoid duplicate IDs across queries
            modified_strategy = dict(strategy)
            if query_idx > 0:
                modified_strategy["id"] = f"{strategy['id']}_q{query_idx}"
            
            batch = await _single_llm_query(modified_strategy, batch_size, model, max_retries)
            
            # Fix IDs: restore original strategy id prefix
            for p in batch:
                if query_idx > 0:
                    p["id"] = p["id"].replace(f"{strategy['id']}_q{query_idx}", strategy["id"])
            
            all_paraphrases.extend(batch)
            remaining -= len(batch)
            query_idx += 1
            
            # If we got nothing, break to avoid infinite loop
            if not batch:
                break
    
    if semaphore:
        async with semaphore:
            await _do_queries()
    else:
        await _do_queries()
    
    return strategy, all_paraphrases


async def create_strategy_space(
    target_size: int,
    base_strategies: list[dict],
    model: str,
    concurrency: int = 100,
    existing_strategies: list[dict] | None = None,
) -> list[dict]:
    """Create a strategy space of the target size by generating paraphrases concurrently.
    
    Args:
        existing_strategies: If provided, use as starting point for incremental generation
    """
    active_strategies = [s for s in base_strategies if s["id"] != "no_strategy"]
    
    # Start with existing strategies or baseline
    if existing_strategies:
        strategies = list(existing_strategies)
        existing_ids = {s["id"] for s in strategies}
        console.print(f"[cyan]Loaded {len(strategies)} existing strategies[/cyan]")
        
        if len(strategies) >= target_size:
            console.print(f"[green]Already have {len(strategies)} strategies, target is {target_size}. Skipping.[/green]")
            return strategies[:target_size]
    else:
        strategies = [base_strategies[0]]  # no_strategy
        existing_ids = {base_strategies[0]["id"]}
    
    if target_size <= len(base_strategies):
        # Just sample from existing strategies
        for s in active_strategies:
            if s["id"] not in existing_ids:
                strategies.append(s)
                if len(strategies) >= target_size:
                    break
        return strategies[:target_size]
    
    # Count existing paraphrases per parent strategy
    existing_by_parent: dict[str, list[dict]] = {}
    for s in strategies:
        parent = s.get("parent")
        if parent:
            existing_by_parent.setdefault(parent, []).append(s)
    
    # Need to generate paraphrases
    n_active = len(active_strategies)
    n_needed = target_size - 1  # excluding baseline
    n_per_strategy = (n_needed - n_active) // n_active + 1
    
    # Determine which strategies need more paraphrases
    strategies_to_generate = []
    for strategy in active_strategies:
        existing_count = len(existing_by_parent.get(strategy["id"], []))
        needed = n_per_strategy - existing_count
        if needed > 0:
            strategies_to_generate.append((strategy, needed))
    
    if not strategies_to_generate:
        console.print(f"[green]All paraphrases already generated[/green]")
        return strategies[:target_size]
    
    console.print(f"[cyan]Generating paraphrases for {len(strategies_to_generate)} strategies (need ~{n_per_strategy} each)[/cyan]")
    console.print(f"[cyan]Using concurrency: {concurrency}[/cyan]")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create all tasks for concurrent execution
    tasks = [
        generate_paraphrases(strategy, n_needed, model, semaphore)
        for strategy, n_needed in strategies_to_generate
    ]
    
    # Execute all tasks concurrently with progress tracking
    with Progress(console=console) as progress:
        task_id = progress.add_task("[green]Generating paraphrases...", total=len(tasks))
        
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            progress.update(task_id, advance=1)
    
    # Add original strategies that are missing
    for strategy in active_strategies:
        if strategy["id"] not in existing_ids:
            strategies.append(strategy)
            existing_ids.add(strategy["id"])
    
    # Collect results
    for strategy, paraphrases in results:
        for p in paraphrases:
            if len(strategies) >= target_size:
                break
            p_id = p["id"]
            if p_id in existing_ids:
                continue  # Skip duplicates
            strategies.append({
                "id": p_id,
                "name": f"{strategy.get('name', strategy['id'])} (Variant)",
                "theory": strategy.get("theory", ""),
                "description": p["description"],
                "parent": strategy["id"],
            })
            existing_ids.add(p_id)
        
        if len(strategies) >= target_size:
            break
    
    return strategies[:target_size]


def save_strategy_space(strategies: list[dict], version: str, output_dir: Path) -> None:
    """Save generated strategy space to a Python file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"social_strategies_{version}.py"

    var_name = f"SOCIAL_STRATEGIES_{version.upper()}"

    with open(output_file, "w") as f:
        f.write(f'"""\nAuto-generated strategy space: {version}\nSize: {len(strategies)} strategies\n"""\n\n')
        f.write(f"{var_name}: list[dict[str, str]] = [\n")
        for s in strategies:
            f.write("    {\n")
            for key, value in s.items():
                if isinstance(value, str):
                    # Escape quotes and handle multiline
                    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                    f.write(f'        "{key}": "{escaped}",\n')
            f.write("    },\n")
        f.write("]\n")

    console.print(f"[green]Saved {len(strategies)} strategies to {output_file}[/green]")

    # Also save as JSON for easier inspection
    json_file = output_dir / f"social_strategies_{version}.json"
    with open(json_file, "w") as f:
        json.dump(strategies, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Saved JSON to {json_file}[/green]")


async def main():
    parser = argparse.ArgumentParser(description="Generate strategy space variants")
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100],
        help="Target sizes for strategy spaces",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/deepseek/deepseek-v3.2",
        help="LLM model for generating paraphrases",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_strategies",
        help="Output directory for generated strategies",
    )
    parser.add_argument(
        "--base-version",
        type=str,
        default="v3",
        help="Base strategy version to expand from",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=100,
        help="Maximum number of concurrent LLM calls (default: 100)",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / args.output_dir
    base_strategies = get_strategies(args.base_version)

    console.print(f"[bold]Base strategies ({args.base_version}): {len(base_strategies)}[/bold]")
    console.print(f"[bold]Target sizes: {args.target_sizes}[/bold]")
    console.print(f"[bold]Concurrency: {args.concurrency}[/bold]")

    # Version mapping: size -> version name
    size_to_version = {
        5: "s5",
        10: "s10",
        20: "s20",
        50: "s50",
        100: "s100",
    }

    for target_size in args.target_sizes:
        version = size_to_version.get(target_size, f"s{target_size}")
        
        # Check if output files already exist
        output_py = output_dir / f"social_strategies_{version}.py"
        output_json = output_dir / f"social_strategies_{version}.json"
        
        # Load existing strategies for incremental generation
        existing_strategies = None
        if output_json.exists():
            with open(output_json) as f:
                existing_strategies = json.load(f)
            if len(existing_strategies) >= target_size:
                console.print(f"[yellow]Skipping {version}: already have {len(existing_strategies)} strategies[/yellow]")
                continue
            console.print(f"[cyan]Found {len(existing_strategies)} existing strategies for {version}[/cyan]")
        
        console.print(f"\n[bold cyan]Generating strategy space: {version} (size={target_size})[/bold cyan]")

        try:
            strategies = await create_strategy_space(
                target_size, base_strategies, args.model, args.concurrency, existing_strategies
            )
            save_strategy_space(strategies, version, output_dir)
        except Exception as e:
            traceback.print_exc()
            console.print(f"[red]Error generating {version}: {e}[/red]")
            continue

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())

