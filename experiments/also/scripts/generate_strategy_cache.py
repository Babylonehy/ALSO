#!/usr/bin/env python3
"""
Pre-generate strategy embedding cache for all scenarios.

This script computes and caches the strategy embeddings for each scenario,
so that running experiments can load from cache instead of calling the embedding API.

Usage:
    cd experiments/also
    uv run python scripts/generate_strategy_cache.py --subset hard
    uv run python scripts/generate_strategy_cache.py --subset hard --strategy-version v2
    uv run python scripts/generate_strategy_cache.py --subset all --cache-dir ./cache/strategy_embeddings
    uv run python scripts/generate_strategy_cache.py --subset hard --concurrency 4  # Async parallel
"""

import argparse
import asyncio
import hashlib
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TaskID

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from sotopia.database import AgentProfile, EnvironmentProfile
from sotopia.database.env_agent_combo_storage import EnvAgentComboStorage
from experiments.also.core.envs_dynamic_parallel import get_bio, render_text_for_agent
from experiments.also.core.bandits.strategy_space import StrategySpace

console = Console()


@dataclass
class ProcessResult:
    """Result of processing a single combo."""
    combo_pk: str
    success: bool
    skipped: bool
    error: str | None = None


def parse_env_ids(file_path: Path) -> dict[str, list[str]]:
    """Parse env_ids.txt file, return hard and all dataset env_id lists."""
    import re
    with open(file_path, "r") as f:
        content = f.read()

    sections = content.split("SOTOPIA-ALL:")
    hard_section = sections[0]
    all_section = sections[1] if len(sections) > 1 else ""

    hard_ids = re.findall(r'"([A-Z0-9]+)"', hard_section)
    all_ids = re.findall(r'"([A-Z0-9]+)"', all_section)

    return {
        "hard": list(set(hard_ids)),
        "all": list(set(all_ids))
    }


def get_combos_for_subset(subset: str) -> list[EnvAgentComboStorage]:
    """Get all combos for the specified dataset subset."""
    env_ids_path = PROJECT_ROOT / "data/env_ids.txt"
    if not env_ids_path.exists():
        raise FileNotFoundError(f"env_ids.txt not found at {env_ids_path}")

    ids_by_category = parse_env_ids(env_ids_path)
    env_ids = ids_by_category.get(subset, [])

    if not env_ids:
        raise ValueError(f"No env_ids found for subset '{subset}'")

    combos = []
    for env_id in env_ids:
        found_combos = list(EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all())
        combos.extend(found_combos)

    return combos


def process_single_combo_sync(
    combo: EnvAgentComboStorage,
    cache_dir: Path,
    embedding_model: str,
    embedding_dim: int,
    skip_existing: bool,
    strategy_version: str,
) -> ProcessResult:
    """Process a single combo and generate its cache (synchronous)."""
    try:
        # Load profiles
        env_profile = EnvironmentProfile.get(pk=combo.env_id)
        agent_profiles = [AgentProfile.get(pk=aid) for aid in combo.agent_ids]

        # Get bios (must match run_bandit_simulation_context.py logic)
        # First get raw bio, then render to remove secrets
        p1_background_raw = get_bio(env_profile.relationship, agent_profiles[0], agent_id=0)
        p2_background_raw = get_bio(env_profile.relationship, agent_profiles[1], agent_id=1)
        p1_background = render_text_for_agent(p1_background_raw, agent_id=0)
        p2_background = render_text_for_agent(p2_background_raw, agent_id=1)

        # Check if cache exists
        bg_hash = hashlib.md5((p1_background + p2_background).encode()).hexdigest()[:12]
        p1_cache = cache_dir / f"strategy_p1_embeddings_{bg_hash}.npy"
        p2_cache = cache_dir / f"strategy_p2_embeddings_{bg_hash}.npy"

        if skip_existing and p1_cache.exists() and p2_cache.exists():
            return ProcessResult(combo_pk=combo.pk, success=True, skipped=True)

        # Create StrategySpace (this triggers embedding computation and caching)
        StrategySpace.from_scenario_backgrounds(
            p1_background=p1_background,
            p2_background=p2_background,
            p1_name=f"{agent_profiles[0].first_name} {agent_profiles[0].last_name}",
            p2_name=f"{agent_profiles[1].first_name} {agent_profiles[1].last_name}",
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            cache_dir=str(cache_dir),
            strategy_version=strategy_version,
        )
        return ProcessResult(combo_pk=combo.pk, success=True, skipped=False)

    except Exception as e:
        traceback.print_exc()
        return ProcessResult(combo_pk=combo.pk, success=False, skipped=False, error=str(e))


async def process_single_combo_async(
    combo: EnvAgentComboStorage,
    cache_dir: Path,
    embedding_model: str,
    embedding_dim: int,
    skip_existing: bool,
    strategy_version: str,
    semaphore: asyncio.Semaphore,
) -> ProcessResult:
    """Process a single combo asynchronously with concurrency control."""
    async with semaphore:
        # Use asyncio.to_thread to run sync code in thread pool
        return await asyncio.to_thread(
            process_single_combo_sync,
            combo, cache_dir, embedding_model, embedding_dim, skip_existing, strategy_version
        )


async def run_async(
    combos: list[EnvAgentComboStorage],
    cache_dir: Path,
    embedding_model: str,
    embedding_dim: int,
    skip_existing: bool,
    strategy_version: str,
    concurrency: int,
    progress: Progress,
    task_id: TaskID,
) -> tuple[int, int, int]:
    """Run all combo processing asynchronously with progress updates."""
    semaphore = asyncio.Semaphore(concurrency)
    success_count = 0
    skip_count = 0
    error_count = 0

    async def process_and_update(combo: EnvAgentComboStorage) -> None:
        nonlocal success_count, skip_count, error_count
        result = await process_single_combo_async(
            combo, cache_dir, embedding_model, embedding_dim, skip_existing, strategy_version, semaphore
        )
        if result.skipped:
            skip_count += 1
        elif result.success:
            success_count += 1
        else:
            error_count += 1
            logger.error(f"Error processing {result.combo_pk}: {result.error}")
        progress.update(task_id, advance=1, description=f"Completed {result.combo_pk[:8]}...")

    # Create all tasks and run concurrently
    await asyncio.gather(*[process_and_update(combo) for combo in combos])

    return success_count, skip_count, error_count


def main():
    parser = argparse.ArgumentParser(description="Pre-generate strategy embedding cache")
    parser.add_argument("--subset", type=str, default="hard", choices=["hard", "all"],
                        help="Dataset subset to process")
    parser.add_argument("--cache-dir", type=str, default="./cache/strategy_embeddings",
                        help="Directory to store cached embeddings")
    parser.add_argument("--embedding-model", type=str, default="qwen/qwen3-embedding-8b",
                        help="Embedding model to use")
    parser.add_argument("--embedding-dim", type=int, default=4096,
                        help="Embedding dimension")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip scenarios that already have cached embeddings")
    parser.add_argument(
        "--strategy-version",
        type=str,
        default="v1",
        choices=[
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "v3_diverse2_12",
            "v3_diverse4_12",
        ],
        help=(
            "Strategy version: v1 (13), v2 (10), v3 (12), v4 (49), "
            "v5 (10 adversarial), v6 (10 hard-optimized), "
            "v3_diverse2_12 (13), v3_diverse4_12 (13)"
        ),
    )
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent async tasks (default: 4)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Generating strategy embedding cache[/bold]")
    console.print(f"  Subset: {args.subset}")
    console.print(f"  Cache dir: {cache_dir.absolute()}")
    console.print(f"  Strategy version: {args.strategy_version}")
    console.print(f"  Embedding model: {args.embedding_model}")
    console.print(f"  Embedding dim: {args.embedding_dim}")
    console.print(f"  Concurrency: {args.concurrency}")

    # Get all combos
    combos = get_combos_for_subset(args.subset)
    console.print(f"  Found {len(combos)} scenarios\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing with concurrency={args.concurrency}...", total=len(combos))

        # Run async event loop
        success_count, skip_count, error_count = asyncio.run(
            run_async(
                combos, cache_dir, args.embedding_model, args.embedding_dim,
                args.skip_existing, args.strategy_version, args.concurrency, progress, task
            )
        )

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Success: {success_count}")
    console.print(f"  Skipped: {skip_count}")
    console.print(f"  Errors: {error_count}")
    console.print(f"\nCache saved to: {cache_dir.absolute()}")


if __name__ == "__main__":
    main()
