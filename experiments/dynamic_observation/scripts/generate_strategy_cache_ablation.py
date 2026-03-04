"""
Generate strategy embedding cache for ablation study on strategy space size.

This script generates embedding caches for different strategy space sizes (s5, s10, s20, s50, s100).

Usage:
    python scripts/generate_strategy_cache_ablation.py --versions s5 s10 s20 s50 s100
    python scripts/generate_strategy_cache_ablation.py --versions s50 s100 --subset hard_small
"""

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing sotopia (which connects to Redis)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from sotopia.database import EnvironmentProfile, AgentProfile, EnvAgentComboStorage

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bandits.strategy_space import StrategySpace

console = Console()


@dataclass
class ProcessResult:
    combo_pk: str
    success: bool
    skipped: bool = False
    error: str = ""


def process_combo(
    combo_pk: str,
    cache_dir: str,
    embedding_model: str,
    embedding_dim: int,
    strategy_version: str,
    skip_existing: bool = True,
) -> ProcessResult:
    """Process a single combo to generate embeddings."""
    try:
        combo = EnvAgentComboStorage.get(combo_pk)
        env = EnvironmentProfile.get(combo.env_id)
        agent_profiles = [AgentProfile.get(pk) for pk in combo.agent_ids]

        p1_background = env.agent_goals[0]
        p2_background = env.agent_goals[1]

        # Check cache
        cache_path = Path(cache_dir)
        import hashlib
        bg_hash = hashlib.md5((p1_background + p2_background).encode()).hexdigest()[:12]
        p1_cache = cache_path / f"strategy_p1_embeddings_{bg_hash}.npy"
        p2_cache = cache_path / f"strategy_p2_embeddings_{bg_hash}.npy"

        if skip_existing and p1_cache.exists() and p2_cache.exists():
            return ProcessResult(combo_pk=combo_pk, success=True, skipped=True)

        # Create StrategySpace (triggers embedding computation and caching)
        StrategySpace.from_scenario_backgrounds(
            p1_background=p1_background,
            p2_background=p2_background,
            p1_name=f"{agent_profiles[0].first_name} {agent_profiles[0].last_name}",
            p2_name=f"{agent_profiles[1].first_name} {agent_profiles[1].last_name}",
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            cache_dir=cache_dir,
            strategy_version=strategy_version,
        )
        return ProcessResult(combo_pk=combo_pk, success=True, skipped=False)

    except Exception as e:
        traceback.print_exc()
        return ProcessResult(combo_pk=combo_pk, success=False, error=str(e))


def get_combos_for_subset(subset: str) -> list[str]:
    """Get combo PKs for a given subset."""
    from lib.database_utils import get_combos_for_subset as _get_combos
    combos = _get_combos(subset)  # type: ignore
    return [c.pk for c in combos]


def main():
    parser = argparse.ArgumentParser(description="Generate strategy embedding cache for ablation")
    parser.add_argument("--versions", type=str, nargs="+", default=["s5", "s10", "s20", "s50", "s100"])
    parser.add_argument("--subset", type=str, default="hard_small", choices=["all", "hard", "hard_small"])
    parser.add_argument("--embedding-model", type=str, default="qwen/qwen3-embedding-8b")
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    combo_pks = get_combos_for_subset(args.subset)
    console.print(f"[bold]Found {len(combo_pks)} combos for subset '{args.subset}'[/bold]")

    for version in args.versions:
        cache_dir = Path(__file__).parent.parent / "cache" / f"strategy_embeddings_{version}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"\n[bold cyan]Generating cache for {version} -> {cache_dir}[/bold cyan]")

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console
        ) as progress:
            task = progress.add_task(f"Processing {version}", total=len(combo_pks))
            
            success, skipped, failed = 0, 0, 0
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_combo, pk, str(cache_dir), args.embedding_model,
                        args.embedding_dim, version, args.skip_existing
                    ): pk for pk in combo_pks
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    if result.success:
                        if result.skipped:
                            skipped += 1
                        else:
                            success += 1
                    else:
                        failed += 1
                        console.print(f"[red]Failed: {result.combo_pk}: {result.error}[/red]")
                    progress.advance(task)
        
        console.print(f"[green]{version}: {success} generated, {skipped} skipped, {failed} failed[/green]")

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()

