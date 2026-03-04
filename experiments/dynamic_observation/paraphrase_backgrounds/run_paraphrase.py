#!/usr/bin/env python3
"""
Main execution script for paraphrasing agent backgrounds.

Usage:
    # Test with 2 scenarios first:
    cd /path/to/project && source .venv/bin/activate && unset ALL_PROXY all_proxy && \
        python -m experiments.dynamic_observation.paraphrase_backgrounds.run_paraphrase --test

    # Run on all data:
    cd /path/to/project && source .venv/bin/activate && unset ALL_PROXY all_proxy && \
        python -m experiments.dynamic_observation.paraphrase_backgrounds.run_paraphrase
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Setup paths and load environment
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from experiments.dynamic_observation.paraphrase_backgrounds.core import (
    DEFAULT_MODEL,
    DEFAULT_NUM_PARAPHRASES,
    DEFAULT_TEMPERATURE,
    collect_scenarios,
    copy_hard_subset,
    process_scenario_pair,
    setup_litellm,
)

console = Console()


# ========== Configuration ==========
INPUT_BASE_DIR = PROJECT_ROOT / "experiments/dynamic_observation/agent_backgrounds"
OUTPUT_BASE_DIR = PROJECT_ROOT / "experiments/dynamic_observation/paraphrased_backgrounds"
MAX_CONCURRENCY = 20
# ===================================


def setup_logger(verbose: bool = False) -> None:
    """Configure loguru logger with rich-compatible format."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Paraphrase agent backgrounds using LLM")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process 2 scenarios",
    )
    parser.add_argument(
        "--hard-only",
        action="store_true",
        help="Only process the 'hard' subset (70 scenarios)",
    )
    parser.add_argument(
        "--num-paraphrases",
        type=int,
        default=DEFAULT_NUM_PARAPHRASES,
        help=f"Number of paraphrased versions to generate (default: {DEFAULT_NUM_PARAPHRASES})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for generation (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=f"Maximum concurrent LLM requests (default: {MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


async def process_scenarios_with_progress(
    scenarios: list[tuple[Path, list[Path]]],
    output_dir: Path,
    llm_call_log_file: Path,
    num_paraphrases: int,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float,
) -> tuple[int, int]:
    """
    Process scenarios with progress bar display.
    Each scenario's files share the same backgrounds, so LLM is called only once per scenario.
    LLM call info is saved to a separate log file.
    Returns (processed_count, skipped_count).
    """
    processed = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing scenarios...", total=len(scenarios))

        # Process scenarios concurrently in batches
        batch_size = semaphore._value  # One scenario per concurrency slot

        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i:i + batch_size]

            async def process_one(scenario_info: tuple[Path, list[Path]]) -> tuple[int, int]:
                scenario_dir, files = scenario_info
                scenario_output_dir = output_dir / scenario_dir.name

                return await process_scenario_pair(
                    scenario_dir,
                    files,
                    scenario_output_dir,
                    num_paraphrases,
                    semaphore,
                    llm_call_log_file,
                    model,
                    temperature,
                )

            results = await asyncio.gather(*[process_one(s) for s in batch])

            for proc_count, skip_count in results:
                processed += proc_count
                skipped += skip_count
                progress.update(task, advance=1)

            # Update description with current scenario
            current_scenario = batch[-1][0].name if batch else ""
            progress.update(
                task,
                description=f"[cyan]Processing: {current_scenario} (done: {processed}, skipped: {skipped})"
            )

    return processed, skipped


async def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logger(args.verbose)

    console.print("[bold green]=" * 60)
    console.print("[bold green]Agent Background Paraphrasing System[/]")
    console.print("[bold green]=" * 60)

    # Setup LiteLLM
    setup_litellm()

    # Prepare directories
    all_input_dir = INPUT_BASE_DIR / "all"
    hard_input_dir = INPUT_BASE_DIR / "hard"
    all_output_dir = OUTPUT_BASE_DIR / "all"
    hard_output_dir = OUTPUT_BASE_DIR / "hard"

    # Determine which directory to process
    if args.hard_only:
        input_dir = hard_input_dir
        output_dir = hard_output_dir
        subset_name = "hard"
    else:
        input_dir = all_input_dir
        output_dir = all_output_dir
        subset_name = "all"

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir} | {__file__}"
        )

    # Collect scenarios (each scenario has 2 files with same backgrounds)
    console.print(f"\n[cyan]Collecting scenarios from {input_dir}...[/]")
    scenarios = collect_scenarios(input_dir)

    # Apply test mode limit
    if args.test:
        scenarios = scenarios[:2]  # Only first 2 scenarios
        console.print(f"[yellow]TEST MODE: Processing only {len(scenarios)} scenarios[/]")

    total_files = sum(len(files) for _, files in scenarios)

    # LLM call log file (separate from paraphrased results)
    llm_call_log_file = OUTPUT_BASE_DIR / "llm_call_log.jsonl"

    console.print(f"[green]Found {len(scenarios)} scenarios ({total_files} files)[/]")
    console.print(f"[blue]Subset: {subset_name}[/]")
    console.print(f"[blue]Model: {args.model}[/]")
    console.print(f"[blue]Temperature: {args.temperature}[/]")
    console.print(f"[blue]Paraphrases per background: {args.num_paraphrases}[/]")
    console.print(f"[blue]Max concurrency: {args.max_concurrency}[/]")
    console.print(f"[blue]LLM call log: {llm_call_log_file}[/]")
    console.print(f"[yellow]Note: LLM called once per scenario (backgrounds shared between files)[/]")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.max_concurrency)

    # Process directory
    console.print(f"\n[bold cyan]Processing '{subset_name}' directory...[/]")
    processed, skipped = await process_scenarios_with_progress(
        scenarios,
        output_dir,
        llm_call_log_file,
        args.num_paraphrases,
        semaphore,
        args.model,
        args.temperature,
    )

    console.print(f"[green]Files processed: {processed}, Files skipped: {skipped}[/]")

    console.print(f"\n[bold green]Done! Results saved to {output_dir}[/]")
    console.print(f"[bold blue]LLM call log saved to {llm_call_log_file}[/]")


if __name__ == "__main__":
    asyncio.run(main())

