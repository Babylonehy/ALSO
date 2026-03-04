"""Execution modes for batch and single scenario runs."""

import argparse
import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from sotopia.database import EnvAgentComboStorage
from sotopia.generation_utils import enable_llm_call_logging

# Import from local modules
from .simulation_runner import BanditSimulationRunner
from .progress_tracking import BatchProgressTracker
from .cost_tracking import calculate_cost_by_model_async
from .display_utils import display_summary, get_output_path, print_cost_breakdown, format_duration
from .database_utils import get_available_scenarios, get_combos_for_subset
from .serialization import save_run_config, get_git_info, load_bandit_config
from .config import parse_args

# Import from core modules  
from experiments.dynamic_observation.core.bandits import BanditConfig
from experiments.dynamic_observation.core.logging_utils import (
    configure_logger,
    setup_terminal_logging,
    cleanup_terminal_logging,
)

# Get PROJECT_ROOT
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

console = Console()


async def run_single_scenario(
    scenario_id: str,
    args: argparse.Namespace,
    experiment_tag: str,
    bandit_config: BanditConfig,
    bandit_type: str = "exp3",
    progress_tracker: BatchProgressTracker | None = None,
    verbose: bool = True,
    experiment_dir: Path | None = None,
    tensorboard_dir: Path | None = None,
) -> dict:
    """
    Run a single scenario with bandit optimization.

    Args:
        scenario_id: The scenario ID (EnvAgentComboStorage pk)
        args: Parsed command line arguments
        experiment_tag: Experiment tag for logging
        bandit_config: Bandit configuration
        bandit_type: Type of bandit algorithm to use
        progress_tracker: Optional progress tracker for batch mode
        verbose: Whether to show detailed output

    Returns:
        Dictionary with episode results
    """
    try:
        runner = BanditSimulationRunner(
            scenario_id=scenario_id,
            model_name=args.model,
            p1_model_name=args.p1_model,
            p2_model_name=args.p2_model,
            env_model_name=args.env_model,
            reward_eval_model_name=args.reward_eval_model,
            terminal_eval_model_name=args.terminal_eval_model,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            embeddings_dir=args.embeddings_dir,  # Pass custom embeddings dir
            bandit_config=bandit_config,
            bandit_type=bandit_type,  # type: ignore
            optimize_mode=args.optimize,  # type: ignore
            push_to_db=args.push_to_db,
            experiment_tag=experiment_tag,
            progress_tracker=progress_tracker,
            verbose=verbose,
            experiment_dir=experiment_dir,
            tensorboard_dir=tensorboard_dir,
            use_context_embedding=args.context_embedding,
            embedding_model=args.embedding_model,
            context_embedding_dim=args.context_embedding_dim,
            alternate_optimization=args.alternate_optimization,
        )

        summary = await runner.run_episode()
        return {
            "success": True,
            "scenario_id": scenario_id,
            "summary": summary,
            "error": None,
        }
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in scenario {scenario_id}: {e}")
        if progress_tracker:
            progress_tracker.complete(scenario_id, success=False)
        return {
            "success": False,
            "scenario_id": scenario_id,
            "summary": None,
            "error": str(e),
        }


async def retry_failed_scenarios(
    failed_scenario_ids: list[str],
    args: argparse.Namespace,
    experiment_tag: str,
    bandit_config: "BanditConfig",  # type: ignore
    bandit_type: str,
    experiment_dir: Path,
    tensorboard_dir: Path,
    retry_attempt: int,
) -> dict:
    """
    Retry failed scenarios with parallel execution.

    Args:
        failed_scenario_ids: List of scenario IDs that failed
        args: Command line arguments
        experiment_tag: Experiment tag for logging
        bandit_config: Bandit configuration
        bandit_type: Type of bandit algorithm
        experiment_dir: Experiment directory path
        tensorboard_dir: TensorBoard directory path
        retry_attempt: Current retry attempt number (1-indexed)

    Returns:
        Dictionary containing:
            - "results": List of retry results
            - "success_count": Number of newly successful scenarios
            - "still_failed": Number of scenarios still failing
            - "failed_ids": List of scenario IDs still failing
    """
    console.print(Panel(
        f"[bold yellow]Retry Attempt {retry_attempt}[/]\n"
        f"Retrying {len(failed_scenario_ids)} failed scenarios...",
        title=f"Automatic Retry #{retry_attempt}",
    ))

    # Initialize progress tracker for retry
    progress_tracker = BatchProgressTracker()
    progress_tracker.total = len(failed_scenario_ids)

    # Use semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.batch_size)
    retry_results: list[dict] = []

    async def run_with_semaphore(scenario_id: str) -> dict:
        async with semaphore:
            return await run_single_scenario(
                scenario_id=scenario_id,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=bandit_config,
                bandit_type=bandit_type,
                progress_tracker=progress_tracker,
                verbose=False,
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
            )

    # Incremental results file for retry
    incremental_results_path = experiment_dir / f"results_retry_{retry_attempt}.jsonl"

    def save_incremental_result(result: dict) -> None:
        """Append a single result to incremental retry file."""
        try:
            with open(incremental_results_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save incremental retry result: {e}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(
            f"[yellow]Retrying {len(failed_scenario_ids)} scenarios...",
            total=len(failed_scenario_ids)
        )

        async def update_progress_description() -> None:
            while progress_tracker.completed < progress_tracker.total:
                status = progress_tracker.get_status()
                progress.update(task, description=status)
                await asyncio.sleep(0.5)

        tasks = [run_with_semaphore(sid) for sid in failed_scenario_ids]
        update_task = asyncio.create_task(update_progress_description())

        for coro in asyncio.as_completed(tasks):
            result = await coro
            retry_results.append(result)
            save_incremental_result(result)
            progress.update(task, advance=1)

        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass

    # Calculate retry statistics
    success_count = sum(1 for r in retry_results if r["success"])
    still_failed_count = len(retry_results) - success_count
    still_failed_ids = [r["scenario_id"] for r in retry_results if not r["success"]]

    console.print(f"[green]Newly Succeeded: {success_count}[/]")
    console.print(f"[red]Still Failed: {still_failed_count}[/]")

    if still_failed_ids:
        console.print(f"\n[bold red]Still Failed Scenario IDs:[/]")
        for sid in still_failed_ids[:5]:
            console.print(f"  • {sid}")
        if len(still_failed_ids) > 5:
            console.print(f"  ... and {len(still_failed_ids) - 5} more")

    return {
        "results": retry_results,
        "success_count": success_count,
        "still_failed": still_failed_count,
        "failed_ids": still_failed_ids,
    }


async def run_batch_episodes(args: argparse.Namespace) -> None:
    """Run episodes in batch mode with parallel execution."""
    batch_start_time = time.time()

    # Resolve model names for P1 and P2 first (needed for tag generation)
    p1_model = args.p1_model or args.model
    p2_model = args.p2_model or args.model

    if args.tag:
        experiment_tag = args.tag
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mask_tag = f"mus{int(args.mask_unselected_scores)}"
        iw_tag = f"iw{int(args.importance_weighted_reward)}"
        # Extract short model name for unique tag (e.g., "qwen-2.5-72b-instruct" -> "qwen25_72b")
        def get_model_short(model: str) -> str:
            name = model.split("/")[-1]  # Get last part after /
            return name.replace(".", "").replace("-", "_")[:20]  # Shorten and sanitize

        p1_short = get_model_short(p1_model)
        p2_short = get_model_short(p2_model)
        # If same model, use single tag; if different, use "p1model_vs_p2model"
        if p1_short == p2_short:
            model_tag = p1_short
        else:
            model_tag = f"{p1_short}_vs_{p2_short}"
        experiment_tag = f"bandit_{args.bandit_type}_{args.optimize}_{args.subset}_{model_tag}_{mask_tag}_{iw_tag}_{timestamp}"


    # Mode description
    mode_desc = {
        "both": "Optimizing BOTH agents",
        "p1": "Optimizing P1 only",
        "p2": "Optimizing P2 only",
        "none": "BASELINE (no optimization)",
    }.get(args.optimize, "Unknown")

    console.print(Panel(
        f"[bold green]Batch Bandit Simulation[/]\n\n"
        f"Subset: {args.subset}\n"
        f"Batch Size (concurrency): {args.batch_size}\n"
        f"Max Episodes: {args.max_episodes or 'all'}\n"
        f"P1 Model: {p1_model}\n"
        f"P2 Model: {p2_model}\n"
        f"Env Model: {args.env_model}\n"
        f"Max Turns: {args.max_turns}\n"
        f"[bold cyan]Bandit Type: {args.bandit_type}[/]\n"
        f"ETA (EXP3): {args.eta}, Alpha (LinUCB): {args.alpha}, Beta (NeuralUCB): {args.beta}\n"
        f"[bold yellow]Mode: {mode_desc}[/]\n"
        f"Push to DB: {args.push_to_db}",
        title="Batch Configuration",
    ))

    # Create experiment directory structure
    experiment_dir = PROJECT_ROOT / "experiments/dynamic_observation/outputs" / experiment_tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = experiment_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    tensorboard_dir = experiment_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    
    models_dir = experiment_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    llm_logs_dir = experiment_dir / "llm_logs"
    llm_logs_dir.mkdir(exist_ok=True)

    # Enable LLM call logging
    log_file = llm_logs_dir / "calls.jsonl"
    enable_llm_call_logging(log_file, experiment_tag=experiment_tag)
    console.print(f"[cyan]LLM call log:[/] {log_file}")
    
    # Setup terminal logging to file in experiment dir
    log_file_path, tee_stdout, tee_stderr = setup_terminal_logging(
        experiment_name=f"bandit_{args.bandit_type}_{args.optimize}",
        log_dir=logs_dir,
    )
    configure_logger(level="DEBUG", include_function=True)


    # Get combos for the subset
    if args.embeddings_dir:
        embeddings_dir = args.embeddings_dir
    else:
        embeddings_dir = PROJECT_ROOT / "experiments/dynamic_observation/embeddings_backgrounds" / args.subset
    available_scenarios = get_available_scenarios(embeddings_dir)
    console.print(f"[cyan]Available scenarios with embeddings:[/] {len(available_scenarios)}")

    # Get combos from database
    combos = get_combos_for_subset(args.subset)
    console.print(f"[green]Found {len(combos)} combos in database[/]")

    # Filter combos to only those with embeddings
    filtered_combos = [c for c in combos if c.pk in available_scenarios]
    console.print(f"[cyan]Combos with embeddings:[/] {len(filtered_combos)}")

    # Limit if max_episodes specified
    if args.max_episodes and args.max_episodes < len(filtered_combos):
        filtered_combos = filtered_combos[:args.max_episodes]
        console.print(f"[yellow]Limited to {args.max_episodes} episodes[/]")

    total_episodes = len(filtered_combos)
    console.print(f"[cyan]Will run {total_episodes} episodes with concurrency={args.batch_size}[/]")

    # Create bandit config with all exploration parameters
    bandit_config = BanditConfig(
        eta=args.eta,
        alpha=args.alpha,
        beta=args.beta,
        update_interval=args.update_interval,
        evolution_interval=args.evolution_interval,
        mask_unselected_scores=args.mask_unselected_scores,
        importance_weighted_reward=args.importance_weighted_reward,
        gamma=args.gamma,
        score_decay=args.score_decay,
        failure_penalty_threshold=args.failure_penalty_threshold,
        failure_penalty_factor=args.failure_penalty_factor,
    )
    run_config_path = save_run_config(
        experiment_dir=experiment_dir,
        experiment_tag=experiment_tag,
        args=args,
        bandit_config=bandit_config,
        bandit_type=args.bandit_type,
    )
    console.print(f"[cyan]Run config:[/] {run_config_path}")

    # Initialize progress tracker
    progress_tracker = BatchProgressTracker()
    progress_tracker.total = total_episodes

    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(args.batch_size)
    results: list[dict] = []

    async def run_with_semaphore(combo: EnvAgentComboStorage) -> dict:
        async with semaphore:
            return await run_single_scenario(
                scenario_id=combo.pk,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=bandit_config,
                bandit_type=args.bandit_type,
                progress_tracker=progress_tracker,
                verbose=False,  # Disable verbose output in batch mode
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
            )


    # Incremental results file for crash recovery
    incremental_results_path = experiment_dir / "results_incremental.jsonl"

    def save_incremental_result(result: dict) -> None:
        """Append a single result to incremental file (JSONL format)."""
        try:
            with open(incremental_results_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save incremental result: {e}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(f"[cyan]Running {total_episodes} episodes...", total=total_episodes)

        # Background task to update progress description
        async def update_progress_description() -> None:
            while progress_tracker.completed < progress_tracker.total:
                status = progress_tracker.get_status()
                progress.update(task, description=status)
                await asyncio.sleep(0.5)

        # Create all tasks
        tasks = [run_with_semaphore(c) for c in filtered_combos]
        update_task = asyncio.create_task(update_progress_description())

        # Process tasks as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            # Save incrementally for crash recovery
            save_incremental_result(result)
            progress.update(task, advance=1)

        # Cancel update task
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass

    # Track retry history
    retry_history: list[dict] = []
    all_results = results  # Start with initial results

    # Auto-retry logic if enabled
    if args.max_retries > 0:
        current_retry = 0

        while current_retry < args.max_retries:
            # Find failed scenarios
            failed_scenario_ids = [
                r["scenario_id"] for r in all_results
                if not r["success"]
            ]

            # If no failures, exit retry loop
            if not failed_scenario_ids:
                console.print("\n[bold green]All scenarios succeeded! No retries needed.[/]")
                break

            console.print(f"\n[yellow]Found {len(failed_scenario_ids)} failed scenarios.[/]")
            console.print(f"[cyan]Starting retry {current_retry + 1}/{args.max_retries}...[/]")

            # Retry failed scenarios
            retry_start_time = time.time()
            retry_result = await retry_failed_scenarios(
                failed_scenario_ids=failed_scenario_ids,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=bandit_config,
                bandit_type=args.bandit_type,
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
                retry_attempt=current_retry + 1,
            )
            retry_duration = time.time() - retry_start_time

            # Record retry history
            retry_history.append({
                "retry_attempt": current_retry + 1,
                "failed_count": len(failed_scenario_ids),
                "failed_ids": failed_scenario_ids,
                "newly_succeeded": retry_result["success_count"],
                "still_failed": retry_result["still_failed"],
                "still_failed_ids": retry_result["failed_ids"],
                "retry_duration_seconds": retry_duration,
                "timestamp": datetime.now().isoformat(),
            })

            # Merge results: replace failed scenarios with retry results
            retry_results_map = {
                r["scenario_id"]: r
                for r in retry_result["results"]
            }

            merged_results = []
            for r in all_results:
                if r["scenario_id"] in retry_results_map:
                    # Replace with retry result
                    merged_results.append(retry_results_map[r["scenario_id"]])
                else:
                    # Keep original result
                    merged_results.append(r)

            all_results = merged_results

            # Check if we should continue retrying
            if retry_result["still_failed"] == 0:
                console.print(f"\n[bold green]All scenarios succeeded after {current_retry + 1} retries![/]")
                break

            current_retry += 1

            if current_retry < args.max_retries:
                console.print(f"\n[yellow]Will retry {retry_result['still_failed']} scenarios in next attempt...[/]")
            else:
                console.print(f"\n[bold red]Max retries ({args.max_retries}) reached. {retry_result['still_failed']} scenarios still failed.[/]")

    # Calculate timing (include retry time if applicable)
    batch_end_time = time.time()
    total_duration = batch_end_time - batch_start_time
    avg_duration = total_duration / total_episodes if total_episodes > 0 else 0

    # Compute statistics (use all_results after retries)
    success_count = sum(1 for r in all_results if r["success"])
    error_count = len(all_results) - success_count

    # Calculate average rewards for successful episodes
    p1_rewards = [r["summary"]["p1_avg_reward"] for r in all_results if r["success"] and r["summary"]]
    p2_rewards = [r["summary"]["p2_avg_reward"] for r in all_results if r["success"] and r["summary"]]

    avg_p1_reward = sum(p1_rewards) / len(p1_rewards) if p1_rewards else 0.0
    avg_p2_reward = sum(p2_rewards) / len(p2_rewards) if p2_rewards else 0.0

    console.print(f"\n[bold green]Batch completed![/]")
    console.print(f"[green]Success: {success_count}[/]")
    console.print(f"[red]Errors: {error_count}[/]")
    console.print(f"[cyan]Total time: {format_duration(total_duration)} ({total_duration:.1f}s)[/]")
    console.print(f"[cyan]Average time per episode: {format_duration(avg_duration)} ({avg_duration:.1f}s)[/]")
    console.print(f"\n[bold]Average Rewards (across all episodes):[/]")
    console.print(f"  P1 Average: {avg_p1_reward:.2f}")
    console.print(f"  P2 Average: {avg_p2_reward:.2f}")

    # Save results to output file FIRST (before cost calculation which may fail)
    output_path = experiment_dir / "results.json"

    if output_path:
        # Convert for JSON serialization
        def convert_to_serializable(obj: object) -> object:
            if hasattr(obj, 'tolist'):
                return obj.tolist()  # type: ignore
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        batch_summary = {
            "experiment_tag": experiment_tag,
            "subset": args.subset,
            "optimize_mode": args.optimize,
            "total_episodes": total_episodes,
            "success_count": success_count,
            "error_count": error_count,
            "total_duration_seconds": total_duration,
            "avg_duration_per_episode": avg_duration,
            "avg_p1_reward": avg_p1_reward,
            "avg_p2_reward": avg_p2_reward,
            "results": convert_to_serializable(all_results),
        }

        # Add retry information if retries were performed
        if args.max_retries > 0 and retry_history:
            batch_summary["retry_info"] = {
                "max_retries_configured": args.max_retries,
                "total_retry_attempts": len(retry_history),
                "retry_history": retry_history,
                "final_failed_count": error_count,
                "final_failed_ids": [
                    r["scenario_id"] for r in all_results
                    if not r["success"]
                ],
            }

        with open(output_path, "w") as f:
            json.dump(batch_summary, f, indent=2)
        console.print(f"\n[green]Results saved to:[/] {output_path}")

    # Calculate API costs (after saving results) - only if requested
    if args.calculate_cost:
        log_path = get_llm_call_log_path()
        if log_path and log_path.exists():
            console.print(f"\n[cyan]LLM call log saved to:[/] {log_path}")
            console.print("\n[bold cyan]Calculating API costs...[/]")
            try:
                cost_info = await calculate_cost_by_model_async(log_path)
                if not cost_info:
                    cost_info = await calculate_cost_async(log_path)
                else:
                    _print_cost_breakdown(cost_info)

                # Save cost info
                cost_info_path = experiment_dir / "cost_info.json"
                with open(cost_info_path, "w") as f:
                    json.dump(cost_info, f, indent=2)
                console.print(f"[green]Cost info saved to:[/] {cost_info_path}")
            except Exception as e:
                logger.warning(f"Failed to calculate cost: {e}")
    else:
        console.print("\n[dim]Skipping cost calculation (use --calculate-cost to enable)[/]")

    # Prepare summary text
    summary_text = (
        f"[bold green]Batch Simulation Completed![/]\n"
        f"Total: {total_episodes} episodes\n"
        f"Success: {success_count}, Errors: {error_count}\n"
        f"Time: {format_duration(total_duration)}\n"
        f"Avg P1 Reward: {avg_p1_reward:.2f}, Avg P2 Reward: {avg_p2_reward:.2f}"
    )

    # Add retry summary if retries were performed
    if args.max_retries > 0 and retry_history:
        summary_text += (
            f"\n\n[bold yellow]Retry Summary:[/]\n"
            f"Total Retries: {len(retry_history)}/{args.max_retries}\n"
            f"Initially Failed: {retry_history[0]['failed_count']}\n"
            f"Finally Succeeded: {retry_history[0]['failed_count'] - error_count}\n"
            f"Still Failed: {error_count}"
        )

    console.print(Panel(summary_text, title="Batch Summary"))

    # Cleanup logging
    cleanup_terminal_logging(tee_stdout, tee_stderr)

    # Explicit cleanup of LiteLLM async clients to prevent "Event loop is closed" errors
    console.print("\n[dim]Cleaning up LiteLLM async clients...[/]")
    try:
        import litellm
        await litellm.close_litellm_async_clients()
        console.print("[dim]✓ LiteLLM async clients closed successfully[/]")
    except Exception as e:
        console.print(f"[yellow]Warning: Error closing LiteLLM clients: {e}[/]")

    # Allow pending aiohttp connections to close gracefully
    await asyncio.sleep(1.5)



async def run_resume_episodes(args: argparse.Namespace) -> None:
    """Resume a previous experiment by re-running only failed/incomplete scenarios."""
    batch_start_time = time.time()

    resume_dir = args.resume
    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    results_file = resume_dir / "results.json"
    incremental_file = resume_dir / "results_incremental.jsonl"

    # Try to load from results.json first, then fallback to incremental file
    prev_results: dict = {}
    prev_results_list: list[dict] = []

    if results_file.exists():
        # Normal case: results.json exists
        with open(results_file) as f:
            prev_results = json.load(f)
        prev_results_list = prev_results.get("results", [])
        experiment_tag = prev_results.get("experiment_tag", resume_dir.name)
    elif incremental_file.exists():
        # Crash recovery: only incremental file exists
        console.print("[yellow]No results.json found, recovering from incremental file...[/]")
        with open(incremental_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        prev_results_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed line in incremental file: {e}")
        # Try to get experiment tag from run_config.json
        run_config_file = resume_dir / "run_config.json"
        if run_config_file.exists():
            with open(run_config_file) as f:
                run_config = json.load(f)
            experiment_tag = run_config.get("experiment_tag", resume_dir.name)
        else:
            experiment_tag = resume_dir.name
        prev_results = {
            "experiment_tag": experiment_tag,
            "results": prev_results_list,
            "success_count": sum(1 for r in prev_results_list if r.get("success", False)),
            "total_episodes": len(prev_results_list),
        }
        console.print(f"[green]Recovered {len(prev_results_list)} results from incremental file[/]")
    else:
        raise FileNotFoundError(
            f"Neither results.json nor results_incremental.jsonl found in {resume_dir}"
        )

    # Get all scenario IDs that were supposed to run (from run_config or embeddings dir)
    run_config_file = resume_dir / "run_config.json"
    all_scenario_ids: set[str] = set()

    if run_config_file.exists():
        with open(run_config_file) as f:
            run_config = json.load(f)

        # Check git commit consistency
        original_git_info = run_config.get("git_info", {})
        current_git_info = get_git_info()

        # Compare git commits if both are available
        if original_git_info.get("commit_id") and current_git_info.get("commit_id"):
            original_commit = original_git_info["commit_id"]
            current_commit = current_git_info["commit_id"]
            original_dirty = original_git_info.get("dirty", False)
            current_dirty = current_git_info.get("dirty", False)

            if original_commit != current_commit or original_dirty != current_dirty:
                # Git state mismatch - warn user and ask for confirmation
                console.print("\n")
                console.print(Panel(
                    f"[bold yellow]⚠️  Git State Mismatch Detected[/]\n\n"
                    f"[bold]Original run:[/]\n"
                    f"  Commit: {original_commit[:8]}\n"
                    f"  Branch: {original_git_info.get('branch', 'unknown')}\n"
                    f"  Dirty: {'Yes' if original_dirty else 'No'}\n\n"
                    f"[bold]Current state:[/]\n"
                    f"  Commit: {current_commit[:8]}\n"
                    f"  Branch: {current_git_info.get('branch', 'unknown')}\n"
                    f"  Dirty: {'Yes' if current_dirty else 'No'}\n\n"
                    f"[red]⚠️  Resuming with different code may lead to inconsistent results![/]",
                    title="⚠️  Warning: Code Version Mismatch",
                    border_style="yellow",
                ))

                # Ask user for confirmation
                from rich.prompt import Confirm
                should_continue = Confirm.ask(
                    "\n[yellow]Do you want to continue anyway?[/]",
                    default=False,
                )

                if not should_continue:
                    console.print("\n[yellow]Resume cancelled by user.[/]")
                    logger.info("Resume cancelled due to git mismatch")
                    return
                else:
                    console.print("\n[yellow]Continuing resume with different code version...[/]")
                    logger.warning("User chose to continue resume despite git mismatch")

        # Try to get total expected from config
        original_args = run_config.get("args", {})
        embeddings_dir = original_args.get("embeddings_dir")
        if embeddings_dir:
            embeddings_path = Path(embeddings_dir)
            if embeddings_path.exists():
                all_scenario_ids = set(get_available_scenarios(embeddings_path))

    # Find completed scenarios (successful ones)
    completed_scenarios = {
        r["scenario_id"] for r in prev_results_list
        if r.get("success", False)
    }

    # Find failed scenarios (explicitly failed)
    failed_scenarios_set = {
        r["scenario_id"] for r in prev_results_list
        if not r.get("success", True)
    }

    # If we know all scenarios, find incomplete ones (never started)
    if all_scenario_ids:
        incomplete_scenarios = all_scenario_ids - completed_scenarios - failed_scenarios_set
    else:
        incomplete_scenarios = set()

    # Scenarios to resume = failed + incomplete
    scenarios_to_resume = list(failed_scenarios_set | incomplete_scenarios)

    if not scenarios_to_resume:
        console.print(Panel(
            "[bold green]No failed or incomplete scenarios to resume![/]\n"
            f"All {len(completed_scenarios)} episodes succeeded.",
            title="Resume Complete",
        ))
        return

    console.print(Panel(
        f"[bold yellow]Resume Mode[/]\n\n"
        f"Resume Directory: {resume_dir}\n"
        f"Experiment Tag: {experiment_tag}\n"
        f"[bold red]Failed Scenarios: {len(failed_scenarios_set)}[/]\n"
        f"[bold yellow]Incomplete Scenarios: {len(incomplete_scenarios)}[/]\n"
        f"[bold cyan]Total to Resume: {len(scenarios_to_resume)}[/]\n"
        f"Previous Success: {len(completed_scenarios)}\n"
        f"Batch Size: {args.batch_size}\n"
        f"Optimize Mode: {args.optimize}",
        title="Resume Configuration",
    ))

    if failed_scenarios_set:
        console.print("\n[bold red]Failed Scenario IDs:[/]")
        for sid in list(failed_scenarios_set)[:10]:
            console.print(f"  • {sid}")
        if len(failed_scenarios_set) > 10:
            console.print(f"  ... and {len(failed_scenarios_set) - 10} more")

    if incomplete_scenarios:
        console.print(f"\n[bold yellow]Incomplete Scenarios:[/] {len(incomplete_scenarios)} (not started)")

    # Use existing experiment directory
    experiment_dir = resume_dir
    logs_dir = experiment_dir / "logs"
    tensorboard_dir = experiment_dir / "tensorboard"
    llm_logs_dir = experiment_dir / "llm_logs"

    # Enable LLM call logging (append mode)
    log_file = llm_logs_dir / "calls_resume.jsonl"
    enable_llm_call_logging(log_file, experiment_tag=f"{experiment_tag}_resume")
    console.print(f"[cyan]LLM call log:[/] {log_file}")

    # Setup terminal logging
    log_file_path, tee_stdout, tee_stderr = setup_terminal_logging(
        experiment_name=f"resume_{experiment_tag}",
        log_dir=logs_dir,
    )
    configure_logger(level="DEBUG", include_function=True)

    # Load saved bandit config instead of creating from args
    run_config_file = resume_dir / "run_config.json"
    if not run_config_file.exists():
        raise FileNotFoundError(
            f"run_config.json not found in {resume_dir}. "
            f"Cannot resume without original configuration."
        )

    with open(run_config_file) as f:
        run_config = json.load(f)

    # Validate bandit type consistency
    saved_bandit_type = run_config.get("bandit_type")
    if saved_bandit_type and saved_bandit_type != args.bandit_type:
        logger.error(
            f"Bandit type mismatch: saved run used '{saved_bandit_type}', "
            f"but --bandit-type='{args.bandit_type}' was specified"
        )
        raise ValueError(
            f"Cannot resume {saved_bandit_type} run with {args.bandit_type}. "
            f"Use --bandit-type={saved_bandit_type} or start a new experiment."
        )

    # Load saved config
    saved_config = run_config.get("bandit_config", {})
    bandit_config = load_bandit_config(saved_config)

    logger.info(f"Loaded saved config: {type(bandit_config).__name__}")
    logger.info(f"Config details: {bandit_config}")

    # Save resume run config for tracking
    resume_config_name = (
        f"run_config_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if (experiment_dir / "run_config.json").exists()
        else "run_config.json"
    )
    save_run_config(
        experiment_dir=experiment_dir,
        experiment_tag=experiment_tag,
        args=args,
        bandit_config=bandit_config,
        bandit_type=args.bandit_type,
        filename=resume_config_name,
    )

    # Initialize progress tracker
    progress_tracker = BatchProgressTracker()
    progress_tracker.total = len(scenarios_to_resume)

    # Use semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.batch_size)
    resume_results: list[dict] = []

    async def run_with_semaphore(scenario_id: str) -> dict:
        async with semaphore:
            return await run_single_scenario(
                scenario_id=scenario_id,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=bandit_config,
                bandit_type=args.bandit_type,
                progress_tracker=progress_tracker,
                verbose=False,
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
            )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Re-running {len(scenarios_to_resume)} scenarios...",
            total=len(scenarios_to_resume)
        )

        async def update_progress_description() -> None:
            while progress_tracker.completed < progress_tracker.total:
                status = progress_tracker.get_status()
                progress.update(task, description=status)
                await asyncio.sleep(0.5)

        tasks = [run_with_semaphore(sid) for sid in scenarios_to_resume]
        update_task = asyncio.create_task(update_progress_description())

        for coro in asyncio.as_completed(tasks):
            result = await coro
            resume_results.append(result)
            progress.update(task, advance=1)

        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass

    # Track retry history for resume mode
    retry_history: list[dict] = []
    all_resume_results = resume_results  # Start with initial resume results

    # Auto-retry logic if enabled
    if args.max_retries > 0:
        current_retry = 0

        while current_retry < args.max_retries:
            # Find failed scenarios from resume results
            failed_scenario_ids = [
                r["scenario_id"] for r in all_resume_results
                if not r["success"]
            ]

            # If no failures, exit retry loop
            if not failed_scenario_ids:
                console.print("\n[bold green]All resumed scenarios succeeded! No retries needed.[/]")
                break

            console.print(f"\n[yellow]Found {len(failed_scenario_ids)} failed scenarios in resume.[/]")
            console.print(f"[cyan]Starting retry {current_retry + 1}/{args.max_retries}...[/]")

            # Retry failed scenarios
            retry_start_time = time.time()
            retry_result = await retry_failed_scenarios(
                failed_scenario_ids=failed_scenario_ids,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=bandit_config,
                bandit_type=args.bandit_type,
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
                retry_attempt=current_retry + 1,
            )
            retry_duration = time.time() - retry_start_time

            # Record retry history
            retry_history.append({
                "retry_attempt": current_retry + 1,
                "failed_count": len(failed_scenario_ids),
                "failed_ids": failed_scenario_ids,
                "newly_succeeded": retry_result["success_count"],
                "still_failed": retry_result["still_failed"],
                "still_failed_ids": retry_result["failed_ids"],
                "retry_duration_seconds": retry_duration,
                "timestamp": datetime.now().isoformat(),
            })

            # Merge results: replace failed scenarios with retry results
            retry_results_map = {
                r["scenario_id"]: r
                for r in retry_result["results"]
            }

            merged_retry_results = []
            for r in all_resume_results:
                if r["scenario_id"] in retry_results_map:
                    # Replace with retry result
                    merged_retry_results.append(retry_results_map[r["scenario_id"]])
                else:
                    # Keep original result
                    merged_retry_results.append(r)

            all_resume_results = merged_retry_results

            # Check if we should continue retrying
            if retry_result["still_failed"] == 0:
                console.print(f"\n[bold green]All resumed scenarios succeeded after {current_retry + 1} retries![/]")
                break

            current_retry += 1

            if current_retry < args.max_retries:
                console.print(f"\n[yellow]Will retry {retry_result['still_failed']} scenarios in next attempt...[/]")
            else:
                console.print(f"\n[bold red]Max retries ({args.max_retries}) reached. {retry_result['still_failed']} scenarios still failed.[/]")

    # Calculate statistics for resumed runs (use all_resume_results after retries)
    batch_end_time = time.time()
    total_duration = batch_end_time - batch_start_time

    resume_success = sum(1 for r in all_resume_results if r["success"])
    resume_error = len(all_resume_results) - resume_success

    console.print(f"\n[bold green]Resume completed![/]")
    console.print(f"[green]Newly Succeeded: {resume_success}[/]")
    console.print(f"[red]Still Failed: {resume_error}[/]")
    console.print(f"[cyan]Resume Duration: {format_duration(total_duration)}[/]")

    # Add retry summary if retries were performed
    if args.max_retries > 0 and retry_history:
        console.print(f"\n[bold yellow]Retry Summary:[/]")
        console.print(f"Total Retries: {len(retry_history)}/{args.max_retries}")
        console.print(f"Initially Failed: {retry_history[0]['failed_count']}")
        console.print(f"Finally Succeeded: {retry_history[0]['failed_count'] - resume_error}")
        console.print(f"Still Failed: {resume_error}")

    # Merge results: replace failed scenarios with new results
    prev_results_list = prev_results.get("results", [])

    # Keep successful results from previous run
    merged_results = [r for r in prev_results_list if r.get("success", True)]

    # Add new results (use all_resume_results which includes retries)
    merged_results.extend(all_resume_results)

    # Recalculate statistics
    merged_success = sum(1 for r in merged_results if r["success"])
    merged_error = len(merged_results) - merged_success

    # Calculate average rewards
    p1_rewards = [r["summary"]["p1_avg_reward"] for r in merged_results if r["success"] and r.get("summary")]
    p2_rewards = [r["summary"]["p2_avg_reward"] for r in merged_results if r["success"] and r.get("summary")]
    avg_p1_reward = sum(p1_rewards) / len(p1_rewards) if p1_rewards else 0.0
    avg_p2_reward = sum(p2_rewards) / len(p2_rewards) if p2_rewards else 0.0

    # Update and save merged results
    def convert_to_serializable(obj: object) -> object:
        if hasattr(obj, 'tolist'):
            return obj.tolist()  # type: ignore
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    merged_summary = {
        "experiment_tag": experiment_tag,
        "subset": prev_results.get("subset", args.subset),
        "optimize_mode": prev_results.get("optimize_mode", args.optimize),
        "total_episodes": len(merged_results),
        "success_count": merged_success,
        "error_count": merged_error,
        "total_duration_seconds": prev_results.get("total_duration_seconds", 0) + total_duration,
        "avg_duration_per_episode": (prev_results.get("total_duration_seconds", 0) + total_duration) / len(merged_results),
        "avg_p1_reward": avg_p1_reward,
        "avg_p2_reward": avg_p2_reward,
        "resume_info": {
            "resumed_at": datetime.now().isoformat(),
            "previously_failed": len(failed_scenarios_set),
            "previously_incomplete": len(incomplete_scenarios),
            "total_resumed": len(scenarios_to_resume),
            "newly_succeeded": resume_success,
            "still_failed": resume_error,
            "resume_duration_seconds": total_duration,
        },
        "results": convert_to_serializable(merged_results),
    }

    with open(results_file, "w") as f:
        json.dump(merged_summary, f, indent=2)
    console.print(f"\n[green]Merged results saved to:[/] {results_file}")

    # Calculate API costs (after saving results) - only if requested
    if args.calculate_cost:
        log_path = get_llm_call_log_path()
        if log_path and log_path.exists():
            console.print(f"\n[cyan]LLM call log saved to:[/] {log_path}")
            console.print("\n[bold cyan]Calculating API costs...[/]")
            try:
                cost_info = await calculate_cost_by_model_async(log_path)
                if not cost_info:
                    cost_info = await calculate_cost_async(log_path)
                else:
                    _print_cost_breakdown(cost_info)

                # Save cost info
                cost_info_path = experiment_dir / "cost_info.json"
                with open(cost_info_path, "w") as f:
                    json.dump(cost_info, f, indent=2)
                console.print(f"[green]Cost info saved to:[/] {cost_info_path}")
            except Exception as e:
                logger.warning(f"Failed to calculate cost: {e}")
    else:
        console.print("\n[dim]Skipping cost calculation (use --calculate-cost to enable)[/]")

    console.print(Panel(
        f"[bold green]Resume Complete![/]\n"
        f"[bold]Final Statistics:[/]\n"
        f"Total Episodes: {len(merged_results)}\n"
        f"Success: {merged_success}, Errors: {merged_error}\n"
        f"[dim]Previously failed: {len(failed_scenarios_set)}, Now succeeded: {resume_success}[/]\n"
        f"Avg P1 Reward: {avg_p1_reward:.2f}, Avg P2 Reward: {avg_p2_reward:.2f}",
        title="Resume Summary",
    ))

    cleanup_terminal_logging(tee_stdout, tee_stderr)

    # Explicit cleanup of LiteLLM async clients to prevent "Event loop is closed" errors
    console.print("\n[dim]Cleaning up LiteLLM async clients...[/]")
    try:
        import litellm
        await litellm.close_litellm_async_clients()
        console.print("[dim]✓ LiteLLM async clients closed successfully[/]")
    except Exception as e:
        console.print(f"[yellow]Warning: Error closing LiteLLM clients: {e}[/]")

    # Allow pending aiohttp connections to close gracefully
    await asyncio.sleep(1.5)

