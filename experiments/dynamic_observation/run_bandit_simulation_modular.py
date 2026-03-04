#!/usr/bin/env python3
"""
Bandit-Based Dynamic Prompt Optimization - Entry Point

This script runs simulations with bandit algorithms for dynamic prompt optimization
in multi-agent social scenarios.

Supports:
- Single scenario mode: Run one scenario with optional optimization
- Batch mode: Run multiple scenarios in parallel
- Resume mode: Continue failed scenarios from previous runs

Examples:
    # Single scenario with optimization
    python run_bandit_simulation_context.py --scenario-id 01H7VKQHT745XAP1A4DDV8H419 --optimize both

    # Batch mode with parallelism
    python run_bandit_simulation_context.py --batch --subset hard --batch-size 5 --optimize both

    # Resume failed scenarios
    python run_bandit_simulation_context.py --resume outputs/experiment_tag
"""

import asyncio
import sys
import traceback
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from sotopia.generation_utils import enable_llm_call_logging, get_llm_call_log_path

# Import from lib modules
from experiments.dynamic_observation.lib import (
    parse_args,
    run_single_scenario,
    run_batch_episodes,
    run_resume_episodes,
    display_summary,
    calculate_cost_by_model_async,
    print_cost_breakdown,
    to_jsonable,
    save_run_config,
)
from experiments.dynamic_observation.core.bandits import BanditConfig
from experiments.dynamic_observation.core.bandits.neural_evolution_bandit import NeuralEvolutionConfig
from experiments.dynamic_observation.core.bandits.prompt_breeder_bandit import PromptBreederConfig
from experiments.dynamic_observation.core.bandits.opro_bandit import OPROConfig
from experiments.dynamic_observation.core.bandits.evoprompt_bandit import EvoPromptConfig
from experiments.dynamic_observation.core.logging_utils import (
    configure_logger,
    setup_terminal_logging,
    cleanup_terminal_logging,
)

# Get PROJECT_ROOT
PROJECT_ROOT = Path(__file__).parent.parent.parent

console = Console()


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Log max_tokens if specified
    if args.max_tokens:
        logger.info(f"Using max_tokens limit: {args.max_tokens}")

    # Check if running in resume mode
    if args.resume:
        await run_resume_episodes(args)
        return

    # Check if running in batch mode
    if args.batch:
        await run_batch_episodes(args)
        return

    # ====== Single Scenario Mode ======

    # Resolve model names for P1 and P2
    p1_model = args.p1_model or args.model
    p2_model = args.p2_model or args.model

    # Generate experiment tag
    if args.tag:
        experiment_tag = args.tag
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mask_tag = f"mus{int(args.mask_unselected_scores)}"
        iw_tag = f"iw{int(args.importance_weighted_reward)}"

        def get_model_short(model: str) -> str:
            name = model.split("/")[-1]
            return name.replace(".", "").replace("-", "_")[:20]

        p1_short = get_model_short(p1_model)
        p2_short = get_model_short(p2_model)
        model_tag = p1_short if p1_short == p2_short else f"{p1_short}_vs_{p2_short}"
        experiment_tag = f"bandit_{args.bandit_type}_{args.optimize}_{model_tag}_{mask_tag}_{iw_tag}_{timestamp}"

    # Mode description
    mode_desc = {
        "both": "Optimizing BOTH agents",
        "p1": "Optimizing P1 only",
        "p2": "Optimizing P2 only",
        "none": "BASELINE (no optimization)",
    }.get(args.optimize, "Unknown")

    console.print(Panel(
        f"[bold green]Bandit-Based Dynamic Prompt Optimization[/]\n\n"
        f"Scenario: {args.scenario_id}\n"
        f"P1 Model: {p1_model}\n"
        f"P2 Model: {p2_model}\n"
        f"Env Model: {args.env_model}\n"
        f"Max Turns: {args.max_turns}\n"
        f"[bold cyan]Bandit Type: {args.bandit_type}[/]\n"
        f"ETA (EXP3): {args.eta}, Alpha (LinUCB): {args.alpha}, Beta (NeuralUCB): {args.beta}\n"
        f"[bold yellow]Mode: {mode_desc}[/]\n"
        f"Push to DB: {args.push_to_db}",
        title="Configuration",
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

    # Enable LiteLLM shared async client to reduce SSL connection overhead
    try:
        import litellm
        litellm.enable_shared_async_client = True
    except ImportError:
        pass

    # Setup terminal logging to file in experiment dir
    log_file_path, tee_stdout, tee_stderr = setup_terminal_logging(
        experiment_name=f"bandit_{args.bandit_type}_{args.optimize}",
        log_dir=logs_dir,
    )
    configure_logger(level="DEBUG", include_function=True)

    # Create bandit config based on bandit type
    base_config_kwargs = {
        "eta": args.eta,
        "alpha": args.alpha,
        "beta": args.beta,
        "update_interval": args.update_interval,
        "evolution_interval": args.evolution_interval,
        "mask_unselected_scores": args.mask_unselected_scores,
        "importance_weighted_reward": args.importance_weighted_reward,
        "gamma": args.gamma,
        "score_decay": args.score_decay,
        "failure_penalty_threshold": args.failure_penalty_threshold,
        "failure_penalty_factor": args.failure_penalty_factor,
    }

    # Select config class based on bandit type
    if args.bandit_type == "neural_evolution":
        bandit_config = NeuralEvolutionConfig(**base_config_kwargs)
    elif args.bandit_type == "prompt_breeder":
        bandit_config = PromptBreederConfig(**base_config_kwargs)
    elif args.bandit_type == "opro":
        bandit_config = OPROConfig(**base_config_kwargs)
    elif args.bandit_type == "evoprompt":
        bandit_config = EvoPromptConfig(**base_config_kwargs)
    else:
        # Default to BanditConfig for other types (exp3, neural_ucb, etc.)
        bandit_config = BanditConfig(**base_config_kwargs)

    run_config_path = save_run_config(
        experiment_dir=experiment_dir,
        experiment_tag=experiment_tag,
        args=args,
        bandit_config=bandit_config,
        bandit_type=args.bandit_type,
    )
    console.print(f"[cyan]Run config:[/] {run_config_path}")

    # Run single scenario
    result = await run_single_scenario(
        scenario_id=args.scenario_id,
        args=args,
        experiment_tag=experiment_tag,
        bandit_config=bandit_config,
        bandit_type=args.bandit_type,
        progress_tracker=None,
        verbose=True,
        experiment_dir=experiment_dir,
        tensorboard_dir=tensorboard_dir,
    )

    if not result["success"]:
        console.print(f"[bold red]Simulation failed:[/] {result['error']}")
        cleanup_terminal_logging(tee_stdout, tee_stderr)
        sys.exit(1)

    summary = result["summary"]

    # Display summary
    display_summary(summary, console)

    # Save results
    output_path = experiment_dir / "results.json"
    serializable_summary = to_jsonable(summary)

    import json
    with open(output_path, "w") as f:
        json.dump(serializable_summary, f, indent=2)
    console.print(f"\n[green]Results saved to:[/] {output_path}")

    # Calculate and display cost (after saving results) - only if requested
    if args.calculate_cost:
        log_path = get_llm_call_log_path()
        if log_path and log_path.exists():
            console.print(f"\n[cyan]LLM call log saved to:[/] {log_path}")
            console.print("\n[bold cyan]Calculating API costs...[/]")
            try:
                cost_info = await calculate_cost_by_model_async(log_path)
                if cost_info:
                    print_cost_breakdown(cost_info, console)

                    # Save cost info
                    cost_info_path = experiment_dir / "cost_info.json"
                    with open(cost_info_path, "w") as f:
                        json.dump(cost_info, f, indent=2)
                    console.print(f"[green]Cost info saved to:[/] {cost_info_path}")
            except Exception as e:
                logger.warning(f"Failed to calculate cost: {e}")
    else:
        console.print("\n[dim]Skipping cost calculation (use --calculate-cost to enable)[/]")

    # Cleanup: restore original streams
    cleanup_terminal_logging(tee_stdout, tee_stderr)

    # Explicit cleanup of LiteLLM async clients to prevent "Event loop is closed" errors
    console.print("\n[dim]Cleaning up LiteLLM async clients...[/]")
    try:
        import litellm
        await litellm.close_litellm_async_clients()
        console.print("[dim]✓ LiteLLM async clients closed successfully[/]")
    except Exception as e:
        console.print(f"[yellow]Warning: Error closing LiteLLM clients: {e}[/]")

    # Allow pending async operations to complete with progress indication
    console.print("[dim]Finalizing async cleanup...[/]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=False,
    ) as progress:
        task = progress.add_task("Waiting for async cleanup...", total=100)
        for i in range(10):
            await asyncio.sleep(0.15)  # Total 1.5s (increased from 0.5s)
            progress.update(task, completed=(i + 1) * 10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
