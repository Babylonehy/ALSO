#!/usr/bin/env python3
"""
Agent-Pair Generalization Experiment.

This script evaluates whether strategies learned with some agent pairs
can transfer to different agent pairs within the SAME scenarios.

Key difference from run_generalization_experiment.py:
- Cross-scenario: Different scenarios for train vs test
- Agent-pair: Same scenarios, different agent pairs for train vs test

This provides a more controlled evaluation of strategy transfer since
the scenario context (goals, constraints) remains the same.

Experimental Design:
1. For each scenario, split 5 agent pairs into 3 train + 2 test
2. Train bandit on all train pairs (with learning)
3. Test trained model on all test pairs (frozen policy)
4. Baseline: Online learning on test pairs without pretraining

Usage:
    # Full experiment (train + test with frozen policy)
    python run_agent_pair_generalization.py --phase both --split A

    # Baseline (no pretraining, online learning on test set)
    python run_agent_pair_generalization.py --phase baseline --split A
"""

import argparse
import asyncio
import json
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env (must be before importing sotopia.database)
load_dotenv(PROJECT_ROOT / ".env")

from experiments.dynamic_observation.lib.agent_pair_splits import (
    get_train_combo_ids,
    get_test_combo_ids,
)

console = Console()


def save_bandit_state(bandit: Any, save_dir: Path) -> None:
    """Save bandit state (model weights + selection history) to directory."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    if hasattr(bandit, "model"):
        torch.save(bandit.model.state_dict(), save_dir / "model_weights.pt")
        logger.info(f"Saved model weights: {save_dir / 'model_weights.pt'}")

    # Save optimizer state (for resuming training)
    if hasattr(bandit, "optimizer"):
        torch.save(bandit.optimizer.state_dict(), save_dir / "optimizer_state.pt")
        logger.info(f"Saved optimizer state: {save_dir / 'optimizer_state.pt'}")

    # Save score traces
    state_data = {
        "score_traces": dict(getattr(bandit, "score_traces", {})),
        "actual_score_traces": dict(getattr(bandit, "actual_score_traces", {})),
        "current_selections": dict(getattr(bandit, "current_selections", {"p1": 0, "p2": 0})),
    }
    with open(save_dir / "state_data.json", "w") as f:
        json.dump(state_data, f, indent=2)

    # Save selection history (full details)
    history_data = []
    for rec in getattr(bandit, "selection_history", []):
        history_data.append({
            "turn": rec.turn,
            "agent": rec.agent,
            "arm_index": rec.arm_index,
            "prompt_text": rec.prompt_text,
            "embedding": rec.embedding.tolist() if rec.embedding is not None else None,
            "reward": rec.reward,
            "cumulative_score": rec.cumulative_score,
            "selection_probability": rec.selection_probability,
        })
    with open(save_dir / "selection_history.pkl", "wb") as f:
        pickle.dump(history_data, f)

    # Also save as JSON for easier inspection
    with open(save_dir / "selection_history.json", "w") as f:
        # Remove embedding from JSON version (too large)
        history_json = [{k: v for k, v in rec.items() if k != "embedding"} for rec in history_data]
        json.dump(history_json, f, indent=2)

    logger.info(f"Bandit state saved to {save_dir} ({len(history_data)} selections)")


def load_bandit_state(bandit: Any, load_dir: Path, freeze: bool = True) -> None:
    """Load bandit state from directory and optionally freeze the model."""
    if not load_dir.exists():
        raise FileNotFoundError(f"State directory not found: {load_dir}")
    
    # Load model weights
    model_path = load_dir / "model_weights.pt"
    if model_path.exists() and hasattr(bandit, "model"):
        device = getattr(bandit.config, "device", "cpu")
        bandit.model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Model weights loaded from {model_path}")
    
    # Load score traces
    state_path = load_dir / "state_data.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            state_data = json.load(f)
        bandit.score_traces = state_data.get("score_traces", {})
        bandit.actual_score_traces = state_data.get("actual_score_traces", {})
        bandit.current_selections = state_data.get("current_selections", {"p1": 0, "p2": 0})
    
    if freeze and hasattr(bandit, "model"):
        bandit.model.eval()
        bandit._frozen = True
        logger.info("Model frozen for evaluation")
    
    logger.info(f"Bandit state loaded from {load_dir}")


class FrozenBanditWrapper:
    """Wrapper that intercepts update calls when bandit is frozen."""

    def __init__(self, bandit: Any):
        self._bandit = bandit
        self._frozen = True

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._bandit, name)

    def update(self, agent: str, arm_index: int, reward: float, turn: int) -> None:
        """Intercept update calls - log but don't actually update."""
        logger.debug(f"[FROZEN] Ignoring update for {agent} arm {arm_index} reward={reward:.2f}")

    def update_with_context(self, agent: str, arm_index: int, reward: float,
                            turn: int, context_embedding: Any = None) -> None:
        """Intercept context update calls - log but don't actually update."""
        logger.debug(f"[FROZEN] Ignoring context update for {agent} arm {arm_index}")

    def select(self, agent: str, turn: int) -> tuple:
        """Use the underlying bandit's selection (uses learned policy)."""
        return self._bandit.select(agent, turn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent-pair generalization experiment for bandit models"
    )

    # Experiment mode
    parser.add_argument(
        "--phase", type=str, choices=["train", "test", "both", "baseline", "finetune"], default="both",
        help="Phase to run: train, test, both, baseline (cold start), or finetune (warm start)"
    )
    parser.add_argument(
        "--split", type=str, choices=["A", "B"], default="A",
        help="Scenario split (A or B)"
    )
    parser.add_argument(
        "--train-ratio", type=int, default=3,
        help="Number of agent pairs per scenario for training"
    )
    parser.add_argument(
        "--test-ratio", type=int, default=2,
        help="Number of agent pairs per scenario for testing"
    )

    # Model settings
    parser.add_argument(
        "--model", type=str, default="openrouter/qwen/qwen-2.5-72b-instruct",
        help="LLM model for agent simulation"
    )
    parser.add_argument(
        "--reward-eval-model", type=str, default="openrouter/deepseek/deepseek-v3.2",
        help="Model for reward evaluation"
    )

    # Bandit settings
    parser.add_argument("--bandit-type", type=str, default="adversarial", help="Bandit algorithm type")
    parser.add_argument("--eta", type=float, default=10.0, help="EXP3 exploration parameter")
    parser.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration parameter")
    parser.add_argument("--beta", type=float, default=1.0, help="UCB exploration parameter")
    parser.add_argument("--epochs", type=int, default=100, help="Neural bandit training epochs")
    parser.add_argument("--update-interval", type=int, default=1, help="Bandit update interval")
    parser.add_argument("--evolution-interval", type=int, default=5, help="Evolution interval")
    parser.add_argument("--mask-unselected-scores", action="store_true", default=False)
    parser.add_argument("--importance-weighted-reward", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.1, help="Discount factor")
    parser.add_argument("--score-decay", type=float, default=0.9, help="Score decay factor")
    parser.add_argument("--dynamic-eta", action="store_true", default=False)
    parser.add_argument("--cumulative-score-mode", type=str, default="nn")
    parser.add_argument("--failure-penalty-threshold", type=float, default=0.3)
    parser.add_argument("--failure-penalty-factor", type=float, default=1.5)
    parser.add_argument("--multi-dim-prediction", action="store_true", default=False)
    parser.add_argument("--selection-strategy", type=str, default=None)
    parser.add_argument("--selection-epsilon", type=float, default=0.1)
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--strategy-version", type=str, default="v3")

    # Simulation settings
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum turns per episode")
    parser.add_argument("--optimize", type=str, choices=["p1", "p2", "both"], default="both")

    # Output settings
    parser.add_argument("--output-dir", type=Path, default=Path("results/generalization_agent_pair"))
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory with trained model")
    parser.add_argument("--push-to-db", action="store_true", help="Push results to database")
    parser.add_argument("--strategy-cache-dir", type=Path, default=None)
    parser.add_argument("--suffix", type=str, default=None, help="Custom suffix for experiment name")

    return parser.parse_args()


async def run_scenarios(
    combo_ids: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    phase: str,
    pretrained_state_dir: Path | None = None,
) -> dict[str, Any]:
    """Run bandit simulation on a list of combo IDs (agent pairs)."""
    from experiments.dynamic_observation.run_bandit_simulation_context import (
        BanditSimulationRunner,
        create_bandit_config_from_args,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config_data = {
        "phase": phase,
        "split": args.split,
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
        "model": args.model,
        "bandit_type": args.bandit_type,
        "optimize": args.optimize,
        "max_turns": args.max_turns,
        "combo_ids": combo_ids,
        "pretrained_state_dir": str(pretrained_state_dir) if pretrained_state_dir else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    results = []
    all_rewards_p1 = []
    all_rewards_p2 = []

    # For training: accumulate model weights across combos
    accumulated_weights_p1: Path | None = None
    accumulated_weights_p2: Path | None = None

    for i, combo_id in enumerate(combo_ids):
        console.print(f"\n[bold cyan]{'='*60}[/]")
        console.print(f"[bold cyan][{phase.upper()}] Combo {i+1}/{len(combo_ids)}: {combo_id}[/]")
        console.print(f"[bold cyan]{'='*60}[/]")

        try:
            # Create runner for this combo
            runner = BanditSimulationRunner(
                scenario_id=combo_id,
                model_name=args.model,
                env_model_name=args.model,
                reward_eval_model_name=args.reward_eval_model,
                max_turns=args.max_turns,
                bandit_config=create_bandit_config_from_args(args),
                bandit_type=args.bandit_type,
                optimize_mode=args.optimize,
                push_to_db=args.push_to_db,
                experiment_tag=f"agent_pair_gen_{phase}_split{args.split}",
                verbose=True,
                experiment_dir=output_dir,
                selection_mode="strategy",
                strategy_cache_dir=args.strategy_cache_dir,
                strategy_version=args.strategy_version,
            )

            # Load weights based on phase
            if phase == "train" and i > 0:
                # Training: load accumulated weights from previous combos
                if accumulated_weights_p1 and runner.bandit_p1:
                    load_bandit_state(runner.bandit_p1, accumulated_weights_p1, freeze=False)
                    logger.info(f"Loaded accumulated P1 weights from combo {i}")
                if accumulated_weights_p2 and runner.bandit_p2:
                    load_bandit_state(runner.bandit_p2, accumulated_weights_p2, freeze=False)
                    logger.info(f"Loaded accumulated P2 weights from combo {i}")

            elif phase == "test" and pretrained_state_dir:
                # Testing: load pre-trained weights and freeze
                if runner.bandit_p1:
                    load_bandit_state(
                        runner.bandit_p1,
                        pretrained_state_dir / "p1_bandit",
                        freeze=True
                    )
                    runner.bandit_p1 = FrozenBanditWrapper(runner.bandit_p1)
                    logger.info("Loaded and froze P1 bandit for testing")
                if runner.bandit_p2:
                    load_bandit_state(
                        runner.bandit_p2,
                        pretrained_state_dir / "p2_bandit",
                        freeze=True
                    )
                    runner.bandit_p2 = FrozenBanditWrapper(runner.bandit_p2)
                    logger.info("Loaded and froze P2 bandit for testing")

            elif phase == "finetune" and pretrained_state_dir:
                # Finetune: load pre-trained weights but DON'T freeze
                if i == 0:
                    if runner.bandit_p1:
                        load_bandit_state(
                            runner.bandit_p1,
                            pretrained_state_dir / "p1_bandit",
                            freeze=False
                        )
                        logger.info("Loaded P1 pretrained weights for finetuning")
                    if runner.bandit_p2:
                        load_bandit_state(
                            runner.bandit_p2,
                            pretrained_state_dir / "p2_bandit",
                            freeze=False
                        )
                        logger.info("Loaded P2 pretrained weights for finetuning")
                else:
                    if accumulated_weights_p1 and runner.bandit_p1:
                        load_bandit_state(runner.bandit_p1, accumulated_weights_p1, freeze=False)
                    if accumulated_weights_p2 and runner.bandit_p2:
                        load_bandit_state(runner.bandit_p2, accumulated_weights_p2, freeze=False)

            # Run episode
            episode_result = await runner.run_episode()
            results.append({
                "combo_id": combo_id,
                "result": episode_result,
            })

            # Collect rewards
            if "final_rewards" in episode_result:
                if episode_result["final_rewards"].get("p1"):
                    p1_data = episode_result["final_rewards"]["p1"]
                    p1_goal = p1_data.get("breakdown", {}).get("goal", p1_data.get("goal", 0))
                    all_rewards_p1.append(p1_goal)
                if episode_result["final_rewards"].get("p2"):
                    p2_data = episode_result["final_rewards"]["p2"]
                    p2_goal = p2_data.get("breakdown", {}).get("goal", p2_data.get("goal", 0))
                    all_rewards_p2.append(p2_goal)

            # Save per-combo result
            combo_results_dir = output_dir / "combos"
            combo_results_dir.mkdir(parents=True, exist_ok=True)
            with open(combo_results_dir / f"{combo_id}.json", "w") as f:
                json.dump({
                    "combo_id": combo_id,
                    "combo_index": i,
                    "phase": phase,
                    "episode_result": episode_result,
                }, f, indent=2, default=str)

            # Training/Finetune: save accumulated weights after each combo
            if phase in ["train", "finetune"]:
                if runner.bandit_p1:
                    bandit_to_save = getattr(runner.bandit_p1, "_bandit", runner.bandit_p1)
                    save_dir = output_dir / "p1_bandit"
                    save_bandit_state(bandit_to_save, save_dir)
                    accumulated_weights_p1 = save_dir
                if runner.bandit_p2:
                    bandit_to_save = getattr(runner.bandit_p2, "_bandit", runner.bandit_p2)
                    save_dir = output_dir / "p2_bandit"
                    save_bandit_state(bandit_to_save, save_dir)
                    accumulated_weights_p2 = save_dir

            console.print(f"[green]Combo {combo_id} completed[/]")

        except Exception as e:
            console.print(f"[red]Error in combo {combo_id}: {e}[/]")
            traceback.print_exc()
            results.append({
                "combo_id": combo_id,
                "error": str(e),
            })

    # Compute summary statistics
    summary = {
        "phase": phase,
        "split": args.split,
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
        "n_combos": len(combo_ids),
        "combo_ids": combo_ids,
        "mean_reward_p1": float(np.mean(all_rewards_p1)) if all_rewards_p1 else None,
        "mean_reward_p2": float(np.mean(all_rewards_p2)) if all_rewards_p2 else None,
        "std_reward_p1": float(np.std(all_rewards_p1)) if all_rewards_p1 else None,
        "std_reward_p2": float(np.std(all_rewards_p2)) if all_rewards_p2 else None,
        "all_rewards_p1": all_rewards_p1,
        "all_rewards_p2": all_rewards_p2,
        "results": results,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def print_comparison_table(train_summary: dict | None, test_summary: dict | None) -> None:
    """Print a comparison table of train vs test performance."""
    table = Table(title="Agent-Pair Generalization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Train", style="green")
    table.add_column("Test", style="yellow")
    table.add_column("Gap", style="red")

    if train_summary and test_summary:
        # P1 Goal
        train_p1 = train_summary.get("mean_reward_p1")
        test_p1 = test_summary.get("mean_reward_p1")
        gap_p1 = (test_p1 - train_p1) if (train_p1 and test_p1) else None
        table.add_row(
            "P1 Mean Goal",
            f"{train_p1:.3f}" if train_p1 else "N/A",
            f"{test_p1:.3f}" if test_p1 else "N/A",
            f"{gap_p1:+.3f}" if gap_p1 else "N/A"
        )

        # P2 Goal
        train_p2 = train_summary.get("mean_reward_p2")
        test_p2 = test_summary.get("mean_reward_p2")
        gap_p2 = (test_p2 - train_p2) if (train_p2 and test_p2) else None
        table.add_row(
            "P2 Mean Goal",
            f"{train_p2:.3f}" if train_p2 else "N/A",
            f"{test_p2:.3f}" if test_p2 else "N/A",
            f"{gap_p2:+.3f}" if gap_p2 else "N/A"
        )

        # N combos
        table.add_row(
            "N Combos",
            str(train_summary.get("n_combos", 0)),
            str(test_summary.get("n_combos", 0)),
            ""
        )

    console.print(table)


async def main():
    args = parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.phase == "baseline":
        phase_suffix = "baseline"
    elif args.phase == "finetune":
        phase_suffix = "finetune"
    else:
        phase_suffix = f"split{args.split}"
    exp_name = f"{args.bandit_type}_agent_pair_{phase_suffix}_{timestamp}"
    if args.suffix:
        exp_name = f"{exp_name}_{args.suffix}"
    output_dir = args.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = vars(args).copy()
    config["timestamp"] = timestamp
    config["output_dir"] = str(output_dir)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Get agent-pair splits
    train_combo_ids = get_train_combo_ids(args.split, args.train_ratio, args.test_ratio)
    test_combo_ids = get_test_combo_ids(args.split, args.train_ratio, args.test_ratio)

    # Baseline mode: online learning on test set without pretraining
    if args.phase == "baseline":
        console.print(Panel(
            f"[bold]Baseline Experiment (No Pretraining)[/]\n\n"
            f"Split: {args.split} (agent-pair split)\n"
            f"Test combos (online learning): {len(test_combo_ids)}\n"
            f"Split ratio: {args.train_ratio} train / {args.test_ratio} test per scenario\n"
            f"Bandit: {args.bandit_type}\n"
            f"Model: {args.model}\n"
            f"Output: {output_dir}\n\n"
            f"[yellow]This baseline runs online learning on test agent pairs WITHOUT pretraining.[/]",
            title="Baseline Configuration"
        ))

        console.print("\n[bold magenta]" + "="*60 + "[/]")
        console.print("[bold magenta]BASELINE: Online Learning on Test Agent Pairs[/]")
        console.print("[bold magenta]" + "="*60 + "[/]")

        baseline_summary = await run_scenarios(
            combo_ids=test_combo_ids,
            args=args,
            output_dir=output_dir / "baseline",
            phase="train",  # Use "train" phase logic (learning enabled)
        )

        console.print(f"\n[green]Baseline complete![/]")
        console.print(f"Mean P1 Goal: {baseline_summary.get('mean_reward_p1', 'N/A')}")
        console.print(f"Mean P2 Goal: {baseline_summary.get('mean_reward_p2', 'N/A')}")

        # Save final summary
        final_summary = {
            "experiment": exp_name,
            "config": config,
            "baseline": baseline_summary,
        }
        with open(output_dir / "final_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2, default=str)

        console.print(f"\n[bold green]Baseline experiment complete![/]")
        console.print(f"Results saved to: {output_dir}")
        return

    # Finetune mode
    if args.phase == "finetune":
        if not args.model_dir:
            console.print("[red]Error: --model-dir is required for finetune mode[/]")
            return

        model_dir = args.model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        console.print(Panel(
            f"[bold]Finetune Experiment[/]\n\n"
            f"Split: {args.split} (agent-pair split)\n"
            f"Test combos (with continued learning): {len(test_combo_ids)}\n"
            f"Pretrained model: {model_dir}\n",
            title="Finetune Configuration"
        ))

        finetune_summary = await run_scenarios(
            combo_ids=test_combo_ids,
            args=args,
            output_dir=output_dir / "finetune",
            phase="finetune",
            pretrained_state_dir=model_dir,
        )

        final_summary = {
            "experiment": exp_name,
            "config": config,
            "finetune": finetune_summary,
            "pretrained_model_dir": str(model_dir),
        }
        with open(output_dir / "final_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2, default=str)

        console.print(f"\n[bold green]Finetune experiment complete![/]")
        console.print(f"Results saved to: {output_dir}")
        return

    # Normal mode: train/test/both
    console.print(Panel(
        f"[bold]Agent-Pair Generalization Experiment[/]\n\n"
        f"Split: {args.split}\n"
        f"Train combos: {len(train_combo_ids)} ({args.train_ratio} per scenario)\n"
        f"Test combos: {len(test_combo_ids)} ({args.test_ratio} per scenario)\n"
        f"[cyan]Same scenarios, different agent pairs[/]\n"
        f"Bandit: {args.bandit_type}\n"
        f"Model: {args.model}\n"
        f"Output: {output_dir}",
        title="Configuration"
    ))

    train_summary = None
    test_summary = None
    train_dir = output_dir / "train"

    # Training phase
    if args.phase in ["train", "both"]:
        console.print("\n[bold magenta]" + "="*60 + "[/]")
        console.print("[bold magenta]TRAINING PHASE[/]")
        console.print("[bold magenta]" + "="*60 + "[/]")

        train_summary = await run_scenarios(
            combo_ids=train_combo_ids,
            args=args,
            output_dir=train_dir,
            phase="train",
        )

        console.print(f"\n[green]Training complete![/]")
        console.print(f"Mean P1 Goal: {train_summary.get('mean_reward_p1', 'N/A')}")
        console.print(f"Mean P2 Goal: {train_summary.get('mean_reward_p2', 'N/A')}")

    # Testing phase
    if args.phase in ["test", "both"]:
        console.print("\n[bold magenta]" + "="*60 + "[/]")
        console.print("[bold magenta]TESTING PHASE (Frozen Policy)[/]")
        console.print("[bold magenta]" + "="*60 + "[/]")

        # Determine pretrained state directory
        if args.phase == "test" and args.model_dir:
            pretrained_dir = args.model_dir
        else:
            pretrained_dir = train_dir

        test_summary = await run_scenarios(
            combo_ids=test_combo_ids,
            args=args,
            output_dir=output_dir / "test",
            phase="test",
            pretrained_state_dir=pretrained_dir,
        )

        console.print(f"\n[green]Testing complete![/]")
        console.print(f"Mean P1 Goal: {test_summary.get('mean_reward_p1', 'N/A')}")
        console.print(f"Mean P2 Goal: {test_summary.get('mean_reward_p2', 'N/A')}")

    # Print comparison table
    if train_summary or test_summary:
        console.print("\n")
        print_comparison_table(train_summary, test_summary)

    # Save final summary
    final_summary = {
        "experiment": exp_name,
        "config": config,
        "train": train_summary,
        "test": test_summary,
    }
    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2, default=str)

    console.print(f"\n[bold green]Experiment complete![/]")
    console.print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
