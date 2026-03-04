#!/usr/bin/env python3
"""
Cross-Scenario Generalization Experiment.

This script evaluates whether strategies learned in one set of scenarios
can transfer to unseen scenarios.

Experimental Design:
1. Training Phase: Train bandit on 7 training scenarios (with learning)
2. Testing Phase: Evaluate trained model on 7 held-out test scenarios (frozen policy)
3. Baseline Phase: Online learning on test set WITHOUT pretraining (cold start)
4. Finetune Phase: Pretrained + continue online learning on test set (warm start)

Test Modes:
| Mode     | Pretrain | Test Learning | Description                    |
|----------|----------|---------------|--------------------------------|
| baseline | ❌       | ✅            | Cold start (from scratch)      |
| test     | ✅       | ❌            | Frozen policy (pure transfer)  |
| finetune | ✅       | ✅            | Warm start (pretrain + adapt)  |

Usage:
    # Full experiment (train + test with frozen policy)
    python run_generalization_experiment.py \
        --phase both --split A

    # Baseline (no pretraining, online learning on test set)
    python run_generalization_experiment.py \
        --phase baseline --split A

    # Finetune (pretrain + continue learning on test set)
    python run_generalization_experiment.py \
        --phase finetune --split A --model-dir results/generalization/xxx/train

Comparison:
- test vs baseline: Generalization benefit of pretraining
- finetune vs baseline: Warm start benefit
- finetune vs test: Benefit of continued learning

This script is completely independent and does not modify any existing code.
All state saving/loading logic is implemented locally.
"""

import argparse
import asyncio
import json
import pickle
import sys
import time
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

from experiments.dynamic_observation.lib.scenario_splits import (
    get_train_scenarios,
    get_test_scenarios,
)

console = Console()


def env_ids_to_combo_ids(env_ids: list[str]) -> list[str]:
    """Convert environment IDs to combo IDs by querying the database.

    For each env_id, finds all associated combos and returns their pks.
    """
    from sotopia.database import EnvAgentComboStorage

    combo_ids = []
    for env_id in env_ids:
        combos = EnvAgentComboStorage.find(
            EnvAgentComboStorage.env_id == env_id
        ).all()
        # Sort by pk for deterministic ordering, take first combo per env
        sorted_combos = sorted(combos, key=lambda c: c.pk)
        if sorted_combos:
            combo_ids.append(sorted_combos[0].pk)

    logger.info(f"Converted {len(env_ids)} env_ids to {len(combo_ids)} combo_ids")
    return combo_ids


def save_bandit_state(bandit: Any, save_dir: Path) -> None:
    """Save bandit state (model weights + selection history) to directory.

    This is a standalone implementation that does not modify the bandit class.
    Saves:
    - model_weights.pt: Neural network weights
    - optimizer_state.pt: Optimizer state (for resuming training)
    - state_data.json: Score traces and current selections
    - selection_history.pkl: Full selection history with rewards
    """
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
    """Load bandit state from directory and optionally freeze the model.
    
    This is a standalone implementation that does not modify the bandit class.
    """
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
        # Mark as frozen by setting a flag
        bandit._frozen = True
        logger.info("Model frozen for evaluation")
    
    logger.info(f"Bandit state loaded from {load_dir}")


def is_bandit_frozen(bandit: Any) -> bool:
    """Check if bandit is frozen."""
    return getattr(bandit, "_frozen", False)


class FrozenBanditWrapper:
    """Wrapper that intercepts update calls when bandit is frozen.

    This ensures that during test phase, no learning occurs while
    still allowing the bandit to make selections using learned policy.
    """

    def __init__(self, bandit: Any):
        self._bandit = bandit
        self._frozen = True

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._bandit, name)

    def update(self, agent: str, arm_index: int, reward: float, turn: int, **kwargs: Any) -> None:
        """Intercept update calls - log but don't actually update."""
        logger.debug(f"[FROZEN] Ignoring update for {agent} arm {arm_index} reward={reward:.2f}")

    def update_with_context(self, agent: str, arm_index: int, reward: float,
                            turn: int, context_embedding: Any = None, **kwargs: Any) -> None:
        """Intercept context update calls - log but don't actually update."""
        logger.debug(f"[FROZEN] Ignoring context update for {agent} arm {arm_index}")

    def select(self, agent: str, turn: int) -> tuple:
        """Use the underlying bandit's selection (uses learned policy)."""
        return self._bandit.select(agent, turn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-scenario generalization experiment for bandit models"
    )

    # Experiment mode
    parser.add_argument(
        "--phase", type=str, choices=["train", "test", "both", "baseline", "finetune"], default="both",
        help="Phase to run: train, test, both, baseline (cold start), or finetune (warm start)"
    )
    parser.add_argument(
        "--split", type=str, choices=["A", "B"], default="A",
        help="Scenario split (A or B for cross-validation)"
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
    parser.add_argument(
        "--bandit-type", type=str, default="adversarial",
        help="Bandit algorithm type"
    )
    parser.add_argument(
        "--eta", type=float, default=10.0,
        help="EXP3 exploration parameter"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="LinUCB exploration parameter"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="UCB exploration parameter"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of epochs for neural bandit training"
    )
    parser.add_argument(
        "--update-interval", type=int, default=1,
        help="Bandit update interval"
    )
    parser.add_argument(
        "--evolution-interval", type=int, default=5,
        help="Evolution interval for evolutionary bandits"
    )
    parser.add_argument(
        "--mask-unselected-scores", action="store_true", default=False,
        help="Mask unselected scores in training"
    )
    parser.add_argument(
        "--importance-weighted-reward", action="store_true", default=True,
        help="Use importance weighted reward"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1,
        help="Discount factor"
    )
    parser.add_argument(
        "--score-decay", type=float, default=0.9,
        help="Score decay factor"
    )
    parser.add_argument(
        "--dynamic-eta", action="store_true", default=False,
        help="Use dynamic eta"
    )
    parser.add_argument(
        "--cumulative-score-mode", type=str, default="nn",
        help="Cumulative score mode"
    )
    parser.add_argument(
        "--failure-penalty-threshold", type=float, default=0.3,
        help="Failure penalty threshold"
    )
    parser.add_argument(
        "--failure-penalty-factor", type=float, default=1.5,
        help="Failure penalty factor"
    )
    parser.add_argument(
        "--multi-dim-prediction", action="store_true", default=False,
        help="Use multi-dimensional prediction"
    )
    parser.add_argument(
        "--selection-strategy", type=str, default=None,
        help="Selection strategy"
    )
    parser.add_argument(
        "--selection-epsilon", type=float, default=0.1,
        help="Epsilon for epsilon-greedy selection"
    )
    parser.add_argument(
        "--selection-temperature", type=float, default=1.0,
        help="Temperature for softmax selection"
    )
    parser.add_argument(
        "--population-size", type=int, default=None,
        help="Population size for evolutionary bandits"
    )
    parser.add_argument(
        "--strategy-version", type=str, default="v3",
        help="Strategy version to use"
    )

    # Simulation settings
    parser.add_argument(
        "--max-turns", type=int, default=20,
        help="Maximum turns per episode"
    )
    parser.add_argument(
        "--optimize", type=str, choices=["p1", "p2", "both"], default="both",
        help="Which agents to optimize"
    )

    # Output settings
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/generalization"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Directory containing trained model (for test phase only)"
    )
    parser.add_argument(
        "--push-to-db", action="store_true",
        help="Push results to database"
    )
    parser.add_argument(
        "--strategy-cache-dir", type=Path, default=None,
        help="Strategy cache directory"
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Custom suffix to append to experiment name"
    )

    # Context embedding settings
    parser.add_argument(
        "--context-embedding", action="store_true",
        help="Use context embedding"
    )
    parser.add_argument(
        "--embedding-model", type=str, default="qwen/qwen3-embedding-8b",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--context-embedding-dim", type=int, default=4096,
        help="Dimension of context embeddings"
    )

    # Single scenario split (for fast verification)
    parser.add_argument(
        "--single-env-split", action="store_true",
        help="Use a single environment with 3 train / 2 test combos (fast verification)"
    )

    # Concurrency settings
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Maximum number of concurrent scenarios"
    )

    return parser.parse_args()


async def run_scenarios(
    scenario_ids: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    phase: str,
    pretrained_state_dir: Path | None = None,
) -> dict[str, Any]:
    """Run bandit simulation on a list of scenarios.

    Key insight for cross-scenario generalization:
    - Each scenario has different agent backgrounds, so StrategySpace is different
    - But the neural network learns embedding → value mapping
    - Strategy embeddings are the same across scenarios (same strategy text)
    - So we can transfer the learned value function (NN weights) to new scenarios

    Training phase:
    - Run each scenario with its own bandit instance
    - After each scenario, save the model weights
    - Accumulate training across scenarios by loading previous weights

    Testing phase:
    - Load pre-trained weights into each scenario's bandit
    - Freeze the model (no updates during testing)
    - Evaluate performance on unseen scenarios

    Args:
        scenario_ids: List of scenario (combo) IDs to run
        args: Command line arguments
        output_dir: Directory to save results
        phase: "train" or "test"
        pretrained_state_dir: Directory containing pre-trained bandit state (for test phase)

    Returns:
        Dictionary with results summary
    """
    from experiments.dynamic_observation.run_bandit_simulation_context import (
        BanditSimulationRunner,
        create_bandit_config_from_args,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config at the start
    config_data = {
        "phase": phase,
        "split": args.split,
        "model": args.model,
        "reward_eval_model": args.reward_eval_model,
        "bandit_type": args.bandit_type,
        "optimize": args.optimize,
        "max_turns": args.max_turns,
        "eta": args.eta,
        "strategy_version": args.strategy_version,
        "scenario_ids": scenario_ids,
        "pretrained_state_dir": str(pretrained_state_dir) if pretrained_state_dir else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    results = []
    all_rewards_p1 = []
    all_rewards_p2 = []

    # For training: accumulate model weights across scenarios
    # We save after each scenario and load into the next
    accumulated_weights_p1: Path | None = None
    accumulated_weights_p2: Path | None = None

    # Concurrency control
    concurrency = getattr(args, "concurrency", 5)  # Default to 5 concurrent tasks
    semaphore = asyncio.Semaphore(concurrency)

    def setup_runner_sync(scenario_id: str) -> Any:
        """Synchronous setup of runner and model loading (runs in thread)."""
        # Create runner
        from experiments.dynamic_observation.run_bandit_simulation_context import (
            BanditSimulationRunner,
            create_bandit_config_from_args,
        )
        runner = BanditSimulationRunner(
            scenario_id=scenario_id,
            model_name=args.model,
            env_model_name=args.model,
            reward_eval_model_name=args.reward_eval_model,
            max_turns=args.max_turns,
            bandit_config=create_bandit_config_from_args(args),
            bandit_type=args.bandit_type,
            optimize_mode=args.optimize,
            push_to_db=args.push_to_db,
            experiment_tag=f"generalization_{phase}_split{args.split}",
            verbose=False,  # Disable verbose UI in parallel mode to avoid "Only one live display" error
            experiment_dir=output_dir,
            selection_mode="strategy",
            strategy_cache_dir=args.strategy_cache_dir,
            strategy_version=args.strategy_version,
            use_context_embedding=args.context_embedding,
            embedding_model=args.embedding_model,
            context_embedding_dim=args.context_embedding_dim,
        )

        # Testing: load pre-trained weights and freeze
        if pretrained_state_dir:
            if runner.bandit_p1:
                load_bandit_state(
                    runner.bandit_p1,
                    pretrained_state_dir / "p1_bandit",
                    freeze=True
                )
                runner.bandit_p1 = FrozenBanditWrapper(runner.bandit_p1)
            if runner.bandit_p2:
                load_bandit_state(
                    runner.bandit_p2,
                    pretrained_state_dir / "p2_bandit",
                    freeze=True
                )
                runner.bandit_p2 = FrozenBanditWrapper(runner.bandit_p2)
        
        return runner

    async def process_scenario_test(i: int, scenario_id: str) -> dict[str, Any]:
        """Process a single scenario in parallel (Test phase only)."""
        async with semaphore:
            console.print(f"[bold cyan][TEST] Starting Scenario {i+1}/{len(scenario_ids)}: {scenario_id}[/]")
            try:
                # Offload blocking initialization and loading to a thread
                loop = asyncio.get_running_loop()
                runner = await loop.run_in_executor(None, setup_runner_sync, scenario_id)

                # Run episode (async)
                episode_result = await runner.run_episode()
                
                # Save per-scenario result immediately
                scenario_results_dir = output_dir / "scenarios"
                scenario_results_dir.mkdir(parents=True, exist_ok=True)
                with open(scenario_results_dir / f"{scenario_id}.json", "w") as f:
                    json.dump({
                        "scenario_id": scenario_id,
                        "scenario_index": i,
                        "phase": phase,
                        "episode_result": episode_result,
                    }, f, indent=2, default=str)
                    
                console.print(f"[green]Scenario {scenario_id} completed[/]")
                
                return {
                    "scenario_id": scenario_id,
                    "result": episode_result,
                }

            except Exception as e:
                console.print(f"[red]Error in scenario {scenario_id}: {e}[/]")
                traceback.print_exc()
                return {
                    "scenario_id": scenario_id,
                    "error": str(e),
                }

    if phase == "test":
        # Parallel execution for Test phase (independent scenarios)
        console.print(f"[bold yellow]Running {len(scenario_ids)} test scenarios in parallel (concurrency={concurrency})...[/]")
        tasks = [process_scenario_test(i, sid) for i, sid in enumerate(scenario_ids)]
        results_list = await asyncio.gather(*tasks)
        
        results.extend(results_list)
        
        # Aggregate rewards from results
        for res in results:
            if "result" in res and "final_rewards" in res["result"]:
                ep_res = res["result"]
                if ep_res["final_rewards"].get("p1"):
                    p1_data = ep_res["final_rewards"]["p1"]
                    p1_goal = p1_data.get("breakdown", {}).get("goal", p1_data.get("goal", 0))
                    all_rewards_p1.append(p1_goal)
                if ep_res["final_rewards"].get("p2"):
                    p2_data = ep_res["final_rewards"]["p2"]
                    p2_goal = p2_data.get("breakdown", {}).get("goal", p2_data.get("goal", 0))
                    all_rewards_p2.append(p2_goal)

    else:
        # Sequential execution for Train/Finetune (accumulating weights)
        for i, scenario_id in enumerate(scenario_ids):
            console.print(f"\n[bold cyan]{'='*60}[/]")
            console.print(f"[bold cyan][{phase.upper()}] Scenario {i+1}/{len(scenario_ids)}: {scenario_id}[/]")
            console.print(f"[bold cyan]{'='*60}[/]")

            try:
                # Create runner for this scenario (creates new bandit with scenario-specific StrategySpace)
                runner = BanditSimulationRunner(
                    scenario_id=scenario_id,
                    model_name=args.model,
                    env_model_name=args.model,
                    reward_eval_model_name=args.reward_eval_model,
                    max_turns=args.max_turns,
                    bandit_config=create_bandit_config_from_args(args),
                    bandit_type=args.bandit_type,
                    optimize_mode=args.optimize,
                    push_to_db=args.push_to_db,
                    experiment_tag=f"generalization_{phase}_split{args.split}",
                    verbose=False,  # Disable verbose UI in parallel mode
                    experiment_dir=output_dir,
                    selection_mode="strategy",
                    strategy_cache_dir=args.strategy_cache_dir,
                    strategy_version=args.strategy_version,
                    use_context_embedding=args.context_embedding,
                    embedding_model=args.embedding_model,
                    context_embedding_dim=args.context_embedding_dim,
                )

                # Load weights based on phase
                if phase == "train" and i > 0:
                    # Training: load accumulated weights from previous scenarios
                    if accumulated_weights_p1 and runner.bandit_p1:
                        load_bandit_state(runner.bandit_p1, accumulated_weights_p1, freeze=False)
                        logger.info(f"Loaded accumulated P1 weights from scenario {i}")
                    if accumulated_weights_p2 and runner.bandit_p2:
                        load_bandit_state(runner.bandit_p2, accumulated_weights_p2, freeze=False)
                        logger.info(f"Loaded accumulated P2 weights from scenario {i}")

                # Note: Test phase logic moved to parallel block above
                # elif phase == "test" and pretrained_state_dir: ...

                elif phase == "finetune" and pretrained_state_dir:
                    # Finetune: load pre-trained weights but DON'T freeze (continue learning)
                    if i == 0:
                        # First scenario: load from pretrained model
                        if runner.bandit_p1:
                            load_bandit_state(
                                runner.bandit_p1,
                                pretrained_state_dir / "p1_bandit",
                                freeze=False  # Don't freeze - allow continued learning
                            )
                            logger.info("Loaded P1 pretrained weights for finetuning (learning enabled)")
                        if runner.bandit_p2:
                            load_bandit_state(
                                runner.bandit_p2,
                                pretrained_state_dir / "p2_bandit",
                                freeze=False
                            )
                            logger.info("Loaded P2 pretrained weights for finetuning (learning enabled)")
                    else:
                        # Subsequent scenarios: load accumulated weights from previous finetune scenarios
                        if accumulated_weights_p1 and runner.bandit_p1:
                            load_bandit_state(runner.bandit_p1, accumulated_weights_p1, freeze=False)
                            logger.info(f"Loaded accumulated P1 finetune weights from scenario {i}")
                        if accumulated_weights_p2 and runner.bandit_p2:
                            load_bandit_state(runner.bandit_p2, accumulated_weights_p2, freeze=False)
                            logger.info(f"Loaded accumulated P2 finetune weights from scenario {i}")

                # Run episode
                episode_result = await runner.run_episode()
                results.append({
                    "scenario_id": scenario_id,
                    "result": episode_result,
                })

                # Collect rewards
                final_rewards = episode_result.get("final_rewards")
                if final_rewards:
                    p1_data = final_rewards.get("p1")
                    if p1_data:
                        # goal is in breakdown dict, not top level
                        p1_goal = p1_data.get("breakdown", {}).get("goal", p1_data.get("goal", 0))
                        all_rewards_p1.append(p1_goal)
                    p2_data = final_rewards.get("p2")
                    if p2_data:
                        p2_goal = p2_data.get("breakdown", {}).get("goal", p2_data.get("goal", 0))
                        all_rewards_p2.append(p2_goal)

                # Save per-scenario episode result
                scenario_results_dir = output_dir / "scenarios"
                scenario_results_dir.mkdir(parents=True, exist_ok=True)
                with open(scenario_results_dir / f"{scenario_id}.json", "w") as f:
                    json.dump({
                        "scenario_id": scenario_id,
                        "scenario_index": i,
                        "phase": phase,
                        "episode_result": episode_result,
                    }, f, indent=2, default=str)

                # Training/Finetune: save accumulated weights after each scenario
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

                console.print(f"[green]Scenario {scenario_id} completed[/]")

            except Exception as e:
                console.print(f"[red]Error in scenario {scenario_id}: {e}[/]")
                traceback.print_exc()
                results.append({
                    "scenario_id": scenario_id,
                    "error": str(e),
                })

    # Compute summary statistics
    summary = {
        "phase": phase,
        "split": args.split,
        "n_scenarios": len(scenario_ids),
        "scenario_ids": scenario_ids,
        "mean_reward_p1": float(np.mean(all_rewards_p1)) if all_rewards_p1 else None,
        "mean_reward_p2": float(np.mean(all_rewards_p2)) if all_rewards_p2 else None,
        "std_reward_p1": float(np.std(all_rewards_p1)) if all_rewards_p1 else None,
        "std_reward_p2": float(np.std(all_rewards_p2)) if all_rewards_p2 else None,
        "all_rewards_p1": all_rewards_p1,
        "all_rewards_p2": all_rewards_p2,
        "results": results,
    }

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def print_comparison_table(train_summary: dict | None, test_summary: dict | None) -> None:
    """Print a comparison table of train vs test performance."""
    table = Table(title="Cross-Scenario Generalization Results")
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

        # N scenarios
        table.add_row(
            "N Scenarios",
            str(train_summary.get("n_scenarios", 0)),
            str(test_summary.get("n_scenarios", 0)),
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
    exp_name = f"{args.bandit_type}_{phase_suffix}_{timestamp}"
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

    # Get scenario splits (env_ids -> combo_ids)
    train_env_ids = get_train_scenarios(args.split)
    
    # Handle single environment split mode
    if args.single_env_split:
        logger.info("Single environment split mode enabled. Searching for environment with >= 5 combos...")
        from sotopia.database import EnvAgentComboStorage
        
        target_env = None
        target_combos = []
        
        # Search in train_env_ids first
        for env_id in train_env_ids:
            combos = EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all()
            if len(combos) >= 5:
                target_env = env_id
                target_combos = sorted(combos, key=lambda c: c.pk)
                break
        
        if not target_env:
            # Try test envs if not found in train
            test_ids = get_test_scenarios(args.split)
            for env_id in test_ids:
                combos = EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all()
                if len(combos) >= 5:
                    target_env = env_id
                    target_combos = sorted(combos, key=lambda c: c.pk)
                    break 
                    
        if not target_env:
             raise ValueError("Could not find any environment with >= 5 combos for single-split verification.")
             
        logger.info(f"Found target environment: {target_env} with {len(target_combos)} combos")
        
        # Split: 3 Train, 2 Test (take first 5)
        # Verify valid count
        if len(target_combos) < 5:
            # Should be caught above, but safe check
             raise ValueError(f"Environment {target_env} has only {len(target_combos)} combos, need 5.")
             
        train_combos = target_combos[:3]
        test_combos = target_combos[3:5]
        
        train_combo_ids = [c.pk for c in train_combos]
        test_combo_ids = [c.pk for c in test_combos]
        
        # Set env_ids for logging purposes
        train_env_ids = [target_env]
        test_env_ids = [target_env]
        
        logger.info(f"Split combos for {target_env}:")
        logger.info(f"  Train ({len(train_combo_ids)}): {train_combo_ids}")
        logger.info(f"  Test  ({len(test_combo_ids)}): {test_combo_ids}")
        
    else:
        # Standard split logic
        test_env_ids = get_test_scenarios(args.split)
    
        # Convert env_ids to combo_ids
        train_combo_ids = env_ids_to_combo_ids(train_env_ids)
        test_combo_ids = env_ids_to_combo_ids(test_env_ids)

    # Baseline mode: online learning on test set without pretraining
    if args.phase == "baseline":
        console.print(Panel(
            f"[bold]Baseline Experiment (No Pretraining)[/]\n\n"
            f"Split: {args.split}\n"
            f"Test scenarios (online learning): {len(test_combo_ids)} (from {len(test_env_ids)} envs)\n"
            f"Bandit: {args.bandit_type}\n"
            f"Model: {args.model}\n"
            f"Output: {output_dir}\n\n"
            f"[yellow]This baseline runs online learning on test scenarios WITHOUT pretraining.[/]\n"
            f"[yellow]Compare with 'test' phase to measure generalization benefit.[/]",
            title="Baseline Configuration"
        ))

        console.print("\n[bold magenta]" + "="*60 + "[/]")
        console.print("[bold magenta]BASELINE: Online Learning on Test Set (No Pretraining)[/]")
        console.print("[bold magenta]" + "="*60 + "[/]")

        baseline_summary = await run_scenarios(
            scenario_ids=test_combo_ids,
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

    # Finetune mode: pretrain + continue online learning on test set
    if args.phase == "finetune":
        # Determine where to load pretrained model from
        if not args.model_dir:
            console.print("[red]Error: --model-dir is required for finetune mode[/]")
            console.print("[yellow]Please provide the path to a trained model directory, e.g.:[/]")
            console.print("[yellow]  --model-dir results/generalization/adversarial_splitA_xxx/train[/]")
            return

        model_dir = args.model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        console.print(Panel(
            f"[bold]Finetune Experiment (Pretrain + Continue Learning)[/]\n\n"
            f"Split: {args.split}\n"
            f"Test scenarios (with continued learning): {len(test_combo_ids)} (from {len(test_env_ids)} envs)\n"
            f"Pretrained model: {model_dir}\n"
            f"Bandit: {args.bandit_type}\n"
            f"Model: {args.model}\n"
            f"Output: {output_dir}\n\n"
            f"[yellow]This mode loads pretrained weights and CONTINUES learning on test scenarios.[/]\n"
            f"[yellow]Compare with 'baseline' to measure warm start benefit.[/]\n"
            f"[yellow]Compare with 'test' to measure benefit of continued learning.[/]",
            title="Finetune Configuration"
        ))

        console.print("\n[bold magenta]" + "="*60 + "[/]")
        console.print("[bold magenta]FINETUNE: Pretrain + Continue Learning on Test Set[/]")
        console.print("[bold magenta]" + "="*60 + "[/]")

        finetune_summary = await run_scenarios(
            scenario_ids=test_combo_ids,
            args=args,
            output_dir=output_dir / "finetune",
            phase="finetune",  # New phase: load weights but don't freeze
            pretrained_state_dir=model_dir,
        )

        console.print(f"\n[green]Finetune complete![/]")
        console.print(f"Mean P1 Goal: {finetune_summary.get('mean_reward_p1', 'N/A')}")
        console.print(f"Mean P2 Goal: {finetune_summary.get('mean_reward_p2', 'N/A')}")

        # Save final summary
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
        f"[bold]Cross-Scenario Generalization Experiment[/]\n\n"
        f"Split: {args.split}\n"
        f"Train scenarios: {len(train_combo_ids)} (from {len(train_env_ids)} envs)\n"
        f"Test scenarios: {len(test_combo_ids)} (from {len(test_env_ids)} envs)\n"
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
            scenario_ids=train_combo_ids,
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
        console.print("[bold magenta]TESTING PHASE[/]")
        console.print("[bold magenta]" + "="*60 + "[/]")

        # Determine where to load model from
        model_dir = args.model_dir or train_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        test_summary = await run_scenarios(
            scenario_ids=test_combo_ids,
            args=args,
            output_dir=output_dir / "test",
            phase="test",
            pretrained_state_dir=model_dir,
        )

        console.print(f"\n[green]Testing complete![/]")
        console.print(f"Mean P1 Goal: {test_summary.get('mean_reward_p1', 'N/A')}")
        console.print(f"Mean P2 Goal: {test_summary.get('mean_reward_p2', 'N/A')}")

    # Print comparison
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            logger.warning(f"Ignored RuntimeError: {e}")
        else:
            raise e
    # Force exit to avoid hanging on background threads (redis-om, litellm, etc.)
    sys.exit(0)
