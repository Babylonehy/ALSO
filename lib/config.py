"""Configuration and command-line argument parsing."""

import argparse
from pathlib import Path
from typing import Literal

# Import BANDIT_TYPES from core module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.dynamic_observation.core.bandits import BANDIT_TYPES

# Type aliases
BanditType = Literal["exp3", "linucb", "neural_ucb", "none"]
OptimizeMode = Literal["p1", "p2", "both", "none"]

# Default constants (must match the model used for prompt space embeddings)
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Run bandit-based dynamic prompt optimization simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scenario: Optimize both agents with separate bandits
  python run_bandit_simulation.py --scenario-id 01H7VKQHT745XAP1A4DDV8H419 --optimize both

  # Single scenario: Baseline (no optimization)
  python run_bandit_simulation.py --scenario-id 01H7VKQHT745XAP1A4DDV8H419 --optimize none

  # Batch mode: Run all hard scenarios with parallelism
  python run_bandit_simulation.py --batch --subset hard --batch-size 5 --optimize both

  # Batch mode: Run first 10 scenarios
  python run_bandit_simulation.py --batch --subset hard --max-episodes 10 --batch-size 3
""",
    )
    # Single scenario mode
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="01H7VKQHT745XAP1A4DDV8H419",
        help="Scenario (EnvAgentComboStorage pk) to run (single mode)",
    )

    # Batch mode arguments
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode (multiple scenarios in parallel)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["hard", "all"],
        default="hard",
        help="Dataset subset to use in batch mode (default: hard)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of concurrent episodes in batch mode (default: 5)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to run in batch mode (default: all)",
    )

    # Common arguments
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/openai/gpt-4o-mini",
        help="Default model for agents (used if p1-model/p2-model not specified)",
    )
    parser.add_argument(
        "--env-model",
        type=str,
        default="openrouter/openai/gpt-4o",
        help="Model for environment/evaluator calls (default: openrouter/openai/gpt-4o)",
    )
    parser.add_argument(
        "--p1-model",
        type=str,
        default=None,
        help="Model for P1 agent (overrides --model for P1)",
    )
    parser.add_argument(
        "--p2-model",
        type=str,
        default=None,
        help="Model for P2 agent (overrides --model for P2)",
    )
    parser.add_argument(
        "--reward-eval-model",
        type=str,
        default=None,
        help="Model for RewardInTurnEvaluator (overrides --env-model for reward evaluation)",
    )
    parser.add_argument(
        "--terminal-eval-model",
        type=str,
        default=None,
        help="Model for TerminalEvaluator (overrides --env-model for terminal evaluation)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per episode",
    )
    parser.add_argument(
        "--bandit-type",
        type=str,
        choices=list(BANDIT_TYPES.keys()),
        default="adversarial",
        help="Type of bandit algorithm: 'adversarial' (Adversarial Bandit), 'linucb' (Linear UCB), 'neural_ucb' (Neural UCB), 'prompt_breeder' (Evolutionary Prompt Optimization). Default: adversarial",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=5.0,
        help="EXP3 exploration parameter (default: 5.0 for stability)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="LinUCB exploration parameter (default: 1.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="NeuralUCB exploration parameter (default: 1.0)",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=1,
        help="Train bandit every N turns (default: 1)",
    )
    parser.add_argument(
        "--evolution-interval",
        type=int,
        default=5,
        help="Evolve population every N turns for evolutionary bandits (default: 5)",
    )
    parser.add_argument(
        "--mask-unselected-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Neural adversarial bandit only: if enabled, only the selected arm's predicted score "
            "is recorded each turn (others are masked to 0). Default: enabled."
        ),
    )
    parser.add_argument(
        "--importance-weighted-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Neural adversarial bandit only: if enabled, training uses importance weighting "
            "(1 - reward) / p_selected; otherwise uses (1 - reward). Default: enabled."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help=(
            "Gamma mixing for exploration guarantee: probs = (1 - gamma) * softmax + gamma / n_arms. "
            "0.0 = disabled, 0.1 = recommended (default)"
        ),
    )
    parser.add_argument(
        "--score-decay",
        type=float,
        default=0.9,
        help=(
            "Score decay factor for cumulative scores: cumulative = decay * old + new. "
            "1.0 = disabled, 0.9 = recommended (default)"
        ),
    )
    parser.add_argument(
        "--failure-penalty-threshold",
        type=float,
        default=0.3,
        help="Reward threshold below which failure penalty is applied (default: 0.3)",
    )
    parser.add_argument(
        "--failure-penalty-factor",
        type=float,
        default=1.5,
        help=(
            "Failure penalty multiplier for selected arm's score when reward < threshold. "
            "1.0 = disabled, 1.5 = recommended (default)"
        ),
    )
    parser.add_argument(
        "--optimize",
        type=str,
        choices=["p1", "p2", "both", "none"],
        default="both",
        help="Which agent(s) to optimize: 'p1', 'p2', 'both', or 'none' (baseline)",
    )
    parser.add_argument(
        "--alternate-optimization",
        action="store_true",
        help="When optimizing both agents, alternate bandit updates between P1 and P2 each turn",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="auto",
        help="Output file for results (JSON). Use 'auto' to auto-generate filename in results/ folder, or specify a path",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable saving output to file",
    )
    parser.add_argument(
        "--push-to-db",
        action="store_true",
        help="Push episode results to database for comparison",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom experiment tag (overrides auto-generated tag)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Directory containing pre-generated embeddings and paraphrased backgrounds",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for LLM output (default: 4096)",
    )
    parser.add_argument(
        "--context-embedding",
        action="store_true",
        help="Enable context embedding for bandit training (uses dialogue history)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Model for context embedding (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--context-embedding-dim",
        type=int,
        default=4096,
        help="Dimension of context embedding (default: 4096 for qwen3-embedding-8b, use 2048 for qwen3-embedding-4b)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a previous experiment directory. Only failed scenarios will be re-run and merged into the same results.",
    )
    parser.add_argument(
        "--calculate-cost",
        action="store_true",
        help="Calculate API costs after experiment (requires OpenRouter API, may be slow)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of automatic retries for failed scenarios (default: 3)",
    )
    return parser.parse_args()
