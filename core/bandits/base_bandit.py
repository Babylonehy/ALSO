"""
Base classes for bandit algorithms in dynamic prompt optimization.

This module defines the abstract base class that all bandit implementations
must inherit from, along with shared configuration and data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from typing import TYPE_CHECKING

from .prompt_space import PromptSpace

if TYPE_CHECKING:
    from .strategy_space import StrategySpace


@dataclass
class BanditConfig:
    """Configuration for bandit algorithms."""

    # Network architecture (matching original NeuralUCB)
    hidden_size: int = 100  # Original NeuralUCB uses 100
    depth: int = 2  # Original NeuralUCB: 2-layer MLP (input -> hidden -> 1)
    use_layer_norm: bool = False  # Original NeuralUCB doesn't use LayerNorm

    # Training parameters (matching original NeuralUCB)
    learning_rate: float = 1e-2  # Original NeuralUCB uses 1e-2 with SGD
    weight_decay: float = 0.01  # Strong L2 regularization for small data
    epochs: int = 100  # Original NeuralUCB uses 100 epochs
    batch_size: int = 8

    # Exploration parameters
    eta: float = 0.5  # EXP3 exploration parameter (use smaller value with importance weighting)
    alpha: float = 0.1  # LinUCB/NeuralUCB lambda parameter (regularization, INSTINCT default: 0.1)
    beta: float = 1.0  # NeuralUCB nu parameter (exploration, INSTINCT default: 1.0)

    # Update interval: train bandit every N turns
    update_interval: int = 1

    # Evolution interval: evolve population every N turns (for evolutionary bandits)
    evolution_interval: int = 5

    # Dynamic eta scheduling: eta_t = eta / sqrt(turn + 1)
    dynamic_eta: bool = False

    # NeuralUCB specific
    n_ensemble: int = 5  # Number of networks in ensemble for uncertainty estimation
    dropout_rate: float = 0.0  # Disabled dropout

    # Context embedding settings
    use_context_embedding: bool = False  # Whether to use context embedding
    embedding_model: str = "qwen/qwen3-embedding-8b"  # Model for context/new prompt embedding (must match prompt space embeddings)
    context_embedding_dim: int = 4096  # Dimension of context embedding (4096 for qwen3-embedding-8b)

    # EXP3-style score trace behavior (used by neural adversarial bandit)
    # If True, only the selected arm's predicted score contributes to the trace each turn.
    # NOTE: Should be False for neural_evolution bandit so NN predictions are preserved for all arms
    mask_unselected_scores: bool = False

    # Importance-weighted reward for training (used by neural adversarial bandit)
    # If True, uses reward / p_selected; if False, uses reward only.
    # EXP3 style: importance sampling corrects for selection bias
    importance_weighted_reward: bool = True

    # Score decay for cumulative scores (used by neural adversarial bandit)
    # cumulative_score = sum(decay^(T-t) * score_t) where T is current turn, t is past turn
    # decay=1.0: no decay (standard sum), decay<1.0: recent scores weighted more
    score_decay: float = 0.9  # Exponential decay factor for cumulative scores

    # Cumulative score mode for arm selection (used by neural adversarial bandit)
    # "nn": use only NN predicted scores (default, stable for all arms)
    # "actual": use only actual observed scores (with importance weighting, sparse)
    # "mean": simple average of NN and actual scores (no normalization)
    # "mean-zscore": Z-score normalize NN and actual separately, then weighted combine
    cumulative_score_mode: str = "nn"

    # Weight for NN predictions when combining with actual scores (cumulative_score_mode="mean-zscore")
    # 0.0 = only actual (EXP3), 1.0 = only NN, 0.5 = equal weight after Z-score normalization
    nn_weight: float = 0.5

    # Adaptive NN weight scheduling: automatically shift from EXP3 to NN over time
    # When enabled, nn_weight is computed as sigmoid((turn - warmup) / scale)
    #   Early turns (turn << warmup): nn_weight ≈ 0, trust EXP3 actual observations
    #   Late turns (turn >> warmup):  nn_weight ≈ 1, trust NN predictions
    adaptive_nn_weight: bool = False
    adaptive_nn_weight_warmup: int = 5  # Midpoint turn where nn_weight ≈ 0.5
    adaptive_nn_weight_scale: float = 2.0  # Controls transition sharpness (larger = smoother)

    # Exploration parameters for improved adversarial bandit
    # Gamma mixing: probs = (1 - gamma) * softmax(scores) + gamma / n_arms
    # Ensures minimum exploration probability for all arms
    gamma: float = 0.1  # 0.0 = disabled, 0.1 = recommended (default)

    # Dynamic gamma scheduling: gamma_t = gamma / sqrt(turn + 1)
    # Starts with more exploration, gradually decreases
    dynamic_gamma: bool = False  # True = enable dynamic gamma scheduling

    # UCB-style exploration bonus: bonus = ucb_c * sqrt(log(t) / (n_arm + 1))
    # Encourages exploration of less-selected arms
    ucb_exploration: bool = False  # True = enable UCB exploration bonus
    ucb_c: float = 2.0  # Exploration coefficient for UCB bonus

    # Score decay: cumulative_score = decay * old_score + new_score
    # Prevents early advantages from dominating forever
    score_decay: float = 0.9  # 1.0 = disabled, 0.9 = recommended (default)

    # Failure penalty: if reward < threshold, multiply arm's cumulative score
    # Encourages exploration after failure
    failure_penalty_threshold: float = 0.3  # Reward below this triggers penalty
    failure_penalty_factor: float = 1.5  # 1.0 = disabled, 1.5 = recommended (default)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed for reproducibility
    seed: int | None = 42  # Default seed for reproducibility, None for random

    # Multi-dimensional prediction mode
    # If True, NN predicts 7 dimension scores separately, compute per-dim loss, then average for final reward
    # If False (default), NN predicts single averaged reward directly
    multi_dim_prediction: bool = False

    # Dimension names for multi-dimensional prediction (order matters for output indexing)
    dimension_names: tuple[str, ...] = (
        "believability",
        "relationship",
        "knowledge",
        "secret",
        "social_rules",
        "financial_and_material_benefits",
        "goal",
    )

    # Strategy version for social strategies selection mode
    # "v1" = SOCIAL_STRATEGIES (13 strategies, more granular negotiation tactics)
    # "v2" = SOCIAL_STRATEGIES_V2 (10 strategies, organized by quadrants: Cooperative/Competitive/Deceptive/Rational)
    strategy_version: str = "v1"


@dataclass
class SelectionRecord:
    """Records a single arm selection and its outcome."""

    turn: int
    agent: Literal["p1", "p2"]
    arm_index: int
    prompt_text: str
    embedding: np.ndarray
    reward: float = 0.0
    cumulative_score: float = 0.0
    selection_probability: float = 0.0
    context_embedding: np.ndarray | None = None  # Dialogue history embedding
    # Multi-dimensional rewards (7 dimension scores, all normalized to 0-1)
    # Keys: believability, relationship, knowledge, secret, social_rules, financial_and_material_benefits, goal
    dimension_rewards: dict[str, float] | None = None


class BaseBandit(ABC):
    """
    Abstract base class for bandit algorithms.
    
    All bandit implementations must inherit from this class and implement
    the abstract methods for arm selection and updates.
    """

    def __init__(
        self,
        prompt_space: "PromptSpace | StrategySpace",
        config: BanditConfig | None = None,
        tensorboard_dir: Path | None = None,
    ) -> None:
        """
        Initialize the bandit.

        Args:
            prompt_space: The prompt space (PromptSpace or StrategySpace) containing embeddings and prompts
            config: Configuration for the bandit algorithm
            tensorboard_dir: Optional directory for TensorBoard logging
        """
        self.prompt_space = prompt_space
        self.config = config or BanditConfig()
        self.tensorboard_dir = tensorboard_dir

        # Determine embedding dimension from loaded embeddings
        self.embedding_dim = prompt_space.p1_embeddings.shape[1]

        # Selection and reward history
        self.selection_history: list[SelectionRecord] = []
        # NN predicted scores (all arms have scores each turn)
        self.score_traces: dict[Literal["p1", "p2"], list[list[float]]] = {
            "p1": [],
            "p2": [],
        }
        # Actual observed scores with importance sampling (only selected arm has score, others 0)
        self.actual_score_traces: dict[Literal["p1", "p2"], list[list[float]]] = {
            "p1": [],
            "p2": [],
        }

        # Current selections (indices)
        self.current_selections: dict[Literal["p1", "p2"], int] = {
            "p1": 0,  # Start with original
            "p2": 0,
        }

        # Flag to stop bandit if numerical issues occur
        self._stopped = False

    @abstractmethod
    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select an arm for the given agent.

        Args:
            agent: Which agent to select for ("p1" or "p2")
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, embedding)
        """
        pass

    @abstractmethod
    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """
        Update the bandit with the observed reward.

        Args:
            agent: Which agent was updated ("p1" or "p2")
            arm_index: The arm index that was selected
            reward: The observed reward
            turn: The turn number when this selection was made
        """
        pass

    @abstractmethod
    def train_model(self, verbose: bool = True) -> float:
        """
        Train the model on accumulated data.

        Args:
            verbose: Whether to show training progress

        Returns:
            Final training loss (or 0.0 if not applicable)
        """
        pass

    def get_current_prompt(self, agent: Literal["p1", "p2"]) -> str:
        """Get the currently selected prompt for an agent."""
        idx = self.current_selections[agent]
        return self.prompt_space.get_prompt(agent, idx)

    def is_stopped(self) -> bool:
        """Check if the bandit has been stopped due to numerical issues."""
        return self._stopped

    def get_selection_summary(self) -> dict:
        """Get a summary of all selections and their rewards."""
        summary = {
            "total_selections": len(self.selection_history),
            "p1_selections": [],
            "p2_selections": [],
            "p1_avg_reward": 0.0,
            "p2_avg_reward": 0.0,
            "reward_progression": [],
        }

        p1_rewards = []
        p2_rewards = []

        for rec in self.selection_history:
            entry = {
                "turn": rec.turn,
                "arm_index": rec.arm_index,
                "prompt_text": rec.prompt_text,  # Include the actual bio text
                "reward": rec.reward,
                "cumulative_score": rec.cumulative_score,
                "selection_probability": rec.selection_probability,
            }

            if rec.agent == "p1":
                summary["p1_selections"].append(entry)
                p1_rewards.append(rec.reward)
            else:
                summary["p2_selections"].append(entry)
                p2_rewards.append(rec.reward)

            summary["reward_progression"].append({
                "turn": rec.turn,
                "agent": rec.agent,
                "arm_index": rec.arm_index,
                "reward": rec.reward,
            })

        if p1_rewards:
            summary["p1_avg_reward"] = sum(p1_rewards) / len(p1_rewards)
        if p2_rewards:
            summary["p2_avg_reward"] = sum(p2_rewards) / len(p2_rewards)

        return summary

    def save_model(self, path: Path) -> None:  # noqa: ARG002
        """Save the model weights to a file (if applicable)."""
        pass  # Override in subclasses that have models to save

    @property
    def bandit_type(self) -> str:
        """Return the type of bandit algorithm."""
        return self.__class__.__name__
