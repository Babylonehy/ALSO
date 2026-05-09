"""
Standard EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
bandit implementation for dynamic prompt optimization.

This is the classic EXP3 algorithm for adversarial multi-armed bandits.
Reference: Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002).
"""

from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .prompt_space import PromptSpace


class EXP3Bandit(BaseBandit):
    """
    Standard EXP3 algorithm for adversarial multi-armed bandits.
    
    The algorithm maintains weights for each arm and uses exponential
    updates based on importance-weighted rewards.
    
    Key features:
    - Exploration through mixing with uniform distribution (gamma parameter)
    - Importance-weighted reward estimation
    - Exponential weight updates with learning rate eta
    """

    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | None = None,
        tensorboard_dir: Path | None = None,
        time_horizon: int = 1000,  # Expected number of turns for gamma calculation
    ) -> None:
        super().__init__(prompt_space, config, tensorboard_dir)

        # Number of arms for each agent
        self.n_arms = {
            "p1": prompt_space.p1_embeddings.shape[0],
            "p2": prompt_space.p2_embeddings.shape[0],
        }

        # Initialize weights to 1 for all arms
        self.weights: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.ones(self.n_arms["p1"], dtype=np.float64),
            "p2": np.ones(self.n_arms["p2"], dtype=np.float64),
        }

        # Time horizon for parameter tuning
        self.time_horizon = time_horizon
        self.max_arms = max(self.n_arms["p1"], self.n_arms["p2"])

        # Standard EXP3 exploration parameter gamma
        # Formula: gamma = min(1, sqrt(K * log(K) / ((e-1) * T)))
        # where K is number of arms and T is time horizon
        self.gamma = min(1.0, np.sqrt(
            self.max_arms * np.log(self.max_arms) / 
            ((np.e - 1) * time_horizon)
        ))
        logger.info(f"Computed gamma={self.gamma:.6f} for T={time_horizon}, K={self.max_arms}")

        # Standard EXP3 learning rate eta
        # Formula: eta = gamma / K
        # This can be overridden by config.eta if provided
        if hasattr(config, 'eta') and config.eta is not None:
            self.eta = config.eta
            logger.info(f"Using config.eta={self.eta} instead of standard eta={self.gamma / self.max_arms:.6f}")
        else:
            self.eta = self.gamma / self.max_arms

        # Store last selection probabilities
        self.last_selection_probs: dict[str, float] = {
            "p1": 1.0 / self.n_arms["p1"],
            "p2": 1.0 / self.n_arms["p2"],
        }
        
        # Store probabilities for importance weighting
        self._current_probs: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.ones(self.n_arms["p1"]) / self.n_arms["p1"],
            "p2": np.ones(self.n_arms["p2"]) / self.n_arms["p2"],
        }

        logger.info(
            f"Initialized Standard EXP3 Bandit with n_arms_p1={self.n_arms['p1']}, "
            f"n_arms_p2={self.n_arms['p2']}, T={time_horizon}, "
            f"gamma={self.gamma:.6f}, eta={self.eta:.6f}"
        )

    def _compute_probabilities(self, agent: Literal["p1", "p2"]) -> np.ndarray:
        """
        Compute arm selection probabilities using EXP3 formula.
        
        p_i = (1 - gamma) * (w_i / sum(w)) + gamma / K
        
        where gamma controls exploration and K is the number of arms.
        """
        weights = self.weights[agent]
        n_arms = self.n_arms[agent]
        
        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum <= 0 or np.isnan(weight_sum) or np.isinf(weight_sum):
            logger.warning(f"Invalid weight sum for {agent}, using uniform distribution")
            return np.ones(n_arms) / n_arms
        
        normalized_weights = weights / weight_sum
        
        # Mix with uniform distribution for exploration
        probs = (1 - self.gamma) * normalized_weights + self.gamma / n_arms
        
        # Ensure valid probability distribution
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / np.sum(probs)
        
        return probs

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select an arm using EXP3 probability distribution.
        """
        if self._stopped:
            logger.warning("EXP3 stopped, using current selection")
            idx = self.current_selections[agent]
            return idx, self.prompt_space.get_prompt(agent, idx), \
                   self.prompt_space.get_embedding(agent, idx)

        # Compute probabilities
        probs = self._compute_probabilities(agent)
        self._current_probs[agent] = probs
        
        # Store weights as scores for compatibility
        self.score_traces[agent].append(self.weights[agent].tolist())

        # Sample arm from probability distribution
        try:
            selected_idx = np.random.choice(self.n_arms[agent], p=probs)
        except ValueError as e:
            logger.error(f"Error sampling from EXP3 distribution: {e}")
            self._stopped = True
            selected_idx = self.current_selections[agent]

        self.current_selections[agent] = selected_idx
        self.last_selection_probs[agent] = float(probs[selected_idx])

        prompt_text = self.prompt_space.get_prompt(agent, selected_idx)
        embedding = self.prompt_space.get_embedding(agent, selected_idx)

        # Log top probabilities
        top_indices = np.argsort(probs)[-5:][::-1]
        logger.debug(
            f"[Turn {turn}] EXP3 {agent}: selected arm {selected_idx}, "
            f"prob={probs[selected_idx]:.4f}, top_arms={top_indices.tolist()}"
        )

        return selected_idx, prompt_text, embedding

    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """
        Update weights using standard EXP3 importance-weighted update rule.

        The weight update is:
        w_i = w_i * exp(eta * r_hat_i)

        where r_hat_i = r_i / p_i is the importance-weighted reward estimate.
        
        This is the standard formulation from Auer et al. (2002).
        """
        # Get probability used for selection
        prob = self._current_probs[agent][arm_index]

        # Importance-weighted reward estimate
        # Only the selected arm gets updated
        reward_estimate = reward / max(prob, 1e-10)

        # Standard EXP3 exponential weight update
        old_weight = self.weights[agent][arm_index]
        self.weights[agent][arm_index] *= np.exp(self.eta * reward_estimate)

        # Prevent weight explosion - normalize if max weight is too large
        max_weight = np.max(self.weights[agent])
        if max_weight > 1e10:
            self.weights[agent] /= max_weight
            logger.debug(f"Normalized {agent} weights to prevent overflow")

        # Record selection
        prompt_text = self.prompt_space.get_prompt(agent, arm_index)
        embedding = self.prompt_space.get_embedding(agent, arm_index)

        record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=prompt_text,
            embedding=embedding,
            reward=reward,
            cumulative_score=float(self.weights[agent][arm_index]),
            selection_probability=prob,
        )
        self.selection_history.append(record)

        logger.info(
            f"[Turn {turn}] EXP3 updated {agent} arm {arm_index}: "
            f"reward={reward:.2f}, prob={prob:.4f}, "
            f"weight: {old_weight:.4f} -> {self.weights[agent][arm_index]:.4f}"
        )

    def train_model(self, verbose: bool = True) -> float:  # noqa: ARG002
        """
        EXP3 is an online algorithm - no batch training needed.

        This method exists for interface compatibility.
        Returns 0.0 as there's no loss to report.
        """
        logger.debug("EXP3 is an online algorithm, no batch training needed")
        return 0.0

    def reset_weights(self) -> None:
        """Reset all weights to initial state."""
        self.weights = {
            "p1": np.ones(self.n_arms["p1"], dtype=np.float64),
            "p2": np.ones(self.n_arms["p2"], dtype=np.float64),
        }
        logger.info("EXP3 weights reset to initial state")

    def get_arm_statistics(self, agent: Literal["p1", "p2"]) -> dict:
        """Get statistics for all arms of an agent."""
        probs = self._compute_probabilities(agent)
        return {
            "weights": self.weights[agent].tolist(),
            "probabilities": probs.tolist(),
            "n_arms": self.n_arms[agent],
            "gamma": self.gamma,
            "eta": self.eta,
        }

