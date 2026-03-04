"""
LinUCB Bandit implementation for dynamic prompt optimization.

This module implements a Linear Upper Confidence Bound (LinUCB) algorithm
that uses ridge regression on embeddings to estimate arm values with
uncertainty-based exploration.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .prompt_space import PromptSpace


class LinUCBBandit(BaseBandit):
    """
    Linear UCB bandit algorithm for selecting optimal bio paraphrases.

    Uses ridge regression on embeddings to estimate arm values, with UCB
    exploration bonus: select arm with highest (predicted_value + alpha * uncertainty).
    """

    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | None = None,
        tensorboard_dir: Path | None = None,
    ) -> None:
        super().__init__(prompt_space, config, tensorboard_dir)

        # LinUCB specific parameters
        self.alpha = self.config.alpha  # Exploration parameter

        # Regularization parameter for ridge regression
        self.lambda_reg = 1.0

        # Initialize A matrix (d x d) and b vector (d x 1) for each agent
        # A = X^T X + lambda * I
        # b = X^T y
        d = self.embedding_dim
        self.A: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": self.lambda_reg * np.eye(d),
            "p2": self.lambda_reg * np.eye(d),
        }
        self.b: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.zeros(d),
            "p2": np.zeros(d),
        }

        # Cached inverse of A for efficiency
        self._A_inv: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.linalg.inv(self.A["p1"]),
            "p2": np.linalg.inv(self.A["p2"]),
        }

        logger.info(
            f"Initialized LinUCBBandit with embedding_dim={self.embedding_dim}, "
            f"alpha={self.alpha}"
        )

    def _update_A_inv(self, agent: Literal["p1", "p2"]) -> None:
        """Update cached inverse of A matrix."""
        try:
            self._A_inv[agent] = np.linalg.inv(self.A[agent])
        except np.linalg.LinAlgError:
            logger.warning(f"Failed to invert A matrix for {agent}, using pseudo-inverse")
            self._A_inv[agent] = np.linalg.pinv(self.A[agent])

    def _compute_ucb(
        self,
        embedding: np.ndarray,
        agent: Literal["p1", "p2"],
    ) -> tuple[float, float, float]:
        """
        Compute UCB value for an arm.

        Returns:
            Tuple of (ucb_value, predicted_value, uncertainty)
        """
        A_inv = self._A_inv[agent]

        # theta = A^{-1} b (ridge regression coefficients)
        theta = A_inv @ self.b[agent]

        # Predicted value
        predicted = float(np.dot(theta, embedding))

        # Uncertainty (confidence bound width)
        # sqrt(x^T A^{-1} x)
        uncertainty = float(np.sqrt(embedding @ A_inv @ embedding))

        # UCB value
        ucb = predicted + self.alpha * uncertainty

        return ucb, predicted, uncertainty

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select an arm for the given agent using LinUCB algorithm.

        Args:
            agent: Which agent to select for ("p1" or "p2")
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, embedding)
        """
        if self._stopped:
            logger.warning("LinUCB stopped due to numerical issues, using current selection")
            idx = self.current_selections[agent]
            return idx, self.prompt_space.get_prompt(agent, idx), self.prompt_space.get_embedding(agent, idx)

        # Get all embeddings for this agent
        embeddings = self.prompt_space.get_all_embeddings(agent)
        n_arms = len(embeddings)

        # Compute UCB for each arm
        ucb_values = []
        predictions = []
        uncertainties = []

        for i in range(n_arms):
            ucb, pred, unc = self._compute_ucb(embeddings[i], agent)
            ucb_values.append(ucb)
            predictions.append(pred)
            uncertainties.append(unc)

        # Store predictions in score traces
        self.score_traces[agent].append(predictions)

        # Select arm with highest UCB
        selected_idx = int(np.argmax(ucb_values))
        self.current_selections[agent] = selected_idx

        prompt_text = self.prompt_space.get_prompt(agent, selected_idx)
        embedding = self.prompt_space.get_embedding(agent, selected_idx)

        logger.info(
            f"[Turn {turn}] Selected {agent} arm {selected_idx}, "
            f"UCB={ucb_values[selected_idx]:.4f}, "
            f"pred={predictions[selected_idx]:.4f}, "
            f"unc={uncertainties[selected_idx]:.4f}"
        )

        # Log top UCB values
        top_indices = np.argsort(ucb_values)[-5:][::-1]
        logger.debug(
            f"Top 5 {agent} arms: indices={top_indices.tolist()}, "
            f"UCBs={[ucb_values[i] for i in top_indices]}"
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
        Update the bandit with the observed reward for the selected arm.

        LinUCB update rule:
        A = A + x x^T
        b = b + r * x

        Args:
            agent: Which agent was updated ("p1" or "p2")
            arm_index: The arm index that was selected
            reward: The observed reward
            turn: The turn number when this selection was made
        """
        prompt_text = self.prompt_space.get_prompt(agent, arm_index)
        embedding = self.prompt_space.get_embedding(agent, arm_index)

        # Update A and b matrices
        x = embedding.reshape(-1, 1)  # Column vector
        self.A[agent] = self.A[agent] + x @ x.T
        self.b[agent] = self.b[agent] + reward * embedding

        # Update cached inverse
        self._update_A_inv(agent)

        # Calculate current UCB for logging
        ucb, pred, unc = self._compute_ucb(embedding, agent)

        record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=prompt_text,
            embedding=embedding,
            reward=reward,
            cumulative_score=ucb,  # Use UCB as cumulative score
        )
        self.selection_history.append(record)

        logger.info(
            f"[Turn {turn}] Updated {agent} arm {arm_index} with reward={reward:.2f}, "
            f"UCB={ucb:.4f}, pred={pred:.4f}, unc={unc:.4f}"
        )

    def train_model(self, verbose: bool = True) -> float:  # noqa: ARG002
        """
        LinUCB doesn't require explicit training - updates are online.

        This method is a no-op for LinUCB but is required by the interface.

        Returns:
            0.0 (no training loss for LinUCB)
        """
        logger.debug("LinUCB uses online updates, no explicit training needed")
        return 0.0

    def save_model(self, path: Path) -> None:
        """Save the LinUCB model parameters to a file."""
        import pickle

        model_data = {
            "A_p1": self.A["p1"],
            "A_p2": self.A["p2"],
            "b_p1": self.b["p1"],
            "b_p2": self.b["p2"],
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"LinUCB model saved to {path}")

