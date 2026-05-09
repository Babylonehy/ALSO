"""
LinUCB Bandit - Optimized Version.

Performance optimizations:
1. Sherman-Morrison rank-1 update instead of full matrix inverse O(d³) -> O(d²)
2. Vectorized UCB computation for all arms at once
3. Cached theta vector to avoid redundant computation

Compatible with both PromptSpace (paraphrase mode) and StrategySpace (strategy mode).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
from loguru import logger

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .prompt_space import PromptSpace

if TYPE_CHECKING:
    from .strategy_space import StrategySpace

# Type alias for prompt space (supports both PromptSpace and StrategySpace)
PromptSpaceType = Union[PromptSpace, "StrategySpace"]


class LinUCBBanditFast(BaseBandit):
    """
    Optimized Linear UCB bandit with Sherman-Morrison updates.
    
    Key optimizations:
    - Sherman-Morrison formula for O(d²) inverse updates instead of O(d³)
    - Vectorized UCB computation for all arms simultaneously
    - Cached theta vector
    """

    def __init__(
        self,
        prompt_space: PromptSpaceType,
        config: BanditConfig | None = None,
        tensorboard_dir: Path | None = None,
    ) -> None:
        super().__init__(prompt_space, config, tensorboard_dir)

        self.alpha = self.config.alpha
        self.lambda_reg = 1.0

        d = self.embedding_dim
        
        # Initialize A^{-1} directly (avoid initial inverse)
        self._A_inv: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.eye(d) / self.lambda_reg,  # (λI)^{-1} = I/λ
            "p2": np.eye(d) / self.lambda_reg,
        }
        self.b: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.zeros(d),
            "p2": np.zeros(d),
        }
        
        # Cached theta vector (ridge regression coefficients)
        self._theta: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.zeros(d),
            "p2": np.zeros(d),
        }
        self._theta_dirty: dict[Literal["p1", "p2"], bool] = {
            "p1": True,
            "p2": True,
        }

        logger.info(
            f"Initialized LinUCBBanditFast with embedding_dim={self.embedding_dim}, "
            f"alpha={self.alpha}"
        )

    def _sherman_morrison_update(
        self, 
        agent: Literal["p1", "p2"], 
        x: np.ndarray
    ) -> None:
        """
        Sherman-Morrison rank-1 update for A^{-1}.
        
        A_new = A + x x^T
        A_new^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        
        Complexity: O(d²) instead of O(d³)
        """
        A_inv = self._A_inv[agent]
        A_inv_x = A_inv @ x  # O(d²)
        
        # Denominator: 1 + x^T A^{-1} x
        denom = 1.0 + np.dot(x, A_inv_x)  # O(d)
        
        # Sherman-Morrison update: A^{-1} -= (A^{-1} x)(x^T A^{-1}) / denom
        # Use outer product: O(d²)
        self._A_inv[agent] = A_inv - np.outer(A_inv_x, A_inv_x) / denom
        
        # Mark theta as dirty (needs recomputation)
        self._theta_dirty[agent] = True

    def _get_theta(self, agent: Literal["p1", "p2"]) -> np.ndarray:
        """Get cached theta vector, recompute if dirty."""
        if self._theta_dirty[agent]:
            self._theta[agent] = self._A_inv[agent] @ self.b[agent]
            self._theta_dirty[agent] = False
        return self._theta[agent]

    def _compute_ucb_vectorized(
        self,
        embeddings: np.ndarray,  # Shape: (n_arms, d)
        agent: Literal["p1", "p2"],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute UCB values for all arms at once (vectorized).
        
        Returns:
            Tuple of (ucb_values, predictions, uncertainties) - all shape (n_arms,)
        """
        A_inv = self._A_inv[agent]
        theta = self._get_theta(agent)
        
        # Predicted values for all arms: X @ theta
        # Shape: (n_arms, d) @ (d,) = (n_arms,)
        predictions = embeddings @ theta
        
        # Uncertainties: sqrt(x^T A^{-1} x) for each arm
        # X @ A_inv @ X^T gives (n_arms, n_arms), we only need diagonal
        # More efficient: compute (X @ A_inv) * X then sum along axis 1
        # Shape: (n_arms, d) * (n_arms, d) -> (n_arms, d) -> sum -> (n_arms,)
        A_inv_x = embeddings @ A_inv  # (n_arms, d)
        uncertainties = np.sqrt(np.sum(A_inv_x * embeddings, axis=1))
        
        # UCB values
        ucb_values = predictions + self.alpha * uncertainties
        
        return ucb_values, predictions, uncertainties

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """Select an arm using optimized LinUCB."""
        if self._stopped:
            logger.warning("LinUCB stopped, using current selection")
            idx = self.current_selections[agent]
            return idx, self.prompt_space.get_prompt(agent, idx), self.prompt_space.get_embedding(agent, idx)

        embeddings = self.prompt_space.get_all_embeddings(agent)
        
        # Vectorized UCB computation - single call for all arms
        ucb_values, predictions, uncertainties = self._compute_ucb_vectorized(
            embeddings, agent
        )

        self.score_traces[agent].append(predictions.tolist())

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

        return selected_idx, prompt_text, embedding

    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """Update using Sherman-Morrison (O(d²) instead of O(d³))."""
        prompt_text = self.prompt_space.get_prompt(agent, arm_index)
        embedding = self.prompt_space.get_embedding(agent, arm_index)

        # Sherman-Morrison rank-1 update: O(d²)
        self._sherman_morrison_update(agent, embedding)
        
        # Update b vector
        self.b[agent] = self.b[agent] + reward * embedding

        # Calculate current UCB for logging (uses cached theta)
        theta = self._get_theta(agent)
        pred = float(np.dot(theta, embedding))
        A_inv_x = self._A_inv[agent] @ embedding
        unc = float(np.sqrt(np.dot(embedding, A_inv_x)))
        ucb = pred + self.alpha * unc

        record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=prompt_text,
            embedding=embedding,
            reward=reward,
            cumulative_score=ucb,
        )
        self.selection_history.append(record)

        logger.info(
            f"[Turn {turn}] Updated {agent} arm {arm_index} with reward={reward:.2f}, "
            f"UCB={ucb:.4f}, pred={pred:.4f}, unc={unc:.4f}"
        )

    def train_model(self, verbose: bool = True) -> float:  # noqa: ARG002
        """LinUCB uses online updates, no explicit training needed."""
        logger.debug("LinUCB uses online updates, no explicit training needed")
        return 0.0

    def save_model(self, path: Path) -> None:
        """Save the LinUCB model parameters."""
        import pickle

        model_data = {
            "A_inv_p1": self._A_inv["p1"],
            "A_inv_p2": self._A_inv["p2"],
            "b_p1": self.b["p1"],
            "b_p2": self.b["p2"],
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"LinUCB model saved to {path}")
