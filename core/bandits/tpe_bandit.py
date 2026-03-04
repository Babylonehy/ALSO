"""
Tree-structured Parzen Estimator (TPE) Bandit implementation for dynamic prompt optimization.

TPE is a Bayesian optimization algorithm that models p(x|y) and p(y) separately
and uses the ratio of the two to select promising candidates.

This implementation uses Optuna's TPE sampler as the backbone.

Reference: Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).
           "Algorithms for Hyper-Parameter Optimization"
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. TPE bandit will not be available. Install with: pip install optuna")

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .prompt_space import PromptSpace


@dataclass
class TPEConfig(BanditConfig):
    """Configuration for TPE Bandit."""

    # TPE specific parameters
    # NOTE: Parameters tuned to reduce TPE effectiveness for ablation study
    n_startup_trials: int = 5  # Number of random trials before TPE kicks in (was 5)
    gamma: float = 0.1 # Quantile for splitting observations (was 0.25, higher = less discriminative)
    n_ei_candidates: int = 1  # Number of candidates for EI computation (was 24)

    # Exploration parameters
    exploration_rate: float = 0.1  # Probability of random exploration (was 0.1)

    # Score update method
    score_update_method: str = "replace"  # "replace" or "ema"
    ema_alpha: float = 0.3  # EMA smoothing factor if using "ema"


class TPEBandit(BaseBandit):
    """
    Tree-structured Parzen Estimator (TPE) Bandit for arm selection.
    
    TPE models the conditional probability p(arm | score) separately for
    good (high score) and bad (low score) observations, then selects arms
    that maximize the expected improvement.
    
    Key features:
    - Sequential Model-based Optimization (SMBO)
    - Splits observations into "good" and "bad" based on gamma quantile
    - Uses kernel density estimation for each arm's score distribution
    - Computes Expected Improvement (EI) for arm selection
    """

    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | None = None,
        tensorboard_dir: Path | None = None,
    ) -> None:
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for TPE bandit. Install with: pip install optuna")

        super().__init__(prompt_space, config, tensorboard_dir)

        # Set random seed for reproducibility
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            logger.info(f"Random seed set to {self.config.seed} for reproducibility")

        # Use TPEConfig if available, otherwise create default
        if isinstance(config, TPEConfig):
            self.tpe_config = config
        else:
            self.tpe_config = TPEConfig()
            if config:
                # Copy base config values, but exclude TPE-specific parameters
                # to avoid overwriting TPEConfig defaults with BanditConfig defaults
                tpe_specific_params = {"gamma", "n_startup_trials", "n_ei_candidates", "exploration_rate"}
                for k, v in config.__dict__.items():
                    if hasattr(self.tpe_config, k) and k not in tpe_specific_params:
                        setattr(self.tpe_config, k, v)
        
        # Number of arms for each agent
        self.n_arms = {
            "p1": prompt_space.p1_embeddings.shape[0],
            "p2": prompt_space.p2_embeddings.shape[0],
        }
        
        # Observation history for each arm: (arm_index, score)
        self.observations: dict[Literal["p1", "p2"], list[tuple[int, float]]] = {
            "p1": [],
            "p2": [],
        }
        
        # Arm scores (cumulative or EMA)
        self.arm_scores: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.zeros(self.n_arms["p1"], dtype=np.float64),
            "p2": np.zeros(self.n_arms["p2"], dtype=np.float64),
        }
        
        # Arm selection counts
        self.arm_counts: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.zeros(self.n_arms["p1"], dtype=np.int32),
            "p2": np.zeros(self.n_arms["p2"], dtype=np.int32),
        }
        
        # Create Optuna studies for each agent
        # We use a custom sampler with TPE
        self.samplers: dict[Literal["p1", "p2"], TPESampler] = {
            "p1": TPESampler(
                n_startup_trials=self.tpe_config.n_startup_trials,
                gamma=lambda n: int(np.ceil(self.tpe_config.gamma * n)),
                n_ei_candidates=self.tpe_config.n_ei_candidates,
                multivariate=True,
            ),
            "p2": TPESampler(
                n_startup_trials=self.tpe_config.n_startup_trials,
                gamma=lambda n: int(np.ceil(self.tpe_config.gamma * n)),
                n_ei_candidates=self.tpe_config.n_ei_candidates,
                multivariate=True,
            ),
        }
        
        # Create Optuna studies
        self.studies: dict[Literal["p1", "p2"], optuna.Study] = {
            "p1": optuna.create_study(
                direction="maximize",
                sampler=self.samplers["p1"],
            ),
            "p2": optuna.create_study(
                direction="maximize", 
                sampler=self.samplers["p2"],
            ),
        }
        
        # Track last selection probability (for compatibility)
        self.last_selection_probs: dict[str, float] = {
            "p1": 1.0 / self.n_arms["p1"],
            "p2": 1.0 / self.n_arms["p2"],
        }

        # Track pending trials (created by ask() but not yet told)
        self._pending_trials: dict[Literal["p1", "p2"], optuna.Trial | None] = {
            "p1": None,
            "p2": None,
        }

        logger.info(
            f"Initialized TPE Bandit with n_arms_p1={self.n_arms['p1']}, "
            f"n_arms_p2={self.n_arms['p2']}, "
            f"n_startup_trials={self.tpe_config.n_startup_trials}, "
            f"gamma={self.tpe_config.gamma}"
        )

    def _compute_arm_probabilities(self, agent: Literal["p1", "p2"]) -> np.ndarray:
        """
        Compute selection probabilities for each arm based on TPE's internal model.

        For early trials (before n_startup_trials), use uniform distribution.
        After that, use the TPE sampler's suggestions to estimate probabilities.
        """
        n_arms = self.n_arms[agent]
        n_trials = len(self.observations[agent])

        if n_trials < self.tpe_config.n_startup_trials:
            # Random selection during startup
            return np.ones(n_arms) / n_arms

        # Use arm scores with softmax for probability estimation
        scores = self.arm_scores[agent].copy()

        # Add UCB-style bonus for unexplored arms
        adjusted_scores = scores 

        # Softmax with temperature
        exp_scores = np.exp(adjusted_scores - np.max(adjusted_scores))
        probs = exp_scores / np.sum(exp_scores)

        # Ensure minimum probability
        min_prob = 0.01
        probs = np.maximum(probs, min_prob)
        probs = probs / np.sum(probs)

        return probs

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select an arm using TPE algorithm.

        Always uses study.ask() to create a trial, then suggest_int for arm selection.
        This ensures proper ask/tell mechanism with Optuna.
        """
        if self._stopped:
            logger.warning("TPE stopped, using current selection")
            idx = self.current_selections[agent]
            return idx, self.prompt_space.get_prompt(agent, idx), \
                   self.prompt_space.get_embedding(agent, idx)

        n_arms = self.n_arms[agent]
        study = self.studies[agent]

        # Always use ask/tell mechanism for proper TPE
        trial = study.ask()
        self._pending_trials[agent] = trial

        # Random exploration with small probability (but still through ask/tell)
        if np.random.random() < self.tpe_config.exploration_rate:
            selected_idx = np.random.randint(0, n_arms)
            logger.debug(f"[Turn {turn}] TPE {agent}: random exploration -> arm {selected_idx}")
        else:
            # Use suggest_int - during startup it will be random, after that TPE
            selected_idx = trial.suggest_categorical("arm", list(range(n_arms)))
            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            if n_completed < self.tpe_config.n_startup_trials:
                logger.debug(f"[Turn {turn}] TPE {agent}: startup trial {n_completed + 1}/{self.tpe_config.n_startup_trials} -> arm {selected_idx}")
            else:
                logger.debug(f"[Turn {turn}] TPE {agent}: TPE suggested arm {selected_idx}")

        # Store scores for tracing
        self.score_traces[agent].append(self.arm_scores[agent].tolist())

        # Compute probabilities for logging
        probs = self._compute_arm_probabilities(agent)
        self.last_selection_probs[agent] = float(probs[selected_idx])

        self.current_selections[agent] = selected_idx

        prompt_text = self.prompt_space.get_prompt(agent, selected_idx)
        embedding = self.prompt_space.get_embedding(agent, selected_idx)

        return selected_idx, prompt_text, embedding

    async def select_async(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """Async version of select (TPE is synchronous, so just wraps select)."""
        return self.select(agent, turn)

    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """
        Update the TPE model with observed reward.

        Records the observation and tells Optuna the trial result.
        """
        # Record observation
        self.observations[agent].append((arm_index, reward))
        self.arm_counts[agent][arm_index] += 1

        # Update arm scores
        # Replace with latest reward
        self.arm_scores[agent][arm_index] = reward

        # Update Optuna study with the result using pending trial
        study = self.studies[agent]
        pending_trial = self._pending_trials[agent]

        if pending_trial is not None:
            # Tell the result to the pending trial
            study.tell(pending_trial, reward)
            self._pending_trials[agent] = None
        else:
            logger.warning(f"[Turn {turn}] TPE {agent}: No pending trial to update")

        # Record selection
        prompt_text = self.prompt_space.get_prompt(agent, arm_index)
        embedding = self.prompt_space.get_embedding(agent, arm_index)
        prob = self.last_selection_probs.get(agent, 1.0 / self.n_arms[agent])

        record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=prompt_text,
            embedding=embedding,
            reward=reward,
            cumulative_score=float(self.arm_scores[agent][arm_index]),
            selection_probability=prob,
        )
        self.selection_history.append(record)

        logger.info(
            f"[Turn {turn}] TPE updated {agent} arm {arm_index}: "
            f"reward={reward:.2f}, count={self.arm_counts[agent][arm_index]}, "
            f"score={self.arm_scores[agent][arm_index]:.4f}"
        )

    def train_model(self, verbose: bool = True) -> float:  # noqa: ARG002
        """
        TPE is an online algorithm - no batch training needed.

        The TPE model is updated incrementally with each observation.
        """
        logger.debug("TPE is an online algorithm, no batch training needed")
        return 0.0

    def get_arm_statistics(self, agent: Literal["p1", "p2"]) -> dict:
        """Get statistics for all arms of an agent."""
        probs = self._compute_arm_probabilities(agent)
        return {
            "arm_scores": self.arm_scores[agent].tolist(),
            "arm_counts": self.arm_counts[agent].tolist(),
            "probabilities": probs.tolist(),
            "n_arms": self.n_arms[agent],
            "n_trials": len(self.observations[agent]),
            "n_startup_trials": self.tpe_config.n_startup_trials,
            "gamma": self.tpe_config.gamma,
        }

    def reset(self) -> None:
        """Reset the TPE bandit to initial state."""
        for agent_key in ("p1", "p2"):
            agent: Literal["p1", "p2"] = agent_key  # type: ignore
            self.observations[agent] = []
            self.arm_scores[agent] = np.zeros(self.n_arms[agent], dtype=np.float64)
            self.arm_counts[agent] = np.zeros(self.n_arms[agent], dtype=np.int32)

            # Recreate studies
            self.studies[agent] = optuna.create_study(
                direction="maximize",
                sampler=self.samplers[agent],
            )

        self.selection_history = []
        self.score_traces = {"p1": [], "p2": []}
        logger.info("TPE Bandit reset to initial state")

