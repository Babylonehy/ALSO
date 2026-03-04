"""
NeuralUCB Bandit implementation for dynamic prompt optimization.

This module implements a Neural Upper Confidence Bound (NeuralUCB) algorithm
based on the NeuralTSDiag implementation from INSTINCT (LlamaForMLPRegression.py).

The key idea is to use per-sample gradients (computed via backpack) to estimate
uncertainty. The UCB is computed as:
    UCB = prediction + sqrt(lambda * nu * feature^2 / U)
where U is the diagonal of the gradient covariance matrix, updated incrementally.

Key differences from standard NeuralUCB:
- U matrix is updated incrementally: U += g[arm] * g[arm] after each selection
- Input standardization using mean/std computed during training
- sigma formula includes lambda: sqrt(sum(lambda * nu * g^2 / U))
"""

import copy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# Import backpack for per-sample gradient computation
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
    BACKPACK_AVAILABLE = True
except ImportError:
    BACKPACK_AVAILABLE = False
    logger.warning("backpack-for-pytorch not available. NeuralUCB will use fallback mode.")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .prompt_space import PromptSpace


class Network(nn.Module):
    """
    Neural network for NeuralUCB value estimation.

    Simple MLP with ReLU activations, matching original NeuralUCB implementation.
    Uses PyTorch default initialization (same as original).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 100,
    ) -> None:
        super().__init__()

        # Match original NeuralUCB: 2-layer MLP (input -> hidden -> 1)
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        # Use PyTorch default initialization (same as original NeuralUCB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.fc2(self.activate(self.fc1(x)))


class NeuralUCBBandit(BaseBandit):
    """
    Neural UCB bandit algorithm for selecting optimal bio paraphrases.

    Based on the NeuralDBDiag implementation from LlamaForMLPRegression.py:
    - Uses per-sample gradients (computed via backpack) to estimate uncertainty
    - UCB = prediction + nu * sqrt(sum(feature^2 / U))
    - U is the diagonal of the gradient covariance matrix
    """

    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | None = None,
        tensorboard_dir: Path | None = None,
    ) -> None:
        super().__init__(prompt_space, config, tensorboard_dir)

        # Set random seed for reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            logger.info(f"Random seed set to {self.config.seed} for reproducibility")

        if not BACKPACK_AVAILABLE:
            raise ImportError(
                "backpack-for-pytorch is required for NeuralUCB. "
                "Install with: pip install backpack-for-pytorch"
            )

        # NeuralUCB specific parameters
        self.lambda_reg = self.config.alpha  # Regularization for covariance matrix
        self.nu = self.config.beta  # Exploration parameter for UCB

        # TensorBoard writer
        if tensorboard_dir and SummaryWriter:
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        else:
            if tensorboard_dir and SummaryWriter is None:
                logger.warning(
                    "TensorBoard requested but torch.utils.tensorboard not found."
                )
            self.writer = None

        # Context embedding settings
        self.use_context_embedding = self.config.use_context_embedding
        self.context_embedding_dim = self.config.context_embedding_dim
        self._context_dim_detected = False  # Flag to track if we've detected the actual dimension

        # Calculate input dimension: bio_embedding + (optional) context_embedding
        if self.use_context_embedding:
            self.input_dim = self.embedding_dim + self.context_embedding_dim
            logger.info(
                f"Context embedding enabled. Input dim = {self.embedding_dim} (bio) + "
                f"{self.context_embedding_dim} (context) = {self.input_dim}"
            )
        else:
            self.input_dim = self.embedding_dim

        # Store current context embedding for each agent (updated each turn)
        self.current_context_embedding: np.ndarray | None = None

        # Initialize the neural network with backpack extension
        # Use float32 (same as original NeuralUCB)
        base_network = Network(
            input_dim=self.input_dim,
            hidden_size=self.config.hidden_size,
        ).to(self.config.device).float()

        self.model = extend(base_network)

        # Store initial state dict for resetting
        self._init_state_dict = copy.deepcopy(self.model.state_dict())

        # Total number of parameters (for gradient covariance matrix)
        self.total_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # INSTINCT-style: Initialize U matrix as lambda * I (diagonal approximation)
        # U is updated incrementally: U += g[arm] * g[arm] after each selection
        self.U: torch.Tensor = self.lambda_reg * torch.ones(self.total_param)

        logger.info(
            f"Initialized NeuralUCBBandit (INSTINCT-style) with input_dim={self.input_dim}, "
            f"bio_dim={self.embedding_dim}, use_context={self.use_context_embedding}, "
            f"lambda={self.lambda_reg}, nu={self.nu}, total_params={self.total_param}"
        )

    def reset_model(self) -> None:
        """Reset model to initial weights before training.

        Also resets U matrix to lambda * I and clears standardization parameters.
        """
        # Recreate model with current input_dim (may have changed due to context embedding)
        base_network = Network(
            input_dim=self.input_dim,
            hidden_size=self.config.hidden_size,
        ).to(self.config.device).float()

        self.model = extend(base_network)
        self._init_state_dict = copy.deepcopy(self.model.state_dict())

        # Update total param count
        self.total_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Reset U matrix to lambda * I (INSTINCT-style)
        self.U = self.lambda_reg * torch.ones(self.total_param)

    def set_context_embedding(self, context_embedding: np.ndarray) -> None:
        """
        Set the current context embedding for use in arm selection.

        This should be called before select() when use_context_embedding is True.

        Args:
            context_embedding: The context embedding from dialogue history
        """
        self.current_context_embedding = context_embedding
        logger.debug(f"Set context embedding: {context_embedding}")

        # Auto-detect context embedding dimension on first call
        if not self._context_dim_detected:
            actual_dim = len(context_embedding)
            if actual_dim != self.context_embedding_dim:
                logger.warning(
                    f"Context embedding dim mismatch: config={self.context_embedding_dim}, "
                    f"actual={actual_dim}. Reinitializing model."
                )
                self.context_embedding_dim = actual_dim
                self.input_dim = self.embedding_dim + actual_dim

                # Reinitialize model with correct input dimension
                base_network = Network(
                    input_dim=self.input_dim,
                    hidden_size=self.config.hidden_size,
                ).to(self.config.device).float()

                self.model = extend(base_network)
                self._init_state_dict = copy.deepcopy(self.model.state_dict())
                self.total_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

                logger.info(f"Reinitialized model with input_dim={self.input_dim}")
            self._context_dim_detected = True

    def _compute_gradients(self, context: torch.Tensor, batch_size: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-sample gradients and predictions for all context embeddings.

        Returns:
            Tuple of (gradients, predictions)
            - gradients: shape (n_samples, total_param)
            - predictions: shape (n_samples, 1)
        """
        context_size = context.shape[0]
        n_batches = context_size // batch_size + int((context_size % batch_size) != 0)

        g_list = []
        mu_list = []

        self.model.train()

        for i in range(n_batches):
            if i == n_batches - 1:
                context_batch = context[(i * batch_size):]
            else:
                context_batch = context[(i * batch_size):((i + 1) * batch_size)]

            # Forward pass
            mu = self.model(context_batch)
            sum_mu = torch.sum(mu)

            # Compute per-sample gradients using backpack
            with backpack(BatchGrad()):
                sum_mu.backward()

            # Concatenate gradients from all parameters
            g_batch = torch.cat(
                [p.grad_batch.flatten(start_dim=1).detach() for p in self.model.parameters()],
                dim=1
            )

            g_list.append(g_batch.cpu())
            mu_list.append(mu.detach().cpu())

        gradients = torch.vstack(g_list)
        predictions = torch.vstack(mu_list)

        return gradients, predictions

    def train_model(self, verbose: bool = True) -> float:  # noqa: ARG002
        """
        Train the neural network on accumulated reward data.

        When use_context_embedding is True, uses [bio_embedding, context_embedding] as input.

        Returns:
            Final training loss
        """
        if len(self.selection_history) == 0:
            logger.warning("No selection history to train on")
            return float("nan")

        # Auto-detect context embedding dimension from training data before reset
        if self.use_context_embedding:
            for rec in self.selection_history:
                if rec.context_embedding is not None:
                    actual_context_dim = len(rec.context_embedding)
                    if actual_context_dim != self.context_embedding_dim:
                        logger.info(
                            f"Updating context embedding dim: {self.context_embedding_dim} -> {actual_context_dim}"
                        )
                        self.context_embedding_dim = actual_context_dim
                        self.input_dim = self.embedding_dim + actual_context_dim
                    break  # Only need to check first record with context

        # Reset to initial weights before training
        self.reset_model()
        self.model.to(self.config.device)

        # Prepare training data - combine bio and context embeddings if enabled
        combined_embeddings = []
        for rec in self.selection_history:
            if self.use_context_embedding and rec.context_embedding is not None:
                # Concatenate bio embedding and context embedding
                combined = np.concatenate([rec.embedding, rec.context_embedding])
                combined_embeddings.append(combined)
            else:
                if self.use_context_embedding:
                    # Context embedding expected but not available - pad with zeros
                    context_zeros = np.zeros(self.context_embedding_dim)
                    combined = np.concatenate([rec.embedding, context_zeros])
                    combined_embeddings.append(combined)
                else:
                    combined_embeddings.append(rec.embedding)

        embeddings = torch.stack([
            torch.tensor(e, dtype=torch.float32)
            for e in combined_embeddings
        ]).to(self.config.device)
        rewards = torch.tensor(
            [rec.reward for rec in self.selection_history],
            dtype=torch.float32
        ).to(self.config.device).reshape(-1, 1)

        n_samples = len(self.selection_history)

        # Weight decay based on sample count (original NeuralUCB: lambda / n_samples)
        weight_decay = self.lambda_reg / n_samples
        # Use SGD with lr=1e-2 (same as original NeuralUCB)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=weight_decay)

        loss_fn = nn.MSELoss()
        self.model.train()

        final_loss = 0.0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Training NeuralUCB:[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("Epoch: {task.completed}/{task.total}"),
            TextColumn("Loss: {task.fields[loss]:.6f}"),
            transient=False,
        ) as progress:
            task_id = progress.add_task("Training", total=self.config.epochs, loss=0.0)

            for epoch in range(self.config.epochs):
                self.model.zero_grad()
                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(embeddings)
                loss = loss_fn(predictions, rewards)

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss at epoch {epoch}, resetting model and stopping training")
                    self.reset_model()
                    return float("nan")

                loss.backward()
                optimizer.step()

                final_loss = loss.item()
                progress.update(task_id, advance=1, loss=final_loss)

                if self.writer:
                    self.writer.add_scalar("Training/Loss", final_loss, epoch)

        # Check for NaN in model weights after training
        has_nan = any(torch.isnan(p).any() or torch.isinf(p).any() for p in self.model.parameters())
        if has_nan:
            logger.warning("NaN/Inf detected in model weights after training, resetting model")
            self.reset_model()
            return float("nan")

        logger.info(f"Training complete. Final loss: {final_loss:.6f}")
        return final_loss

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select an arm using NeuralUCB algorithm (INSTINCT-style).

        When use_context_embedding is True, combines bio embedding with the current
        context embedding (set via set_context_embedding) for scoring.

        INSTINCT-style UCB computation:
        - Input standardization using mean/std from training
        - sigma = sqrt(sum(lambda * nu * g^2 / U))
        - UCB = prediction + sigma
        - U is updated incrementally: U += g[arm] * g[arm]

        Args:
            agent: Which agent to select for ("p1" or "p2")
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, embedding)
        """
        if self._stopped:
            logger.warning("NeuralUCB stopped, using current selection")
            idx = self.current_selections[agent]
            return idx, self.prompt_space.get_prompt(agent, idx), self.prompt_space.get_embedding(agent, idx)

        # Get all bio embeddings for this agent
        bio_embeddings = self.prompt_space.get_all_embeddings(agent)

        # Combine with context embedding if enabled
        if self.use_context_embedding:
            if self.current_context_embedding is not None:
                # Update context_embedding_dim if it changed
                actual_context_dim = len(self.current_context_embedding)
                if actual_context_dim != self.context_embedding_dim:
                    logger.info(f"Updating context_embedding_dim from {self.context_embedding_dim} to {actual_context_dim}")
                    self.context_embedding_dim = actual_context_dim
                # Replicate context embedding for each arm
                n_arms = bio_embeddings.shape[0]
                context_replicated = np.tile(self.current_context_embedding, (n_arms, 1))
                combined_embeddings = np.concatenate([bio_embeddings, context_replicated], axis=1)
            else:
                # No context yet - pad with zeros
                n_arms = bio_embeddings.shape[0]
                context_zeros = np.zeros((n_arms, self.context_embedding_dim))
                combined_embeddings = np.concatenate([bio_embeddings, context_zeros], axis=1)

            # Check if model input dimension matches combined embedding dimension
            combined_dim = combined_embeddings.shape[1]
            if combined_dim != self.input_dim:
                logger.warning(
                    f"Input dimension mismatch in select: model expects {self.input_dim}, "
                    f"but combined embedding is {combined_dim}. Reinitializing model."
                )
                self.input_dim = combined_dim
                # Reinitialize model with correct input dimension
                base_network = Network(
                    input_dim=self.input_dim,
                    hidden_size=self.config.hidden_size,
                ).to(self.config.device).float()

                self.model = extend(base_network)
                self._init_state_dict = copy.deepcopy(self.model.state_dict())
                self.total_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                # Reset U matrix for new model
                self.U = self.lambda_reg * torch.ones(self.total_param)
                logger.info(f"Reinitialized model with input_dim={self.input_dim}")

            context = torch.tensor(combined_embeddings, dtype=torch.float32).to(self.config.device)
        else:
            context = torch.tensor(bio_embeddings, dtype=torch.float32).to(self.config.device)

        # Compute gradients and predictions for current arms
        g_list, mu = self._compute_gradients(context)

        predictions_np = mu.squeeze(-1).numpy()

        # INSTINCT-style UCB: sigma = sqrt(sum(lambda * nu * g^2 / U))
        # U is maintained incrementally (initialized as lambda * I, updated after each selection)
        sigma = torch.sqrt(torch.sum(self.lambda_reg * self.nu * g_list * g_list / self.U, dim=1))
        sigma_np = sigma.numpy()

        # UCB = prediction + sigma
        ucb_values = predictions_np + sigma_np

        logger.debug(f"UCB values for pred {predictions_np},\n uc:{sigma_np}, \n sum: {ucb_values}, \n nu={self.nu}, lambda={self.lambda_reg}")

        # Store predictions in score traces
        self.score_traces[agent].append(predictions_np.tolist())

        # Check for numerical issues - now with recovery instead of stopping
        if np.isnan(ucb_values).any() or np.isinf(ucb_values).any():
            logger.warning(
                f"Numerical issues in NeuralUCB predictions (nan={np.isnan(ucb_values).sum()}, "
                f"inf={np.isinf(ucb_values).sum()}). Replacing with predictions only."
            )
            # Replace NaN/Inf with just predictions (no exploration bonus)
            ucb_values = np.where(np.isfinite(ucb_values), ucb_values, predictions_np)
            # If predictions also have issues, use uniform random
            if np.isnan(ucb_values).any() or np.isinf(ucb_values).any():
                logger.error("Predictions also have numerical issues. Using random selection.")
                ucb_values = np.random.rand(len(ucb_values))

        # Select arm with highest UCB
        selected_idx = int(np.argmax(ucb_values))
        self.current_selections[agent] = selected_idx

        # INSTINCT-style: Update U matrix incrementally with selected arm's gradient
        # U += g[arm] * g[arm]
        self.U = self.U + g_list[selected_idx] * g_list[selected_idx]

        prompt_text = self.prompt_space.get_prompt(agent, selected_idx)
        embedding = self.prompt_space.get_embedding(agent, selected_idx)

        logger.info(
            f"[Turn {turn}] NeuralUCB selected {agent} arm {selected_idx}, "
            f"UCB={ucb_values[selected_idx]:.4f}, "
            f"pred={predictions_np[selected_idx]:.4f}, "
            f"uncertainty={sigma_np[selected_idx]:.4f}, "
            f"has_context={self.current_context_embedding is not None}"
        )

        # Log top UCB values
        top_indices = np.argsort(ucb_values)[-5:][::-1]
        logger.debug(
            f"Top 5 {agent} arms: indices={top_indices.tolist()}, "
            f"UCBs=[{', '.join(f'{ucb_values[i]:.4f}' for i in top_indices)}]"
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

        Stores the gradient of the selected arm for future UCB computation.

        Args:
            agent: Which agent was updated ("p1" or "p2")
            arm_index: The arm index that was selected
            reward: The observed reward
            turn: The turn number when this selection was made
        """
        self.update_with_context(agent, arm_index, reward, turn, context_embedding=None)

    def update_with_context(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
        context_embedding: np.ndarray | None,
        dimension_rewards: dict[str, float] | None = None,  # noqa: ARG002
    ) -> None:
        """
        Update the bandit with the observed reward and optional context embedding.

        Stores the gradient of the selected arm for future UCB computation.

        Args:
            agent: Which agent was updated ("p1" or "p2")
            arm_index: The arm index that was selected
            reward: The observed reward
            turn: The turn number when this selection was made
            context_embedding: Optional context embedding from dialogue history
            dimension_rewards: Optional per-dimension rewards (not used in NeuralUCB,
                but accepted for API compatibility with other bandits)
        """
        prompt_text = self.prompt_space.get_prompt(agent, arm_index)
        bio_embedding = self.prompt_space.get_embedding(agent, arm_index)

        # Combine bio embedding with context if provided
        if self.use_context_embedding and context_embedding is not None:
            combined_embedding = np.concatenate([bio_embedding, context_embedding])
            # Update context_embedding_dim if it changed
            actual_context_dim = len(context_embedding)
            if actual_context_dim != self.context_embedding_dim:
                logger.info(f"Updating context_embedding_dim from {self.context_embedding_dim} to {actual_context_dim}")
                self.context_embedding_dim = actual_context_dim
        elif self.use_context_embedding:
            # Context expected but not provided - pad with zeros
            context_zeros = np.zeros(self.context_embedding_dim)
            combined_embedding = np.concatenate([bio_embedding, context_zeros])
        else:
            combined_embedding = bio_embedding

        # Check if model input dimension matches combined embedding dimension
        combined_dim = len(combined_embedding)
        if combined_dim != self.input_dim:
            logger.warning(
                f"Input dimension mismatch: model expects {self.input_dim}, "
                f"but combined embedding is {combined_dim}. Reinitializing model."
            )
            self.input_dim = combined_dim
            # Reinitialize model with correct input dimension
            base_network = Network(
                input_dim=self.input_dim,
                hidden_size=self.config.hidden_size,
            ).to(self.config.device).float()

            self.model = extend(base_network)
            self._init_state_dict = copy.deepcopy(self.model.state_dict())
            self.total_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Reinitialized model with input_dim={self.input_dim}")

        # Get current prediction for logging
        embedding_tensor = torch.tensor(combined_embedding, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            pred = self.model(embedding_tensor)
        pred_value = pred.item()

        record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=prompt_text,
            embedding=bio_embedding,
            reward=reward,
            cumulative_score=pred_value,
            context_embedding=context_embedding,
        )
        self.selection_history.append(record)

        logger.info(
            f"[Turn {turn}] Updated {agent} arm {arm_index} with reward={reward:.2f}, "
            f"pred={pred_value:.4f}, history_size={len(self.selection_history)}, "
            f"has_context={context_embedding is not None}"
        )

    def save_model(self, path: Path) -> None:
        """Save the model weights to a file."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"NeuralUCB model saved to {path}")

