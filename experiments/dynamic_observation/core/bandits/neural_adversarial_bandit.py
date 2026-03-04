"""
Neural Adversarial Bandit implementation for dynamic prompt optimization.

This module implements a neural network-based adversarial bandit algorithm that uses
value estimation with EXP3-style exploration to select optimal agent bio
paraphrases during Sotopia simulation.

Note: This is NOT the standard EXP3 algorithm. For standard EXP3, use exp3_bandit.py.
"""

import copy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .prompt_space import PromptSpace


console = Console()


class ValueNetwork(nn.Module):
    """Neural network for estimating arm values from embeddings.

    Supports two modes:
    - Single output (output_dim=1): Predicts averaged reward directly
    - Multi-dim output (output_dim=7): Predicts each dimension score separately

    Uses Pre-LN structure: LayerNorm → Linear → ReLU (more stable training)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 512,
        depth: int = 1,
        output_dim: int = 1,  # 1 for single reward, 7 for multi-dimensional
        use_layer_norm: bool = True,  # Whether to use LayerNorm before each linear layer
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm
        self.activate = nn.GELU()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # Input layer: (optional LN on input) → Linear → ReLU
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(input_dim))
        self.layers.append(nn.Linear(input_dim, hidden_size))

        # Hidden layers: LN → Linear → ReLU
        for _ in range(depth - 1):
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_size))
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer: LN → Linear (no activation)
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights with standard initialization."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network (Pre-LN structure).

        Structure: [LN → Linear → ReLU] × (depth) → [LN → Linear]

        Returns:
            If output_dim=1: shape (batch,) squeezed
            If output_dim>1: shape (batch, output_dim)
        """
        # Hidden layers with activation
        for i, layer in enumerate(self.layers[:-1]):
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = layer(x)
            x = self.activate(x)

        # Output layer (no activation)
        if self.use_layer_norm:
            x = self.layer_norms[-1](x)
        x = self.layers[-1](x)

        if self.output_dim == 1:
            return x.squeeze(-1)
        return x  # (batch, output_dim)


class NeuralAdversarialBandit(BaseBandit):
    """
    Neural network-based adversarial bandit for selecting optimal bio paraphrases.

    Uses a neural network to estimate values from embeddings, with EXP3-style
    exploration to balance exploration and exploitation.

    Note: This is NOT the standard EXP3 algorithm. For standard EXP3, use EXP3Bandit.
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

        # TensorBoard writer
        if tensorboard_dir and SummaryWriter:
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        else:
            if tensorboard_dir and SummaryWriter is None:
                logger.warning(
                    "TensorBoard requested but torch.utils.tensorboard not found. Skipping TB logging."
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

        # Multi-dimensional prediction mode
        self.multi_dim_prediction = self.config.multi_dim_prediction
        self.dimension_names = self.config.dimension_names
        self.n_dims = len(self.dimension_names) if self.multi_dim_prediction else 1

        # Initialize the value network
        self.model = ValueNetwork(
            input_dim=self.input_dim,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            output_dim=self.n_dims,  # 7 for multi-dim, 1 for single
            use_layer_norm=self.config.use_layer_norm,
        ).to(self.config.device)

        # Store initial weights for resetting
        self._init_weights = copy.deepcopy(self.model.state_dict())

        # Optimizer (will be recreated on each training)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

        # Store last selection probability for recording
        # Initialize to uniform probability (1/n_arms)
        n_p1_arms = self.prompt_space.p1_embeddings.shape[0]
        n_p2_arms = self.prompt_space.p2_embeddings.shape[0]
        self.last_selection_probs: dict[str, float] = {
            "p1": 1.0 / n_p1_arms,
            "p2": 1.0 / n_p2_arms,
        }

        # Arm selection counts for UCB exploration bonus
        self.arm_counts: dict[str, torch.Tensor] = {
            "p1": torch.zeros(n_p1_arms, dtype=torch.float64),
            "p2": torch.zeros(n_p2_arms, dtype=torch.float64),
        }
        self.total_selections: dict[str, int] = {"p1": 0, "p2": 0}

        logger.info(
            f"Initialized NeuralAdversarialBandit with input_dim={self.input_dim}, "
            f"bio_dim={self.embedding_dim}, use_context={self.use_context_embedding}, "
            f"eta={self.config.eta}, device={self.config.device}, "
            f"multi_dim_prediction={self.multi_dim_prediction}"
        )

    def reset_model(self) -> None:
        """Reset model with fresh weights and adjusted weight decay.

        Uses EXPO-style dynamic weight decay: weight_decay = 1 / num_samples
        - Few samples: strong regularization to prevent overfitting
        - Many samples: weak regularization to trust the data more
        """
        logger.warning("Resetting model with fresh weights")

        # Recreate model with current input_dim (may have changed due to context embedding)
        self.model = ValueNetwork(
            input_dim=self.input_dim,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            output_dim=self.n_dims,
            use_layer_norm=self.config.use_layer_norm,
        ).to(self.config.device).double()

        # Dynamic weight decay: 0.01 / num_samples (capped at 0.01)
        # Few samples -> stronger regularization; many samples -> weaker regularization
        num_samples = max(1, len(self.selection_history))
        wd = min(0.01, 0.01 / num_samples)
        logger.info(f"Dynamic weight decay: min(0.01, 0.01/{num_samples}) = {wd:.6f}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=wd,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.learning_rate * 0.01,
        )

    def train_model(self, verbose: bool = True) -> float:  # noqa: ARG002
        """
        Train the value network on accumulated reward data.

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

        self.reset_model()
        self.model.to(self.config.device)

        # Prepare training data - combine bio and context embeddings if enabled
        combined_embeddings = []
        for rec in self.selection_history:
            # logger.debug(f"rec.context_embedding: {rec.context_embedding}")
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

        # NN predicts raw reward directly (range 0-1)
        # NOTE: We always use raw reward for NN training, NOT importance-weighted reward.
        # Importance weighting is applied separately in fitness update (EXP3 style).
        # Using IW reward for training would cause NN to predict values >> 1,
        # which then causes fitness explosion when used in exp(eta * predicted_reward).
        arm_indices = [rec.arm_index for rec in self.selection_history]

        embeddings_tensor = torch.stack([
            torch.tensor(e, dtype=torch.float64) for e in combined_embeddings
        ]).to(self.config.device)

        # Prepare rewards tensor based on prediction mode
        if self.multi_dim_prediction:
            # Multi-dimensional: extract 7 dimension scores for each record
            dim_rewards_list = []
            for rec in self.selection_history:
                if rec.dimension_rewards is not None:
                    # Use actual dimension rewards (order matters!)
                    dim_scores = [rec.dimension_rewards.get(dim, rec.reward) for dim in self.dimension_names]
                else:
                    # Fallback: use single reward for all dimensions
                    dim_scores = [rec.reward] * self.n_dims
                dim_rewards_list.append(dim_scores)
            rewards_tensor = torch.tensor(dim_rewards_list, dtype=torch.float64).to(self.config.device)
            # rewards_tensor shape: (n_samples, 7)
            rewards = [sum(r) / len(r) for r in dim_rewards_list]  # For display
        else:
            # Single-dimensional: use averaged reward
            rewards = [rec.reward for rec in self.selection_history]
            rewards_tensor = torch.tensor(rewards,dtype=torch.float64).to(self.config.device)

        if arm_indices and rewards:
            # Display current training samples with Rich so we can inspect without running simulations
            table = Table(
                show_header=True,
                header_style="bold cyan",
                title_style="bold white",
            )
            table.add_column("#", justify="right", style="dim")
            table.add_column("Arm", justify="center")

            if self.multi_dim_prediction:
                # Add columns for each dimension (short names)
                dim_short_names = ["bel", "rel", "kno", "sec", "soc", "fin", "goal"]
                for short_name in dim_short_names:
                    table.add_column(short_name, justify="right", style="cyan")
                table.add_column("avg", justify="right", style="bold green")

                for idx, (arm_idx, dim_scores) in enumerate(zip(arm_indices, dim_rewards_list), start=1):
                    row = [str(idx), str(arm_idx)]
                    for score in dim_scores:
                        row.append(f"{score:.3f}")
                    row.append(f"{sum(dim_scores)/len(dim_scores):.3f}")
                    table.add_row(*row)
            else:
                table.add_column("Normalized Reward", justify="right")
                for idx, (arm_idx, reward) in enumerate(zip(arm_indices, rewards), start=1):
                    table.add_row(str(idx), str(arm_idx), f"{reward:.6f}")

            scenario_label = getattr(self.prompt_space, "scenario_id", None)
            panel_title = "Training Samples"
            if scenario_label:
                panel_title += f" • Scenario {scenario_label}"

            console.print(
                Panel(
                    table,
                    title=panel_title,
                    border_style="green",
                )
            )

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(embeddings_tensor, rewards_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        logger.debug(f"Training dataset size: {len(dataset)}")
        loss_fn = nn.MSELoss()
        self.model.train()

        final_loss = 0.0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold red]Training NeuralAdversarial:[/bold red]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("Epoch: {task.completed}/{task.total}"),
            TextColumn("Loss: {task.fields[loss]:.6f}"),
            transient=False,
        ) as progress:
            task_id = progress.add_task("Training", total=self.config.epochs, loss=0.0)

            early_stop_patience = 10
            unchanged_eps = 1e-5
            unchanged_count = 0
            last_epoch_loss: float | None = None

            for epoch in range(self.config.epochs):
                for batch_embeddings, batch_rewards in dataloader:
                    self.optimizer.zero_grad()
                    predictions = self.model(batch_embeddings)
                    loss = loss_fn(predictions, batch_rewards)
                    loss.backward()
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()
                final_loss = loss.item()

                progress.update(task_id, advance=1, loss=final_loss)

                if self.writer:
                    self.writer.add_scalar("Training/Loss", final_loss, epoch)

                if last_epoch_loss is not None and abs(final_loss - last_epoch_loss) <= unchanged_eps:
                    unchanged_count += 1
                    if unchanged_count >= early_stop_patience:
                        logger.info(
                            f"Early stopping: loss unchanged for {early_stop_patience} consecutive epochs "
                            f"(eps={unchanged_eps:g}). Final loss={final_loss:.6f}"
                        )
                        break
                else:
                    unchanged_count = 0
                last_epoch_loss = final_loss

        logger.info(f"Training complete. Final loss: {final_loss:.6f}")
        
        return final_loss

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
                self.model = ValueNetwork(
                    input_dim=self.input_dim,
                    hidden_size=self.config.hidden_size,
                    depth=self.config.depth,
                    use_layer_norm=self.config.use_layer_norm,
                ).to(self.config.device).double()
                self._init_weights = copy.deepcopy(self.model.state_dict())

                logger.info(f"Reinitialized model with input_dim={self.input_dim}")
            self._context_dim_detected = True

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select an arm for the given agent using EXP3-style algorithm.

        When use_context_embedding is True, combines bio embedding with the current
        context embedding (set via set_context_embedding) for scoring.

        Args:
            agent: Which agent to select for ("p1" or "p2")
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, bio_embedding)
        """
        if self._stopped:
            logger.warning("Bandit stopped due to numerical issues, using current selection")
            idx = self.current_selections[agent]
            return idx, self.prompt_space.get_prompt(agent, idx), self.prompt_space.get_embedding(agent, idx)

        # Get all bio embeddings for this agent
        bio_embeddings = self.prompt_space.get_all_embeddings(agent)

        # Combine with context embedding if enabled
        if self.use_context_embedding:
            if self.current_context_embedding is not None:
                # Replicate context embedding for each arm
                n_arms = bio_embeddings.shape[0]
                context_replicated = np.tile(self.current_context_embedding, (n_arms, 1))
                combined_embeddings = np.concatenate([bio_embeddings, context_replicated], axis=1)
            else:
                # No context yet - pad with zeros
                n_arms = bio_embeddings.shape[0]
                context_zeros = np.zeros((n_arms, self.context_embedding_dim))
                combined_embeddings = np.concatenate([bio_embeddings, context_zeros], axis=1)
            embeddings_tensor = torch.tensor(combined_embeddings, dtype=torch.float64).to(self.config.device)
        else:
            embeddings_tensor = torch.tensor(bio_embeddings, dtype=torch.float64).to(self.config.device)

        # Predict scores with the model
        self.model.eval()
        with torch.no_grad():
            raw_scores = self.model(embeddings_tensor).cpu().numpy()
            if self.multi_dim_prediction:
                # raw_scores shape: (n_arms, 7) - average across dimensions for arm selection
                scores = np.mean(raw_scores, axis=1)  # (n_arms,)
            else:
                # raw_scores shape: (n_arms,) or (n_arms, 1)
                scores = raw_scores.flatten()

        # Store NN predicted scores - ALL arms should have scores every turn
        n_arms = len(scores)
        self.score_traces[agent].append(scores.tolist())

        # Initialize actual_score_traces for this turn (zeros, will be updated in update_with_context)
        self.actual_score_traces[agent].append([0.0] * n_arms)

        # Compute cumulative scores from both traces with exponential decay
        decay = self.config.score_decay
        n_turns = len(self.score_traces[agent])

        # Create decay weights
        if decay < 1.0 and n_turns > 1:
            weights = torch.tensor([decay ** (n_turns - 1 - t) for t in range(n_turns)], dtype=torch.float64)
        else:
            weights = torch.ones(n_turns, dtype=torch.float64)
        logger.debug(f"[Turn {turn}] Decay weights: {weights}")
        # 1. NN predicted cumulative scores
        nn_score_tensor = torch.tensor(self.score_traces[agent], dtype=torch.float64)
        nn_cumulative = torch.sum(nn_score_tensor * weights.unsqueeze(1), dim=0)

        # 2. Actual observed cumulative scores (with importance sampling)
        actual_score_tensor = torch.tensor(self.actual_score_traces[agent], dtype=torch.float64)
        actual_cumulative = torch.sum(actual_score_tensor * weights.unsqueeze(1), dim=0)
        
        logger.debug(f"[Turn {turn}] {agent} actual cumulative: {actual_cumulative}")
        
        # 3. Compute cumulative scores based on configured mode
        mode = self.config.cumulative_score_mode
        if mode == "nn":
            cumulative_scores = nn_cumulative
            logger.debug(f"[Turn {turn}] {agent} cumulative (nn only): {cumulative_scores}")
        elif mode == "actual":
            cumulative_scores = actual_cumulative
            logger.debug(f"[Turn {turn}] {agent} cumulative (actual only): {cumulative_scores}")
        elif mode == "mean":
            # Simple average (original behavior, no normalization)
            cumulative_scores = (nn_cumulative + actual_cumulative) / 2.0
            logger.debug(
                f"[Turn {turn}] {agent} cumulative (mean): nn={nn_cumulative}, "
                f"actual={actual_cumulative}, combined={cumulative_scores}"
            )
        elif mode == "mean-zscore":
            # Z-score normalize each component to eliminate scale mismatch
            nn_std = nn_cumulative.std()
            actual_std = actual_cumulative.std()

            if nn_std > 1e-10:
                nn_norm = (nn_cumulative - nn_cumulative.mean()) / nn_std
            else:
                nn_norm = nn_cumulative - nn_cumulative.mean()

            if actual_std > 1e-10:
                actual_norm = (actual_cumulative - actual_cumulative.mean()) / actual_std
            else:
                actual_norm = actual_cumulative - actual_cumulative.mean()

            # Compute NN weight (fixed or adaptive)
            if self.config.adaptive_nn_weight:
                # Sigmoid scheduling: early → trust EXP3, late → trust NN
                # nn_weight = sigmoid((turn - warmup) / scale)
                warmup = self.config.adaptive_nn_weight_warmup
                scale = self.config.adaptive_nn_weight_scale
                nn_w = 1.0 / (1.0 + np.exp(-(turn - warmup) / scale))
                logger.debug(
                    f"[Turn {turn}] Adaptive nn_weight: sigmoid(({turn} - {warmup}) / {scale}) = {nn_w:.4f}"
                )
            else:
                nn_w = self.config.nn_weight

            cumulative_scores = nn_w * nn_norm + (1 - nn_w) * actual_norm
            logger.debug(
                f"[Turn {turn}] {agent} cumulative (zscore mean, nn_w={nn_w:.4f}): "
                f"nn_raw={nn_cumulative}, actual_raw={actual_cumulative}, "
                f"nn_norm={nn_norm}, actual_norm={actual_norm}, combined={cumulative_scores}"
            )
        else:
            raise ValueError(f"Unknown cumulative_score_mode: {mode}. Use 'nn', 'actual', 'mean', or 'mean-zscore'.")

        # Apply UCB exploration bonus if enabled
        # UCB bonus = ucb_c * sqrt(log(t) / (n_arm + 1))
        # Encourages exploration of less-selected arms
        # if self.config.ucb_exploration:
        #     t = self.total_selections[agent] + 1  # Total selections so far
        #     arm_counts = self.arm_counts[agent]
        #     # UCB bonus: higher for less-selected arms
        #     ucb_bonus = self.config.ucb_c * torch.sqrt(
        #         torch.log(torch.tensor(t, dtype=torch.float64)) / (arm_counts + 1)
        #     )
        #     cumulative_scores = cumulative_scores + ucb_bonus
        #     logger.debug(
        #         f"[Turn {turn}] {agent} UCB bonus (c={self.config.ucb_c}): "
        #         f"max={ucb_bonus.max():.4f}, min={ucb_bonus.min():.4f}"
        #     )

        # Apply dynamic eta scheduling if enabled: eta_t = eta * sqrt(turn + 1)
        # Early turns: small eta -> flat distribution -> exploration
        # Later turns: large eta -> sharp distribution -> exploitation
        if self.config.dynamic_eta:
            effective_eta = self.config.eta * np.sqrt(turn + 1)
            logger.debug(f"[Turn {turn}] Dynamic eta: {self.config.eta:.4f} * sqrt({turn + 1}) = {effective_eta:.4f}")
        else:
            effective_eta = self.config.eta
        logger.debug(f"[Turn {turn}] effective_eta: {effective_eta}, cumulative_scores: {effective_eta * cumulative_scores}.")
        # Apply softmax with eta scaling
        probabilities = torch.softmax(effective_eta * cumulative_scores, dim=0)
        logger.debug(f"[Turn {turn}] {agent} probabilities: {probabilities}")

        # Check for numerical issues
        if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
            logger.error("Numerical issues in probabilities. Stopping bandit.")
            self._stopped = True
            idx = self.current_selections[agent]
            return idx, self.prompt_space.get_prompt(agent, idx), self.prompt_space.get_embedding(agent, idx)

        # Log top probabilities
        top_probs, top_indices = torch.topk(probabilities, min(5, len(probabilities)))
        logger.debug(
            f"Top 5 {agent} arm probabilities: {top_probs.tolist()}, indices: {top_indices.tolist()}"
        )

        # Sample from the distribution
        logger.debug(f"probabilities: {probabilities}")
        selected_idx = torch.multinomial(probabilities, 1).item()
        self.current_selections[agent] = selected_idx

        # Store probability for recording
        self.last_selection_probs[agent] = float(probabilities[selected_idx])

        # Update arm selection counts for UCB exploration
        self.arm_counts[agent][selected_idx] += 1
        self.total_selections[agent] += 1

        # Only update selected arm score estimator (optional)
        # if self.config.mask_unselected_scores:
        #     logger.warning("!!!!!!!Masking unselected scores!!!")
        #     n_arms = len(self.score_traces[agent][-1])
        #     masked_scores = [0.0] * n_arms
        #     masked_scores[selected_idx] = self.score_traces[agent][-1][selected_idx]
        #     self.score_traces[agent][-1] = masked_scores

        prompt_text = self.prompt_space.get_prompt(agent, selected_idx)
        bio_embedding = self.prompt_space.get_embedding(agent, selected_idx)

        logger.info(
            f"[Turn {turn}] Selected {agent} arm {selected_idx}, "
            f"cumulative_score = {cumulative_scores[selected_idx]:.4f}, "
            f"prob={probabilities[selected_idx]:.4f}, "
            f"has_context={self.current_context_embedding is not None}"
        )

        return selected_idx, prompt_text, bio_embedding

    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """
        Update the bandit with the observed reward for the selected arm.

        Args:
            agent: Which agent was updated ("p1" or "p2")
            arm_index: The arm index that was selected
            reward: The observed reward (p1_rate or p2_rate from evaluator)
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
        dimension_rewards: dict[str, float] | None = None,
    ) -> None:
        """
        Update the bandit with the observed reward and optional context embedding.

        Args:
            agent: Which agent was updated ("p1" or "p2")
            arm_index: The arm index that was selected
            reward: The observed reward (p1_rate or p2_rate from evaluator, averaged score)
            turn: The turn number when this selection was made
            context_embedding: Optional context embedding from dialogue history
            dimension_rewards: Optional dict of dimension->score for multi-dim prediction mode
                Keys: believability, relationship, knowledge, secret, social_rules,
                      financial_and_material_benefits, goal (all normalized to 0-1)
        """
        prompt_text = self.prompt_space.get_prompt(agent, arm_index)
        bio_embedding = self.prompt_space.get_embedding(agent, arm_index)

        # Update actual_score_traces with importance-weighted reward for selected arm
        # Only selected arm gets the reward, others remain 0
        selection_prob = self.last_selection_probs.get(agent, 1.0)
        selection_prob = max(selection_prob, 0.01)  # Clip to avoid extreme values
        iw_reward = reward / selection_prob  # Importance-weighted reward

        # Update the last entry in actual_score_traces (created in select())
        if len(self.actual_score_traces[agent]) > 0:
            last_trace = self.actual_score_traces[agent][-1]
            if arm_index < len(last_trace):
                last_trace[arm_index] = iw_reward
                logger.debug(
                    f"[Turn {turn}] {agent} actual_score_traces: arm {arm_index} = {iw_reward:.3f} "
                    f"(reward={reward:.2f}, prob={selection_prob:.3f})"
                )

        # Calculate cumulative score for this arm (average of NN and actual)
        cumulative_score = 0.0
        if len(self.score_traces[agent]) > 0:
            nn_score_tensor = torch.tensor(self.score_traces[agent], dtype=torch.float64)
            actual_score_tensor = torch.tensor(self.actual_score_traces[agent], dtype=torch.float64)
            nn_cumulative = torch.sum(nn_score_tensor, dim=0)
            actual_cumulative = torch.sum(actual_score_tensor, dim=0)
            combined_cumulative = (nn_cumulative + actual_cumulative) / 2.0
            if arm_index < len(combined_cumulative):
                cumulative_score = combined_cumulative[arm_index].item()

        record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=prompt_text,
            embedding=bio_embedding,
            reward=reward,
            cumulative_score=cumulative_score,
            selection_probability=self.last_selection_probs.get(agent, 1.0),
            context_embedding=context_embedding,
            dimension_rewards=dimension_rewards,
        )
        self.selection_history.append(record)

        logger.info(
            f"[Turn {turn}] Updated {agent} arm {arm_index} with reward={reward:.2f}, "
            f"cumulative_score={cumulative_score:.4f}, has_context={context_embedding is not None}, "
            f"has_dim_rewards={dimension_rewards is not None}"
        )

    def save_model(self, path: Path) -> None:
        """Save the model weights to a file."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load the model weights from a file."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.config.device))
        self.model.eval()
        logger.info(f"Model loaded from {path}")


# Backward compatibility alias
AdversarialBandit = NeuralAdversarialBandit
