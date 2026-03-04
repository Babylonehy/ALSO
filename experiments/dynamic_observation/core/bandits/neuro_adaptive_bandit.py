"""
Neuro-Adaptive Prompting Bandit for dynamic prompt optimization.

This module implements the "EvoPrompt for Generation + Neural Contextual Bandit for Selection"
architecture, combining evolutionary prompt generation with neural network-based context-aware
selection.

Key Architecture:
- **Producer (EvoPrompt)**: Background process that continuously generates new high-quality prompts
- **Manager (Neural Contextual Bandit)**: Context-aware arm selection using [State_Embedding, Prompt_Embedding]
- **Adversarial Robustness**: EXP3-style exploration handles non-stationary environments

Why this works:
- Traditional OPRO/EvoPrompt find a single "best" prompt, which fails in non-stationary environments
- This architecture learns a **policy** π(prompt | context) instead of a single optimal prompt
- Neural network generalizes across similar contexts (e.g., "aggressive opponent" patterns)
- Bandit exploration handles distribution shift and adversarial opponents
"""

import asyncio
import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table

from .base_bandit import BaseBandit, BanditConfig
from .evoprompt_templates import parse_bio_from_response
from .llm_utils import acompletion_with_retry
from .prompt_space import PromptSpace

load_dotenv()
console = Console()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NeuroAdaptiveConfig(BanditConfig):
    """Configuration for Neuro-Adaptive Prompting Bandit."""

    # === Neural Network Architecture ===
    hidden_size: int = 256  # Smaller network for faster training
    depth: int = 2  # 2 hidden layers
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 1000

    # === Context Embedding ===
    use_context_embedding: bool = True  # Critical: enable context-aware selection
    context_embedding_dim: int = 4096  # Dimension of context embedding (scenario/opponent)

    # === Prompt Pool (Action Space) ===
    pool_size: int = 10  # Maximum prompts in action pool
    initial_pool_size: int = 5  # Start with N prompts from PromptSpace

    # === Evolution (Producer) ===
    evolution_enabled: bool = True
    evolution_interval: int = 10  # Generate new prompts every N turns
    evolution_batch_size: int = 2  # Generate N new prompts per evolution
    elite_ratio: float = 0.3  # Protect top 30% from replacement
    evolution_model: str = "openrouter/qwen/qwen-2.5-7b-instruct"
    evolution_temperature: float = 0.7

    # === Selection (Manager) - Neural Contextual Bandit ===
    selection_mode: str = "neural_ucb"  # "neural_ucb", "neural_exp3", "epsilon_greedy"
    eta: float = 1.0  # EXP3 exploration parameter
    beta: float = 0.5  # UCB exploration parameter
    epsilon: float = 0.1  # Epsilon-greedy exploration rate
    gamma: float = 0.1  # Minimum exploration probability

    # === Fitness Tracking ===
    fitness_ema_alpha: float = 0.3  # EMA: new_fitness = α * reward + (1-α) * old
    initial_fitness: float = 0.5  # Neutral starting fitness

    # === Embedding Model ===
    embedding_model: str = "qwen/qwen3-embedding-8b"

    # === Parallelization ===
    max_concurrent_evolutions: int = 3


# =============================================================================
# Prompt Unit (Action)
# =============================================================================


@dataclass
class PromptUnit:
    """A single prompt in the action pool."""

    prompt_text: str
    embedding: np.ndarray
    fitness: float = 0.5  # Accumulated performance score
    times_selected: int = 0
    times_won: int = 0  # Times this prompt led to good reward
    generation: int = 0  # Which evolution generation created this
    parent_prompt: str | None = None  # Parent prompt if evolved
    step_added: int = -1  # Turn when this prompt was added
    is_original: bool = False  # True if from original PromptSpace


# =============================================================================
# Value Network (Neural Contextual Bandit Core)
# =============================================================================


class ContextualValueNetwork(nn.Module):
    """
    Neural network for predicting reward from [context_embedding, prompt_embedding].

    Input: concatenation of:
    - Context embedding (opponent persona, goal, dialogue history)
    - Prompt embedding (bio/instruction embedding)

    Output: Predicted reward (0-1 scale)
    """

    def __init__(
        self,
        context_dim: int,
        prompt_dim: int,
        hidden_size: int = 256,
        depth: int = 2,
    ) -> None:
        super().__init__()

        self.input_dim = context_dim + prompt_dim
        self.activate = nn.ReLU()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(self.input_dim, hidden_size))

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns (batch,) shaped predictions."""
        for layer in self.layers[:-1]:
            x = self.activate(layer(x))
        x = self.layers[-1](x)
        return x.squeeze(-1)


# =============================================================================
# Main Bandit Implementation
# =============================================================================


class NeuroAdaptiveBandit(BaseBandit):
    """
    Neuro-Adaptive Prompting: EvoPrompt (Producer) + Neural Contextual Bandit (Manager).
    
    Architecture:
    1. Maintains a pool of prompts (action space)
    2. Neural network predicts reward for each prompt given current context
    3. UCB/EXP3 style exploration balances exploitation and exploration
    4. Background evolution generates new prompts to expand action space
    5. Low-performing prompts are replaced by evolved variants
    
    Key Innovation:
    - Learns π(prompt | context) instead of finding single optimal prompt
    - Context embedding captures opponent/scenario characteristics
    - Neural network generalizes across similar contexts
    """

    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | NeuroAdaptiveConfig | None = None,
        tensorboard_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        # Convert config
        if config is None:
            self.na_config = NeuroAdaptiveConfig()
        elif isinstance(config, NeuroAdaptiveConfig):
            self.na_config = config
        else:
            self.na_config = NeuroAdaptiveConfig(**{
                k: v for k, v in config.__dict__.items()
                if k in NeuroAdaptiveConfig.__dataclass_fields__
            })

        super().__init__(prompt_space, self.na_config, tensorboard_dir)

        # Set random seed for reproducibility
        if self.na_config.seed is not None:
            torch.manual_seed(self.na_config.seed)
            np.random.seed(self.na_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.na_config.seed)
            logger.info(f"Random seed set to {self.na_config.seed} for reproducibility")

        self.output_dir = Path(output_dir) if output_dir else None

        # Prompt pools (action space)
        self.pools: dict[Literal["p1", "p2"], list[PromptUnit]] = {"p1": [], "p2": []}
        self.prompt_hashes: dict[Literal["p1", "p2"], set[str]] = {"p1": set(), "p2": set()}
        self._initialize_pools()

        # Current context embedding (set before each selection)
        self.current_context_embedding: np.ndarray | None = None

        # Evolution state
        self.generation: dict[Literal["p1", "p2"], int] = {"p1": 0, "p2": 0}
        self.last_evolution_turn: dict[Literal["p1", "p2"], int] = {"p1": -1, "p2": -1}

        # Neural network (initialized lazily when we know context dim)
        self.model: ContextualValueNetwork | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._model_initialized = False

        # UCB: track gradient features for uncertainty estimation
        self.arm_counts: dict[Literal["p1", "p2"], np.ndarray] = {
            "p1": np.ones(self.na_config.pool_size),
            "p2": np.ones(self.na_config.pool_size),
        }

        # Last selection probabilities for logging
        self.last_selection_probs: dict[str, float] = {"p1": 0.0, "p2": 0.0}

        logger.info(
            f"NeuroAdaptiveBandit initialized: pool_size={self.na_config.pool_size}, "
            f"selection_mode={self.na_config.selection_mode}, "
            f"evolution_interval={self.na_config.evolution_interval}"
        )

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize_pools(self) -> None:
        """Initialize prompt pools from PromptSpace."""
        for agent in ("p1", "p2"):
            agent_typed: Literal["p1", "p2"] = agent  # type: ignore
            n_available = self.prompt_space.get_num_arms(agent_typed)
            n_to_use = min(n_available, self.na_config.initial_pool_size)

            pool = []
            for i in range(n_to_use):
                prompt = self.prompt_space.get_prompt(agent_typed, i)
                embedding = self.prompt_space.get_embedding(agent_typed, i)
                unit = PromptUnit(
                    prompt_text=prompt,
                    embedding=embedding,
                    fitness=self.na_config.initial_fitness,
                    is_original=(i == 0),
                    step_added=-1,
                )
                pool.append(unit)
                self.prompt_hashes[agent].add(self._get_hash(prompt))

            self.pools[agent_typed] = pool
            logger.info(f"Initialized {agent_typed} pool with {len(pool)} prompts")

    def _get_hash(self, text: str) -> str:
        """MD5 hash for deduplication."""
        return hashlib.md5(text.encode()).hexdigest()

    def _init_model(self, context_dim: int) -> None:
        """Lazily initialize the neural network when context dim is known."""
        if self._model_initialized:
            return

        prompt_dim = self.embedding_dim
        self.model = ContextualValueNetwork(
            context_dim=context_dim,
            prompt_dim=prompt_dim,
            hidden_size=self.na_config.hidden_size,
            depth=self.na_config.depth,
        ).to(self.na_config.device).double()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.na_config.learning_rate,
            weight_decay=self.na_config.weight_decay,
        )

        self._model_initialized = True
        logger.info(
            f"Initialized ContextualValueNetwork: context_dim={context_dim}, "
            f"prompt_dim={prompt_dim}, hidden_size={self.na_config.hidden_size}"
        )

    def set_context_embedding(self, embedding: np.ndarray) -> None:
        """Set the current context embedding for next selection."""
        self.current_context_embedding = embedding
        # Initialize model if not done yet
        if not self._model_initialized:
            self._init_model(len(embedding))

    # =========================================================================
    # Selection (Manager - Neural Contextual Bandit)
    # =========================================================================

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """Sync version of select."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.select_async(agent, turn))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.select_async(agent, turn))

    async def select_async(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select a prompt using Neural Contextual Bandit.

        Algorithm:
        1. Check if evolution should trigger (background producer)
        2. Compute predicted rewards for all prompts given current context
        3. Apply exploration strategy (UCB, EXP3, or epsilon-greedy)
        4. Return selected prompt
        """
        if self._stopped:
            idx = self.current_selections[agent]
            pool = self.pools[agent]
            if idx < len(pool):
                unit = pool[idx]
                return idx, unit.prompt_text, unit.embedding
            return 0, self.prompt_space.get_prompt(agent, 0), self.prompt_space.get_embedding(agent, 0)

        # 1. Check evolution trigger
        if self._should_evolve(agent, turn):
            await self._evolve_async(agent, turn)

        pool = self.pools[agent]
        if not pool:
            return 0, self.prompt_space.get_prompt(agent, 0), self.prompt_space.get_embedding(agent, 0)

        n_arms = len(pool)

        # 2. Get predictions for all arms
        predictions = self._get_predictions(agent)

        # 3. Apply exploration strategy
        if self.na_config.selection_mode == "neural_ucb":
            selected_idx, prob = self._select_ucb(agent, predictions, turn)
        elif self.na_config.selection_mode == "neural_exp3":
            selected_idx, prob = self._select_exp3(agent, predictions, turn)
        else:  # epsilon_greedy
            selected_idx, prob = self._select_epsilon_greedy(agent, predictions)

        # Record selection
        self.current_selections[agent] = selected_idx
        self.last_selection_probs[agent] = prob
        pool[selected_idx].times_selected += 1

        # Update arm counts for UCB
        if selected_idx < len(self.arm_counts[agent]):
            self.arm_counts[agent][selected_idx] += 1

        unit = pool[selected_idx]
        logger.info(
            f"[Turn {turn}] NeuroAdaptive {agent}: selected arm {selected_idx}, "
            f"pred={predictions[selected_idx]:.4f}, prob={prob:.4f}, "
            f"fitness={unit.fitness:.4f}"
        )

        return selected_idx, unit.prompt_text, unit.embedding

    def _get_predictions(self, agent: Literal["p1", "p2"]) -> np.ndarray:
        """Get neural network predictions for all arms in the pool."""
        pool = self.pools[agent]
        n_arms = len(pool)

        # If no context or model not trained, return fitness as predictions
        if self.current_context_embedding is None or not self._model_initialized:
            return np.array([unit.fitness for unit in pool])

        if len(self.selection_history) < 2:
            # Not enough data to train - use fitness
            return np.array([unit.fitness for unit in pool])

        # Prepare inputs: [context, prompt_embedding] for each arm
        context = self.current_context_embedding
        inputs = []
        for unit in pool:
            combined = np.concatenate([context, unit.embedding])
            inputs.append(combined)

        inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.float64).to(self.na_config.device)

        with torch.no_grad():
            predictions = self.model(inputs_tensor).cpu().numpy()

        return predictions

    def _select_ucb(
        self,
        agent: Literal["p1", "p2"],
        predictions: np.ndarray,
        turn: int,
    ) -> tuple[int, float]:
        """Neural UCB selection: prediction + exploration bonus."""
        n_arms = len(predictions)
        counts = self.arm_counts[agent][:n_arms]

        # UCB bonus: beta * sqrt(log(t) / n_arm)
        bonus = self.na_config.beta * np.sqrt(np.log(turn + 1) / (counts + 1))
        ucb_values = predictions + bonus

        selected_idx = int(np.argmax(ucb_values))
        # Approximate probability (for logging)
        prob = 1.0 / n_arms + 0.5 * (1 - 1.0 / n_arms)  # Rough estimate

        return selected_idx, prob

    def _select_exp3(
        self,
        agent: Literal["p1", "p2"],
        predictions: np.ndarray,
        turn: int,
    ) -> tuple[int, float]:
        """EXP3-style selection: softmax with gamma mixing."""
        n_arms = len(predictions)
        eta = self.na_config.eta

        # Softmax probabilities
        exp_pred = np.exp(eta * (predictions - np.max(predictions)))
        probs = exp_pred / np.sum(exp_pred)

        # Gamma mixing for minimum exploration
        gamma = self.na_config.gamma
        probs = (1 - gamma) * probs + gamma / n_arms

        # Handle numerical issues
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / np.sum(probs)

        selected_idx = np.random.choice(n_arms, p=probs)
        prob = probs[selected_idx]

        return int(selected_idx), float(prob)

    def _select_epsilon_greedy(
        self,
        agent: Literal["p1", "p2"],
        predictions: np.ndarray,
    ) -> tuple[int, float]:
        """Epsilon-greedy selection."""
        n_arms = len(predictions)
        epsilon = self.na_config.epsilon

        if np.random.random() < epsilon:
            # Explore: random selection
            selected_idx = np.random.randint(n_arms)
            prob = epsilon / n_arms
        else:
            # Exploit: best prediction
            selected_idx = int(np.argmax(predictions))
            prob = 1 - epsilon + epsilon / n_arms

        return selected_idx, prob

    # =========================================================================
    # Update (Reward Feedback)
    # =========================================================================

    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """Update fitness based on observed reward."""
        pool = self.pools[agent]

        if arm_index >= len(pool):
            logger.warning(f"Invalid arm_index {arm_index} for {agent}")
            return

        unit = pool[arm_index]
        old_fitness = unit.fitness

        # EMA update
        alpha = self.na_config.fitness_ema_alpha
        unit.fitness = alpha * reward + (1 - alpha) * old_fitness

        if reward > 0.6:  # Good reward threshold
            unit.times_won += 1

        logger.debug(
            f"NeuroAdaptive update {agent}: arm={arm_index}, "
            f"reward={reward:.4f}, fitness: {old_fitness:.4f} -> {unit.fitness:.4f}"
        )

    # =========================================================================
    # Evolution (Producer - Background Prompt Generation)
    # =========================================================================

    def _should_evolve(self, agent: Literal["p1", "p2"], turn: int) -> bool:
        """Check if evolution should trigger."""
        if not self.na_config.evolution_enabled:
            return False
        if turn < self.na_config.evolution_interval:
            return False
        turns_since_last = turn - self.last_evolution_turn[agent]
        return turns_since_last >= self.na_config.evolution_interval

    async def _evolve_async(self, agent: Literal["p1", "p2"], turn: int) -> None:
        """
        Evolve the prompt pool (Producer role).

        Strategy:
        1. Select parents using roulette wheel (fitness-proportional)
        2. Generate new prompts via LLM mutation
        3. Replace worst-performing prompts (respecting elite ratio)
        """
        pool = self.pools[agent]
        if len(pool) < 2:
            return

        self.generation[agent] += 1
        self.last_evolution_turn[agent] = turn
        gen = self.generation[agent]

        logger.info(f"[Turn {turn}] Evolution triggered for {agent}, generation {gen}")

        # Sort by fitness (ascending, worst first)
        sorted_indices = np.argsort([u.fitness for u in pool])
        n_elite = max(1, int(len(pool) * self.na_config.elite_ratio))
        n_to_replace = min(self.na_config.evolution_batch_size, len(pool) - n_elite)

        if n_to_replace <= 0:
            return

        # Indices to replace (worst performers, excluding elites)
        replace_indices = sorted_indices[:n_to_replace].tolist()

        # Generate new prompts
        tasks = []
        for _ in range(n_to_replace):
            parent = self._select_parent_roulette(pool)
            tasks.append(self._mutate_prompt(parent.prompt_text, agent))

        new_prompts = await asyncio.gather(*tasks, return_exceptions=True)

        # Replace worst prompts with new ones
        replaced_count = 0
        for i, (idx, new_prompt_result) in enumerate(zip(replace_indices, new_prompts)):
            if isinstance(new_prompt_result, Exception):
                logger.warning(f"Mutation failed: {new_prompt_result}")
                continue
            new_prompt = str(new_prompt_result)
            if not new_prompt or not new_prompt.strip():
                continue

            # Check for duplicates
            hash_str = self._get_hash(new_prompt)
            if hash_str in self.prompt_hashes[agent]:
                continue

            # Get embedding for new prompt
            embedding = await self._get_embedding_async(new_prompt)
            if embedding is None:
                continue

            # Create new unit
            new_unit = PromptUnit(
                prompt_text=new_prompt,
                embedding=embedding,
                fitness=self.na_config.initial_fitness,  # Reset fitness
                generation=gen,
                parent_prompt=pool[idx].prompt_text,
                step_added=turn,
            )

            # Replace
            old_hash = self._get_hash(pool[idx].prompt_text)
            self.prompt_hashes[agent].discard(old_hash)
            self.prompt_hashes[agent].add(hash_str)
            pool[idx] = new_unit
            replaced_count += 1

        logger.info(f"Evolution complete: replaced {replaced_count}/{n_to_replace} prompts")

    def _select_parent_roulette(self, pool: list[PromptUnit]) -> PromptUnit:
        """Roulette wheel selection (fitness-proportional)."""
        fitnesses = np.array([u.fitness for u in pool])
        # Shift to positive
        min_fit = np.min(fitnesses)
        shifted = fitnesses - min_fit + 0.1
        probs = shifted / np.sum(shifted)
        idx = np.random.choice(len(pool), p=probs)
        return pool[idx]

    async def _mutate_prompt(self, parent_prompt: str, agent: Literal["p1", "p2"]) -> str:
        """Generate a mutated prompt using LLM."""
        mutation_prompt = (
            f"You are improving a character instruction for a multi-agent social simulation.\n\n"
            f"## Original Character Instruction\n{parent_prompt}\n\n"
            f"## Requirements for Improvement\n"
            f"1. Fact Preservation: Keep ALL original identity markers intact (name, age, occupation, relationships).\n"
            f"2. Strategic Re-alignment: Emphasize traits that help achieve the character's Goal.\n"
            f"3. Behavioral Consistency: Ensure the rewritten traits will guide the agent toward their Goal without 'Topic Drift' or 'Parroting'.\n"
            f"4. Social Resilience: Frame the character's values to be robust against adversarial 'Deadlocks' by emphasizing flexible strategic pathways.\n"
            f"5. Output Format: Provide ONLY the modified instruction text. Do not include meta-commentary or reasoning steps.\n\n"
            f"## Improved Character Instruction:"
        )

        response = await acompletion_with_retry(
            model=self.na_config.evolution_model,
            messages=[{"role": "user", "content": mutation_prompt}],
            temperature=self.na_config.evolution_temperature,
            max_tokens=1024,
        )

        new_prompt = response.choices[0].message.content
        if new_prompt is None:
            return ""
        parsed = parse_bio_from_response(new_prompt.strip())
        return parsed if parsed else ""

    async def _get_embedding_async(self, text: str) -> np.ndarray | None:
        """Get embedding for a text using the embedding model."""
        try:
            import litellm
            response = await litellm.aembedding(
                model=self.na_config.embedding_model,
                input=[text],
            )
            # litellm returns EmbeddingResponse with data attribute
            return np.array(response["data"][0]["embedding"])  # type: ignore
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            traceback.print_exc()
            return None

    # =========================================================================
    # Model Training
    # =========================================================================

    def train_model(self, verbose: bool = True) -> float:  # type: ignore[override]
        """
        Train the neural network on collected experience.

        Uses selection history with context embeddings to learn:
        f([context_embedding, prompt_embedding]) -> reward
        """
        if not self._model_initialized or self.model is None or self.optimizer is None:
            return 0.0

        # Filter records with context embeddings
        records_with_context = [
            r for r in self.selection_history
            if r.context_embedding is not None
        ]

        if len(records_with_context) < 5:
            return 0.0

        # Prepare training data
        X = []
        y = []
        for record in records_with_context:
            ctx_emb = record.context_embedding
            if ctx_emb is None:
                continue
            combined = np.concatenate([ctx_emb, record.embedding])
            X.append(combined)
            y.append(record.reward)

        X_tensor = torch.tensor(np.array(X), dtype=torch.float64).to(self.na_config.device)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float64).to(self.na_config.device)

        # Training loop
        self.model.train()
        total_loss = 0.0
        n_epochs = min(self.na_config.epochs, 100)  # Cap epochs for online learning

        model = self.model
        optimizer = self.optimizer
        assert model is not None and optimizer is not None  # Already checked above
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            predictions = model(X_tensor)
            loss = nn.functional.mse_loss(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            total_loss = loss.item()

        model.eval()

        logger.info(
            f"Model trained: {len(records_with_context)} samples, "
            f"final_loss={total_loss:.6f}"
        )

        return total_loss

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_pool_stats(self, agent: Literal["p1", "p2"]) -> dict[str, Any]:
        """Get statistics about the prompt pool."""
        pool = self.pools[agent]
        if not pool:
            return {"size": 0}

        fitnesses = [u.fitness for u in pool]
        return {
            "size": len(pool),
            "mean_fitness": np.mean(fitnesses),
            "max_fitness": np.max(fitnesses),
            "min_fitness": np.min(fitnesses),
            "std_fitness": np.std(fitnesses),
            "generation": self.generation[agent],
            "n_original": sum(1 for u in pool if u.is_original),
            "n_evolved": sum(1 for u in pool if not u.is_original),
        }

    def get_best_prompt(self, agent: Literal["p1", "p2"]) -> tuple[str, float]:
        """Get the best performing prompt."""
        pool = self.pools[agent]
        if not pool:
            return "", 0.0
        best = max(pool, key=lambda u: u.fitness)
        return best.prompt_text, best.fitness

    def print_pool_summary(self, agent: Literal["p1", "p2"]) -> None:
        """Print a summary table of the prompt pool."""
        pool = self.pools[agent]
        if not pool:
            console.print(f"[yellow]No prompts in {agent} pool[/yellow]")
            return

        table = Table(title=f"{agent.upper()} Prompt Pool (Gen {self.generation[agent]})")
        table.add_column("Idx", style="cyan")
        table.add_column("Fitness", style="green")
        table.add_column("Selected", style="yellow")
        table.add_column("Won", style="magenta")
        table.add_column("Gen", style="blue")
        table.add_column("Prompt (first 50 chars)", style="white")

        for i, unit in enumerate(pool):
            table.add_row(
                str(i),
                f"{unit.fitness:.4f}",
                str(unit.times_selected),
                str(unit.times_won),
                str(unit.generation),
                unit.prompt_text[:50] + "..." if len(unit.prompt_text) > 50 else unit.prompt_text,
            )

        console.print(table)

    def stop(self) -> None:
        """Stop the bandit and save state."""
        self._stopped = True
        # Could save model weights here if needed
        logger.info("NeuroAdaptiveBandit stopped")

