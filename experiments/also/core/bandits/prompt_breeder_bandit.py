"""
PromptBreeder Bandit implementation for dynamic prompt optimization.

This module integrates the PromptBreeder evolutionary algorithm as a bandit strategy,
combining evolutionary prompt mutation with bandit-style exploration-exploitation.

Reference: Fernando, C., et al. (2023). PromptBreeder: Self-Referential Self-Improvement
Via Prompt Evolution.
"""

import asyncio
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .llm_utils import acompletion_with_retry
from .prompt_space import PromptSpace
from .mutation_prompts import (
    PROMPT_BREEDER_MUTATION_PROMPTS,
    PROMPT_BREEDER_THINKING_STYLES,
)
import traceback

# Load environment variables for API keys
load_dotenv()


@dataclass
class EvolutionUnit:
    """A single unit in the prompt population."""

    thinking_style: str  # T: thinking style used for generation
    mutation_prompt: str  # M: mutation prompt
    task_prompt: str  # P: the actual prompt/background
    fitness: float = 0.0  # Performance score
    embedding: np.ndarray | None = None  # Cached embedding
    history: list[str] = field(default_factory=list)  # Historical prompts
    is_original: bool = False  # True if this is the original (un-paraphrased) prompt


@dataclass
class PromptBreederConfig(BanditConfig):
    """Extended configuration for PromptBreeder bandit."""

    # Population parameters
    population_size: int = 10  # Number of prompts to maintain per agent
    elite_ratio: float = 0.2  # Fraction of population to keep as elites

    # Evolution parameters
    mutation_rate: float = 0.3  # Probability of mutation per generation
    evolution_interval: int = 1  # Evolve population every N turns

    # LLM parameters for mutation
    mutation_model: str = "openrouter/qwen/qwen-2.5-7b-instruct"  # Model for prompt mutation
    mutation_temperature: float = 0.8  # Temperature for creative mutations

    # Selection strategy: "greedy", "epsilon_greedy", "softmax", "round_robin", "fitness_weighted", "tournament", "elite_only"
    selection_strategy: str = "fitness_weighted"
    tournament_size: int = 3  # For tournament selection

    # Epsilon for epsilon-greedy exploration (only used if selection_strategy="epsilon_greedy")
    epsilon: float = 0.1

    # Temperature for softmax selection (only used if selection_strategy="softmax")
    selection_temperature: float = 1.0

    # ===== Initial random selection parameters =====
    # Use random selection for initial turns, then switch to fitness-weighted
    random_initial_selection: bool = True
    # Number of turns to use random selection before switching to fitness-weighted
    random_selection_turns: int = 5


# Default mutation prompts and thinking styles are now imported from mutation_prompts.py


class PromptBreederBandit(BaseBandit):
    """
    PromptBreeder-style evolutionary bandit for dynamic prompt optimization.

    Combines evolutionary algorithm concepts with bandit exploration:
    - Maintains a population of prompts for each agent
    - Uses fitness-weighted selection (like EXP3) for arm selection
    - Periodically evolves the population through mutation
    - Tracks elite prompts for lineage-based mutation
    """

    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | PromptBreederConfig | None = None,
        tensorboard_dir: Path | None = None,
    ) -> None:
        super().__init__(prompt_space, config, tensorboard_dir)

        # Use PromptBreederConfig if not provided
        if config is None:
            self.pb_config = PromptBreederConfig()
        elif isinstance(config, PromptBreederConfig):
            self.pb_config = config
        else:
            # Convert BanditConfig to PromptBreederConfig
            self.pb_config = PromptBreederConfig(**{
                k: v for k, v in config.__dict__.items()
                if k in PromptBreederConfig.__dataclass_fields__
            })

        # Initialize populations for each agent
        self.populations: dict[Literal["p1", "p2"], list[EvolutionUnit]] = {
            "p1": [],
            "p2": [],
        }

        # Elite history for lineage-based mutation
        self.elites: dict[Literal["p1", "p2"], list[EvolutionUnit]] = {
            "p1": [],
            "p2": [],
        }

        # Track current generation
        self.generation: dict[Literal["p1", "p2"], int] = {"p1": 0, "p2": 0}

        # Initialize populations from prompt space
        self._initialize_populations()

        # LLM client for mutations (lazy initialization)
        self._llm_client = None

        logger.info(
            f"Initialized PromptBreeder Bandit with population_size={self.pb_config.population_size}, "
            f"evolution_interval={self.pb_config.evolution_interval}, "
            f"mutation_rate={self.pb_config.mutation_rate}"
        )

    def _initialize_populations(self) -> None:
        """Initialize populations from the existing prompt space."""
        for agent in ["p1", "p2"]:
            agent_typed: Literal["p1", "p2"] = agent  # type: ignore
            n_available = self.prompt_space.get_num_arms(agent_typed)
            n_to_use = min(n_available, self.pb_config.population_size)

            population = []
            for idx in range(n_to_use):
                prompt_text = self.prompt_space.get_prompt(agent_typed, idx)
                embedding = self.prompt_space.get_embedding(agent_typed, idx)

                # Create evolution unit with random thinking style and mutation prompt
                unit = EvolutionUnit(
                    thinking_style=random.choice(PROMPT_BREEDER_THINKING_STYLES),
                    mutation_prompt=random.choice(PROMPT_BREEDER_MUTATION_PROMPTS),
                    task_prompt=prompt_text,
                    fitness=1.0,  # Start with equal fitness
                    embedding=embedding,
                    history=[prompt_text],
                )
                population.append(unit)

            self.populations[agent_typed] = population
            logger.info(f"Initialized {agent} population with {len(population)} units")

    def _get_llm_client(self):
        """Lazily initialize the LLM client for mutations."""
        if self._llm_client is None:
            try:
                import litellm
                self._llm_client = litellm
                # Configure for openrouter
                api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
                if api_key:
                    os.environ["OPENROUTER_API_KEY"] = api_key
            except ImportError as e:
                logger.error(f"Failed to import litellm: {e}")
                raise ImportError("litellm is required for PromptBreeder mutations") from e
        return self._llm_client

    def _compute_selection_probabilities(
        self, agent: Literal["p1", "p2"]
    ) -> np.ndarray:
        """Compute selection probabilities based on fitness (EXP3-style)."""
        population = self.populations[agent]
        if not population:
            raise ValueError(f"Empty population for {agent}")

        fitnesses = np.array([u.fitness for u in population], dtype=np.float64)

        # Ensure positive fitnesses
        fitnesses = np.maximum(fitnesses, 1e-10)

        # Softmax-style probability with temperature (eta)
        eta = self.config.eta if hasattr(self.config, 'eta') else 1.0
        exp_fitnesses = np.exp(eta * fitnesses)

        # Normalize
        total = np.sum(exp_fitnesses)
        if total <= 0 or np.isnan(total) or np.isinf(total):
            # Fallback to uniform
            return np.ones(len(population)) / len(population)

        probs = exp_fitnesses / total
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / np.sum(probs)

        return probs

    async def select_async(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Async version of select. Select a prompt using fitness-weighted sampling.

        Args:
            agent: Which agent to select for ("p1" or "p2")
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, embedding)
        """
        if self._stopped:
            logger.warning("PromptBreeder stopped, using current selection")
            idx = self.current_selections[agent]
            unit = self.populations[agent][idx]
            # Ensure embedding is not None
            embedding = unit.embedding if unit.embedding is not None else np.zeros(self.embedding_dim)
            return idx, unit.task_prompt, embedding

        population = self.populations[agent]
        if not population:
            raise ValueError(f"No population available for {agent}")

        # Check if evolution should trigger (async)
        if turn > 0 and turn % self.pb_config.evolution_interval == 0:
            await self._evolve_population_async(agent, turn)

        # Store fitness scores for tracing
        fitnesses = [u.fitness for u in population]
        self.score_traces[agent].append(fitnesses)

        # Check if we should use random selection (for initial turns)
        use_random = (
            self.pb_config.random_initial_selection
            and turn <= self.pb_config.random_selection_turns
        )

        if use_random:
            # Random selection for initial exploration
            selected_idx = int(np.random.randint(0, len(population)))
            prob = 1.0 / len(population)
            logger.debug(f"[Turn {turn}] Using random selection (turn <= {self.pb_config.random_selection_turns})")
        else:
            # After initial phase: use configured selection strategy
            strategy = self.pb_config.selection_strategy
            fitnesses_arr = np.array(fitnesses)

            if strategy == "greedy":
                # Pure greedy: always pick the best
                selected_idx = int(np.argmax(fitnesses_arr))
                prob = 1.0
                logger.debug(f"[Turn {turn}] Using greedy selection: best_idx={selected_idx}")
            elif strategy == "epsilon_greedy":
                # Epsilon-greedy: greedy with probability (1-epsilon), random otherwise
                if np.random.random() < self.pb_config.epsilon:
                    selected_idx = int(np.random.randint(0, len(population)))
                    prob = self.pb_config.epsilon / len(population)
                    logger.debug(f"[Turn {turn}] Using epsilon-greedy: random exploration")
                else:
                    selected_idx = int(np.argmax(fitnesses_arr))
                    prob = 1.0 - self.pb_config.epsilon
                    logger.debug(f"[Turn {turn}] Using epsilon-greedy: greedy selection")
            elif strategy == "softmax":
                # Softmax: probability proportional to exp(fitness / temperature)
                temp = self.pb_config.selection_temperature
                exp_fitnesses = np.exp(fitnesses_arr / temp)
                probs = exp_fitnesses / np.sum(exp_fitnesses)
                selected_idx = int(np.random.choice(len(population), p=probs))
                prob = probs[selected_idx]
                logger.debug(f"[Turn {turn}] Using softmax selection: idx={selected_idx}, prob={prob:.4f}")
            else:
                # Default: round-robin
                last_idx = self.current_selections.get(agent, -1)
                selected_idx = (last_idx + 1) % len(population)
                prob = 1.0 / len(population)
                logger.debug(f"[Turn {turn}] Using round-robin selection: {last_idx} -> {selected_idx}")

        self.current_selections[agent] = selected_idx
        selected_unit = population[selected_idx]

        prompt_text = selected_unit.task_prompt
        # Ensure embedding is not None for interface compatibility
        embedding = selected_unit.embedding if selected_unit.embedding is not None else np.zeros(self.embedding_dim)

        logger.info(
            f"[Turn {turn}] PromptBreeder {agent}: selected unit {selected_idx}, "
            f"fitness={selected_unit.fitness:.4f}, prob={prob:.4f}"
        )

        return selected_idx, prompt_text, embedding

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Sync version of select. Select a prompt using fitness-weighted sampling.

        Note: If evolution is triggered, this will use asyncio.run() which
        WILL FAIL if called from within an already running event loop.
        Use select_async() when calling from async code.

        Args:
            agent: Which agent to select for ("p1" or "p2")
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, embedding)

        Raises:
            RuntimeError: If called from within an async context
        """
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - cannot use asyncio.run()
            raise RuntimeError(
                "Cannot call select() from async context. "
                "Use select_async() instead to avoid event loop conflicts."
            )
        except RuntimeError as e:
            # Check if this is our error or the "no running loop" error
            if "Cannot call select()" in str(e):
                raise
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self.select_async(agent, turn))


    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """
        Update the fitness of the selected prompt unit.

        Args:
            agent: Which agent was updated ("p1" or "p2")
            arm_index: The unit index that was selected
            reward: The observed reward
            turn: The turn number when this selection was made
        """
        population = self.populations[agent]
        if arm_index >= len(population):
            logger.error(f"Invalid arm_index {arm_index} for population size {len(population)}")
            return

        selected_unit = population[arm_index]

        # Directly use environment reward as fitness
        old_fitness = selected_unit.fitness
        selected_unit.fitness = reward
        logger.info(
            f"[Turn {turn}] PromptBreeder updated {agent} unit {arm_index}: "
            f"reward={reward:.2f}, fitness: {old_fitness:.4f} -> {selected_unit.fitness:.4f}"
        )

        # Record selection
        # Ensure embedding is not None for interface compatibility
        embedding = selected_unit.embedding if selected_unit.embedding is not None else np.zeros(self.embedding_dim)
        record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=selected_unit.task_prompt,
            embedding=embedding,
            reward=reward,
            cumulative_score=selected_unit.fitness,
            selection_probability=self._compute_selection_probabilities(agent)[arm_index],
        )
        self.selection_history.append(record)

    async def _evolve_population_async(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> None:
        """
        Async version of evolve population through mutation.

        Evolution strategy: High-fitness arms are mutated to generate new prompts,
        which then replace low-fitness arms. This is a "survival of the fittest"
        approach where good prompts spawn offspring that replace weak ones.

        Elites are stored for lineage mutation reference only, not protected from replacement.
        """
        population = self.populations[agent]
        if len(population) < 2:
            logger.warning(f"Population too small for evolution: {len(population)}")
            return

        self.generation[agent] += 1
        logger.info(f"[Turn {turn}] Evolving {agent} population - Generation {self.generation[agent]}")

        # Sort population indices by fitness (descending)
        sorted_indices = sorted(
            range(len(population)),
            key=lambda i: population[i].fitness,
            reverse=True
        )

        # Identify high-fitness (parents) and low-fitness (to be replaced) arms
        n_elites = max(1, int(len(population) * self.pb_config.elite_ratio))
        high_fitness_indices = sorted_indices[:n_elites]  # Top performers (parents)
        low_fitness_indices = sorted_indices[-n_elites:]  # Bottom performers (to replace)

        # Store elites for lineage mutation (reference only, not protection)
        for idx in high_fitness_indices:
            elite = population[idx]
            elite_copy = EvolutionUnit(
                thinking_style=elite.thinking_style,
                mutation_prompt=elite.mutation_prompt,
                task_prompt=elite.task_prompt,
                fitness=elite.fitness,
                embedding=elite.embedding,
                history=elite.history.copy(),
                is_original=elite.is_original,
            )
            self.elites[agent].append(elite_copy)

        # Sort by fitness descending and keep top 10
        self.elites[agent].sort(key=lambda u: u.fitness, reverse=True)
        self.elites[agent] = self.elites[agent][:10]

        logger.debug(
            f"Elite pool for {agent}: {len(self.elites[agent])} units, "
            f"fitness range: [{self.elites[agent][-1].fitness:.4f}, {self.elites[agent][0].fitness:.4f}]"
            if self.elites[agent] else "Elite pool empty"
        )

        # Mutate high-fitness arms to replace low-fitness arms
        # mutation_rate determines what fraction of population to replace
        # e.g., mutation_rate=0.3 with 10 arms -> replace 3 low-score arms
        n_to_replace = max(1, int(len(population) * self.pb_config.mutation_rate))
        n_to_mutate = min(n_to_replace, len(high_fitness_indices), len(low_fitness_indices))
        mutated_count = 0

        for i in range(n_to_mutate):
            parent_idx = high_fitness_indices[i]
            replace_idx = low_fitness_indices[i]

            # Mutate the parent and use result to replace the low-fitness arm
            success = await self._mutate_and_replace_async(
                agent, parent_idx, replace_idx, turn
            )
            if success:
                mutated_count += 1

        logger.info(
            f"[Turn {turn}] Evolution complete for {agent}: "
            f"{mutated_count} mutations, {n_elites} elites stored for lineage"
        )

    async def _mutate_and_replace_async(
        self,
        agent: Literal["p1", "p2"],
        parent_idx: int,
        replace_idx: int,
        turn: int,
    ) -> bool:
        """
        Mutate a high-fitness parent arm and use the result to replace a low-fitness arm.

        Args:
            agent: Which agent's population to evolve
            parent_idx: Index of high-fitness arm to mutate (parent)
            replace_idx: Index of low-fitness arm to replace with offspring
            turn: Current turn number

        Returns:
            True if mutation and replacement succeeded, False otherwise.
        """
        population = self.populations[agent]
        parent = population[parent_idx]
        target = population[replace_idx]

        # Default to first_order mutation (most stable)
        mutation_type = "first_order"

        try:
            # Generate mutation prompt based on mutation type
            if mutation_type == "first_order":
                prompt = f"{parent.mutation_prompt}\n\nOriginal instruction: {parent.task_prompt}\n\nImproved instruction:"
            elif mutation_type == "lineage":
                elites = self.elites[agent]
                if elites:
                    elite_prompts = "\n".join([f"{i+1}. {e.task_prompt}" for i, e in enumerate(elites[:5])])
                    prompt = f"Here are some high-quality instructions in order of quality (best first):\n{elite_prompts}\n\nGenerate a new improved instruction that builds on these:"
                else:
                    prompt = f"{parent.mutation_prompt}\n\nOriginal: {parent.task_prompt}\n\nImproved:"
            else:
                prompt = f"{parent.mutation_prompt}\n\nOriginal instruction: {parent.task_prompt}\n\nImproved instruction:"

            # Call LLM for mutation
            logger.debug(f"Calling LLM for mutation: parent={parent_idx} -> replace={replace_idx}")
            response = await acompletion_with_retry(
                model=self.pb_config.mutation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.pb_config.mutation_temperature,
                max_tokens=4096,
                caller_name=f"PromptBreeder({mutation_type})",
            )
            new_prompt = response.choices[0].message.content.strip()

            if new_prompt and len(new_prompt) > 10:
                # Replace low-fitness arm with mutated offspring
                old_prompt = target.task_prompt
                old_fitness = target.fitness

                target.task_prompt = new_prompt
                target.history = [new_prompt]  # Start fresh history for offspring
                target.thinking_style = parent.thinking_style  # Inherit from parent
                target.mutation_prompt = parent.mutation_prompt  # Inherit from parent
                target.embedding = None  # Will be computed if needed
                target.is_original = False  # Offspring is not original

                # Initialize offspring fitness to initial value (1.0) for fair exploration
                target.fitness = 1.0

                logger.info(
                    f"[Turn {turn}] {agent} evolution: parent[{parent_idx}](fit={parent.fitness:.4f}) "
                    f"spawned offspring to replace arm[{replace_idx}](fit={old_fitness:.4f} -> {target.fitness:.4f})"
                )
                logger.debug(f"  old prompt: '{old_prompt[:50]}...' -> new: '{new_prompt[:50]}...'")
                return True
            else:
                logger.warning(f"Invalid mutation result, keeping original prompt at {replace_idx}")
                return False

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Mutation failed for {agent} parent={parent_idx} -> replace={replace_idx}: {e}")
            return False

    async def _mutate_unit_async(
        self,
        agent: Literal["p1", "p2"],
        unit_idx: int,
        turn: int,
    ) -> bool:
        """
        Apply a mutation operator to a unit in-place (async version).

        Note: This method mutates the unit in-place. For the new evolution strategy
        (high-fitness mutates to replace low-fitness), use _mutate_and_replace_async instead.

        Mutation operators from PromptBreeder:
        - First-order prompt generation: M + P -> P'
        - Zero-order hypermutation: D + T -> M'
        - Lineage-based mutation: elites history -> P'

        Returns:
            True if mutation succeeded, False otherwise.
        """
        unit = self.populations[agent][unit_idx]
        old_prompt = unit.task_prompt

        # Default to first_order mutation (most stable)
        mutation_type = "first_order"

        try:
            # Generate mutation prompt based on mutation type
            if mutation_type == "first_order":
                # Apply mutation prompt to task prompt
                prompt = f"{unit.mutation_prompt}\n\nOriginal instruction: {unit.task_prompt}\n\nImproved instruction:"

            elif mutation_type == "hypermutation":
                # Generate new mutation prompt, then apply
                hyper_prompt = "Please summarize and improve the following instruction for generating better prompts:\n"
                prompt = f"{hyper_prompt}{unit.mutation_prompt}\n\nImproved mutation instruction:"

            elif mutation_type == "lineage":
                # Use elite history for inspiration
                elites = self.elites[agent]
                if elites:
                    elite_prompts = "\n".join([f"{i+1}. {e.task_prompt}" for i, e in enumerate(elites[-5:])])
                    prompt = f"Here are some high-quality instructions in order of quality:\n{elite_prompts}\n\nGenerate a new improved instruction that builds on these:"
                else:
                    # Fallback to first-order
                    prompt = f"{unit.mutation_prompt}\n\nOriginal: {unit.task_prompt}\n\nImproved:"
            else:
                # Default fallback to first-order mutation
                prompt = f"{unit.mutation_prompt}\n\nOriginal instruction: {unit.task_prompt}\n\nImproved instruction:"

            # Call LLM for mutation with retry logic
            logger.debug(f"Calling LLM for mutation: {mutation_type}")
            response = await acompletion_with_retry(
                model=self.pb_config.mutation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.pb_config.mutation_temperature,
                max_tokens=4096,
                caller_name=f"PromptBreeder({mutation_type})",
            )
            new_prompt = response.choices[0].message.content.strip()

            if new_prompt and len(new_prompt) > 10:
                # Update unit
                unit.task_prompt = new_prompt
                unit.history.append(new_prompt)

                # Note: PromptBreeder doesn't use embeddings for selection (uses fitness),
                # so we skip embedding computation to save API calls.
                # Set to None - the embedding field is only for interface compatibility.
                unit.embedding = None

                # Reset fitness for exploration
                # Use simple decay without floor to avoid score boosting exploit
                unit.fitness = unit.fitness * 0.8

                logger.info(
                    f"[Turn {turn}] Mutated {agent} unit {unit_idx} via {mutation_type}: "
                    f"'old: {old_prompt}...'\n -> 'new: {new_prompt}...'"
                )
                return True
            else:
                logger.warning(f"Invalid mutation result, keeping original prompt")
                return False

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Mutation failed for {agent} unit {unit_idx}: {e}")
            return False

    def train_model(self, verbose: bool = True) -> float:
        """
        PromptBreeder uses evolutionary optimization, not gradient-based training.

        This method is a no-op for interface compatibility.
        Evolution happens automatically during selection when evolution_interval is reached.

        Returns:
            0.0 (no loss to report)
        """
        if verbose:
            logger.debug("PromptBreeder uses evolution, no batch training needed")
        return 0.0

    def get_population_summary(self, agent: Literal["p1", "p2"]) -> dict:
        """Get summary statistics for an agent's population."""
        population = self.populations[agent]
        if not population:
            return {"size": 0, "units": []}

        fitnesses = [u.fitness for u in population]
        return {
            "size": len(population),
            "generation": self.generation[agent],
            "mean_fitness": np.mean(fitnesses),
            "max_fitness": np.max(fitnesses),
            "min_fitness": np.min(fitnesses),
            "std_fitness": np.std(fitnesses),
            "n_elites": len(self.elites[agent]),
            "units": [
                {
                    "idx": i,
                    "fitness": u.fitness,
                    "prompt_preview": u.task_prompt[:50] + "..." if len(u.task_prompt) > 50 else u.task_prompt,
                    "n_mutations": len(u.history) - 1,
                }
                for i, u in enumerate(population)
            ],
        }

    def save_model(self, path: Path) -> None:
        """Save population state to files."""
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for agent_str in ["p1", "p2"]:
            agent: Literal["p1", "p2"] = agent_str  # type: ignore
            pop_data = {
                "generation": self.generation[agent],
                "units": [
                    {
                        "thinking_style": u.thinking_style,
                        "mutation_prompt": u.mutation_prompt,
                        "task_prompt": u.task_prompt,
                        "fitness": u.fitness,
                        "history": u.history,
                    }
                    for u in self.populations[agent]
                ],
                "elites": [
                    {
                        "task_prompt": e.task_prompt,
                        "fitness": e.fitness,
                    }
                    for e in self.elites[agent]
                ],
            }

            with open(path / f"{agent_str}_population.json", "w") as f:
                json.dump(pop_data, f, indent=2)

        logger.info(f"Saved PromptBreeder populations to {path}")

    def reset(self) -> None:
        """Reset the bandit to initial state."""
        self.generation = {"p1": 0, "p2": 0}
        self.elites = {"p1": [], "p2": []}
        self.selection_history = []
        self.score_traces = {"p1": [], "p2": []}
        self._stopped = False
        self._initialize_populations()
        logger.info("PromptBreeder bandit reset to initial state")

    @property
    def bandit_type(self) -> str:
        """Return the type of bandit algorithm."""
        return "prompt_breeder"
