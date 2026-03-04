"""
Progressive PromptBreeder Bandit implementation for dynamic prompt optimization.

This variant starts with only N original prompts and gradually mutates them.
Unlike the standard PromptBreeder, it doesn't expose the full strategy space upfront,
but progressively evolves the population through mutation.

Key differences from standard PromptBreeder:
- Only starts with N original prompts (no access to full strategy space)
- Mutation strategies are introduced gradually (not all at once)
- Focus on evolving from a small initial seed population
"""

import asyncio
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .llm_utils import acompletion_with_retry
from .prompt_space import PromptSpace
from .mutation_prompts import (
    PROMPT_BREEDER_MUTATION_PROMPTS,
    PROMPT_BREEDER_THINKING_STYLES,
)
import traceback


@dataclass
class ProgressiveEvolutionUnit:
    """A single unit in the progressive prompt population."""

    task_prompt: str  # P: the actual prompt/background
    mutation_prompt: str  # M: current mutation prompt
    thinking_style: str  # T: thinking style used for generation
    fitness: float = 0.0  # Performance score
    embedding: np.ndarray | None = None  # Cached embedding
    generation: int = 0  # Which generation this prompt was born in
    parent_prompt: str | None = None  # Parent prompt (if mutated)
    history: list[str] = field(default_factory=list)  # Historical prompts


@dataclass
class ProgressivePromptBreederConfig(BanditConfig):
    """Configuration for Progressive PromptBreeder bandit."""

    # Population parameters
    initial_population_size: int = 10  # N: number of original prompts to start with
    max_population_size: int = 10  # Maximum population size after growth

    # Evolution parameters
    mutation_rate: float = 0.3  # Probability of mutation per generation
    evolution_interval: int = 1  # Evolve population every N turns
    grow_population: bool = True  # Whether to grow population through mutation

    # Strategy pool parameters (only expose N strategies at a time)
    active_strategy_pool_size: int = 1  # N: number of active mutation strategies (reduced from 3)
    strategy_rotation_interval: int = 5  # Rotate strategies every N generations

    # LLM parameters for mutation
    mutation_model: str = "openrouter/qwen/qwen-2.5-7b-instruct"
    mutation_temperature: float = 0.8

    # Selection strategy: "greedy", "epsilon_greedy", "softmax", "round_robin", "fitness_weighted"
    selection_strategy: str = "round_robin"
    elite_ratio: float = 0.2

    # Epsilon for epsilon-greedy exploration (only used if selection_strategy="epsilon_greedy")
    epsilon: float = 0.1

    # Temperature for softmax selection (only used if selection_strategy="softmax")
    selection_temperature: float = 1.0

    # Weakening parameters for ablation study
    random_initial_selection: bool = True  # Use random selection instead of fitness-weighted at start
    random_selection_turns: int = 5  # Number of turns to use random selection
    mutated_fitness_init: float = 0.0  # Initial fitness for mutated prompts (0.0 = cold start)


class ProgressivePromptBreederBandit(BaseBandit):
    """
    Progressive PromptBreeder for gradual prompt evolution.

    Starts with only N original prompts and evolves them through mutation,
    introducing mutation strategies progressively rather than all at once.
    """

    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | ProgressivePromptBreederConfig | None = None,
        tensorboard_dir: Path | None = None,
    ) -> None:
        super().__init__(prompt_space, config, tensorboard_dir)

        # Set random seed for reproducibility
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            logger.info(f"Random seed set to {self.config.seed} for reproducibility")

        # Use ProgressivePromptBreederConfig if not provided
        if config is None:
            self.pb_config = ProgressivePromptBreederConfig()
        elif isinstance(config, ProgressivePromptBreederConfig):
            self.pb_config = config
        else:
            self.pb_config = ProgressivePromptBreederConfig(**{
                k: v for k, v in config.__dict__.items()
                if k in ProgressivePromptBreederConfig.__dataclass_fields__
            })

        # Initialize populations for each agent (start with N original prompts)
        self.populations: dict[Literal["p1", "p2"], list[ProgressiveEvolutionUnit]] = {
            "p1": [],
            "p2": [],
        }

        # Track current generation
        self.generation: dict[Literal["p1", "p2"], int] = {"p1": 0, "p2": 0}

        # Selection temperature for softmax (eta parameter)
        self.selection_eta: float = 1.0

        # Active mutation strategies (limited to N at a time)
        self._init_active_strategies()

        # Initialize populations with only N original prompts
        self._initialize_seed_population()

        # LLM client for mutations (lazy initialization)
        self._llm_client = None

        logger.info(
            f"Initialized Progressive PromptBreeder with "
            f"initial_size={self.pb_config.initial_population_size}, "
            f"active_strategies={self.pb_config.active_strategy_pool_size}"
        )

    def _init_active_strategies(self) -> None:
        """Initialize the active strategy pool with N random strategies."""
        n = self.pb_config.active_strategy_pool_size

        # Randomly select N mutation prompts from the full pool
        all_mutations = list(PROMPT_BREEDER_MUTATION_PROMPTS)
        all_styles = list(PROMPT_BREEDER_THINKING_STYLES)

        random.shuffle(all_mutations)
        random.shuffle(all_styles)

        self.active_mutation_prompts = all_mutations[:n]
        self.active_thinking_styles = all_styles[:n]

        logger.info(f"Active strategy pool: {n} mutation prompts, {n} thinking styles")

    def _rotate_strategies(self) -> None:
        """Rotate active strategies by replacing one with a new random one."""
        all_mutations = list(PROMPT_BREEDER_MUTATION_PROMPTS)
        all_styles = list(PROMPT_BREEDER_THINKING_STYLES)

        # Remove current active ones from pool
        available_mutations = [m for m in all_mutations if m not in self.active_mutation_prompts]
        available_styles = [s for s in all_styles if s not in self.active_thinking_styles]

        if available_mutations:
            # Replace one random active strategy with a new one
            idx = random.randrange(len(self.active_mutation_prompts))
            new_mutation = random.choice(available_mutations)
            old_mutation = self.active_mutation_prompts[idx]
            self.active_mutation_prompts[idx] = new_mutation
            logger.debug(f"Rotated mutation strategy: '{old_mutation[:30]}...' -> '{new_mutation[:30]}...'")

        if available_styles:
            idx = random.randrange(len(self.active_thinking_styles))
            new_style = random.choice(available_styles)
            self.active_thinking_styles[idx] = new_style

    def _initialize_seed_population(self) -> None:
        """Initialize populations with only N original prompts."""
        for agent in ["p1", "p2"]:
            agent_typed: Literal["p1", "p2"] = agent  # type: ignore
            n_available = self.prompt_space.get_num_arms(agent_typed)
            n_to_use = min(n_available, self.pb_config.initial_population_size)

            population = []
            for idx in range(n_to_use):
                prompt_text = self.prompt_space.get_prompt(agent_typed, idx)
                embedding = self.prompt_space.get_embedding(agent_typed, idx)

                # Create evolution unit with active strategies only
                unit = ProgressiveEvolutionUnit(
                    task_prompt=prompt_text,
                    mutation_prompt=random.choice(self.active_mutation_prompts),
                    thinking_style=random.choice(self.active_thinking_styles),
                    fitness=1.0,  # Start with equal fitness
                    embedding=embedding,
                    generation=0,
                    parent_prompt=None,
                    history=[prompt_text],
                )
                population.append(unit)

            self.populations[agent_typed] = population
            logger.info(f"Initialized {agent} seed population with {len(population)} original prompts")

    async def select_async(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """Async version of select using fitness-proportional (roulette wheel) selection."""
        if self._stopped:
            idx = self.current_selections[agent]
            unit = self.populations[agent][idx]
            embedding = unit.embedding if unit.embedding is not None else np.zeros(self.embedding_dim)
            return idx, unit.task_prompt, embedding

        population = self.populations[agent]
        if not population:
            raise ValueError(f"No population available for {agent}")

        # Check if evolution should trigger
        if turn > 0 and turn % self.pb_config.evolution_interval == 0:
            await self._evolve_population_async(agent, turn)

        # Check if strategy rotation should occur
        if self.generation[agent] > 0 and self.generation[agent] % self.pb_config.strategy_rotation_interval == 0:
            self._rotate_strategies()

        # Record fitness trace
        fitnesses = np.array([u.fitness for u in population])
        self.score_traces[agent].append(fitnesses.tolist())

        # Check if we should use random selection (for initial turns)
        use_random = (
            self.pb_config.random_initial_selection
            and turn <= self.pb_config.random_selection_turns
        )

        if use_random:
            # Random selection for initial exploration (weakening strategy)
            selected_idx = int(np.random.randint(0, len(population)))
            prob = 1.0 / len(population)
            logger.debug(f"[Turn {turn}] Using random selection (turn <= {self.pb_config.random_selection_turns})")
        else:
            # After initial phase: use configured selection strategy
            strategy = self.pb_config.selection_strategy

            if strategy == "greedy":
                # Pure greedy: always pick the best
                selected_idx = int(np.argmax(fitnesses))
                prob = 1.0
                logger.debug(f"[Turn {turn}] Using greedy selection: best_idx={selected_idx}")
            elif strategy == "epsilon_greedy":
                # Epsilon-greedy: greedy with probability (1-epsilon), random otherwise
                if np.random.random() < self.pb_config.epsilon:
                    selected_idx = int(np.random.randint(0, len(population)))
                    prob = self.pb_config.epsilon / len(population)
                    logger.debug(f"[Turn {turn}] Using epsilon-greedy: random exploration")
                else:
                    selected_idx = int(np.argmax(fitnesses))
                    prob = 1.0 - self.pb_config.epsilon
                    logger.debug(f"[Turn {turn}] Using epsilon-greedy: greedy selection")
            elif strategy == "softmax":
                # Softmax: probability proportional to exp(fitness / temperature)
                temp = self.pb_config.selection_temperature
                exp_fitnesses = np.exp(fitnesses / temp)
                probs = exp_fitnesses / np.sum(exp_fitnesses)
                selected_idx = int(np.random.choice(len(population), p=probs))
                prob = probs[selected_idx]
                logger.debug(f"[Turn {turn}] Using softmax selection: idx={selected_idx}, prob={prob:.4f}")
            elif strategy == "round_robin":
                # Round-robin: sequential selection
                last_idx = self.current_selections.get(agent, -1)
                selected_idx = (last_idx + 1) % len(population)
                prob = 1.0 / len(population)
                logger.debug(f"[Turn {turn}] Using round-robin selection: {last_idx} -> {selected_idx}")
            else:
                # Default: fitness_weighted (roulette wheel)
                exp_fitnesses = np.exp(self.selection_eta * fitnesses)
                probs = exp_fitnesses / np.sum(exp_fitnesses)
                selected_idx = int(np.random.choice(len(population), p=probs))
                prob = probs[selected_idx]
                logger.debug(f"[Turn {turn}] Using fitness-weighted selection: idx={selected_idx}, prob={prob:.4f}")

        self.current_selections[agent] = selected_idx
        selected_unit = population[selected_idx]

        embedding = selected_unit.embedding if selected_unit.embedding is not None else np.zeros(self.embedding_dim)

        logger.info(
            f"[Turn {turn}] Progressive PromptBreeder {agent}: selected unit {selected_idx}, "
            f"fitness={selected_unit.fitness:.4f}, prob={prob:.3f}, gen={selected_unit.generation}"
        )

        return selected_idx, selected_unit.task_prompt, embedding

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """Sync version of select."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Cannot call select() from async context. Use select_async() instead.")
        except RuntimeError as e:
            if "Cannot call select()" in str(e):
                raise
            return asyncio.run(self.select_async(agent, turn))

    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """Update the fitness of the selected prompt unit."""
        population = self.populations[agent]
        if arm_index >= len(population):
            logger.error(f"Invalid arm_index {arm_index}")
            return

        selected_unit = population[arm_index]
        old_fitness = selected_unit.fitness
        selected_unit.fitness = reward

        logger.info(
            f"[Turn {turn}] Progressive PromptBreeder updated {agent} unit {arm_index}: "
            f"reward={reward:.2f}, fitness: {old_fitness:.4f} -> {selected_unit.fitness:.4f}"
        )

        embedding = selected_unit.embedding if selected_unit.embedding is not None else np.zeros(self.embedding_dim)
        # Compute current selection probability (roulette wheel)
        fitnesses = np.array([u.fitness for u in population])
        exp_fitnesses = np.exp(self.selection_eta * fitnesses)
        probs = exp_fitnesses / np.sum(exp_fitnesses)
        selection_prob = float(probs[arm_index])
        record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=selected_unit.task_prompt,
            embedding=embedding,
            reward=reward,
            cumulative_score=selected_unit.fitness,
            selection_probability=selection_prob,
        )
        self.selection_history.append(record)

    async def _evolve_population_async(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> None:
        """Evolve population using Binary Tournament Selection.

        This follows the original PromptBreeder algorithm:
        1. Randomly shuffle and pair population units
        2. For each pair, the unit with higher fitness wins
        3. The loser gets mutated using a randomly selected mutation operator
        """
        population = self.populations[agent]
        if len(population) < 2:
            return

        self.generation[agent] += 1
        logger.info(f"[Turn {turn}] Evolving {agent} - Generation {self.generation[agent]} (Binary Tournament)")

        # Step 1: Create random pairs (Binary Tournament)
        indices = list(range(len(population)))
        random.shuffle(indices)
        pairs = [indices[2 * x : 2 * x + 2] for x in range(len(indices) // 2)]

        mutated_count = 0
        for pair in pairs:
            first_idx, second_idx = pair[0], pair[1]
            first_unit = population[first_idx]
            second_unit = population[second_idx]

            # Step 2: Determine winner and loser
            # Higher fitness wins, loser gets mutated
            if first_unit.fitness >= second_unit.fitness:
                winner_idx = first_idx
                loser_idx = second_idx
            else:
                winner_idx = second_idx
                loser_idx = first_idx

            logger.debug(
                f"Tournament: unit[{first_idx}](fit={first_unit.fitness:.4f}) vs "
                f"unit[{second_idx}](fit={second_unit.fitness:.4f}) -> "
                f"winner={winner_idx}, loser={loser_idx}"
            )

            # Step 3: Mutate the loser using a uniformly random mutation operator
            success = await self._mutate_loser_async(agent, loser_idx, turn)
            if success:
                mutated_count += 1

        logger.info(f"[Turn {turn}] Evolution complete: {mutated_count} mutations")

    async def _mutate_loser_async(
        self,
        agent: Literal["p1", "p2"],
        loser_idx: int,
        turn: int,
    ) -> bool:
        """Mutate the losing unit in Binary Tournament.

        Following PromptBreeder:
        - Uniformly randomly select a mutation operator
        - Apply it to the loser's task prompt
        """
        population = self.populations[agent]
        loser = population[loser_idx]

        # Uniformly randomly select a mutation prompt (not fitness-weighted!)
        mutation_prompt = random.choice(self.active_mutation_prompts)

        try:
            # Build structured mutation prompt for social simulation character improvement
            prompt = (
                f"You are improving a character instruction for a multi-agent social simulation.\n\n"
                f"## Mutation Guidance: \n{mutation_prompt}\n\n"
                f"## Original Character Instruction\n{loser.task_prompt}\n\n"
                f"## Requirements for Improvement\n"
                f"1. Fact Preservation: Keep ALL original identity markers intact (name, age, gender, occupation, and personal secrets).\n"
                f"2. Strategic Re-alignment: Modify the bio to reflect more effective social strategies.\n"
                f"3. Behavioral Consistency: Ensure the rewritten traits will guide the agent toward their Goal without 'Topic Drift' or 'Parroting'.\n"
                f"4. Social Resilience: Frame the character's values to be robust against adversarial 'Deadlocks' by emphasizing flexible strategic pathways.\n"
                f"5. Output Format: Provide ONLY the modified instruction text. Do not include meta-commentary or reasoning steps.\n\n"
                f"## Improved Character Instruction:"
            )

            logger.debug(f"Mutating loser unit[{loser_idx}] with random mutation operator")
            response = await acompletion_with_retry(
                model=self.pb_config.mutation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.pb_config.mutation_temperature,
                max_tokens=4096,
                caller_name="ProgressivePromptBreeder",
            )
            new_prompt = response.choices[0].message.content.strip()

            if new_prompt and len(new_prompt) > 10:
                old_prompt = loser.task_prompt
                old_fitness = loser.fitness

                # Update the loser with mutated prompt
                loser.task_prompt = new_prompt
                loser.history.append(new_prompt)
                loser.parent_prompt = old_prompt
                loser.mutation_prompt = mutation_prompt
                loser.thinking_style = random.choice(self.active_thinking_styles)
                loser.embedding = None
                loser.generation = self.generation[agent]
                # Use configured initial fitness for mutated prompts (0.0 = cold start, harder to recover)
                new_fitness = self.pb_config.mutated_fitness_init
                loser.fitness = new_fitness

                logger.info(
                    f"[Turn {turn}] {agent}: mutated loser[{loser_idx}] "
                    f"(fit={old_fitness:.4f} -> {new_fitness}, gen={loser.generation})"
                )
                return True
            else:
                logger.warning("Invalid mutation result, loser unchanged")
                return False

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Mutation failed: {e}")
            return False

    def train_model(self, verbose: bool = True) -> float:
        """No batch training needed for evolution."""
        if verbose:
            logger.debug("Progressive PromptBreeder uses evolution, no batch training")
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
            "active_strategies": len(self.active_mutation_prompts),
            "mean_fitness": np.mean(fitnesses),
            "max_fitness": np.max(fitnesses),
            "min_fitness": np.min(fitnesses),
            "units": [
                {
                    "idx": i,
                    "fitness": u.fitness,
                    "generation": u.generation,
                    "prompt_preview": u.task_prompt[:50] + "..." if len(u.task_prompt) > 50 else u.task_prompt,
                }
                for i, u in enumerate(population)
            ],
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self.generation = {"p1": 0, "p2": 0}
        self.selection_history = []
        self.score_traces = {"p1": [], "p2": []}
        self._stopped = False
        self._init_active_strategies()
        self._initialize_seed_population()
        logger.info("Progressive PromptBreeder reset")

    @property
    def bandit_type(self) -> str:
        """Return the type of bandit algorithm."""
        return "progressive_prompt_breeder"

