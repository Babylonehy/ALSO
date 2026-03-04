"""
EvoPrompt-style Bandit for dynamic prompt optimization in Sotopia.

This module implements the EvoPrompt algorithm (GA and DE variants) as a bandit,
providing evolutionary prompt optimization that can be compared against other
bandit methods like NeuralAdversarialBandit, OPROBandit, etc.

Reference: "Connecting Large Language Models with Evolutionary Algorithms 
           Yields Powerful Prompt Optimizers" (Guo et al., 2023)
"""

import asyncio
import hashlib
import json
import random
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .evoprompt_templates import (
    build_ga_prompt,
    build_de_prompt,
    parse_bio_from_response,
)
from .llm_utils import acompletion_with_retry
from .prompt_space import PromptSpace

load_dotenv()
console = Console()


@dataclass
class EvoPromptConfig(BanditConfig):
    """Configuration for EvoPrompt-style bandit optimization."""

    # Evolution mode: "ga" (Genetic Algorithm) or "de" (Differential Evolution)
    mode: str = "ga"

    # Population size (降低: 10 -> 5)
    population_size: int = 5

    # Evolution interval: evolve every N turns (increased from 5 to 10 for performance)
    evolution_interval: int = 10

    # ===== Mutation rate parameters (NEW - for selective evolution) =====
    # Fraction of population to mutate each generation (降低: 0.3 -> 0.2)
    mutation_rate: float = 0.2
    # Fraction of top performers to protect as elites (增加: 0.2 -> 0.4)
    elite_ratio: float = 0.4

    # ===== GA-specific parameters =====
    # Selection mode: "tournament", "wheel", "random"
    ga_selection_mode: str = "wheel"
    # GA population update mode: "topk" (keep best N) or "std" (replace all)
    ga_mode: str = "topk"

    # ===== DE-specific parameters =====
    # Use best prompt as the third donor (instead of random)
    de_use_best_donor: bool = True

    # ===== LLM parameters =====
    # Model for mutation/crossover
    mutation_model: str = "openrouter/qwen/qwen-2.5-7b-instruct"
    # Mutation temperature (降低: 0.5 -> 0.3)
    mutation_temperature: float = 0.3

    # Embedding model for new prompts
    embedding_model: str = "qwen/qwen3-embedding-8b"

    # ===== Selection parameters =====
    # Selection strategy for arm selection: "greedy", "epsilon_greedy", "softmax", or "round_robin"
    selection_strategy: str = "round_robin"
    # Epsilon for epsilon-greedy (if used)
    epsilon: float = 0.1
    # Temperature for softmax selection
    selection_temperature: float = 1.0

    # ===== Fitness update parameters =====
    # Score update method: "ema" or "replace"
    score_update_method: str = "replace"
    # EMA alpha (learning rate, only used if score_update_method="ema")
    ema_alpha: float = 0.3

    # ===== Initial random selection parameters =====
    # Use random selection for initial turns, then switch to greedy
    random_initial_selection: bool = True
    # Number of turns to use random selection before switching to greedy
    random_selection_turns: int = 5

    # ===== Parallelization parameters (NEW) =====
    # Maximum concurrent LLM calls for mutation
    max_concurrent_mutations: int = 5


@dataclass
class EvolutionUnit:
    """A single unit in the evolutionary population."""
    task_prompt: str
    embedding: np.ndarray
    fitness: float = 0.5
    history: list[str] = field(default_factory=list)
    step_added: int = -1
    times_selected: int = 0
    is_original: bool = False


class EvoPromptBandit(BaseBandit):
    """
    EvoPrompt-style optimization as a bandit algorithm.
    
    Supports both GA (Genetic Algorithm) and DE (Differential Evolution) modes.
    
    Key features:
    - Population-based prompt evolution
    - GA: Tournament/wheel selection + crossover + mutation
    - DE: Differential mutation with best-donor strategy
    - Fitness-weighted arm selection (softmax or greedy)
    - EMA fitness updates from observed rewards
    """
    
    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | EvoPromptConfig | None = None,
        tensorboard_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize EvoPrompt Bandit."""
        # Convert to EvoPromptConfig if needed
        if config is None:
            config = EvoPromptConfig()
        elif not isinstance(config, EvoPromptConfig):
            config = EvoPromptConfig(**{
                k: v for k, v in config.__dict__.items()
                if k in EvoPromptConfig.__dataclass_fields__
            })
        
        super().__init__(prompt_space, config, tensorboard_dir)
        self.evo_config: EvoPromptConfig = config
        self.output_dir = Path(output_dir) if output_dir else None

        # Random state with seed for reproducibility
        self.rng = np.random.default_rng(config.seed)
        if config.seed is not None:
            random.seed(config.seed)
            logger.info(f"Random seed set to {config.seed} for reproducibility")

        # Hash set for deduplication (must be before _initialize_populations)
        self.prompt_hashes: dict[Literal["p1", "p2"], set[str]] = {"p1": set(), "p2": set()}
        
        # Initialize populations
        self.populations: dict[Literal["p1", "p2"], list[EvolutionUnit]] = {
            "p1": [],
            "p2": [],
        }
        self._initialize_populations()
        
        # Evolution state
        self.generation: dict[Literal["p1", "p2"], int] = {"p1": 0, "p2": 0}
        self.last_evolution_turn: dict[Literal["p1", "p2"], int] = {"p1": -1, "p2": -1}
        
        # Evolution history for logging
        self.evolution_history: list[dict[str, Any]] = []
        
        # Setup evolution log directory
        if self.output_dir:
            self.evolution_log_dir = self.output_dir / "evolution_logs"
            self.evolution_log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"EvoPromptBandit initialized: mode={self.evo_config.mode}, "
            f"population_size={self.evo_config.population_size}, "
            f"evolution_interval={self.evo_config.evolution_interval}"
        )
    
    def _initialize_populations(self) -> None:
        """Initialize populations from the existing prompt space."""
        for agent in ["p1", "p2"]:
            n_available = self.prompt_space.get_num_arms(agent)
            n_to_use = min(n_available, self.evo_config.population_size)
            
            population = []
            for i in range(n_to_use):
                prompt = self.prompt_space.get_prompt(agent, i)
                embedding = self.prompt_space.get_embedding(agent, i)
                is_original = (i == 0)  # First prompt is the original
                unit = EvolutionUnit(
                    task_prompt=prompt,
                    embedding=embedding,
                    fitness=0.5,  # Neutral initial fitness
                    history=[prompt],
                    step_added=-1,
                    is_original=is_original,
                )
                population.append(unit)
                
                # Track hash
                hash_str = self._get_hash(prompt)
                self.prompt_hashes[agent].add(hash_str)
            
            self.populations[agent] = population
            logger.info(f"Initialized {agent} population with {len(population)} units")
    
    def _get_hash(self, text: str) -> str:
        """Get MD5 hash of text for deduplication."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select an arm for the given agent.
        
        Triggers evolution if interval reached, then selects based on fitness.
        """
        # Check evolution trigger
        if self._should_evolve(agent, turn):
            try:
                asyncio.get_running_loop()
                # In async context, use ThreadPoolExecutor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._evolve_async(agent, turn))
                    future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                asyncio.run(self._evolve_async(agent, turn))
        
        population = self.populations[agent]
        
        if not population:
            # Fallback
            idx = 0
            prompt = self.prompt_space.get_prompt(agent, idx)
            embedding = self.prompt_space.get_embedding(agent, idx)
            return idx, prompt, embedding
        
        # Check if we should use random selection (for initial turns)
        use_random = (
            self.evo_config.random_initial_selection
            and turn <= self.evo_config.random_selection_turns
        )

        if use_random:
            # Random selection for initial exploration
            selected_idx = int(self.rng.integers(0, len(population)))
            logger.debug(f"[Turn {turn}] Using random selection (turn <= {self.evo_config.random_selection_turns})")
        else:
            # After initial phase: use configured selection strategy
            strategy = self.evo_config.selection_strategy
            fitnesses = np.array([u.fitness for u in population])

            if strategy == "greedy":
                # Pure greedy: always pick the best
                selected_idx = int(np.argmax(fitnesses))
                logger.debug(f"[Turn {turn}] Using greedy selection: best_idx={selected_idx}")
            elif strategy == "epsilon_greedy":
                # Epsilon-greedy: greedy with probability (1-epsilon), random otherwise
                if self.rng.random() < self.evo_config.epsilon:
                    selected_idx = int(self.rng.integers(0, len(population)))
                    logger.debug(f"[Turn {turn}] Using epsilon-greedy: random exploration")
                else:
                    selected_idx = int(np.argmax(fitnesses))
                    logger.debug(f"[Turn {turn}] Using epsilon-greedy: greedy selection")
            elif strategy == "softmax":
                # Softmax: probability proportional to exp(fitness / temperature)
                temp = self.evo_config.selection_temperature
                exp_fitnesses = np.exp(fitnesses / temp)
                probs = exp_fitnesses / np.sum(exp_fitnesses)
                selected_idx = int(self.rng.choice(len(population), p=probs))
                logger.debug(f"[Turn {turn}] Using softmax selection: idx={selected_idx}, prob={probs[selected_idx]:.4f}")
            else:
                # Default: round-robin
                last_idx = self.current_selections.get(agent, -1)
                selected_idx = (last_idx + 1) % len(population)
                logger.debug(f"[Turn {turn}] Using round-robin selection: {last_idx} -> {selected_idx}")
        
        unit = population[selected_idx]
        unit.times_selected += 1
        self.current_selections[agent] = selected_idx
        
        logger.debug(
            f"EvoPrompt select {agent}: idx={selected_idx}, "
            f"fitness={unit.fitness:.3f}, pool_size={len(population)}"
        )
        
        return selected_idx, unit.task_prompt, unit.embedding
    
    def _should_evolve(self, agent: Literal["p1", "p2"], turn: int) -> bool:
        """Check if evolution should be triggered."""
        if turn < self.evo_config.evolution_interval:
            return False
        
        turns_since_last = turn - self.last_evolution_turn[agent]
        return turns_since_last >= self.evo_config.evolution_interval
    
    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """
        Update the bandit with the observed reward.
        
        Updates fitness using EMA or direct replacement.
        """
        population = self.populations[agent]
        
        if arm_index >= len(population):
            logger.warning(f"Invalid arm_index {arm_index} for {agent} pool size {len(population)}")
            return
        
        unit = population[arm_index]
        old_fitness = unit.fitness
        
        if self.evo_config.score_update_method == "ema":
            alpha = self.evo_config.ema_alpha
            unit.fitness = alpha * reward + (1 - alpha) * old_fitness
        else:
            unit.fitness = reward
        
        # Record selection
        selection_record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=unit.task_prompt,
            embedding=unit.embedding,
            reward=reward,
            cumulative_score=unit.fitness,
        )
        self.selection_history.append(selection_record)
        
        logger.debug(
            f"EvoPrompt update {agent}: idx={arm_index}, "
            f"reward={reward:.3f}, fitness: {old_fitness:.3f} -> {unit.fitness:.3f}"
        )
    
    async def _evolve_async(self, agent: Literal["p1", "p2"], turn: int) -> None:
        """Evolve population using GA or DE."""
        if self.evo_config.mode == "ga":
            await self._evolve_ga_async(agent, turn)
        else:
            await self._evolve_de_async(agent, turn)
        
        # Sync to prompt_space
        self._sync_population_to_prompt_space(agent)
        
        # Update tracking
        self.generation[agent] += 1
        self.last_evolution_turn[agent] = turn
        
        # Save log
        if self.output_dir:
            self._save_evolution_log(agent, turn)
    
    async def _evolve_ga_async(self, agent: Literal["p1", "p2"], turn: int) -> None:
        """
        GA-style evolution: crossover + mutation.

        OPTIMIZED:
        - Only mutate low-fitness individuals (controlled by mutation_rate)
        - Protect elite individuals (controlled by elite_ratio)
        - Parallel LLM calls with asyncio.gather()
        """
        population = self.populations[agent]
        n = len(population)

        if n < 2:
            logger.warning(f"Population too small for GA evolution: {n}")
            return

        logger.info(f"[Turn {turn}] GA evolution for {agent}, generation {self.generation[agent] + 1}")

        fitness = np.array([unit.fitness for unit in population])

        # Identify protected indices (original + elites)
        original_indices = {i for i, unit in enumerate(population) if unit.is_original}
        n_elites = max(1, int(n * self.evo_config.elite_ratio))
        elite_indices = set(np.argsort(fitness)[-n_elites:])
        protected_indices = original_indices | elite_indices

        # Select mutation candidates (low fitness, non-protected)
        non_protected = [i for i in range(n) if i not in protected_indices]
        n_to_mutate = max(1, int(n * self.evo_config.mutation_rate))

        # Sort by fitness ascending and select lowest
        sorted_candidates = sorted(non_protected, key=lambda i: fitness[i])
        mutate_indices = sorted_candidates[:min(n_to_mutate, len(sorted_candidates))]

        logger.info(
            f"GA evolution: {len(mutate_indices)} mutations, "
            f"{len(protected_indices)} protected (elites + original)"
        )

        # Build mutation tasks for parallel execution
        mutation_tasks = []
        mutation_info = []  # Track which index each task corresponds to

        for j in mutate_indices:
            # Parent selection from high-fitness individuals
            if self.evo_config.ga_selection_mode == "tournament":
                group_a = random.sample(range(n), min(2, n))
                group_b = random.sample(range(n), min(2, n))
                parent_a_idx = max(group_a, key=lambda i: fitness[i])
                parent_b_idx = max(group_b, key=lambda i: fitness[i])
            elif self.evo_config.ga_selection_mode == "wheel":
                probs = fitness / np.sum(fitness) if np.sum(fitness) > 0 else np.ones(n) / n
                parent_a_idx, parent_b_idx = self.rng.choice(n, size=2, replace=False, p=probs)
            else:  # random
                parent_a_idx, parent_b_idx = random.sample(range(n), 2)

            bio_a = population[parent_a_idx].task_prompt
            bio_b = population[parent_b_idx].task_prompt
            prompt = build_ga_prompt(bio_a, bio_b)

            mutation_tasks.append(self._call_llm_mutation_async(prompt))
            mutation_info.append({
                "replace_idx": j,
                "parent_a": parent_a_idx,
                "parent_b": parent_b_idx,
                "parent_fitness": (fitness[parent_a_idx] + fitness[parent_b_idx]) / 2,
            })

        # Execute mutations in parallel with concurrency limit
        if mutation_tasks:
            semaphore = asyncio.Semaphore(self.evo_config.max_concurrent_mutations)

            async def limited_mutation(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(
                *[limited_mutation(t) for t in mutation_tasks],
                return_exceptions=True
            )

            # Process results
            for result, info in zip(results, mutation_info):
                j = info["replace_idx"]

                if isinstance(result, BaseException):
                    logger.warning(f"Mutation failed for idx {j}: {result}")
                    continue

                child_bio: str | None = result
                if child_bio and isinstance(child_bio, str) and self._is_valid_bio(child_bio):
                    hash_str = self._get_hash(child_bio)
                    if hash_str not in self.prompt_hashes[agent]:
                        embedding = self._compute_embedding(child_bio)
                        if embedding is not None:
                            # Replace low-fitness individual with new offspring
                            # Inherit average parent fitness for fair comparison
                            new_unit = EvolutionUnit(
                                task_prompt=child_bio,
                                embedding=embedding,
                                fitness=info["parent_fitness"],
                                history=[child_bio],
                                step_added=self.generation[agent] + 1,
                                is_original=False,
                            )
                            population[j] = new_unit
                            self.prompt_hashes[agent].add(hash_str)
                            logger.debug(f"Replaced idx {j} with new offspring")

        # Apply update mode
        if self.evo_config.ga_mode == "topk":
            # Sort and keep best
            population.sort(key=lambda x: x.fitness, reverse=True)
            # Always keep original at front
            originals = [u for u in population if u.is_original]
            others = [u for u in population if not u.is_original]
            self.populations[agent] = originals + others[:self.evo_config.population_size - len(originals)]
        else:
            self.populations[agent] = population[:self.evo_config.population_size]

        self._log_evolution_summary(agent, "GA")
    
    async def _evolve_de_async(self, agent: Literal["p1", "p2"], turn: int) -> None:
        """
        DE-style evolution: differential mutation.

        OPTIMIZED:
        - Only mutate low-fitness individuals (controlled by mutation_rate)
        - Protect elite individuals (controlled by elite_ratio)
        - Parallel LLM calls with asyncio.gather()
        """
        population = self.populations[agent]
        n = len(population)

        if n < 4:
            logger.warning(f"Population too small for DE evolution: {n}")
            return

        logger.info(f"[Turn {turn}] DE evolution for {agent}, generation {self.generation[agent] + 1}")

        fitness = np.array([unit.fitness for unit in population])
        best_idx = int(np.argmax(fitness))

        # Identify protected indices (original + elites)
        original_indices = {i for i, unit in enumerate(population) if unit.is_original}
        n_elites = max(1, int(n * self.evo_config.elite_ratio))
        elite_indices = set(np.argsort(fitness)[-n_elites:])
        protected_indices = original_indices | elite_indices

        # Select mutation candidates (low fitness, non-protected)
        non_protected = [i for i in range(n) if i not in protected_indices]
        n_to_mutate = max(1, int(n * self.evo_config.mutation_rate))

        # Sort by fitness ascending and select lowest
        sorted_candidates = sorted(non_protected, key=lambda i: fitness[i])
        mutate_indices = sorted_candidates[:min(n_to_mutate, len(sorted_candidates))]

        logger.info(
            f"DE evolution: {len(mutate_indices)} mutations, "
            f"{len(protected_indices)} protected (elites + original)"
        )

        # Build mutation tasks for parallel execution
        mutation_tasks = []
        mutation_info = []

        for j in mutate_indices:
            old_prompt = population[j].task_prompt

            # Select 3 candidates (excluding j)
            candidates = [k for k in range(n) if k != j]
            a, b, c = random.sample(candidates, 3)

            # Use best as donor if configured
            if self.evo_config.de_use_best_donor:
                c = best_idx

            bio_a = population[a].task_prompt
            bio_b = population[b].task_prompt
            bio_c = population[c].task_prompt

            prompt = build_de_prompt(old_prompt, bio_a, bio_b, bio_c)
            mutation_tasks.append(self._call_llm_mutation_async(prompt))
            mutation_info.append({
                "replace_idx": j,
                "donor_fitness": fitness[c],  # Use best donor's fitness
            })

        # Execute mutations in parallel with concurrency limit
        if mutation_tasks:
            semaphore = asyncio.Semaphore(self.evo_config.max_concurrent_mutations)

            async def limited_mutation(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(
                *[limited_mutation(t) for t in mutation_tasks],
                return_exceptions=True
            )

            # Process results
            for result, info in zip(results, mutation_info):
                j = info["replace_idx"]

                if isinstance(result, BaseException):
                    logger.warning(f"DE mutation failed for idx {j}: {result}")
                    continue

                child_bio: str | None = result
                if child_bio and isinstance(child_bio, str) and self._is_valid_bio(child_bio):
                    hash_str = self._get_hash(child_bio)
                    if hash_str not in self.prompt_hashes[agent]:
                        embedding = self._compute_embedding(child_bio)
                        if embedding is not None:
                            # Replace low-fitness individual with new offspring
                            new_unit = EvolutionUnit(
                                task_prompt=child_bio,
                                embedding=embedding,
                                fitness=info["donor_fitness"],  # Inherit donor fitness
                                history=[child_bio],
                                step_added=self.generation[agent] + 1,
                                is_original=False,
                            )
                            population[j] = new_unit
                            self.prompt_hashes[agent].add(hash_str)
                            logger.debug(f"DE replaced idx {j} with new offspring")

        self.populations[agent] = population[:self.evo_config.population_size]
        self._log_evolution_summary(agent, "DE")
    
    async def _call_llm_mutation_async(self, prompt: str) -> str | None:
        """Call LLM for mutation using async acompletion with retry logic."""
        try:
            logger.debug(f"Calling LLM for mutation: {self.evo_config.mutation_model}, temperature: {self.evo_config.mutation_temperature}")

            response = await acompletion_with_retry(
                model=self.evo_config.mutation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.evo_config.mutation_temperature,
                max_tokens=4096,
                caller_name="EvoPrompt mutation",
            )
            content = response.choices[0].message.content
            if content:
                return parse_bio_from_response(content)
            return None
        except Exception as e:
            logger.error(f"EvoPrompt mutation failed: {e}")
            return None
    
    def _is_valid_bio(self, bio: str) -> bool:
        """Check if bio is valid (not too short, etc.)."""
        return bio and len(bio) >= 20
    
    def _compute_embedding(self, text: str) -> np.ndarray | None:
        """
        Compute dummy embedding for new prompt (all zeros).
        
        Since EvoPrompt is a text-based evolutionary algorithm, it doesn't strictly 
        require embeddings for its core logic (though they are part of the BaseBandit interface).
        We return a zero vector to satisfy the interface without incurring API costs.
        """
        try:
            # Create dummy zero embedding matching expected dimension
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            traceback.print_exc()
            return None
    
    def _sync_population_to_prompt_space(self, agent: Literal["p1", "p2"]) -> None:
        """Sync population back to prompt_space.
        
        Note: PromptSpace uses original_p{1,2}_background + paraphrased_p{1,2}_backgrounds.
        We sync the evolved population back by updating the paraphrased list and embeddings.
        """
        population = self.populations[agent]
        if not population:
            return
        
        # Extract prompts and embeddings from population
        # Index 0 is original, rest are paraphrased
        prompts = [unit.task_prompt for unit in population]
        embeddings = np.array([unit.embedding for unit in population])
        
        if agent == "p1":
            # First prompt is original, rest are paraphrased
            if len(prompts) > 0:
                self.prompt_space.original_p1_background = prompts[0]
            if len(prompts) > 1:
                self.prompt_space.paraphrased_p1_backgrounds = prompts[1:]
            self.prompt_space.p1_embeddings = embeddings
        else:
            if len(prompts) > 0:
                self.prompt_space.original_p2_background = prompts[0]
            if len(prompts) > 1:
                self.prompt_space.paraphrased_p2_backgrounds = prompts[1:]
            self.prompt_space.p2_embeddings = embeddings
    
    def _log_evolution_summary(self, agent: Literal["p1", "p2"], mode: str) -> None:
        """Log evolution summary with Rich table."""
        population = self.populations[agent]
        
        table = Table(title=f"{mode} Evolution - {agent} Gen {self.generation[agent] + 1}")
        table.add_column("Idx", justify="right", style="dim")
        table.add_column("Fitness", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Bio Preview", justify="left", max_width=50)
        
        for i, unit in enumerate(population):
            fitness_str = f"{unit.fitness:.4f}"
            if unit.is_original:
                status = "[cyan]Original[/cyan]"
            elif unit.step_added == self.generation[agent] + 1:
                status = "[green]New[/green]"
            else:
                status = "-"
            
            preview = unit.task_prompt[:47] + "..." if len(unit.task_prompt) > 50 else unit.task_prompt
            table.add_row(str(i), fitness_str, status, preview)
        
        console.print(table)
    
    def _save_evolution_log(self, agent: Literal["p1", "p2"], turn: int) -> None:
        """Save evolution log to file."""
        if not self.output_dir:
            return
        
        population = self.populations[agent]
        
        log = {
            "agent": agent,
            "turn": turn,
            "generation": self.generation[agent],
            "mode": self.evo_config.mode,
            "pool_size": len(population),
            "timestamp": datetime.now().isoformat(),
            "units": [
                {
                    "idx": i,
                    "fitness": unit.fitness,
                    "is_original": unit.is_original,
                    "step_added": unit.step_added,
                    "times_selected": unit.times_selected,
                    "task_prompt": unit.task_prompt,
                }
                for i, unit in enumerate(population)
            ],
        }
        
        self.evolution_history.append(log)
        
        filename = f"evoprompt_{self.evo_config.mode}_{agent}_gen{self.generation[agent]:03d}.json"
        filepath = self.evolution_log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(log, f, indent=2, default=str)
    
    def train_model(self, verbose: bool = True) -> float:
        """EvoPrompt does not use neural network training."""
        return 0.0
    
    def get_current_prompt(self, agent: Literal["p1", "p2"]) -> str:
        """Get the currently selected prompt."""
        idx = self.current_selections.get(agent, 0)
        population = self.populations[agent]
        
        if idx < len(population):
            return population[idx].task_prompt
        
        return self.prompt_space.get_prompt(agent, 0)
    
    @property
    def bandit_type(self) -> str:
        """Return bandit type identifier."""
        return f"evoprompt_{self.evo_config.mode}"
    
    def get_pool_summary(self, agent: Literal["p1", "p2"]) -> dict[str, Any]:
        """Get summary of population for an agent."""
        population = self.populations[agent]
        
        if not population:
            return {"pool_size": 0}
        
        fitness = [unit.fitness for unit in population]
        return {
            "pool_size": len(population),
            "mode": self.evo_config.mode,
            "generation": self.generation[agent],
            "avg_fitness": np.mean(fitness),
            "max_fitness": np.max(fitness),
            "min_fitness": np.min(fitness),
            "top_3": [
                {"bio": unit.task_prompt[:100], "fitness": unit.fitness}
                for unit in sorted(population, key=lambda x: x.fitness, reverse=True)[:3]
            ],
        }
