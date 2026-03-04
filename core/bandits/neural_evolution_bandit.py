"""
Neural Adversarial Evolution Bandit for dynamic prompt optimization.

This module combines NeuralAdversarialBandit's neural network value estimation and
EXP3-style selection with PromptBreeder's evolutionary optimization to create a
powerful bandit that can both evaluate and evolve prompts.

Key features:
- Neural network predicts reward directly from embeddings (HIGH score = GOOD)
- EXPO-style: no importance sampling, cumulative NN scores + softmax selection
- Evolutionary mutation generates new prompts (expands arm space)
- NN scores guide evolution: low-score (bad) arms get mutation opportunities
"""

import asyncio
import copy
import os
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table

from .base_bandit import BanditConfig
from .llm_utils import acompletion_with_retry
from .neural_adversarial_bandit import NeuralAdversarialBandit
from .prompt_breeder_bandit import EvolutionUnit
from .mutation_prompts import (
    NEURAL_EVOLUTION_MUTATION_PROMPTS,
    NEURAL_EVOLUTION_THINKING_STYLES,
)
from .prompt_space import PromptSpace

load_dotenv()
console = Console()


@dataclass
class NeuralEvolutionConfig(BanditConfig):
    """Configuration for Neural Adversarial Evolution Bandit.

    Optimized based on Neural UCB's best performing configuration:
    - eta=5.0 (balanced exploration-exploitation)
    - epochs=1000 (better NN training)
    - use_context_embedding=False (simpler, more robust)
    """

    # === NeuralAdversarial parameters ===
    eta: float = 5.0  # EXP3 exploration (Neural UCB uses 5.0)
    dynamic_eta: bool = False
    hidden_size: int = 512
    depth: int = 2  # Increased from 1 to match Neural UCB
    use_context_embedding: bool = False  # Disabled to match Neural UCB
    context_embedding_dim: int = 4096
    mask_unselected_scores: bool = False  # Update all arms
    importance_weighted_reward: bool = False

    # === Evolution parameters ===
    evolution_enabled: bool = True
    population_size: int = 13
    evolution_interval: int = 5  # Evolve every N turns
    mutation_rate: float = 0.2  # Increased from 0.1 for more exploration
    elite_ratio: float = 0.3  # Fraction to protect as elites

    # === LLM mutation parameters ===
    mutation_model: str = "openrouter/qwen/qwen-2.5-72b-instruct"
    mutation_temperature: float = 0.3
    embedding_model: str = "qwen/qwen3-embedding-8b"  # Must match the model used in prompt space embeddings
    mutation_type: str = "first_order"  # Options: "first_order", "hypermutation", "lineage", "random"
    max_concurrent_mutations: int = 5  # Max parallel LLM calls for mutation

    # === Combination strategy ===
    use_nn_score_for_evolution: bool = True  # Use NN scores to guide mutation selection
    new_prompt_score_init: str = "predict"  # predict, mean, zero

    # === Selection mode ===
    # "nn_cumulative": Use NN cumulative scores + EXP3 softmax (original)
    # "fitness": Use PopulationUnit.fitness + EXP3 softmax (new)
    selection_mode: str = "nn_cumulative"

    # === Fitness update mode ===
    # If True: dual-track update (selected arm uses real reward, unselected use NN prediction)
    # If False: only update selected arm's fitness
    dual_track_fitness_update: bool = True
    fitness_ema_alpha: float = 0.3 # EMA smoothing factor: fitness = α*new + (1-α)*old

    # === Mutation replacement strategy ===
    # If True: new mutated arm replaces lowest fitness non-protected arm
    # If False: mutation modifies the arm in-place (original behavior)
    mutation_replaces_lowest: bool = True


class NeuralAdversarialEvolutionBandit(NeuralAdversarialBandit):
    """
    Neural Adversarial Bandit with evolutionary prompt optimization.

    Inherits NeuralAdversarialBandit for:
    - Neural network value estimation
    - EXP3-style cumulative score selection

    Adds PromptBreeder-style evolution:
    - Population of prompts with thinking styles and mutation prompts
    - Periodic mutation based on NN scores (low scores = bad, get mutated)
    - High scores = good, protected as elites
    - New prompt embeddings can be immediately evaluated by NN
    """

    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | NeuralEvolutionConfig | None = None,
        tensorboard_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        # Convert to NeuralEvolutionConfig if needed
        if config is None:
            self.evo_config = NeuralEvolutionConfig()
        elif isinstance(config, NeuralEvolutionConfig):
            self.evo_config = config
        else:
            # Convert BanditConfig to NeuralEvolutionConfig
            self.evo_config = NeuralEvolutionConfig(**{
                k: v for k, v in config.__dict__.items()
                if k in NeuralEvolutionConfig.__dataclass_fields__
            })

        # Initialize parent NeuralAdversarialBandit
        super().__init__(prompt_space, self.evo_config, tensorboard_dir)

        # Set random seed for reproducibility (for random module used in evolution)
        if self.evo_config.seed is not None:
            random.seed(self.evo_config.seed)

        # Output directory for saving evolution snapshots
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.evolution_log_dir = self.output_dir / "evolution_logs"
            self.evolution_log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Evolution logs will be saved to: {self.evolution_log_dir}")

        # Initialize evolution components
        self.populations: dict[Literal["p1", "p2"], list[EvolutionUnit]] = {
            "p1": [],
            "p2": [],
        }
        self.elites: dict[Literal["p1", "p2"], list[EvolutionUnit]] = {
            "p1": [],
            "p2": [],
        }
        self.generation: dict[Literal["p1", "p2"], int] = {"p1": 0, "p2": 0}

        # Track which population index was selected
        self._last_selected_unit: dict[Literal["p1", "p2"], int] = {"p1": 0, "p2": 0}

        # Initialize populations from prompt space
        self._initialize_populations()

        # Save initial prompt space snapshot
        if self.output_dir:
            self._save_evolution_snapshot(turn=0, event="initialization")

        logger.info(
            f"Initialized NeuralAdversarialEvolutionBandit: "
            f"evolution_enabled={self.evo_config.evolution_enabled}, "
            f"population_size={self.evo_config.population_size}, "
            f"evolution_interval={self.evo_config.evolution_interval}, "
            f"mutation_rate={self.evo_config.mutation_rate}"
        )

    def _initialize_populations(self) -> None:
        """Initialize populations from the existing prompt space.

        Note: arm_index=0 is the ORIGINAL prompt (not paraphrased),
        which should be protected from mutation.
        """
        for agent in ["p1", "p2"]:
            n_available = self.prompt_space.get_num_arms(agent)
            n_to_use = min(n_available, self.evo_config.population_size)

            population = []
            for idx in range(n_to_use):
                prompt_text = self.prompt_space.get_prompt(agent, idx)
                embedding = self.prompt_space.get_embedding(agent, idx)

                # Mark if this is the original prompt (idx=0)
                is_original = (idx == 0)

                unit = EvolutionUnit(
                    thinking_style=random.choice(NEURAL_EVOLUTION_THINKING_STYLES),
                    mutation_prompt=random.choice(NEURAL_EVOLUTION_MUTATION_PROMPTS),
                    task_prompt=prompt_text,
                    fitness=1.0,
                    embedding=embedding,
                    history=[prompt_text],
                    is_original=is_original,  # Track original prompt
                )
                population.append(unit)

            self.populations[agent] = population
            logger.info(f"Initialized {agent} population with {len(population)} units (idx=0 is original)")

    async def select_async(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Async version of select with two selection modes.

        Selection modes:
        - "nn_cumulative": Use NN cumulative scores + EXP3 softmax (parent behavior)
        - "fitness": Use PopulationUnit.fitness + EXP3 softmax

        1. Check if evolution should trigger (based on turn and interval)
        2. If triggered, evolve population and sync to prompt_space
        3. Select based on configured mode

        Args:
            agent: Which agent to select for ("p1" or "p2")
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, embedding)
        """
        # 1. Check evolution trigger
        if self.evo_config.evolution_enabled:
            if turn > 0 and turn % self.evo_config.evolution_interval == 0:
                await self._evolve_population_async(agent, turn)
                self._sync_population_to_prompt_space(agent)

        # 2. Select based on configured mode
        if self.evo_config.selection_mode == "fitness":
            idx, prompt_text, embedding = self._select_by_fitness(agent, turn)
        else:
            # Default: use parent NeuralAdversarial selection (nn_cumulative)
            idx, prompt_text, embedding = super().select(agent, turn)

        # 3. Track selected unit
        self._last_selected_unit[agent] = idx

        return idx, prompt_text, embedding

    def _select_by_fitness(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select an arm based on PopulationUnit.fitness using EXP3-style softmax.

        probabilities = softmax(eta * fitness_scores)

        Also stores NN predicted scores for ALL arms in score_traces.

        Args:
            agent: Which agent to select for
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, embedding)
        """
        population = self.populations.get(agent, [])
        if not population:
            # Fallback to parent selection if no population
            return super().select(agent, turn)

        n_arms = len(population)

        # ========== Store NN predictions for ALL arms ==========
        # Skip NN prediction if model hasn't been trained yet (no selection history)
        if len(self.selection_history) > 0:
            nn_scores = self._get_nn_scores(agent)  # NN predictions for all arms
            self.score_traces[agent].append(nn_scores.tolist())
        else:
            # Use zeros before first training
            self.score_traces[agent].append([0.0] * n_arms)
        self.actual_score_traces[agent].append([0.0] * n_arms)

        # Get fitness scores from population
        fitness_scores = torch.tensor([unit.fitness for unit in population], dtype=torch.float64)

        # Apply dynamic eta scheduling if enabled: eta_t = eta * sqrt(turn + 1)
        # Early turns: small eta -> flat distribution -> exploration
        # Later turns: large eta -> sharp distribution -> exploitation
        if self.config.dynamic_eta:
            effective_eta = self.config.eta * np.sqrt(turn + 1)
            logger.debug(f"[Turn {turn}] Dynamic eta: {self.config.eta:.4f} * sqrt({turn + 1}) = {effective_eta:.4f}")
        else:
            effective_eta = self.config.eta

        # Compute probabilities: softmax(eta * fitness)
        probabilities = torch.softmax(effective_eta * fitness_scores, dim=0)

        # Apply gamma mixing for minimum exploration if configured
        if self.config.gamma > 0:
            uniform = torch.ones(n_arms, dtype=torch.float64) / n_arms
            probabilities = (1 - self.config.gamma) * probabilities + self.config.gamma * uniform

        # Check for numerical issues
        if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
            logger.error("Numerical issues in probabilities. Using uniform distribution.")
            probabilities = torch.ones(n_arms, dtype=torch.float64) / n_arms

        # Sample from the distribution
        selected_idx = int(torch.multinomial(probabilities, 1).item())

        # Store selection probability
        self.last_selection_probs[agent] = float(probabilities[selected_idx])

        # Get selected unit's prompt and embedding
        unit = population[selected_idx]
        prompt_text = unit.task_prompt
        embedding = unit.embedding

        logger.info(
            f"[Turn {turn}] Selected {agent} arm {selected_idx} (fitness-based), "
            f"fitness={fitness_scores[selected_idx]:.4f}, "
            f"prob={probabilities[selected_idx]:.4f}"
        )

        return selected_idx, prompt_text, embedding

    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Sync version of select. Select an arm using NeuralAdversarial mechanism with evolution trigger.

        Note: If evolution is enabled and triggered, this will use asyncio.run() which
        WILL FAIL if called from within an already running event loop.
        Use select_async() when calling from async code.

        Args:
            agent: Which agent to select for ("p1" or "p2")
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, embedding)
        """
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - cannot use asyncio.run()
            # Use nest_asyncio or run in executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.select_async(agent, turn))
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self.select_async(agent, turn))

    def update_with_context(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
        context_embedding: np.ndarray | None = None,
        dimension_rewards: dict[str, float] | None = None,
    ) -> None:
        """
        Update bandit with dual-track fitness update.

        Dual-track update mechanism (when enabled):
        - Selected arm: EMA update with actual reward
        - Unselected arms: EMA update with NN-predicted reward (only after NN is trained)

        This ensures all arms' fitness values are updated each turn,
        allowing the bandit to track changing arm qualities.

        Args:
            agent: Which agent was updated ("p1" or "p2")
            arm_index: The arm/unit index that was selected
            reward: The observed reward
            turn: The turn number
            context_embedding: Optional context embedding
            dimension_rewards: Optional dict of dimension->score for multi-dim prediction mode
        """
        # 1. Call parent update (trains NN, records history, etc.)
        super().update_with_context(agent, arm_index, reward, turn, context_embedding, dimension_rewards)

        population = self.populations.get(agent, [])
        if not population:
            return

        eta = self.config.eta  # EXP3 learning rate
        logger.debug(f"[Turn {turn}] eta: {eta}")
        # Get selection probability for importance sampling
        selection_prob = self.last_selection_probs.get(agent, 1.0)
        # Clip probability to avoid division by very small numbers
        selection_prob = max(selection_prob, 0.01)

        # 2. Get NN predictions for all arms (for unselected arm updates)
        # Only use NN predictions after model has been trained (need at least 2 samples)
        nn_predictions = None
        if self.evo_config.dual_track_fitness_update and len(self.selection_history) >= 2:
            try:
                nn_predictions = self._get_nn_scores(agent).flatten()
            except Exception:
                logger.warning(f"[Turn {turn}] NN prediction failed, only updating selected arm")
        elif self.evo_config.dual_track_fitness_update:
            logger.debug(f"[Turn {turn}] NN not trained yet (samples={len(self.selection_history)}), skip unselected arm updates")

        # 3. Update fitness for all arms using EMA
        # EMA: fitness = α * new_value + (1 - α) * old_fitness
        alpha = self.evo_config.fitness_ema_alpha
        for idx, unit in enumerate(population):
            old_fitness = unit.fitness

            if idx == arm_index:
                # Selected arm: use actual reward (importance-weighted)
                iw_reward = reward / selection_prob
                # Combine actual reward with predicted if available
                if nn_predictions is not None and idx < len(nn_predictions):
                    predicted_reward = float(nn_predictions[idx])
                    # Use weighted combination of actual and predicted
                    combined_reward = (iw_reward + predicted_reward) / 2
                    unit.fitness = alpha * combined_reward + (1 - alpha) * old_fitness
                    logger.debug(
                        f"[Turn {turn}] {agent} arm {idx} (SELECTED): "
                        f"fitness {old_fitness:.4f} -> {unit.fitness:.4f} "
                        f"(iw_actual={iw_reward:.2f}, predicted={predicted_reward:.2f}, α={alpha})"
                    )
                else:
                    unit.fitness = alpha * iw_reward + (1 - alpha) * old_fitness
                    logger.debug(
                        f"[Turn {turn}] {agent} arm {idx} (SELECTED): "
                        f"fitness {old_fitness:.4f} -> {unit.fitness:.4f} "
                        f"(iw_actual={iw_reward:.2f}, α={alpha})"
                    )
            elif nn_predictions is not None and idx < len(nn_predictions):
                # Unselected arm: use NN-predicted reward (dual-track mode)
                predicted_reward = float(nn_predictions[idx])
                unit.fitness = alpha * predicted_reward + (1 - alpha) * old_fitness
                logger.debug(
                    f"[Turn {turn}] {agent} arm {idx} (unselected): "
                    f"fitness {old_fitness:.4f} -> {unit.fitness:.4f} (predicted={predicted_reward:.2f}, α={alpha})"
                )

    def _sync_population_to_prompt_space(self, agent: Literal["p1", "p2"]) -> None:
        """
        Sync evolved population back to prompt_space.

        This ensures parent's select() uses updated prompts and embeddings.
        Also adjusts score_traces if population size changed.
        """
        population = self.populations[agent]
        if not population:
            return

        new_size = len(population)
        old_size = self.prompt_space.get_num_arms(agent)

        new_prompts = [unit.task_prompt for unit in population]
        new_embeddings = np.stack([unit.embedding for unit in population])

        if agent == "p1":
            self.prompt_space.p1_prompts = new_prompts
            self.prompt_space.p1_embeddings = new_embeddings
        else:
            self.prompt_space.p2_prompts = new_prompts
            self.prompt_space.p2_embeddings = new_embeddings

        # Adjust score_traces and actual_score_traces if population size changed
        if new_size != old_size and len(self.score_traces[agent]) > 0:
            logger.warning(
                f"Population size changed from {old_size} to {new_size}, "
                f"adjusting score_traces for {agent}"
            )
            # Adjust NN score_traces
            adjusted_traces = []
            for trace in self.score_traces[agent]:
                if len(trace) > new_size:
                    adjusted_traces.append(trace[:new_size])
                elif len(trace) < new_size:
                    adjusted_traces.append(trace + [0.0] * (new_size - len(trace)))
                else:
                    adjusted_traces.append(trace)
            self.score_traces[agent] = adjusted_traces

            # Adjust actual_score_traces
            adjusted_actual_traces = []
            for trace in self.actual_score_traces[agent]:
                if len(trace) > new_size:
                    adjusted_actual_traces.append(trace[:new_size])
                elif len(trace) < new_size:
                    adjusted_actual_traces.append(trace + [0.0] * (new_size - len(trace)))
                else:
                    adjusted_actual_traces.append(trace)
            self.actual_score_traces[agent] = adjusted_actual_traces

        logger.debug(f"Synced {agent} population ({len(new_prompts)} prompts) to prompt_space")

    async def _evolve_population_async(self, agent: Literal["p1", "p2"], turn: int) -> None:
        """
        Async version of evolve population using NN scores to guide mutation selection.

        Key innovation (EXPO-style, NN predicts reward directly):
        - Use neural network predicted scores (HIGH score = GOOD)
        - Low-score arms (bad prompts) get mutation opportunities
        - High-score arms (good prompts) are protected as elites
        """
        population = self.populations[agent]
        n_arms = len(population)

        if n_arms < 2:
            logger.warning(f"Population too small for evolution: {n_arms}")
            return

        self.generation[agent] += 1
        logger.info(
            f"[Turn {turn}] Evolving {agent} population - Generation {self.generation[agent]}"
        )

        # 1. Get scores for evolution guidance
        if self.evo_config.use_nn_score_for_evolution:
            scores = self._get_nn_scores(agent)
        else:
            # Fallback to cumulative scores from score_traces
            if len(self.score_traces[agent]) > 0:
                scores = np.sum(self.score_traces[agent], axis=0)
            else:
                scores = np.ones(n_arms)

        # Ensure scores array matches population size - raise error if mismatch
        if len(scores) != n_arms:
            raise ValueError(
                f"Scores array size ({len(scores)}) does not match population size ({n_arms}) for {agent}. "
                f"score_traces shape: {[len(t) for t in self.score_traces[agent]] if self.score_traces[agent] else 'empty'}, "
                f"population size: {n_arms}"
            )

        # 2. Get fitness values for elite selection (fitness = accumulated performance)
        fitness_values = np.array([unit.fitness for unit in population])

        # 3. Identify protected indices:
        #    - Original prompt (idx=0) is ALWAYS protected
        #    - Elites (highest FITNESS) are protected and used as parents
        original_indices = {i for i, unit in enumerate(population) if unit.is_original}

        n_elites = max(1, int(n_arms * self.evo_config.elite_ratio))
        # Select elites by FITNESS (not NN score) - fitness represents accumulated real performance
        elite_indices = set(np.argsort(fitness_values)[-n_elites:])

        # Combine protected: original + elites
        protected_indices = original_indices | elite_indices

        # Save elites for lineage mutation
        for idx in elite_indices:
            self.elites[agent].append(copy.deepcopy(population[idx]))
        self.elites[agent] = self.elites[agent][-10:]  # Keep last 10

        # 4. Select mutation candidates (low FITNESS are replaced deterministically)
        # Deterministic selection: sort by fitness and select lowest fitness arms
        n_to_mutate = max(1, int(n_arms * self.evo_config.mutation_rate))
        logger.debug(f"[Turn {turn}] Selecting {n_to_mutate} mutation candidates from {n_arms} arms")
        
        # Exclude protected indices (original + elites) from mutation candidates
        non_protected_indices = [i for i in range(n_arms) if i not in protected_indices]

        if non_protected_indices:
            # Sort non-protected indices by fitness (ascending) and select lowest fitness arms
            sorted_non_protected = sorted(
                non_protected_indices,
                key=lambda i: fitness_values[i]
            )
            mutate_count = min(n_to_mutate, len(non_protected_indices))
            mutate_indices = sorted_non_protected[:mutate_count]  # Deterministic: lowest fitness first
            
            logger.debug(
                f"[Turn {turn}] Deterministic selection: replacing arms {mutate_indices} "
                f"with fitness {[fitness_values[i] for i in mutate_indices]}"
            )
        else:
            mutate_indices = []

        # 5. Select elite parents for mutation (highest FITNESS elites)
        # Low-fitness arms will be REPLACED, high-fitness elite parents provide the genetic material
        elite_list = list(elite_indices)

        # Build mutation plan: which low-fitness arm to replace, which elite parent to use
        logger.info(f"[Turn {turn}] Building mutation plan for {len(mutate_indices)} candidates, elite_list={elite_list}")
        mutation_plan: list[dict] = []
        for i, replace_idx in enumerate(mutate_indices):
            logger.info(f"[Turn {turn}] Plan #{i}: replace_idx={replace_idx}")
            # Select parent from elites by FITNESS (best fitness first)
            if elite_list:
                sorted_elites = sorted(elite_list, key=lambda x: fitness_values[x], reverse=True)
                parent_idx = sorted_elites[i % len(sorted_elites)]
            else:
                parent_idx = replace_idx  # Fallback to self (shouldn't happen)

            # Handle multi-dim scores: take mean if array, otherwise use as-is
            replace_score = scores[replace_idx]
            parent_score = scores[parent_idx]
            if hasattr(replace_score, '__len__') and len(replace_score) > 1:
                replace_score = float(np.mean(replace_score))
            else:
                replace_score = float(replace_score)
            if hasattr(parent_score, '__len__') and len(parent_score) > 1:
                parent_score = float(np.mean(parent_score))
            else:
                parent_score = float(parent_score)

            mutation_plan.append({
                "replace_idx": int(replace_idx),
                "replace_score": replace_score,
                "replace_fitness": float(fitness_values[replace_idx]),
                "parent_idx": int(parent_idx),
                "parent_score": parent_score,
                "parent_fitness": float(fitness_values[parent_idx]),
            })

        # 5. Execute mutations from elite parents (async) - PARALLEL with semaphore
        logger.info(f"[Turn {turn}] Starting {len(mutation_plan)} mutation(s) in parallel using model: {self.evo_config.mutation_model}")

        # Use semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(self.evo_config.max_concurrent_mutations)

        async def mutate_with_limit(plan: dict, idx: int) -> tuple[EvolutionUnit | None, dict, int]:
            async with semaphore:
                logger.debug(f"[Turn {turn}] Mutation {idx+1}/{len(mutation_plan)}: parent={plan['parent_idx']} -> replace={plan['replace_idx']}")
                new_unit = await self._mutate_unit_async(
                    agent, plan["parent_idx"], turn,
                    inherit_fitness=plan["parent_fitness"],
                    inherit_score=plan["parent_score"]
                )
                return new_unit, plan, idx

        # Execute all mutations in parallel
        results = await asyncio.gather(
            *[mutate_with_limit(plan, i) for i, plan in enumerate(mutation_plan)],
            return_exceptions=True
        )

        # Collect successful mutations
        new_units: list[tuple[EvolutionUnit, dict]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"[Turn {turn}] Mutation failed: {result}")
                continue
            new_unit, plan, idx = result
            if new_unit is not None:
                new_units.append((new_unit, plan))
                logger.info(f"[Turn {turn}] Mutation {idx+1} completed successfully")

        # 6. Replace low-score arms with new units from elite parents
        mutated_count = 0
        replacement_info: list[dict] = []
        pre_mutation_state: list[dict] = []

        for new_unit, plan in new_units:
            replace_idx = plan["replace_idx"]
            old_fitness = plan["replace_fitness"]
            old_score = plan["replace_score"]

            # Record pre-mutation state
            pre_mutation_state.append({
                "child_idx": replace_idx,
                "child_score": old_score,
                "child_fitness": old_fitness,
                "parent_idx": plan["parent_idx"],
                "parent_score": plan["parent_score"],
                "parent_fitness": plan["parent_fitness"],
            })

            # Record replacement info
            replacement_info.append({
                "replaced_idx": replace_idx,
                "old_score": old_score,
                "old_fitness": old_fitness,
                "new_fitness": new_unit.fitness,
                "parent_idx": plan["parent_idx"],
            })

            # Replace the unit
            population[replace_idx] = new_unit
            fitness_values[replace_idx] = new_unit.fitness

            logger.debug(
                f"[Turn {turn}] Replaced {agent} arm {replace_idx} (score={old_score:.4f}, fitness={old_fitness:.4f}) "
                f"with child of elite {plan['parent_idx']} (new fitness={new_unit.fitness:.4f})"
            )
            mutated_count += 1

            # Update score traces: inherit parent's score for the replaced position
            self._init_new_prompt_score_from_parent(
                agent, replace_idx, plan["parent_idx"], new_unit.embedding
            )

        # Log evolution summary with mutation details
        self._log_evolution_summary(
            agent, scores, elite_indices, original_indices, mutate_indices,
            mutated_count, pre_mutation_state, replacement_info
        )

        # Save evolution snapshot to output directory
        if self.output_dir:
            self._save_evolution_snapshot(
                turn=turn,
                event=f"evolution_{agent}_gen{self.generation[agent]}",
                agent=agent,
                scores=scores,
                elite_indices=elite_indices,
                original_indices=original_indices,
                mutate_indices=list(mutate_indices) if len(mutate_indices) > 0 else [],
            )

    def _get_nn_scores(self, agent: Literal["p1", "p2"]) -> np.ndarray:
        """Get neural network predicted scores for all arms.

        Returns scores. Shape depends on multi_dim_prediction:
        - multi_dim_prediction=False: (n_arms,) - single score per arm
        - multi_dim_prediction=True: (n_arms, 7) - 7 dimension scores per arm
        """
        population = self.populations[agent]
        embeddings = np.stack([unit.embedding for unit in population])

        # Combine with context if enabled
        if self.use_context_embedding:
            if self.current_context_embedding is not None:
                n_arms = embeddings.shape[0]
                context_replicated = np.tile(self.current_context_embedding, (n_arms, 1))
                combined = np.concatenate([embeddings, context_replicated], axis=1)
            else:
                context_zeros = np.zeros((embeddings.shape[0], self.context_embedding_dim))
                combined = np.concatenate([embeddings, context_zeros], axis=1)
            embeddings_tensor = torch.tensor(combined, dtype=torch.float64).to(self.config.device)
        else:
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float64).to(self.config.device)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(embeddings_tensor).cpu().numpy()

        # Log multi-dimensional scores if enabled (skip during initialization when model is untrained)
        if self.config.multi_dim_prediction and len(scores.shape) > 1 and scores.shape[1] > 1:
            # Only log after first training (check if we have selection history)
            if len(self.selection_history) > 0:
                dim_names = self.config.dimension_names
                logger.info(f"[{agent}] Multi-dim prediction scores (shape={scores.shape}):")
                for arm_idx in range(min(len(population), scores.shape[0])):
                    dim_scores = {dim_names[i]: f"{scores[arm_idx, i]:.3f}" for i in range(min(len(dim_names), scores.shape[1]))}
                    avg_score = np.mean(scores[arm_idx])
                    logger.info(f"  Arm {arm_idx}: {dim_scores} => avg={avg_score:.3f}")

        return scores

    async def _mutate_unit_async(
        self, agent: Literal["p1", "p2"], unit_idx: int, turn: int,
        inherit_fitness: float | None = None,
        inherit_score: float | None = None,
    ) -> EvolutionUnit | None:
        """
        Mutate a unit using LLM-based mutation operators (async version).

        Mutation types from PromptBreeder:
        - first_order: M + P -> P'
        - hypermutation: D + T -> M'
        - lineage: elite history -> P'

        Args:
            agent: Which agent's population to mutate
            unit_idx: Index of the parent unit to mutate from
            turn: Current turn number
            inherit_fitness: Override fitness to inherit (from elite parent)
            inherit_score: Score of parent (for logging)

        Returns:
            New EvolutionUnit with inherited parent fitness, or None if mutation failed
        """
        parent_unit = self.populations[agent][unit_idx]
        old_prompt = parent_unit.task_prompt
        # Use provided inherit_fitness or fall back to parent's fitness
        parent_fitness = inherit_fitness if inherit_fitness is not None else parent_unit.fitness

        # Use configured mutation type, or random if set to "random"
        if self.evo_config.mutation_type == "random":
            mutation_type = random.choice(["first_order", "hypermutation", "lineage"])
        else:
            mutation_type = self.evo_config.mutation_type

        try:
            if mutation_type == "first_order":
                # Social simulation-aware mutation prompt
                prompt = (
                    f"You are improving a character instruction for a multi-agent social simulation.\n\n"
                    f"## Mutation Guidance: \n{parent_unit.mutation_prompt}\n\n"
                    f"## Original Character Instruction\n{parent_unit.task_prompt}\n\n"
                    f"## Requirements for Improvement\n"
                    f"1. Fact Preservation: Keep ALL original identity markers intact (name, age, gender, occupation, and personal secrets).\n"
                    f"2. Strategic Re-alignment: Modify the bio to reflect more effective social strategies.\n"
                    f"3. Behavioral Consistency: Ensure the rewritten traits will guide the agent toward their Goal without 'Topic Drift' or 'Parroting'.\n"
                    f"4. Social Resilience: Frame the character's values to be robust against adversarial 'Deadlocks' by emphasizing flexible strategic pathways.\n"
                    f"5. Output Format: Provide ONLY the modified instruction text. Do not include meta-commentary or reasoning steps.\n\n"
                    f"## Improved Character Instruction:"
                )
            elif mutation_type == "hypermutation":
                # Social simulation-aware hypermutation (evolve the mutation prompt itself)
                prompt = (
                    f"You are improving a meta-instruction that guides how to evolve character behaviors "
                    f"in multi-agent social simulations.\n\n"
                    f"## Current Meta-Instruction\n{parent_unit.mutation_prompt}\n\n"
                    f"## Context\n"
                    f"This meta-instruction is used to mutate character instructions for social dialogues. "
                    f"Characters need to: achieve their goals, maintain relationships, read social cues, "
                    f"adapt to resistance, and stay believable.\n\n"
                    f"## Improved Meta-Instruction (make it more effective for social simulation):"
                )
            elif mutation_type == "lineage":
                elites = self.elites[agent]
                if elites:
                    elite_prompts = "\n".join(
                        [f"{i+1}. {e.task_prompt}" for i, e in enumerate(elites[-5:])]
                    )
                    prompt = (
                        f"You are creating a new character instruction for a multi-agent social simulation.\n\n"
                        f"## High-Performing Character Instructions (ranked by effectiveness)\n"
                        f"{elite_prompts}\n\n"
                        f"## Task\n"
                        f"Analyze the patterns that made these instructions effective, then synthesize a new instruction "
                        f"that combines their strengths while adding novel elements.\n\n"
                        f"## Requirements\n"
                        f"1. Fact Preservation: Keep ALL original identity markers intact (name, age, gender, occupation, and personal secrets).\n"
                        f"2. Behavioral Consistency: Avoid 'Topic Drift' or 'Parroting'.\n"
                        f"3. Social Resilience: Emphasize flexible strategic pathways to avoid 'Deadlocks'.\n"
                        f"4. Output Format: Provide ONLY the instruction text, no meta-commentary.\n\n"
                        f"## New Character Instruction:"
                    )
                else:
                    # Fallback to first_order style if no elites
                    prompt = (
                        f"You are improving a character instruction for a multi-agent social simulation.\n\n"
                        f"## Mutation Guidance\n{parent_unit.mutation_prompt}\n\n"
                        f"## Original Character Instruction\n{parent_unit.task_prompt}\n\n"
                        f"## Requirements\n"
                        f"1. Fact Preservation: Keep ALL identity markers intact.\n"
                        f"2. Behavioral Consistency: Avoid 'Topic Drift' or 'Parroting'.\n"
                        f"3. Social Resilience: Emphasize flexible strategic pathways.\n\n"
                        f"## Improved Character Instruction:"
                    )
            else:
                # Default to first_order mutation (social simulation-aware)
                prompt = (
                    f"You are improving a character instruction for a multi-agent social simulation.\n\n"
                    f"## Mutation Guidance\n{parent_unit.mutation_prompt}\n\n"
                    f"## Original Character Instruction\n{parent_unit.task_prompt}\n\n"
                    f"## Requirements\n"
                    f"1. Fact Preservation: Keep ALL identity markers intact.\n"
                    f"2. Behavioral Consistency: Avoid 'Topic Drift' or 'Parroting'.\n"
                    f"3. Social Resilience: Emphasize flexible strategic pathways.\n\n"
                    f"## Improved Character Instruction:"
                )

            # Call LLM for mutation with retry logic
            logger.debug(f"Calling LLM for mutation: {mutation_type}, model: {self.evo_config.mutation_model}, temperature: {self.evo_config.mutation_temperature}")

            response = await acompletion_with_retry(
                model=self.evo_config.mutation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.evo_config.mutation_temperature,
                max_tokens=4096,
                caller_name=f"Mutation({mutation_type})",
            )
            new_prompt = response.choices[0].message.content.strip()

            if new_prompt and len(new_prompt) > 10:
                # Compute new embedding
                new_embedding = self._compute_embedding(new_prompt)

                # Create new unit inheriting parent's fitness
                new_unit = EvolutionUnit(
                    task_prompt=new_prompt,
                    thinking_style=parent_unit.thinking_style,
                    mutation_prompt=parent_unit.mutation_prompt,
                    embedding=new_embedding,
                    fitness=parent_fitness,  # Inherit parent fitness (avoid cold start)
                    is_original=False,
                    history=parent_unit.history + [new_prompt],
                )

                logger.debug(
                    f"[Turn {turn}] Created mutated unit from {agent} parent {unit_idx} via {mutation_type}, "
                    f"inherited fitness={parent_fitness:.4f}"
                )
                return new_unit
            else:
                logger.warning(f"Invalid mutation result, keeping original prompt")
                return None

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Mutation failed for {agent} unit {unit_idx}: {e}")
            return None

    def _mutate_unit(
        self, agent: Literal["p1", "p2"], unit_idx: int, turn: int
    ) -> EvolutionUnit | None:
        """
        Mutate a unit using LLM-based mutation operators (sync wrapper).

        This wraps the async version for compatibility with sync callers.
        Note: Will fail if called from within an async context. Use _mutate_unit_async instead.

        Returns:
            New EvolutionUnit with inherited parent fitness, or None if mutation failed
        """
        try:
            asyncio.get_running_loop()
            # We're in an async context - cannot use asyncio.run()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._mutate_unit_async(agent, unit_idx, turn))
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self._mutate_unit_async(agent, unit_idx, turn))

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a new prompt text using OpenRouter API directly.

        Note: If the API returns a different embedding dimension than expected,
        we will pad with zeros or truncate to match self.embedding_dim.
        """
        max_retries = 5
        base_delay = 2.0

        # Get model name (remove openrouter/ prefix if present)
        model = self.evo_config.embedding_model
        if model.startswith("openrouter/"):
            model = model[len("openrouter/"):]

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": text,
        }
        url = "https://openrouter.ai/api/v1/embeddings"

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    if "data" not in data:
                        logger.error(f"OpenRouter API response missing 'data' field: {data}")
                        raise ValueError("OpenRouter API response missing 'data' field")
                    raw_embedding = data["data"][0]["embedding"]
                    if raw_embedding is None:
                        raise ValueError("Embedding returned None for text")

                    # Convert to numpy array
                    embedding = np.array(raw_embedding, dtype=np.float32)

                    # Verify dimension matches expected
                    if len(embedding) != self.embedding_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch: got {len(embedding)}, "
                            f"expected {self.embedding_dim}. Check embedding_model config "
                            f"(current: {model}) matches the model used for prompt space embeddings."
                        )

                    return embedding
                elif resp.status_code in (500, 502, 503, 504, 429):
                    # Server error or rate limit, retry
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"OpenRouter API error {resp.status_code}, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise ValueError(f"OpenRouter API error: {resp.status_code} - {resp.text}")
            except requests.exceptions.RequestException as e:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Request failed: {e}, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)

        raise RuntimeError(f"Max retries ({max_retries}) exceeded for embedding request")

    def _init_new_prompt_score(
        self, agent: Literal["p1", "p2"], idx: int, new_embedding: np.ndarray
    ) -> None:
        """
        Initialize new mutated prompt's score in score_traces.

        For NN scores (score_traces): Use NN prediction (NN can generalize to new embeddings)
        For actual scores (actual_score_traces): Keep as 0 (no actual observation for new arm)

        Strategies for NN scores:
        - "predict": Use NN to predict initial score (leverages generalization)
        - "mean": Use current mean score across all arms
        - "zero": Start from zero (most conservative)
        """
        strategy = self.evo_config.new_prompt_score_init

        if len(self.score_traces[agent]) == 0:
            return  # No history yet

        def _extend_trace_if_needed(trace: list, target_idx: int) -> None:
            """Extend trace list to accommodate target_idx if needed."""
            while len(trace) <= target_idx:
                trace.append(0.0)

        if strategy == "predict":
            if self.use_context_embedding and self.current_context_embedding is not None:
                combined = np.concatenate([new_embedding, self.current_context_embedding])
                emb_tensor = torch.tensor(combined.reshape(1, -1), dtype=torch.float64)
            else:
                if self.use_context_embedding:
                    context_zeros = np.zeros(self.context_embedding_dim)
                    combined = np.concatenate([new_embedding, context_zeros])
                    emb_tensor = torch.tensor(combined.reshape(1, -1), dtype=torch.float64)
                else:
                    emb_tensor = torch.tensor(new_embedding.reshape(1, -1), dtype=torch.float64)

            emb_tensor = emb_tensor.to(self.config.device)
            self.model.eval()
            with torch.no_grad():
                predicted_score = self.model(emb_tensor).cpu().item()

            # Update all historical NN scores for this arm
            for t in range(len(self.score_traces[agent])):
                trace = self.score_traces[agent][t]
                _extend_trace_if_needed(trace, idx)
                trace[idx] = predicted_score

            logger.debug(
                f"Initialized new prompt NN score using prediction: {predicted_score:.4f}"
            )

        elif strategy == "mean":
            if self.score_traces[agent]:
                # Mean across all arms for each turn
                mean_score = float(np.mean([np.mean(s) for s in self.score_traces[agent]]))
                for t in range(len(self.score_traces[agent])):
                    trace = self.score_traces[agent][t]
                    _extend_trace_if_needed(trace, idx)
                    trace[idx] = mean_score

        elif strategy == "zero":
            for t in range(len(self.score_traces[agent])):
                trace = self.score_traces[agent][t]
                _extend_trace_if_needed(trace, idx)
                trace[idx] = 0.0

        # actual_score_traces: new arm has no actual observations, keep as 0
        for t in range(len(self.actual_score_traces[agent])):
            trace = self.actual_score_traces[agent][t]
            _extend_trace_if_needed(trace, idx)
            trace[idx] = 0.0

    def _init_new_prompt_score_from_parent(
        self, agent: Literal["p1", "p2"], child_idx: int, parent_idx: int,
        new_embedding: np.ndarray
    ) -> None:
        """
        Initialize new mutated prompt's score.

        For NN scores: Use current NN prediction for the new embedding (NN can generalize)
        For actual scores: Inherit from parent (actual observations can only be inherited)
        """
        if len(self.score_traces[agent]) == 0:
            return  # No history yet

        # 1. NN scores: predict with current NN for the new embedding
        if self.use_context_embedding and self.current_context_embedding is not None:
            combined = np.concatenate([new_embedding, self.current_context_embedding])
            emb_tensor = torch.tensor(combined.reshape(1, -1), dtype=torch.float64)
        else:
            if self.use_context_embedding:
                context_zeros = np.zeros(self.context_embedding_dim)
                combined = np.concatenate([new_embedding, context_zeros])
                emb_tensor = torch.tensor(combined.reshape(1, -1), dtype=torch.float64)
            else:
                emb_tensor = torch.tensor(new_embedding.reshape(1, -1), dtype=torch.float64)

        emb_tensor = emb_tensor.to(self.config.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(emb_tensor).cpu()
            # Handle multi-dim prediction: take mean if output has multiple dimensions
            if output.numel() > 1:
                predicted_score = float(output.mean().item())
            else:
                predicted_score = output.item()

        # Fill all historical NN scores with current prediction
        for t in range(len(self.score_traces[agent])):
            trace = self.score_traces[agent][t]
            # Extend trace if needed
            while len(trace) <= child_idx:
                trace.append(0.0)
            trace[child_idx] = predicted_score

        # 2. Actual scores: new arm has no actual observations, keep as 0
        # (actual scores only record real observations, not inherited from parent)
        for t in range(len(self.actual_score_traces[agent])):
            trace = self.actual_score_traces[agent][t]
            # Extend trace if needed
            while len(trace) <= child_idx:
                trace.append(0.0)
            trace[child_idx] = 0.0

        nn_cum = sum(self.score_traces[agent][t][child_idx] for t in range(len(self.score_traces[agent])))
        logger.debug(
            f"Child arm {child_idx}: NN predicted={predicted_score:.4f}, nn_cum={nn_cum:.3f}, "
            f"actual_cum=0.0 (new arm has no observations)"
        )

    def _log_evolution_summary(
        self,
        agent: Literal["p1", "p2"],
        scores: np.ndarray,
        elite_indices: set,
        original_indices: set,
        mutate_indices: np.ndarray,
        mutated_count: int,
        pre_mutation_state: list[dict] | None = None,
        replacement_info: list[dict] | None = None,
    ) -> None:
        """Log evolution summary with Rich table.

        Shows population status with original prompt marked as protected.
        Idx=0 is typically the original (un-paraphrased) prompt.
        """
        population = self.populations[agent]
        n_arms = len(population)

        # Compute cumulative scores from both traces
        decay = self.config.score_decay
        n_turns = len(self.score_traces[agent])

        nn_cumulative = np.zeros(n_arms)
        actual_cumulative = np.zeros(n_arms)

        if n_turns > 0:
            # Create decay weights
            if decay < 1.0 and n_turns > 1:
                weights = np.array([decay ** (n_turns - 1 - t) for t in range(n_turns)])
            else:
                weights = np.ones(n_turns)

            # NN cumulative - ensure size matches n_arms
            nn_scores = np.array(self.score_traces[agent])  # (n_turns, n_arms_trace)
            nn_sum = np.sum(nn_scores * weights[:, None], axis=0)
            # Align to n_arms
            if len(nn_sum) >= n_arms:
                nn_cumulative = nn_sum[:n_arms]
            else:
                nn_cumulative[:len(nn_sum)] = nn_sum

            # Actual cumulative - ensure size matches n_arms
            actual_scores = np.array(self.actual_score_traces[agent])  # (n_turns, n_arms_trace)
            actual_sum = np.sum(actual_scores * weights[:, None], axis=0)
            # Align to n_arms
            if len(actual_sum) >= n_arms:
                actual_cumulative = actual_sum[:n_arms]
            else:
                actual_cumulative[:len(actual_sum)] = actual_sum

        combined_cumulative = (nn_cumulative + actual_cumulative) / 2.0

        # Final validation: ensure all arrays match n_arms
        assert len(combined_cumulative) == n_arms, \
            f"combined_cumulative size ({len(combined_cumulative)}) != n_arms ({n_arms})"

        # ========== 1. Pre-Mutation State Table ==========
        if pre_mutation_state:
            pre_table = Table(title=f"Pre-Mutation State - {agent} (before evolution)")
            pre_table.add_column("Child Idx", justify="right", style="yellow")
            pre_table.add_column("Child Score", justify="right")
            pre_table.add_column("Child Fitness", justify="right")
            pre_table.add_column("→", justify="center", style="dim")
            pre_table.add_column("Parent Idx", justify="right", style="green")
            pre_table.add_column("Parent Score", justify="right")
            pre_table.add_column("Parent Fitness", justify="right")
            pre_table.add_column("Reason", justify="left")

            for info in pre_mutation_state:
                pre_table.add_row(
                    str(info["child_idx"]),
                    f"{info['child_score']:.4f}",
                    f"{info['child_fitness']:.4f}",
                    "←",
                    str(info["parent_idx"]),
                    f"{info['parent_score']:.4f}",
                    f"{info['parent_fitness']:.4f}",
                    f"Low score arm {info['child_idx']} mutated from elite {info['parent_idx']}"
                )
            console.print(pre_table)

        # ========== 2. Replacement Info Table ==========
        if replacement_info:
            replace_table = Table(title=f"Replacement Details - {agent}")
            replace_table.add_column("Replaced Idx", justify="right", style="red")
            replace_table.add_column("Old Score", justify="right")
            replace_table.add_column("Old Fitness", justify="right")
            replace_table.add_column("→", justify="center", style="dim")
            replace_table.add_column("New Fitness", justify="right", style="green")

            for info in replacement_info:
                replace_table.add_row(
                    str(info["replaced_idx"]),
                    f"{info['old_score']:.4f}",
                    f"{info['old_fitness']:.4f}",
                    "→",
                    f"{info['new_fitness']:.4f}",
                )
            console.print(replace_table)

        # ========== 3. Main Evolution Summary Table ==========
        # Sort by combined score to show ranking
        sorted_indices = np.argsort(combined_cumulative)[::-1]  # Descending

        table = Table(title=f"Evolution Summary - {agent} Gen {self.generation[agent]}")
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Idx", justify="right", style="dim")
        table.add_column("NN Score", justify="right")
        table.add_column("NN Cum", justify="right")
        table.add_column("Actual Cum", justify="right")
        table.add_column("Combined", justify="right")
        table.add_column("Fitness", justify="right")
        table.add_column("Status", justify="center")

        # Validate scores array size matches population
        if len(scores) != n_arms:
            raise ValueError(
                f"Scores array size ({len(scores)}) does not match population size ({n_arms}) in _log_evolution_summary"
            )

        for rank, i in enumerate(sorted_indices):
            # Handle multi-dim scores: take mean if array
            score_val = scores[i]
            if hasattr(score_val, '__len__') and len(score_val) > 1:
                score_val = float(np.mean(score_val))
            nn_score_str = f"{score_val:.4f}"
            nn_cum_str = f"{nn_cumulative[i]:.3f}"
            actual_cum_str = f"{actual_cumulative[i]:.3f}"
            combined_str = f"{combined_cumulative[i]:.3f}"
            fitness_str = f"{population[i].fitness:.4f}"

            # Determine status
            if i in original_indices:
                status = "[cyan]Original[/cyan]"
            elif i in elite_indices:
                status = "[green]Elite[/green]"
            elif i in mutate_indices:
                status = "[yellow]Mutated[/yellow]"
            else:
                status = "-"

            table.add_row(
                str(rank + 1), str(i), nn_score_str, nn_cum_str,
                actual_cum_str, combined_str, fitness_str, status
            )

        console.print(table)

        # ========== 4. Score Traces Table (last 5 turns) ==========
        if n_turns > 0:
            show_turns = min(5, n_turns)
            traces_table = Table(title=f"Score Traces (last {show_turns} turns) - {agent}")
            traces_table.add_column("Arm", justify="right", style="dim")

            for t in range(n_turns - show_turns, n_turns):
                traces_table.add_column(f"T{t} NN", justify="right")
                traces_table.add_column(f"T{t} Act", justify="right")

            for i in range(n_arms):
                row = [str(i)]
                for t in range(n_turns - show_turns, n_turns):
                    nn_val = self.score_traces[agent][t][i]
                    actual_val = self.actual_score_traces[agent][t][i]
                    row.append(f"{nn_val:.3f}")
                    row.append(f"{actual_val:.3f}")
                traces_table.add_row(*row)

            console.print(traces_table)

        logger.info(
            f"Evolution complete: {mutated_count} mutations, {len(elite_indices)} elites, "
            f"{len(original_indices)} original(s) protected"
        )

    def _save_evolution_snapshot(
        self,
        turn: int,
        event: str,
        agent: Literal["p1", "p2"] | None = None,
        scores: np.ndarray | None = None,
        elite_indices: set | None = None,
        original_indices: set | None = None,
        mutate_indices: list | None = None,
    ) -> None:
        """Save a complete snapshot of the current prompt space to a JSON file.

        Args:
            turn: Current turn number
            event: Event name (e.g., "initialization", "evolution_p1_gen1")
            agent: Agent that was evolved (None for initialization)
            scores: NN scores for the agent's population
            elite_indices: Indices of elite units
            original_indices: Indices of original (non-paraphrased) units
            mutate_indices: Indices of mutated units
        """
        import json
        from datetime import datetime

        if not self.output_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"turn{turn:03d}_{event}_{timestamp}.json"
        filepath = self.evolution_log_dir / filename

        # Build snapshot data
        snapshot = {
            "metadata": {
                "turn": turn,
                "event": event,
                "timestamp": timestamp,
                "scenario_id": self.prompt_space.scenario_id,
            },
            "config": {
                "evolution_enabled": self.evo_config.evolution_enabled,
                "population_size": self.evo_config.population_size,
                "evolution_interval": self.evo_config.evolution_interval,
                "mutation_rate": self.evo_config.mutation_rate,
                "elite_ratio": self.evo_config.elite_ratio,
            },
            "generations": {
                "p1": self.generation["p1"],
                "p2": self.generation["p2"],
            },
            "populations": {},
        }

        # Add population data for each agent
        for a in ["p1", "p2"]:
            population = self.populations[a]
            if not population:
                continue

            # Get scores for this agent (skip if model not trained yet)
            if len(self.selection_history) > 0:
                try:
                    agent_scores = self._get_nn_scores(a)
                except Exception:
                    agent_scores = np.zeros(len(population))
            else:
                # Use zeros before first training
                agent_scores = np.zeros(len(population))

            units_data = []
            for i, unit in enumerate(population):
                # Handle multi-dimensional scores (shape: n_arms x n_dims) vs single-dim (shape: n_arms,)
                if i < len(agent_scores):
                    score_val = agent_scores[i]
                    # If multi-dim prediction, score_val is an array of 7 dims - take mean
                    if hasattr(score_val, '__len__') and len(score_val) > 1:
                        nn_score = float(np.mean(score_val))
                    else:
                        nn_score = float(score_val)
                else:
                    nn_score = 0.0

                unit_data = {
                    "idx": i,
                    "is_original": unit.is_original,
                    "nn_score": nn_score,
                    "fitness": unit.fitness,
                    "thinking_style": unit.thinking_style,
                    "mutation_prompt": unit.mutation_prompt,
                    "task_prompt": unit.task_prompt,
                    "history_length": len(unit.history),
                }

                # Add status for the evolved agent
                if a == agent and scores is not None:
                    if original_indices and i in original_indices:
                        unit_data["status"] = "original"
                    elif elite_indices and i in elite_indices:
                        unit_data["status"] = "elite"
                    elif mutate_indices and i in mutate_indices:
                        unit_data["status"] = "mutated"
                    else:
                        unit_data["status"] = "unchanged"

                units_data.append(unit_data)

            snapshot["populations"][a] = {
                "size": len(population),
                "generation": self.generation[a],
                "units": units_data,
            }

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved evolution snapshot: {filepath}")

    def get_population_summary(self, agent: Literal["p1", "p2"]) -> dict:
        """Get summary statistics for an agent's population."""
        population = self.populations[agent]
        if not population:
            return {"size": 0, "units": []}

        # Use NN scores for summary
        scores = self._get_nn_scores(agent)

        return {
            "size": len(population),
            "generation": self.generation[agent],
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "std_score": float(np.std(scores)),
            "n_elites": len(self.elites[agent]),
            "units": [
                {
                    "idx": i,
                    "nn_score": float(scores[i]),
                    "prompt_preview": (
                        u.task_prompt[:50] + "..."
                        if len(u.task_prompt) > 50
                        else u.task_prompt
                    ),
                    "n_mutations": len(u.history) - 1,
                }
                for i, u in enumerate(population)
            ],
        }

    def save_model(self, path: Path) -> None:
        """Save model weights and population state."""
        import json

        path = Path(path)

        # Save neural network weights (parent method)
        nn_path = path.with_suffix(".pt") if path.suffix != ".pt" else path
        super().save_model(nn_path)

        # Save population state
        pop_dir = path.parent / f"{path.stem}_population"
        pop_dir.mkdir(parents=True, exist_ok=True)

        for agent in ["p1", "p2"]:
            pop_data = {
                "generation": self.generation[agent],
                "units": [
                    {
                        "thinking_style": u.thinking_style,
                        "mutation_prompt": u.mutation_prompt,
                        "task_prompt": u.task_prompt,
                        "history": u.history,
                    }
                    for u in self.populations[agent]
                ],
                "elites": [
                    {"task_prompt": e.task_prompt}
                    for e in self.elites[agent]
                ],
            }
            with open(pop_dir / f"{agent}_population.json", "w") as f:
                json.dump(pop_data, f, indent=2)

        logger.info(f"Saved NeuralEvolution model to {nn_path} and population to {pop_dir}")

    def reset(self) -> None:
        """Reset the bandit to initial state."""
        # Reset parent
        super().reset() if hasattr(super(), 'reset') else None

        # Reset evolution state
        self.generation = {"p1": 0, "p2": 0}
        self.elites = {"p1": [], "p2": []}
        self._last_selected_unit = {"p1": 0, "p2": 0}

        # Reinitialize populations
        self._initialize_populations()
        self._sync_population_to_prompt_space("p1")
        self._sync_population_to_prompt_space("p2")

        logger.info("NeuralAdversarialEvolutionBandit reset to initial state")

    @property
    def bandit_type(self) -> str:
        """Return the type of bandit algorithm."""
        return "neural_evolution"
