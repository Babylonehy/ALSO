"""
OPRO-style Bandit for dynamic prompt optimization in Sotopia.

This module implements the OPRO (LLM as Optimizer) algorithm as a bandit,
using meta-prompts with instruction-score history to guide LLM generation
of new, potentially better agent bio descriptions.

Reference: "Large Language Models as Optimizers" (Yang et al., 2023)
"""

import asyncio
import hashlib
import json
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import requests
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console

from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .llm_utils import acompletion_with_retry
from .opro_prompts import gen_sotopia_meta_prompt, parse_bio_from_response
from .prompt_space import PromptSpace

load_dotenv()
console = Console()


@dataclass
class OPROConfig(BanditConfig):
    """Configuration for OPRO-style bandit optimization."""

    # ===== Population-based evolution parameters =====
    # Population size (N): number of instructions to maintain
    population_size: int = 5

    # Number of new instructions to generate per evolution step (default = population_size)
    # If None, will use population_size
    num_generated_per_step: int | None = None

    # Maximum number of instructions to keep in the meta prompt (for LLM context)
    max_num_instructions: int = 10

    # Score threshold: only include instructions with score >= threshold in meta prompt
    score_threshold: float = 0.5

    # Evolution trigger: "round_complete" (after all N selected once) or "interval" (every N turns)
    evolution_trigger: str = "round_complete"

    # Evolution interval: trigger evolution every N turns (only used if evolution_trigger="interval")
    evolution_interval: int = 5

    # LLM optimizer temperature (higher = more creative, lower = more conservative)
    optimizer_temperature: float = 1.0

    # Whether to include scenario context in meta prompt
    use_context_in_prompt: bool = False

    # Optimizer LLM model
    optimizer_model: str = "openrouter/qwen/qwen-2.5-7b-instruct"

    # Embedding model for new prompts
    embedding_model: str = "qwen/qwen3-embedding-8b"

    # Selection strategy: "greedy", "epsilon_greedy", "softmax", or "round_robin"
    # Default is "round_robin" to ensure all instructions are evaluated before evolution
    selection_strategy: str = "round_robin"

    # Epsilon for epsilon-greedy exploration (only used if selection_strategy="epsilon_greedy")
    epsilon: float = 0.1

    # Temperature for softmax selection (only used if selection_strategy="softmax")
    selection_temperature: float = 1.0

    # Score update method: "ema", "replace", or "no"
    score_update_method: str = "replace"

    # EMA decay factor (only used if score_update_method="ema")
    ema_alpha: float = 0.3

    # Seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        """Set defaults based on population_size."""
        if self.num_generated_per_step is None:
            self.num_generated_per_step = self.population_size


@dataclass
class InstructionRecord:
    """Record for a single instruction in the pool."""
    instruction: str
    score: float
    step_added: int
    embedding: np.ndarray
    times_selected: int = 0
    last_selected_turn: int = -1
    needs_evaluation: bool = True  # True for new instructions that haven't been evaluated yet


class OPROBandit(BaseBandit):
    """
    OPRO-style optimization as a bandit algorithm.
    
    Key features:
    - Uses meta-prompts with instruction-score history to guide LLM
    - LLM optimizer generates new instructions based on history
    - Greedy or epsilon-greedy selection from instruction pool
    - EMA or replace score updates based on observed rewards
    """
    
    def __init__(
        self,
        prompt_space: PromptSpace,
        config: BanditConfig | OPROConfig | None = None,
        tensorboard_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """
        Initialize OPRO Bandit.
        
        Args:
            prompt_space: The prompt space containing initial embeddings and prompts
            config: Configuration for the bandit algorithm
            tensorboard_dir: Optional directory for TensorBoard logging
            output_dir: Optional directory for saving evolution logs
        """
        # Use OPROConfig as default
        if config is None:
            config = OPROConfig()
        elif not isinstance(config, OPROConfig):
            # Convert BanditConfig to OPROConfig
            config = OPROConfig(**{
                k: v for k, v in config.__dict__.items() 
                if k in OPROConfig.__dataclass_fields__
            })
        
        super().__init__(prompt_space, config, tensorboard_dir)
        self.opro_config: OPROConfig = config
        self.output_dir = output_dir
        
        # Initialize instruction pools for each agent
        self.instruction_pools: dict[Literal["p1", "p2"], list[InstructionRecord]] = {
            "p1": [],
            "p2": [],
        }
        
        # Track evolution steps
        self.evolution_step: dict[Literal["p1", "p2"], int] = {"p1": 0, "p2": 0}
        
        # Track last evolution turn
        self.last_evolution_turn: dict[Literal["p1", "p2"], int] = {"p1": -1, "p2": -1}

        # Evolution history for logging
        self.evolution_history: list[dict[str, Any]] = []

        # MD5 hash set to avoid duplicate instructions
        self.instruction_hashes: dict[Literal["p1", "p2"], set[str]] = {
            "p1": set(),
            "p2": set(),
        }

        # Track round-robin progress for evolution trigger
        # Stores indices that have been selected in current round
        self.round_selected: dict[Literal["p1", "p2"], set[int]] = {
            "p1": set(),
            "p2": set(),
        }

        # Random state
        self.rng = np.random.default_rng(config.seed)

        # Initialize pools from prompt_space
        self._initialize_pools()

        # Log initialization
        population_size = self.opro_config.population_size
        logger.info(
            f"OPROBandit initialized: population_size={population_size}, "
            f"P1 pool={len(self.instruction_pools['p1'])}, "
            f"P2 pool={len(self.instruction_pools['p2'])}, "
            f"evolution_trigger={self.opro_config.evolution_trigger}, "
            f"selection_strategy={self.opro_config.selection_strategy}"
        )

    def _initialize_pools(self) -> None:
        """Initialize instruction pools from the existing prompt space.

        Only initializes population_size instructions (takes first N if more available).
        """
        population_size = self.opro_config.population_size

        for agent_str in ["p1", "p2"]:
            agent: Literal["p1", "p2"] = "p1" if agent_str == "p1" else "p2"
            n_arms = self.prompt_space.get_num_arms(agent)

            # Take first population_size indices (consistent with EvoPrompt/PromptBreeder)
            n_to_use = min(n_arms, population_size)

            for i in range(n_to_use):
                prompt = self.prompt_space.get_prompt(agent, i)
                embedding = self.prompt_space.get_embedding(agent, i)
                # Score initial prompts at 0.5 (neutral, not yet evaluated)
                # Initial instructions need evaluation in the first round
                record = InstructionRecord(
                    instruction=prompt,
                    score=0.5,
                    step_added=-1,  # -1 means initial instruction
                    embedding=embedding,
                    needs_evaluation=True,  # Will be evaluated in first round
                )
                self.instruction_pools[agent].append(record)

                # Add hash to avoid duplicates later
                hash_str = self._get_instruction_hash(prompt)
                self.instruction_hashes[agent].add(hash_str)

            logger.debug(f"Initialized {len(self.instruction_pools[agent])} instructions for {agent}")
    
    def _get_instruction_hash(self, instruction: str) -> str:
        """Get MD5 hash of instruction for deduplication."""
        return hashlib.md5(instruction.encode()).hexdigest()
    
    def select(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        """
        Select an arm for the given agent.

        Default behavior (round_robin with priority on unevaluated):
        - Prioritize instructions that need evaluation (needs_evaluation=True)
        - When all unevaluated instructions have been selected, trigger evolution
        - After evolution, new instructions are marked as needs_evaluation=True

        Args:
            agent: Which agent to select for ("p1" or "p2")
            turn: Current turn number

        Returns:
            Tuple of (arm_index, prompt_text, embedding)
        """
        pool = self.instruction_pools[agent]

        if not pool:
            # Fallback to original prompt space
            idx = 0
            prompt = self.prompt_space.get_prompt(agent, idx)
            embedding = self.prompt_space.p1_embeddings[idx] if agent == "p1" else self.prompt_space.p2_embeddings[idx]
            return idx, prompt, embedding

        # Find indices of instructions that need evaluation
        unevaluated_indices = [i for i, r in enumerate(pool) if r.needs_evaluation]

        # Select using configured strategy
        strategy = self.opro_config.selection_strategy
        scores = np.array([r.score for r in pool])

        if strategy == "greedy":
            # Pure greedy: always pick the best
            selected_idx = int(np.argmax(scores))
            logger.debug(f"[Turn {turn}] Using greedy selection: best_idx={selected_idx}")
        elif strategy == "epsilon_greedy":
            # Epsilon-greedy: greedy with probability (1-epsilon), random otherwise
            if self.rng.random() < self.opro_config.epsilon:
                selected_idx = int(self.rng.integers(0, len(pool)))
                logger.debug(f"[Turn {turn}] Using epsilon-greedy: random exploration")
            else:
                selected_idx = int(np.argmax(scores))
                logger.debug(f"[Turn {turn}] Using epsilon-greedy: greedy selection")
        elif strategy == "softmax":
            # Softmax: probability proportional to exp(score / temperature)
            temp = self.opro_config.selection_temperature
            exp_scores = np.exp(scores / temp)
            probs = exp_scores / np.sum(exp_scores)
            selected_idx = int(self.rng.choice(len(pool), p=probs))
            logger.debug(f"[Turn {turn}] Using softmax selection: idx={selected_idx}, prob={probs[selected_idx]:.4f}")
        else:
            # Default: round-robin with priority on unevaluated instructions
            if unevaluated_indices:
                # Pick next unevaluated instruction in order
                # Find the first unevaluated that hasn't been selected this round
                unselected_unevaluated = [i for i in unevaluated_indices if i not in self.round_selected[agent]]
                if unselected_unevaluated:
                    selected_idx = unselected_unevaluated[0]
                else:
                    # All unevaluated have been selected, pick first unevaluated
                    selected_idx = unevaluated_indices[0]
                logger.debug(
                    f"[Turn {turn}] Round-robin (unevaluated priority): idx={selected_idx}, "
                    f"unevaluated={len(unevaluated_indices)}/{len(pool)}"
                )
            else:
                # All evaluated, use regular round-robin (shouldn't happen if evolution works)
                last_idx = self.current_selections.get(agent, -1)
                selected_idx = (last_idx + 1) % len(pool)
                logger.debug(f"[Turn {turn}] Round-robin (all evaluated): {last_idx} -> {selected_idx}")

        record = pool[selected_idx]
        record.times_selected += 1
        record.last_selected_turn = turn

        # Track selection for round-complete evolution trigger
        self.round_selected[agent].add(selected_idx)

        # Update current selection
        self.current_selections[agent] = selected_idx

        # Count remaining unevaluated for logging
        remaining_unevaluated = sum(1 for r in pool if r.needs_evaluation)

        # Log selection
        logger.debug(
            f"OPRO select {agent}: idx={selected_idx}, "
            f"score={record.score:.3f}, pool_size={len(pool)}, "
            f"needs_eval={record.needs_evaluation}, remaining_unevaluated={remaining_unevaluated}"
        )

        return selected_idx, record.instruction, record.embedding

    def _should_evolve(self, agent: Literal["p1", "p2"], turn: int) -> bool:
        """Check if evolution should be triggered.

        Two modes:
        - "round_complete": evolve when all unevaluated instructions have been evaluated
        - "interval": evolve every N turns (legacy behavior)
        """
        trigger_mode = self.opro_config.evolution_trigger
        pool = self.instruction_pools[agent]

        if trigger_mode == "round_complete":
            # Check if all unevaluated instructions have been evaluated
            unevaluated_count = sum(1 for r in pool if r.needs_evaluation)

            if unevaluated_count == 0 and len(pool) > 0:
                logger.info(
                    f"[{agent}] All {len(pool)} instructions evaluated (no unevaluated left), "
                    f"triggering evolution at turn {turn}"
                )
                return True
            return False
        else:
            # Legacy interval-based evolution
            if turn < self.opro_config.evolution_interval:
                return False
            turns_since_last = turn - self.last_evolution_turn[agent]
            return turns_since_last >= self.opro_config.evolution_interval
    
    def update(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        reward: float,
        turn: int,
    ) -> None:
        """
        Update the bandit with the observed reward.

        Args:
            agent: Which agent was updated ("p1" or "p2")
            arm_index: The arm index that was selected
            reward: The observed reward (should be in [0, 1])
            turn: The turn number when this selection was made
        """
        pool = self.instruction_pools[agent]

        if arm_index >= len(pool):
            logger.warning(f"Invalid arm_index {arm_index} for {agent} pool size {len(pool)}")
            return

        record = pool[arm_index]
        old_score = record.score
        was_unevaluated = record.needs_evaluation

        # Update score based on method
        if self.opro_config.score_update_method == "ema":
            alpha = self.opro_config.ema_alpha
            record.score = alpha * reward + (1 - alpha) * old_score
        else:
            # Replace
            record.score = reward

        # Mark as evaluated (no longer needs evaluation)
        record.needs_evaluation = False

        # Record in selection history
        selection_record = SelectionRecord(
            turn=turn,
            agent=agent,
            arm_index=arm_index,
            prompt_text=record.instruction,
            embedding=record.embedding,
            reward=reward,
            cumulative_score=record.score,
        )
        self.selection_history.append(selection_record)

        logger.debug(
            f"OPRO update {agent}: idx={arm_index}, "
            f"reward={reward:.3f}, score: {old_score:.3f} -> {record.score:.3f}, "
            f"was_unevaluated={was_unevaluated}"
        )
    
    def _evolve(self, agent: Literal["p1", "p2"], turn: int) -> None:
        """
        Evolve the instruction pool using LLM optimizer (sync wrapper).
        """
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._evolve_async(agent, turn))
                future.result()
        except RuntimeError:
            asyncio.run(self._evolve_async(agent, turn))

    async def _evolve_async(self, agent: Literal["p1", "p2"], turn: int) -> None:
        """
        Evolve the instruction pool using LLM optimizer.

        New logic:
        1. Sort current pool by score, keep top-N (population_size)
        2. Mark top-N as already evaluated (needs_evaluation=False)
        3. Build meta prompt from top-N instruction-score history
        4. Call LLM to generate N new instructions
        5. New instructions marked as needs_evaluation=True
        6. Final pool = top-N (evaluated) + N new (unevaluated)
        7. Next round only evaluates the N new ones
        """
        population_size = self.opro_config.population_size
        logger.info(
            f"OPRO evolution triggered for {agent} at turn {turn}, "
            f"population_size={population_size}"
        )

        pool = self.instruction_pools[agent]

        # Step 1: Sort by score and keep top-N
        pool.sort(key=lambda x: x.score, reverse=True)
        top_n = pool[:population_size]
        removed_old = pool[population_size:]

        # Clear hashes for removed instructions
        for rec in removed_old:
            hash_str = self._get_instruction_hash(rec.instruction)
            self.instruction_hashes[agent].discard(hash_str)

        if removed_old:
            logger.info(
                f"[{agent}] Kept top {len(top_n)} instructions, "
                f"removed {len(removed_old)} low-scoring ones"
            )

        # Step 2: Mark top-N as already evaluated (they have real scores)
        for rec in top_n:
            rec.needs_evaluation = False

        # Log top-N scores
        top_scores = [f"{rec.score:.3f}" for rec in top_n]
        logger.info(f"[{agent}] Top-{population_size} scores: [{', '.join(top_scores)}]")

        # Step 3: Build meta prompt from top-N
        instructions_and_scores = [
            (rec.instruction, rec.score, rec.step_added)
            for rec in top_n
        ]

        meta_prompt = gen_sotopia_meta_prompt(
            instructions_and_scores,
            max_num_instructions=self.opro_config.max_num_instructions,
            score_threshold=self.opro_config.score_threshold,
            include_context=self.opro_config.use_context_in_prompt,
        )

        # Log the constructed meta prompt for debugging
        logger.debug(
            f"\n{'='*60}\n"
            f"[OPRO Evolution] Agent: {agent}, Turn: {turn}, Step: {self.evolution_step[agent]}\n"
            f"{'='*60}\n"
            f"Instructions in pool: {len(instructions_and_scores)}\n"
            f"Meta prompt:\n{'-'*40}\n{meta_prompt}\n{'-'*40}"
        )

        # Step 4: Call LLM optimizer to generate N new instructions
        num_to_generate = self.opro_config.num_generated_per_step or population_size
        new_instructions = await self._generate_new_instructions_async(meta_prompt, num_to_generate)

        # Log the generated new instructions
        logger.info(
            f"\n{'='*60}\n"
            f"[OPRO Generated] Agent: {agent}, Turn: {turn}\n"
            f"{'='*60}\n"
            f"Generated {len(new_instructions)} new instructions:\n"
            + "\n".join([
                f"{'-'*40}\n[Instruction {i+1}]\n{instr}\n"
                for i, instr in enumerate(new_instructions)
            ])
            + f"{'-'*40}"
        )

        # Step 5: Build new pool = top-N (evaluated) + new instructions (unevaluated)
        new_pool: list[InstructionRecord] = list(top_n)
        added_count = 0

        for instruction in new_instructions:
            hash_str = self._get_instruction_hash(instruction)

            # Skip duplicates
            if hash_str in self.instruction_hashes[agent]:
                logger.debug(f"Skipping duplicate instruction: {instruction[:50]}...")
                continue

            # Compute embedding
            embedding = self._compute_embedding(instruction)
            if embedding is None:
                logger.warning(f"Failed to compute embedding for: {instruction[:50]}...")
                continue

            # New instructions need evaluation, start with score 0.5 (neutral)
            record = InstructionRecord(
                instruction=instruction,
                score=0.5,  # Neutral score, will be evaluated in next round
                step_added=self.evolution_step[agent],
                embedding=embedding,
                needs_evaluation=True,  # Must be evaluated before next evolution
            )

            new_pool.append(record)
            self.instruction_hashes[agent].add(hash_str)
            added_count += 1

        # Replace pool
        pool.clear()
        pool.extend(new_pool)

        # Step 6: Reset round tracking for next evaluation cycle
        self.round_selected[agent].clear()

        # Count unevaluated for logging
        unevaluated_count = sum(1 for r in pool if r.needs_evaluation)

        logger.info(
            f"[{agent}] Evolution complete: kept {len(top_n)} top (evaluated), "
            f"added {added_count} new (unevaluated), total pool={len(pool)}, "
            f"need_to_evaluate={unevaluated_count}"
        )

        # Sync to prompt_space for compatibility
        self._sync_pool_to_prompt_space(agent)
        
        # Update evolution tracking
        self.evolution_step[agent] += 1
        self.last_evolution_turn[agent] = turn
        
        # Log evolution
        evolution_log = {
            "agent": agent,
            "turn": turn,
            "step": self.evolution_step[agent],
            "added_count": added_count,
            "pool_size": len(pool),
            "top_scores": [rec.score for rec in sorted(pool, key=lambda x: x.score, reverse=True)[:5]],
            "timestamp": datetime.now().isoformat(),
        }
        self.evolution_history.append(evolution_log)
        
        # Save to file if output_dir is set
        if self.output_dir:
            self._save_evolution_log(evolution_log)
        
        logger.info(
            f"OPRO evolution complete for {agent}: "
            f"added {added_count} new instructions, pool size={len(pool)}"
        )
    
    async def _generate_new_instructions_async(self, meta_prompt: str, num_to_generate: int) -> list[str]:
        """
        Call LLM optimizer to generate new instructions with retry logic (async).

        Args:
            meta_prompt: The meta prompt to send to the optimizer
            num_to_generate: Number of new instructions to generate

        Returns:
            List of new instruction strings
        """
        try:
            # Use litellm for LLM calls with retry logic
            response = await acompletion_with_retry(
                model=self.opro_config.optimizer_model,
                messages=[{"role": "user", "content": meta_prompt}],
                temperature=self.opro_config.optimizer_temperature,
                n=num_to_generate,
                max_tokens=4096,
                caller_name="OPRO optimizer",
            )

            new_instructions = []
            for choice in response.choices:
                content = choice.message.content
                if content:
                    parsed = parse_bio_from_response(content)
                    if parsed:
                        new_instructions.append(parsed)

            return new_instructions

        except Exception as e:
            logger.error(f"Failed to generate new instructions via LLM: {e}")
            traceback.print_exc()
            return []
    
    def _compute_embedding(self, text: str) -> np.ndarray | None:
        """
        Compute dummy embedding for new prompt (all zeros).
        
        OPRO relies on text-based meta-prompts and does not strictly require 
        embeddings for its core logic. We return a zero vector to satisfy 
        the BaseBandit interface without incurring API costs.
        """
        try:
            # Create dummy zero embedding matching expected dimension
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Failed to compute embedding for text: {text[:50]}...")
            traceback.print_exc()
            return None
    
    def _sync_pool_to_prompt_space(self, agent: Literal["p1", "p2"]) -> None:
        """Sync instruction pool back to prompt_space for compatibility.
        
        Note: PromptSpace uses original_p{1,2}_background + paraphrased_p{1,2}_backgrounds.
        We sync the evolved pool back by updating these attributes and embeddings.
        """
        pool = self.instruction_pools[agent]
        if not pool:
            return
        
        prompts = [rec.instruction for rec in pool]
        embeddings = np.array([rec.embedding for rec in pool])
        
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
    
    def _save_evolution_log(self, log: dict[str, Any]) -> None:
        """Save evolution log to file."""
        if not self.output_dir:
            return
        
        log_dir = Path(self.output_dir) / "evolution_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"opro_evolution_{log['agent']}_step{log['step']:03d}.json"
        filepath = log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(log, f, indent=2, default=str)
    
    def train_model(self, verbose: bool = True) -> float:
        """
        OPRO does not use neural network training.
        
        Returns:
            Always returns 0.0
        """
        return 0.0
    
    def get_current_prompt(self, agent: Literal["p1", "p2"]) -> str:
        """Get the currently selected prompt for an agent."""
        idx = self.current_selections.get(agent, 0)
        pool = self.instruction_pools[agent]
        
        if idx < len(pool):
            return pool[idx].instruction
        
        # Fallback
        return self.prompt_space.get_prompt(agent, 0)
    
    @property
    def bandit_type(self) -> str:
        """Return the type of bandit algorithm."""
        return "opro"
    
    def get_pool_summary(self, agent: Literal["p1", "p2"]) -> dict[str, Any]:
        """Get summary of instruction pool for an agent."""
        pool = self.instruction_pools[agent]
        
        if not pool:
            return {"pool_size": 0}
        
        scores = [rec.score for rec in pool]
        return {
            "pool_size": len(pool),
            "avg_score": np.mean(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores),
            "evolution_steps": self.evolution_step[agent],
            "top_3_instructions": [
                {"instruction": rec.instruction[:100], "score": rec.score}
                for rec in sorted(pool, key=lambda x: x.score, reverse=True)[:3]
            ],
        }
