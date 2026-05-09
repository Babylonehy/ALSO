"""
Neural adversarial bandit with online learnable strategy replacement, v2 naming path.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger

from .base_bandit import BanditConfig
from .learnable_strategy_space_v2 import LearnableStrategySpaceV2
from .llm_utils import acompletion_with_retry
from .neural_adversarial_bandit import NeuralAdversarialBandit


@dataclass
class LearnableAdversarialV2Config(BanditConfig):
    """Configuration for NeuralAdversarialLearnableV2Bandit."""

    proposal_interval: int = 5
    proposal_warmup_turns: int = 5
    proposal_context_turns: int = 3
    proposal_model: str = "openrouter/qwen/qwen-2.5-72b-instruct"
    proposal_temperature: float = 0.3
    proposal_max_tokens: int = 512
    proposal_top_k: int = 2
    protect_baseline: bool = True


class NeuralAdversarialLearnableV2Bandit(NeuralAdversarialBandit):
    """Neural adversarial bandit with proposal-replace strategy learning."""

    def __init__(
        self,
        prompt_space: LearnableStrategySpaceV2,
        config: BanditConfig | LearnableAdversarialV2Config | None = None,
        tensorboard_dir: Path | None = None,
    ) -> None:
        if config is None:
            self.learnable_config = LearnableAdversarialV2Config()
        elif isinstance(config, LearnableAdversarialV2Config):
            self.learnable_config = config
        else:
            self.learnable_config = LearnableAdversarialV2Config(
                **{
                    key: value
                    for key, value in config.__dict__.items()
                    if key in LearnableAdversarialV2Config.__dataclass_fields__
                }
            )

        if not isinstance(prompt_space, LearnableStrategySpaceV2):
            raise TypeError(
                "NeuralAdversarialLearnableV2Bandit requires LearnableStrategySpaceV2"
            )

        super().__init__(prompt_space, self.learnable_config, tensorboard_dir)
        self.prompt_space: LearnableStrategySpaceV2 = prompt_space
        self.current_dialogue_context = ""
        self.last_proposal_turn: dict[Literal["p1", "p2"], int] = {"p1": -1, "p2": -1}
        self.proposal_events: dict[Literal["p1", "p2"], list[dict[str, Any]]] = {
            "p1": [],
            "p2": [],
        }

    def set_generation_context(
        self,
        dialogue_text: str,
        context_embedding: np.ndarray | None = None,
    ) -> None:
        self.current_dialogue_context = dialogue_text.strip()
        if context_embedding is not None and self.use_context_embedding:
            self.set_context_embedding(context_embedding)

    async def select_async(
        self,
        agent: Literal["p1", "p2"],
        turn: int,
    ) -> tuple[int, str, np.ndarray]:
        await self._maybe_replace_strategy(agent, turn)
        return super().select(agent, turn)

    async def _maybe_replace_strategy(
        self, agent: Literal["p1", "p2"], turn: int
    ) -> None:
        if not self._should_trigger_proposal(agent, turn):
            return

        target_slot = self._get_replacement_slot(agent)
        if target_slot is None:
            return

        ranked_slots = self._rank_strategy_slots(agent)
        top_slots = [
            slot for slot in reversed(ranked_slots)
            if slot["slot_index"] != target_slot["slot_index"]
        ][: self.learnable_config.proposal_top_k]

        proposal_prompt = self._build_proposal_prompt(agent, target_slot, top_slots)
        response = await acompletion_with_retry(
            model=self.learnable_config.proposal_model,
            messages=[{"role": "user", "content": proposal_prompt}],
            temperature=self.learnable_config.proposal_temperature,
            max_tokens=self.learnable_config.proposal_max_tokens,
            caller_name="LearnableStrategyProposalV2",
        )
        proposed_strategy = response.choices[0].message.content.strip().strip("` \n")
        if not proposed_strategy:
            logger.warning(f"[Turn {turn}] Empty strategy proposal for {agent}, skipping")
            return

        arm_index = target_slot["slot_index"]
        old_metadata = self.prompt_space.get_slot_metadata(agent, arm_index)
        self.prompt_space.replace_strategy(
            agent,
            arm_index,
            proposed_strategy,
            source="proposal_replace_v2",
            created_turn=turn,
            parent_slot=top_slots[0]["slot_index"] if top_slots else None,
            force_recompute_embedding=True,
        )
        self._reset_slot_history(agent, arm_index)
        self.last_proposal_turn[agent] = turn
        self.proposal_events[agent].append(
            {
                "turn": turn,
                "agent": agent,
                "slot_index": arm_index,
                "replaced_strategy_id": old_metadata["strategy_id"],
                "replaced_strategy_text": old_metadata["strategy_text"],
                "new_strategy_text": proposed_strategy,
                "dialogue_excerpt": self.current_dialogue_context,
            }
        )
        logger.info(
            f"[Turn {turn}] Replaced {agent} slot {arm_index} with a learned v2 strategy"
        )

    def _should_trigger_proposal(
        self, agent: Literal["p1", "p2"], turn: int
    ) -> bool:
        if self._stopped:
            return False
        if turn < self.learnable_config.proposal_warmup_turns:
            return False
        if self.learnable_config.proposal_interval <= 0:
            return False
        if turn % self.learnable_config.proposal_interval != 0:
            return False
        if self.last_proposal_turn[agent] == turn:
            return False
        return len(self.selection_history) > 0 and self.prompt_space.get_num_arms(agent) > 1

    def _predict_current_scores(self, agent: Literal["p1", "p2"]) -> np.ndarray:
        embeddings = self.prompt_space.get_all_embeddings(agent)
        if self.use_context_embedding:
            if self.current_context_embedding is not None:
                context = np.tile(self.current_context_embedding, (embeddings.shape[0], 1))
            else:
                context = np.zeros((embeddings.shape[0], self.context_embedding_dim))
            features = np.concatenate([embeddings, context], axis=1)
        else:
            features = embeddings

        tensor = torch.tensor(features, dtype=torch.float64).to(self.config.device)
        self.model.double()
        self.model.eval()
        with torch.no_grad():
            raw_scores = self.model(tensor).cpu().numpy()
        if self.multi_dim_prediction:
            return np.mean(raw_scores, axis=1)
        return raw_scores.flatten()

    def _rank_strategy_slots(self, agent: Literal["p1", "p2"]) -> list[dict[str, Any]]:
        predicted_scores = self._predict_current_scores(agent)
        ranked_slots: list[dict[str, Any]] = []
        for arm_index in range(self.prompt_space.get_num_arms(agent)):
            created_turn = self.prompt_space.get_slot_metadata(agent, arm_index).get(
                "created_turn", 0
            )
            history = [
                rec.reward
                for rec in self.selection_history
                if rec.agent == agent
                and rec.arm_index == arm_index
                and rec.turn >= created_turn
            ]
            recent_rewards = history[-5:]
            observed_mean = float(np.mean(recent_rewards)) if recent_rewards else None
            quality_score = (
                observed_mean if observed_mean is not None else float(predicted_scores[arm_index])
            )
            metadata = self.prompt_space.get_slot_metadata(agent, arm_index)
            ranked_slots.append(
                {
                    "slot_index": arm_index,
                    "quality_score": quality_score,
                    "observed_mean_reward": observed_mean,
                    "selection_count": len(history),
                    "strategy_text": metadata["strategy_text"],
                }
            )
        ranked_slots.sort(key=lambda item: item["quality_score"])
        return ranked_slots

    def _get_replacement_slot(
        self, agent: Literal["p1", "p2"]
    ) -> dict[str, Any] | None:
        ranked_slots = self._rank_strategy_slots(agent)
        protected = {0} if self.learnable_config.protect_baseline else set()
        for slot in ranked_slots:
            if slot["slot_index"] not in protected:
                return slot
        return None

    def _build_proposal_prompt(
        self,
        agent: Literal["p1", "p2"],
        target_slot: dict[str, Any],
        top_slots: list[dict[str, Any]],
    ) -> str:
        original_bio = (
            self.prompt_space.original_p1_background
            if agent == "p1"
            else self.prompt_space.original_p2_background
        )
        top_examples = "\n".join(
            [
                (
                    f"- Slot {slot['slot_index']} | mean_reward={slot['observed_mean_reward']!s} | "
                    f"strategy: {slot['strategy_text'] or '(baseline/no addendum)'}"
                )
                for slot in top_slots
            ]
        ) or "- No successful examples yet."

        recent_dialogue = self.current_dialogue_context or "No recent dialogue available."
        current_strategy = target_slot["strategy_text"] or "(baseline/no addendum)"
        return (
            "You are improving a high-level social strategy addendum for a multi-agent social simulation.\n\n"
            "Write only the new strategy addendum, not the full biography.\n\n"
            f"Agent bio facts that must remain unchanged:\n{original_bio}\n\n"
            f"Recent dialogue excerpt:\n{recent_dialogue}\n\n"
            f"High-performing strategy examples:\n{top_examples}\n\n"
            "Underperforming strategy to replace:\n"
            f"- Slot {target_slot['slot_index']} | "
            f"mean_reward={target_slot['observed_mean_reward']!s} | "
            f"strategy: {current_strategy}\n\n"
            "Requirements:\n"
            "1. Output only a strategy addendum in 2-4 sentences.\n"
            "2. Do not restate or edit identity facts, secrets, age, job, or pronouns.\n"
            "3. Make the strategy responsive to the recent dialogue.\n"
            "4. Favor concrete social tactics such as probing, reciprocity, framing, de-escalation, "
            "commitment, or information control.\n"
            "5. Avoid meta-commentary, bullet points, or JSON.\n"
        )

    def _reset_slot_history(self, agent: Literal["p1", "p2"], arm_index: int) -> None:
        for trace in self.score_traces[agent]:
            if arm_index < len(trace):
                trace[arm_index] = 0.0
        for trace in self.actual_score_traces[agent]:
            if arm_index < len(trace):
                trace[arm_index] = 0.0
