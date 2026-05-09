"""
Mutable strategy space for online-learned social strategies, v2 naming path.

This extends StrategySpace with slot replacement support so a bandit can
overwrite individual strategies at runtime while keeping a fixed arm count.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .strategy_space import StrategySpace


@dataclass
class LearnableStrategySpaceV2(StrategySpace):
    """Strategy space that supports replacing non-baseline slots online."""

    def __post_init__(self) -> None:
        super().__post_init__()

    def _get_prompt_list(self, agent: Literal["p1", "p2"]) -> list[str]:
        return self.p1_prompts if agent == "p1" else self.p2_prompts

    def _get_metadata_list(self, agent: Literal["p1", "p2"]) -> list[dict[str, Any]]:
        return self.p1_slot_metadata if agent == "p1" else self.p2_slot_metadata

    def _compose_prompt(self, agent: Literal["p1", "p2"], strategy_text: str) -> str:
        original_bio = (
            self.original_p1_background if agent == "p1" else self.original_p2_background
        )
        cleaned = strategy_text.strip()
        if not cleaned:
            return original_bio
        return (
            f"{original_bio}\n{cleaned}\n"
            "If you think this strategy is not helpful, please ignore it."
        )

    def get_num_arms(self, agent: Literal["p1", "p2"]) -> int:
        return len(self._get_prompt_list(agent))

    def get_prompt(self, agent: Literal["p1", "p2"], arm_index: int) -> str:
        prompts = self._get_prompt_list(agent)
        if arm_index < 0 or arm_index >= len(prompts):
            raise IndexError(f"Strategy index {arm_index} out of range")
        return prompts[arm_index]

    def get_slot_metadata(
        self, agent: Literal["p1", "p2"], arm_index: int
    ) -> dict[str, Any]:
        metadata = self._get_metadata_list(agent)
        if arm_index < 0 or arm_index >= len(metadata):
            raise IndexError(f"Strategy index {arm_index} out of range")
        return dict(metadata[arm_index])

    def replace_strategy(
        self,
        agent: Literal["p1", "p2"],
        arm_index: int,
        strategy_text: str,
        *,
        source: str,
        created_turn: int,
        parent_slot: int | None = None,
        strategy_name: str | None = None,
        strategy_id: str | None = None,
        embedding: np.ndarray | None = None,
        force_recompute_embedding: bool = False,
    ) -> tuple[str, np.ndarray]:
        return super().replace_strategy(
            agent,
            arm_index,
            strategy_text,
            source=source,
            created_turn=created_turn,
            parent_slot=parent_slot,
            strategy_name=strategy_name,
            strategy_id=strategy_id,
            embedding=embedding,
            force_recompute_embedding=force_recompute_embedding,
        )
