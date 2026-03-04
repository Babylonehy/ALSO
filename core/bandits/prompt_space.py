"""
Prompt space management for loading pre-generated embeddings and paraphrased backgrounds.

This module supports two modes:
1. Paraphrase mode (default): Loads scenario-specific bio paraphrases from files
2. Strategy mode: Uses shared social strategies appended to original bios
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger


@dataclass
class PromptSpace:
    """
    Manages the space of available prompts (paraphrased backgrounds) and their embeddings.
    
    Loads pre-generated data from the embeddings_backgrounds directory.
    """
    
    scenario_id: str
    base_dir: Path
    
    # Original and paraphrased texts
    original_p1_background: str = ""
    original_p2_background: str = ""
    paraphrased_p1_backgrounds: list[str] = field(default_factory=list)
    paraphrased_p2_backgrounds: list[str] = field(default_factory=list)
    
    # Embeddings as numpy arrays
    p1_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    p2_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Agent names
    p1_name: str = ""
    p2_name: str = ""
    
    def __post_init__(self) -> None:
        """Load data from files after initialization."""
        self._load_data()
    
    def _load_data(self) -> None:
        """Load texts and embeddings from the pre-generated files."""
        scenario_dir = self.base_dir / self.scenario_id
        
        if not scenario_dir.exists():
            raise FileNotFoundError(
                f"Scenario directory not found: {scenario_dir}. "
                f"Expected directory at: {scenario_dir.absolute()}"
            )
        
        # Load texts
        texts_file = scenario_dir / "texts.json"
        if not texts_file.exists():
            raise FileNotFoundError(f"texts.json not found in {scenario_dir}")
        
        with open(texts_file, "r", encoding="utf-8") as f:
            texts = json.load(f)
        
        self.original_p1_background = texts["original"]["p1_background"]
        self.original_p2_background = texts["original"]["p2_background"]
        self.paraphrased_p1_backgrounds = texts["paraphrased"]["p1_background"]
        self.paraphrased_p2_backgrounds = texts["paraphrased"]["p2_background"]
        
        # Load statistics for agent names
        stats_file = scenario_dir / "statistics.json"
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
            self.p1_name = stats.get("p1_name", "")
            self.p2_name = stats.get("p2_name", "")
        
        # Load embeddings
        p1_emb_file = scenario_dir / "p1_embeddings.npy"
        p2_emb_file = scenario_dir / "p2_embeddings.npy"
        
        if p1_emb_file.exists():
            self.p1_embeddings = np.load(str(p1_emb_file))
            logger.debug(
                f"Loaded p1 embeddings shape: {self.p1_embeddings.shape}"
            )
        else:
            raise FileNotFoundError(f"p1_embeddings.npy not found in {scenario_dir}")
        
        if p2_emb_file.exists():
            self.p2_embeddings = np.load(str(p2_emb_file))
            logger.debug(
                f"Loaded p2 embeddings shape: {self.p2_embeddings.shape}"
            )
        else:
            raise FileNotFoundError(f"p2_embeddings.npy not found in {scenario_dir}")
        
        logger.info(
            f"Loaded prompt space for {self.scenario_id}: "
            f"{len(self.paraphrased_p1_backgrounds)} p1 paraphrases, "
            f"{len(self.paraphrased_p2_backgrounds)} p2 paraphrases"
        )
    
    def get_num_arms(self, agent: Literal["p1", "p2"]) -> int:
        """Get the number of available arms (prompts) for the specified agent."""
        if agent == "p1":
            # +1 for the original
            return len(self.paraphrased_p1_backgrounds) + 1
        else:
            return len(self.paraphrased_p2_backgrounds) + 1
    
    def get_prompt(self, agent: Literal["p1", "p2"], arm_index: int) -> str:
        """Get the prompt text for a given arm index."""
        if agent == "p1":
            if arm_index == 0:
                return self.original_p1_background
            return self.paraphrased_p1_backgrounds[arm_index - 1]
        else:
            if arm_index == 0:
                return self.original_p2_background
            return self.paraphrased_p2_backgrounds[arm_index - 1]
    
    def get_embedding(self, agent: Literal["p1", "p2"], arm_index: int) -> np.ndarray:
        """Get the embedding vector for a given arm index."""
        if agent == "p1":
            return self.p1_embeddings[arm_index]
        else:
            return self.p2_embeddings[arm_index]
    
    def get_all_embeddings(self, agent: Literal["p1", "p2"]) -> np.ndarray:
        """Get all embeddings for the specified agent."""
        if agent == "p1":
            return self.p1_embeddings
        else:
            return self.p2_embeddings

    @property
    def p1_prompts(self) -> list[str]:
        """Get all p1 prompts (original + paraphrased) for bandit sync compatibility."""
        return [self.original_p1_background] + self.paraphrased_p1_backgrounds

    @p1_prompts.setter
    def p1_prompts(self, value: list[str]) -> None:
        """Set p1 prompts from a list (for bandit sync)."""
        if value:
            self.original_p1_background = value[0]
            self.paraphrased_p1_backgrounds = value[1:] if len(value) > 1 else []

    @property
    def p2_prompts(self) -> list[str]:
        """Get all p2 prompts (original + paraphrased) for bandit sync compatibility."""
        return [self.original_p2_background] + self.paraphrased_p2_backgrounds

    @p2_prompts.setter
    def p2_prompts(self, value: list[str]) -> None:
        """Set p2 prompts from a list (for bandit sync)."""
        if value:
            self.original_p2_background = value[0]
            self.paraphrased_p2_backgrounds = value[1:] if len(value) > 1 else []

