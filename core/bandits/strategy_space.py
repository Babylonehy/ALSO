"""
Strategy Space for Social Strategy Selection.

This module provides a StrategySpace class that manages social strategies
and their embeddings, designed to work with the existing bandit infrastructure.

Unlike PromptSpace which loads scenario-specific bio paraphrases,
StrategySpace uses a shared set of social strategies that can be appended
to any agent's original bio.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import requests
from dotenv import load_dotenv
from loguru import logger

from .social_strategies import (
    StrategyVersion,
    create_enhanced_bio,
    get_num_strategies,
    get_strategies,
)

load_dotenv()


@dataclass
class StrategySpace:
    """
    Manages social strategies and their embeddings for bandit selection.
    
    Key differences from PromptSpace:
    - Uses a shared set of social strategies (not scenario-specific)
    - Appends strategies to original bios instead of replacing them
    - Returns enhanced_bio = original_bio + strategy when selecting
    
    The interface is designed to be compatible with existing bandit algorithms.
    """
    
    # Original agent bios (from scenario)
    original_p1_background: str
    original_p2_background: str

    # Agent names
    p1_name: str = ""
    p2_name: str = ""

    # Scenario ID (for compatibility with PromptSpace - set to "strategy_mode")
    scenario_id: str = "strategy_mode"

    # Embedding configuration
    embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_dim: int = 4096

    # Strategy embeddings (shared across agents)
    # Shape: (num_strategies, embedding_dim)
    strategy_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))

    # Cache directory for embeddings
    cache_dir: Path | None = None

    # Strategy version: "v1" = SOCIAL_STRATEGIES, "v2" = SOCIAL_STRATEGIES_V2
    strategy_version: StrategyVersion = "v1"

    # Skip embedding computation (for OPRO/PromptBreeder which don't need embeddings)
    skip_embeddings: bool = False

    # Direct storage for prompts (for compatibility with bandit sync)
    p1_prompts: list[str] = field(default_factory=list)
    p2_prompts: list[str] = field(default_factory=list)
    p1_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    p2_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))

    # Paraphrased backgrounds (for compatibility with PromptSpace - maps to strategies)
    paraphrased_p1_backgrounds: list[str] = field(default_factory=list)
    paraphrased_p2_backgrounds: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize strategy embeddings after dataclass initialization."""
        if self.skip_embeddings:
            # Use zero embeddings for bandits that don't need real embeddings
            # (OPRO, PromptBreeder use fitness-based selection, not embedding-based)
            self._initialize_dummy_embeddings()
            logger.info("Skipping embedding computation (skip_embeddings=True)")
        else:
            self._initialize_embeddings()
        self._initialize_prompts()

    def _initialize_dummy_embeddings(self) -> None:
        """Initialize zero embeddings for bandits that don't need real embeddings."""
        n_strategies = get_num_strategies(self.strategy_version)
        self.p1_embeddings = np.zeros((n_strategies, self.embedding_dim), dtype=np.float32)
        self.p2_embeddings = np.zeros((n_strategies, self.embedding_dim), dtype=np.float32)
        self.strategy_embeddings = self.p1_embeddings.copy()
        logger.info(f"Initialized dummy zero embeddings: shape=({n_strategies}, {self.embedding_dim}), version={self.strategy_version}")
    
    def _initialize_prompts(self) -> None:
        """Initialize the prompt lists for each agent (enhanced bios)."""
        n_strategies = get_num_strategies(self.strategy_version)
        self.p1_prompts = [
            create_enhanced_bio(self.original_p1_background, i, self.strategy_version)
            for i in range(n_strategies)
        ]
        self.p2_prompts = [
            create_enhanced_bio(self.original_p2_background, i, self.strategy_version)
            for i in range(n_strategies)
        ]
        # Set paraphrased backgrounds for PromptSpace compatibility
        # Index 0 is "no strategy", indices 1+ are strategies
        self.paraphrased_p1_backgrounds = self.p1_prompts[1:]
        self.paraphrased_p2_backgrounds = self.p2_prompts[1:]
        logger.info(f"Initialized {n_strategies} enhanced bios for each agent (version={self.strategy_version})")
    
    def _initialize_embeddings(self) -> None:
        """Initialize or load enhanced bio embeddings for each agent."""
        # Try to load from cache first
        # Cache is based on hash of original backgrounds to handle different scenarios
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            # Create unique cache key based on backgrounds
            import hashlib
            bg_hash = hashlib.md5(
                (self.original_p1_background + self.original_p2_background).encode()
            ).hexdigest()[:12]
            p1_cache = cache_dir / f"strategy_p1_embeddings_{bg_hash}.npy"
            p2_cache = cache_dir / f"strategy_p2_embeddings_{bg_hash}.npy"

            if p1_cache.exists() and p2_cache.exists():
                self.p1_embeddings = np.load(str(p1_cache))
                self.p2_embeddings = np.load(str(p2_cache))
                # strategy_embeddings kept for backward compat (use p1's)
                self.strategy_embeddings = self.p1_embeddings.copy()
                logger.info(f"Loaded enhanced bio embeddings from cache: p1={self.p1_embeddings.shape}, p2={self.p2_embeddings.shape}")
                return

        # Compute embeddings for enhanced bios (original bio + strategy)
        logger.info("Computing enhanced bio embeddings for each agent...")
        self.p1_embeddings = self._compute_enhanced_bio_embeddings("p1")
        self.p2_embeddings = self._compute_enhanced_bio_embeddings("p2")
        # strategy_embeddings kept for backward compat (use p1's)
        self.strategy_embeddings = self.p1_embeddings.copy()

        # Save to cache
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            import hashlib
            bg_hash = hashlib.md5(
                (self.original_p1_background + self.original_p2_background).encode()
            ).hexdigest()[:12]
            np.save(str(cache_dir / f"strategy_p1_embeddings_{bg_hash}.npy"), self.p1_embeddings)
            np.save(str(cache_dir / f"strategy_p2_embeddings_{bg_hash}.npy"), self.p2_embeddings)
            logger.info(f"Saved enhanced bio embeddings to cache")
    
    def _compute_enhanced_bio_embeddings(self, agent: Literal["p1", "p2"]) -> np.ndarray:
        """Compute embeddings for all enhanced bios (original bio + strategy) for an agent."""
        original_bio = self.original_p1_background if agent == "p1" else self.original_p2_background
        embeddings = []

        n_strategies = get_num_strategies(self.strategy_version)
        for i in range(n_strategies):
            # Create enhanced bio and compute its embedding
            enhanced_bio = create_enhanced_bio(original_bio, i, self.strategy_version)
            embedding = self._compute_embedding(enhanced_bio)
            embeddings.append(embedding)

        logger.info(f"Computed {n_strategies} enhanced bio embeddings for {agent} (version={self.strategy_version})")
        return np.array(embeddings)
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text using OpenRouter API."""
        max_retries = 5
        base_delay = 2.0
        
        model = self.embedding_model
        if model.startswith("openrouter/"):
            model = model[len("openrouter/"):]
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "input": text}
        url = "https://openrouter.ai/api/v1/embeddings"
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
                    if len(embedding) != self.embedding_dim:
                        logger.warning(
                            f"Embedding dim mismatch: got {len(embedding)}, expected {self.embedding_dim}"
                        )
                    return embedding
                elif resp.status_code in (429, 500, 502, 503, 504):
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"API error {resp.status_code}, retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    raise ValueError(f"API error: {resp.status_code} - {resp.text}")
            except requests.exceptions.RequestException as e:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Request failed: {e}, retrying in {delay:.1f}s")
                time.sleep(delay)
        
        raise RuntimeError(f"Max retries ({max_retries}) exceeded for embedding request")

    def get_num_arms(self, agent: Literal["p1", "p2"]) -> int:
        """Get the number of available arms (strategies)."""
        return get_num_strategies(self.strategy_version)

    def get_prompt(self, agent: Literal["p1", "p2"], arm_index: int) -> str:
        """
        Get the enhanced bio for a given strategy index.

        Returns: original_bio + strategy_description
        """
        original_bio = self.original_p1_background if agent == "p1" else self.original_p2_background
        return create_enhanced_bio(original_bio, arm_index, self.strategy_version)

    def get_embedding(self, agent: Literal["p1", "p2"], arm_index: int) -> np.ndarray:
        """Get the embedding for a given agent's enhanced bio at arm_index."""
        embeddings = self.p1_embeddings if agent == "p1" else self.p2_embeddings
        if arm_index < 0 or arm_index >= len(embeddings):
            raise IndexError(f"Strategy index {arm_index} out of range")
        return embeddings[arm_index]

    def get_all_embeddings(self, agent: Literal["p1", "p2"]) -> np.ndarray:
        """Get all enhanced bio embeddings for the specified agent."""
        return self.p1_embeddings if agent == "p1" else self.p2_embeddings

    def get_strategy_name(self, arm_index: int) -> str:
        """Get the strategy name for a given index."""
        strategies = get_strategies(self.strategy_version)
        if 0 <= arm_index < len(strategies):
            return strategies[arm_index]["name"]
        return "Unknown"

    def get_strategy_id(self, arm_index: int) -> str:
        """Get the strategy ID for a given index."""
        strategies = get_strategies(self.strategy_version)
        if 0 <= arm_index < len(strategies):
            return strategies[arm_index]["id"]
        return "unknown"

    @classmethod
    def from_scenario_backgrounds(
        cls,
        p1_background: str,
        p2_background: str,
        p1_name: str = "",
        p2_name: str = "",
        embedding_model: str = "qwen/qwen3-embedding-8b",
        embedding_dim: int = 4096,
        cache_dir: Path | str | None = None,
        skip_embeddings: bool = False,
        strategy_version: StrategyVersion = "v1",
    ) -> "StrategySpace":
        """
        Create a StrategySpace from scenario background information.

        This is the primary factory method for creating StrategySpace instances.

        Args:
            p1_background: Agent 1's background/bio text
            p2_background: Agent 2's background/bio text
            p1_name: Agent 1's name
            p2_name: Agent 2's name
            embedding_model: Model for computing embeddings
            embedding_dim: Expected embedding dimension
            cache_dir: Directory for caching embeddings
            skip_embeddings: If True, use zero embeddings (for OPRO/PromptBreeder
                           which don't need real embeddings for selection)
            strategy_version: "v1" for SOCIAL_STRATEGIES, "v2" for SOCIAL_STRATEGIES_V2
        """
        return cls(
            original_p1_background=p1_background,
            original_p2_background=p2_background,
            p1_name=p1_name,
            p2_name=p2_name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            cache_dir=Path(cache_dir) if cache_dir else None,
            strategy_version=strategy_version,
            skip_embeddings=skip_embeddings,
        )

