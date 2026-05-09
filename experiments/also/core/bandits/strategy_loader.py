"""
Dynamic strategy loader for generated strategy spaces.

This module loads strategy spaces of different sizes (s5, s10, s20, s50, s100)
from generated JSON files.
"""

import json
from pathlib import Path
from typing import Literal

from loguru import logger

# Base directory for generated strategies
GENERATED_STRATEGIES_DIR = Path(__file__).parent.parent.parent / "generated_strategies"

# Cache for loaded strategies
_strategy_cache: dict[str, list[dict]] = {}

# Extended strategy version type
ExtendedStrategyVersion = Literal["v1", "v2", "v3", "v4", "s5", "s6", "s8", "s10", "s20", "s24", "s48", "s50", "s100"]


def load_generated_strategies(version: str) -> list[dict[str, str]]:
    """Load generated strategies from JSON file.
    
    Args:
        version: Strategy version (s5, s10, s20, s50, s100)
        
    Returns:
        List of strategy dictionaries
    """
    if version in _strategy_cache:
        return _strategy_cache[version]
    
    json_file = GENERATED_STRATEGIES_DIR / f"social_strategies_{version}.json"
    
    if not json_file.exists():
        raise FileNotFoundError(
            f"Strategy file not found: {json_file}\n"
            f"Please run: python scripts/generate_strategy_variants.py --target-sizes {version[1:]}"
        )
    
    with open(json_file, "r") as f:
        strategies = json.load(f)
    
    _strategy_cache[version] = strategies
    logger.info(f"Loaded {len(strategies)} strategies from {json_file}")
    
    return strategies


def is_generated_version(version: str) -> bool:
    """Check if version is a generated strategy space."""
    return version.startswith("s") and version[1:].isdigit()


def get_generated_strategy_sizes() -> list[int]:
    """Get available generated strategy space sizes."""
    sizes = []
    if GENERATED_STRATEGIES_DIR.exists():
        for f in GENERATED_STRATEGIES_DIR.glob("social_strategies_s*.json"):
            try:
                size = int(f.stem.split("_")[-1][1:])  # Extract number from s5, s10, etc.
                sizes.append(size)
            except ValueError:
                continue
    return sorted(sizes)

