"""
Agent-pair splits for within-scenario generalization testing.

This module defines train/test splits based on agent pairs within the same scenarios,
rather than splitting by scenarios. This allows testing whether strategies learned
with some agent pairs can transfer to different agent pairs in the same scenario context.

Key differences from scenario_splits.py:
- scenario_splits: Different scenarios for train vs test
- agent_pair_splits: Same scenarios, different agent pairs for train vs test
"""

from typing import Literal

from loguru import logger
from sotopia.database import EnvAgentComboStorage


# Use the same scenarios as the main experiments (from the cached embeddings)
# These are different from HARD_SCENARIO_IDS - they come from actual experiment runs
EXPERIMENT_SCENARIO_ENV_IDS = [
    # Split A scenarios (found in DB)
    "01H7VFHNFVGFY578101R2PCV3T",
    "01H7VFHNJHK2W1P8JSWKAMBG4Z",
    "01H7VFHPDE1AM74JSR8KBJJF3A",
    "01H7VFHQ1Q67B1ADNBD9WBAG3X",
    "01H7VFHNBYXD48NDRY02VCWXFN",
    "01H7VFHNKVTCAGBA299VQG1QS2",
    "01H7VFHNGJEVGSVPPT0784H6P8",
    # Additional scenarios used in training (from SplitA train or SplitB)
    "01H7VFHPBTC4ES406NQ4ET12EQ",
    "01H7VFHNH8A88C4XJ7X4PVAHV4",
    "01H7VFHP1JEP91TTK5PEK39D2S",
    "01H7VFHPB2RC4RHAJ80ESYF1HW",
    "01H7VFHN56ZT2Z4C0EFX79Q31F",
    "01H7VFHNSV5BKMP61H535PPTSG",
    "01H7VFHPHWA2CYG7BC82NS4XH1",
]


def get_combos_for_env(env_id: str) -> list[EnvAgentComboStorage]:
    """Get all agent-pair combos for a given environment ID.
    
    Args:
        env_id: Environment (scenario) ID
        
    Returns:
        List of EnvAgentComboStorage, sorted by pk for determinism
    """
    combos = list(EnvAgentComboStorage.find(
        EnvAgentComboStorage.env_id == env_id
    ).all())
    # Sort by pk for deterministic ordering
    return sorted(combos, key=lambda c: c.pk)


def split_combos_by_agent_pair(
    env_ids: list[str],
    train_ratio: int = 3,
    test_ratio: int = 2,
) -> dict[str, list[str]]:
    """Split combos within each scenario by agent pairs.
    
    For each env_id:
    1. Get all combos (typically 5 per scenario)
    2. Sort by pk for determinism
    3. Assign first `train_ratio` combos to train, rest to test
    
    Args:
        env_ids: List of environment IDs to process
        train_ratio: Number of combos per scenario for training
        test_ratio: Number of combos per scenario for testing
    
    Returns:
        Dict with "train" and "test" keys containing lists of combo PKs
    """
    train_combo_ids = []
    test_combo_ids = []
    
    for env_id in env_ids:
        combos = get_combos_for_env(env_id)
        
        if len(combos) == 0:
            logger.warning(f"No combos found for env_id: {env_id}")
            continue
            
        # Calculate split point
        total_expected = train_ratio + test_ratio
        if len(combos) < total_expected:
            logger.warning(
                f"Env {env_id} has only {len(combos)} combos, "
                f"expected at least {total_expected}. Using all for train."
            )
            train_combo_ids.extend([c.pk for c in combos])
            continue
        
        # Split: first train_ratio for training, next test_ratio for testing
        train_combos = combos[:train_ratio]
        test_combos = combos[train_ratio:train_ratio + test_ratio]
        
        train_combo_ids.extend([c.pk for c in train_combos])
        test_combo_ids.extend([c.pk for c in test_combos])
        
        logger.debug(
            f"Env {env_id}: {len(train_combos)} train, {len(test_combos)} test combos"
        )
    
    logger.info(
        f"Agent-pair split: {len(train_combo_ids)} train, "
        f"{len(test_combo_ids)} test combos from {len(env_ids)} scenarios"
    )
    
    return {
        "train": train_combo_ids,
        "test": test_combo_ids,
    }


def get_agent_pair_split(
    split_name: Literal["A", "B"] = "A",
    train_ratio: int = 3,
    test_ratio: int = 2,
) -> dict[str, list[str]]:
    """Get train/test combo IDs split by agent pairs within scenarios.
    
    Args:
        split_name: Split name ("A" or "B")
            - A: Use first 7 scenarios
            - B: Use last 7 scenarios
        train_ratio: Number of combos per scenario for training (default: 3)
        test_ratio: Number of combos per scenario for testing (default: 2)
    
    Returns:
        Dict with "train" and "test" keys containing lists of combo PKs
    """
    if split_name == "A":
        env_ids = EXPERIMENT_SCENARIO_ENV_IDS[:7]
    elif split_name == "B":
        env_ids = EXPERIMENT_SCENARIO_ENV_IDS[7:14]
    else:
        raise ValueError(f"Unknown split name: {split_name}")
    
    return split_combos_by_agent_pair(
        env_ids=env_ids,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
    )


def get_train_combo_ids(
    split_name: Literal["A", "B"] = "A",
    train_ratio: int = 3,
    test_ratio: int = 2,
) -> list[str]:
    """Get training combo IDs for the specified split."""
    return get_agent_pair_split(split_name, train_ratio, test_ratio)["train"]


def get_test_combo_ids(
    split_name: Literal["A", "B"] = "A",
    train_ratio: int = 3,
    test_ratio: int = 2,
) -> list[str]:
    """Get test combo IDs for the specified split."""
    return get_agent_pair_split(split_name, train_ratio, test_ratio)["test"]
