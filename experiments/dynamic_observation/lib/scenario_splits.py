"""
Scenario splits for cross-scenario generalization testing.

This module defines train/test splits for evaluating whether strategies
learned in one set of scenarios can transfer to unseen scenarios.
"""

from typing import Literal

# SOTOPIA-HARD has 14 scenarios, split into 7 train + 7 test
# Fixed split for reproducibility
HARD_SCENARIO_IDS = [
    "01H7VFHNV13MHN97GAH73E3KM8",
    "01H7VFHN5WVC5HKKVBHZBA553R",
    "01H7VFHN9W0WAFZCBT09PKJJNK",
    "01H7VFHPDZVVCDZR3AARA547CY",
    "01H7VFHPQQQY6H4DNC6NBQ8XTG",
    "01H7VFHN7WJK7VWVRZZTQ6DX9T",
    "01H7VFHPS5WJW2694R1MNC8JFY",
    "01H7VFHNN7XTR99319DS8KZCQM",
    "01H7VFHQ11NAMZS4A2RDGDB01V",
    "01H7VFHPSWGDGEYRP63H2DJKV0",
    "01H7VFHNF4G18PC9JHGRC8A1R6",
    "01H7VFHNNYH3W0VRWVY178K2TK",
    "01H7VFHP8AN5643B0NR0NP00VE",
    "01H7VFHN7A1ZX5KSMT2YN9RXC4",
]

# Split A: First 7 for training, last 7 for testing
SPLIT_A = {
    "train": HARD_SCENARIO_IDS[:7],
    "test": HARD_SCENARIO_IDS[7:],
}

# Split B: Last 7 for training, first 7 for testing (for 2-fold cross-validation)
SPLIT_B = {
    "train": HARD_SCENARIO_IDS[7:],
    "test": HARD_SCENARIO_IDS[:7],
}

# Default split
DEFAULT_SPLIT = SPLIT_A


def get_scenario_split(
    split_name: Literal["A", "B"] = "A"
) -> dict[str, list[str]]:
    """
    Get train/test scenario split.
    
    Args:
        split_name: "A" for first 7 train / last 7 test,
                   "B" for last 7 train / first 7 test
    
    Returns:
        Dict with "train" and "test" keys containing scenario IDs
    """
    if split_name == "A":
        return SPLIT_A
    elif split_name == "B":
        return SPLIT_B
    else:
        raise ValueError(f"Unknown split name: {split_name}")


def get_train_scenarios(split_name: Literal["A", "B"] = "A") -> list[str]:
    """Get training scenario IDs for the specified split."""
    return get_scenario_split(split_name)["train"]


def get_test_scenarios(split_name: Literal["A", "B"] = "A") -> list[str]:
    """Get test scenario IDs for the specified split."""
    return get_scenario_split(split_name)["test"]

