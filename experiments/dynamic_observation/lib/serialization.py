"""Data serialization utilities for converting objects to JSON-safe formats."""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# Import all bandit config classes for deserialization
from experiments.dynamic_observation.core.bandits.base_bandit import BanditConfig
from experiments.dynamic_observation.core.bandits.prompt_breeder_bandit import PromptBreederConfig
from experiments.dynamic_observation.core.bandits.neural_evolution_bandit import NeuralEvolutionConfig
from experiments.dynamic_observation.core.bandits.opro_bandit import OPROConfig
from experiments.dynamic_observation.core.bandits.evoprompt_bandit import EvoPromptConfig


def get_git_info() -> dict[str, Any]:
    """
    Get current git repository information.

    Returns:
        Dictionary containing:
            - commit_id: Current commit hash (or None if not in git repo)
            - branch: Current branch name (or None)
            - dirty: Whether there are uncommitted changes (bool)
            - error: Error message if git command failed (or None)
    """
    git_info: dict[str, Any] = {
        "commit_id": None,
        "branch": None,
        "dirty": False,
        "error": None,
    }

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["commit_id"] = result.stdout.strip()

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["branch"] = result.stdout.strip()

        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["dirty"] = bool(result.stdout.strip())

    except subprocess.CalledProcessError as e:
        git_info["error"] = f"Git command failed: {e}"
        logger.warning(f"Failed to get git info: {e}")
    except subprocess.TimeoutExpired:
        git_info["error"] = "Git command timed out"
        logger.warning("Git command timed out")
    except FileNotFoundError:
        git_info["error"] = "Git not found in PATH"
        logger.warning("Git executable not found")
    except Exception as e:
        git_info["error"] = f"Unexpected error: {e}"
        logger.warning(f"Unexpected error getting git info: {e}")

    return git_info


def to_jsonable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable types.

    Handles:
    - Path -> str
    - np.ndarray -> list
    - dict, list, tuple, set recursively

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of obj
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(to_jsonable(v) for v in obj)
    return obj


# Config type registry for deserialization
CONFIG_TYPES = {
    "BanditConfig": BanditConfig,
    "PromptBreederConfig": PromptBreederConfig,
    "NeuralEvolutionConfig": NeuralEvolutionConfig,
    "OPROConfig": OPROConfig,
    "EvoPromptConfig": EvoPromptConfig,
}


def load_bandit_config(config_dict: dict) -> BanditConfig:
    """
    Reconstruct bandit config from saved JSON with type information.

    Args:
        config_dict: Dictionary loaded from run_config.json["bandit_config"]

    Returns:
        Appropriate config instance (BanditConfig, PromptBreederConfig, etc.)

    Raises:
        ValueError: If config type is unknown or invalid
    """
    if not config_dict:
        logger.warning("Empty config dict provided, returning default BanditConfig")
        return BanditConfig()

    # Extract config type metadata (if present)
    config_type_name = config_dict.get("__config_type__", "BanditConfig")

    # Create a clean copy without metadata fields
    clean_dict = {k: v for k, v in config_dict.items() if not k.startswith("__")}

    # Backward compatibility: if no type info, default to BanditConfig
    if "__config_type__" not in config_dict:
        logger.warning(
            "No config type found in saved config, defaulting to BanditConfig. "
            "This may be an old experiment save file."
        )
        config_type_name = "BanditConfig"

    # Look up config class
    config_class = CONFIG_TYPES.get(config_type_name)
    if config_class is None:
        raise ValueError(
            f"Unknown config type: {config_type_name}. "
            f"Valid types: {list(CONFIG_TYPES.keys())}"
        )

    # Instantiate config with saved parameters
    try:
        return config_class(**clean_dict)
    except TypeError as e:
        logger.error(f"Failed to instantiate {config_type_name}: {e}")
        logger.error(f"Config dict keys: {list(clean_dict.keys())}")
        raise ValueError(
            f"Failed to create {config_type_name} from saved config. "
            f"This may indicate a config schema change. Error: {e}"
        ) from e


def save_run_config(
    experiment_dir: Path,
    experiment_tag: str,
    args: argparse.Namespace,
    bandit_config: Any,  # BanditConfig dataclass
    bandit_type: str,
    filename: str = "run_config.json",
) -> Path:
    """
    Save a JSON snapshot of runtime args and bandit config into experiment_dir.

    Args:
        experiment_dir: Directory to save config in
        experiment_tag: Tag identifying this experiment run
        args: Parsed command-line arguments
        bandit_config: Bandit configuration dataclass
        bandit_type: Type of bandit algorithm (e.g., "prompt_breeder", "neural_ucb")
        filename: Name of config file to create

    Returns:
        Path to created config file
    """
    payload = {
        "experiment_tag": experiment_tag,
        "bandit_type": bandit_type,
        "saved_at": datetime.now().isoformat(),
        "argv": list(sys.argv),
        "args": to_jsonable(vars(args)),
        "bandit_config": {
            "__config_type__": type(bandit_config).__name__,
            **to_jsonable(asdict(bandit_config)),
        },
        "git_info": get_git_info(),
    }
    path = experiment_dir / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
