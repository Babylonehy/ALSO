"""Database operations and profile loading utilities."""

import re
from pathlib import Path
from typing import Literal

from loguru import logger
from sotopia.database import AgentProfile, EnvironmentProfile, EnvAgentComboStorage


# Assuming PROJECT_ROOT is accessible from parent directory
def _get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


def find_combo_by_pk(combo_pk: str) -> EnvAgentComboStorage | None:
    """
    Find an EnvAgentComboStorage by its primary key.

    Args:
        combo_pk: Primary key of the combo

    Returns:
        EnvAgentComboStorage instance or None if not found
    """
    try:
        combo = EnvAgentComboStorage.get(pk=combo_pk)
        return combo
    except Exception as e:
        logger.warning(f"Combo not found by pk: {e}")
        return None


def load_profiles(combo: EnvAgentComboStorage) -> tuple[EnvironmentProfile, list[AgentProfile]]:
    """
    Load environment and agent profiles from combo.

    Args:
        combo: Environment-agent combination storage

    Returns:
        Tuple of (environment_profile, list of agent_profiles)
    """
    env_profile = EnvironmentProfile.get(pk=combo.env_id)
    agent_profiles = [AgentProfile.get(pk=agent_id) for agent_id in combo.agent_ids]
    return env_profile, agent_profiles


def get_available_scenarios(embeddings_dir: Path) -> list[str]:
    """
    Get list of available scenario IDs from the embeddings directory.

    Args:
        embeddings_dir: Directory containing scenario subdirectories

    Returns:
        Sorted list of scenario IDs

    Raises:
        FileNotFoundError: If embeddings directory doesn't exist
    """
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    scenarios = []
    for subdir in embeddings_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            scenarios.append(subdir.name)

    return sorted(scenarios)


def parse_env_ids(file_path: Path) -> dict[str, list[str]]:
    """
    Parse env_ids.txt file, return hard and all dataset env_id lists.

    Args:
        file_path: Path to env_ids.txt

    Returns:
        Dict with keys "hard" and "all" containing lists of env IDs
    """
    with open(file_path, "r") as f:
        content = f.read()

    sections = content.split("SOTOPIA-ALL:")
    hard_section = sections[0]
    all_section = sections[1] if len(sections) > 1 else ""

    hard_ids = re.findall(r'"([A-Z0-9]+)"', hard_section)
    all_ids = re.findall(r'"([A-Z0-9]+)"', all_section)

    return {
        "hard": list(set(hard_ids)),
        "all": list(set(all_ids))
    }


def get_combos_for_subset(subset: Literal["hard", "all", "hard_small"]) -> list[EnvAgentComboStorage]:
    """
    Get all combos for the specified dataset subset.

    Args:
        subset: Dataset subset name ("hard", "all", or "hard_small")
            - "hard": All 70 combos from 14 hard environments (5 combos per env)
            - "all": All combos from 90 environments
            - "hard_small": Only 14 combos (first combo per hard environment, deterministic)

    Returns:
        List of EnvAgentComboStorage instances

    Raises:
        FileNotFoundError: If env_ids.txt not found
    """
    PROJECT_ROOT = _get_project_root()
    env_ids_path = PROJECT_ROOT / "data/env_ids.txt"
    if not env_ids_path.exists():
        raise FileNotFoundError(f"env_ids.txt not found at {env_ids_path}")

    ids_by_category = parse_env_ids(env_ids_path)

    # For hard_small, use hard env_ids but only take first combo per env
    is_hard_small = subset == "hard_small"
    base_subset = "hard" if is_hard_small else subset
    env_ids = ids_by_category.get(base_subset, [])

    if not env_ids:
        logger.warning(f"No environment IDs found for subset '{subset}'")
        return []

    # Sort env_ids for deterministic ordering
    env_ids = sorted(env_ids)

    # Query all combos for these env_ids
    all_combos: list[EnvAgentComboStorage] = []
    for env_id in env_ids:
        combos = EnvAgentComboStorage.find(
            EnvAgentComboStorage.env_id == env_id
        ).all()

        if is_hard_small:
            # For hard_small: sort by pk and take only the first combo
            sorted_combos = sorted(combos, key=lambda c: c.pk)  # type: ignore[arg-type]
            if sorted_combos:
                all_combos.append(sorted_combos[0])  # type: ignore[arg-type]
        else:
            all_combos.extend(combos)  # type: ignore[arg-type]

    logger.info(f"Found {len(all_combos)} combos for subset '{subset}' ({len(env_ids)} env_ids)")
    return all_combos
