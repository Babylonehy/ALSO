"""
Bandit simulation library - modular components.

Public API:
- Execution: run_single_scenario, run_batch_episodes, run_resume_episodes
- Configuration: parse_args, DEFAULT_* constants
- Utilities: display_summary, calculate_cost_by_model_async
"""

# Execution modes
from .execution_modes import (
    run_single_scenario,
    run_batch_episodes,
    run_resume_episodes,
)

# Configuration
from .config import (
    parse_args,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_API_BASE,
    BanditType,
    OptimizeMode,
)

# Display utilities
from .display_utils import (
    display_summary,
    format_duration,
    get_output_path,
    print_cost_breakdown,
)

# Cost tracking
from .cost_tracking import calculate_cost_by_model_async

# Core components (for advanced usage)
from .simulation_runner import BanditSimulationRunner
from .embedding_client import EmbeddingClient, extract_agent_history
from .database_utils import find_combo_by_pk, load_profiles
from .serialization import to_jsonable, save_run_config
from .progress_tracking import BatchProgressTracker

__all__ = [
    # Execution
    "run_single_scenario",
    "run_batch_episodes",
    "run_resume_episodes",
    # Config
    "parse_args",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_API_BASE",
    # Display
    "display_summary",
    "format_duration",
    "get_output_path",
    "print_cost_breakdown",
    # Cost
    "calculate_cost_by_model_async",
    # Core
    "BanditSimulationRunner",
    "EmbeddingClient",
    "extract_agent_history",
    "find_combo_by_pk",
    "load_profiles",
    "to_jsonable",
    "save_run_config",
    "BatchProgressTracker",
]
