"""
Paraphrase Backgrounds Module

This module provides functionality to paraphrase agent background information
using large language models via litellm with OpenRouter.
"""

from .core import (
    DEFAULT_MODEL,
    DEFAULT_NUM_PARAPHRASES,
    DEFAULT_TEMPERATURE,
    append_llm_call_log,
    check_existing_paraphrases,
    collect_all_scenario_files,
    collect_scenarios,
    copy_hard_subset,
    extract_backgrounds,
    load_existing_paraphrases,
    load_scenario_file,
    paraphrase_backgrounds,
    paraphrase_single_text,
    process_scenario_pair,
    process_single_scenario_file,
    save_paraphrased_result,
    setup_litellm,
)
from .prompts import (
    PARAPHRASE_BACKGROUND_PROMPT,
    STYLE_VARIATION_PROMPTS,
    SYSTEM_PROMPT,
)

__all__ = [
    # Core functions
    "setup_litellm",
    "load_scenario_file",
    "save_paraphrased_result",
    "append_llm_call_log",
    "extract_backgrounds",
    "check_existing_paraphrases",
    "load_existing_paraphrases",
    "paraphrase_single_text",
    "paraphrase_backgrounds",
    "process_single_scenario_file",
    "process_scenario_pair",
    "collect_all_scenario_files",
    "collect_scenarios",
    "copy_hard_subset",
    # Constants
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_NUM_PARAPHRASES",
    # Prompts
    "PARAPHRASE_BACKGROUND_PROMPT",
    "SYSTEM_PROMPT",
    "STYLE_VARIATION_PROMPTS",
]

