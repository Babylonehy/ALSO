"""
Core functions for paraphrasing agent backgrounds.

This module contains:
- File I/O operations
- LLM paraphrasing logic
- Utility functions for incremental storage
"""

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
from loguru import logger

from .prompts import PARAPHRASE_BACKGROUND_PROMPT, SYSTEM_PROMPT


# ========== Configuration ==========
DEFAULT_MODEL = "openrouter/openai/gpt-4o"
DEFAULT_TEMPERATURE = 1.5
DEFAULT_NUM_PARAPHRASES = 10
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
# ===================================


def setup_litellm(api_key: str | None = None) -> None:
    """Setup litellm with OpenRouter API key."""
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            f"OPENROUTER_API_KEY not found. "
            f"Please set it in .env file or pass it as argument. "
            f"[{__file__}:{setup_litellm.__code__.co_firstlineno}]"
        )
    litellm.api_key = key
    logger.info(f"LiteLLM configured with OpenRouter API key | {__file__}:{setup_litellm.__code__.co_firstlineno + 5}")


def load_scenario_file(file_path: Path) -> dict[str, Any]:
    """Load a scenario JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {file_path} | {__file__}:{load_scenario_file.__code__.co_firstlineno}"
        )
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_paraphrased_result(result: dict[str, Any], output_path: Path) -> None:
    """Save paraphrased result to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.debug(f"Saved result to {output_path} | {__file__}:{save_paraphrased_result.__code__.co_firstlineno + 3}")


def append_llm_call_log(
    llm_call_infos: list[dict[str, Any]],
    log_file: Path,
    scenario_id: str,
    source_file: str,
) -> None:
    """
    Append LLM call info to a JSONL log file.
    Each line is a JSON object with call details for cost tracking.
    """
    if not llm_call_infos:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        for call_info in llm_call_infos:
            log_entry = {
                "scenario_id": scenario_id,
                "source_file": source_file,
                "timestamp": datetime.now().isoformat(),
                **call_info,
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    logger.debug(
        f"Appended {len(llm_call_infos)} LLM calls to {log_file} | "
        f"{__file__}:{append_llm_call_log.__code__.co_firstlineno}"
    )


def extract_backgrounds(scenario_data: dict[str, Any]) -> dict[str, str]:
    """Extract background fields from scenario data."""
    backgrounds = {}
    for key, value in scenario_data.items():
        if key.endswith("_background") and isinstance(value, str) and value.strip():
            backgrounds[key] = value
    return backgrounds


def check_existing_paraphrases(output_path: Path, num_required: int) -> int:
    """
    Check how many valid paraphrases already exist.
    Returns the number of existing valid paraphrases for each background field.
    """
    if not output_path.exists():
        return 0

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check the minimum number of valid paraphrases across all fields
    min_count = num_required
    paraphrased_data = data.get("paraphrased", {})

    for field_name, versions in paraphrased_data.items():
        valid_count = sum(1 for v in versions if v and v.strip())
        min_count = min(min_count, valid_count)

    return min_count


def load_existing_paraphrases(output_path: Path) -> dict[str, Any] | None:
    """Load existing paraphrased data if available."""
    if not output_path.exists():
        return None

    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def paraphrase_single_text(
    original_text: str,
    semaphore: asyncio.Semaphore,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[str, dict[str, Any]]:
    """
    Use LLM to paraphrase a single text with retry mechanism.

    Returns:
        Tuple of (paraphrased_text, llm_call_info).
        llm_call_info contains: id, model, usage (prompt_tokens, completion_tokens, total_tokens)
    """
    async with semaphore:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": PARAPHRASE_BACKGROUND_PROMPT.format(original_text=original_text)},
                    ],
                    temperature=temperature,
                )
                result = response.choices[0].message.content
                if result is None:
                    raise ValueError(
                        f"LLM returned None for paraphrase. "
                        f"Original text preview: {original_text[:500]}... | "
                        f"{__file__}:{paraphrase_single_text.__code__.co_firstlineno}"
                    )

                # Extract LLM call info for cost tracking
                llm_call_info = {
                    "id": getattr(response, "id", None),
                    "model": getattr(response, "model", model),
                    "usage": {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
                        "total_tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0,
                    },
                }

                return result.strip(), llm_call_info
            except Exception as e:
                last_error = e
                error_str = str(e)
                should_retry = any(str(code) in error_str for code in RETRY_STATUS_CODES)
                if not should_retry:
                    raise

                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {error_str[:100]}... "
                    f"Retrying in {delay:.1f}s | {__file__}:{paraphrase_single_text.__code__.co_firstlineno + 20}"
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Failed after {MAX_RETRIES} retries. Last error: {last_error} | "
            f"{__file__}:{paraphrase_single_text.__code__.co_firstlineno}"
        )



async def paraphrase_backgrounds(
    backgrounds: dict[str, str],
    num_paraphrases: int,
    semaphore: asyncio.Semaphore,
    existing_data: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """
    Paraphrase all background fields with incremental storage support.
    Only generates missing paraphrases.

    Returns:
        Tuple of (paraphrased_results, llm_call_infos).
        llm_call_infos is a list of all LLM call info dicts for cost tracking.
    """
    result = {}
    all_llm_call_infos: list[dict[str, Any]] = []
    existing_paraphrased = existing_data.get("paraphrased", {}) if existing_data else {}

    for field_name, original_text in backgrounds.items():
        existing_versions = existing_paraphrased.get(field_name, [])
        valid_existing = [v for v in existing_versions if v and v.strip()]

        num_needed = num_paraphrases - len(valid_existing)

        if num_needed <= 0:
            result[field_name] = valid_existing[:num_paraphrases]
            logger.debug(f"Skipping {field_name}: already has {len(valid_existing)} versions")
            continue

        logger.debug(f"Generating {num_needed} new paraphrases for {field_name}")

        # Generate missing paraphrases concurrently
        tasks = [
            paraphrase_single_text(original_text, semaphore, model, temperature)
            for _ in range(num_needed)
        ]
        responses = await asyncio.gather(*tasks)

        # Separate text and call info
        new_versions = [text for text, _ in responses]
        new_call_infos = [info for _, info in responses]

        # Add field name to each call info for traceability
        for info in new_call_infos:
            info["field_name"] = field_name

        all_llm_call_infos.extend(new_call_infos)

        # Combine existing and new versions
        result[field_name] = valid_existing + new_versions

    return result, all_llm_call_infos


async def process_single_scenario_file(
    input_path: Path,
    output_path: Path,
    num_paraphrases: int,
    semaphore: asyncio.Semaphore,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[dict[str, Any], bool]:
    """
    Process a single scenario file: extract backgrounds, paraphrase, and save.
    Supports incremental storage.

    Returns:
        Tuple of (result_data, was_processed).
        was_processed is True if new paraphrases were generated, False if skipped.
    """
    # Load original scenario
    scenario_data = load_scenario_file(input_path)

    # Extract backgrounds
    backgrounds = extract_backgrounds(scenario_data)

    if not backgrounds:
        logger.warning(f"No background fields found in {input_path}")
        return {"original": scenario_data, "paraphrased": {}, "metadata": {}}, False

    # Check existing data for incremental storage
    existing_data = load_existing_paraphrases(output_path)

    # Check if we need to generate more
    existing_count = check_existing_paraphrases(output_path, num_paraphrases)
    if existing_count >= num_paraphrases:
        logger.debug(f"Skipping {input_path.name}: already has {existing_count} paraphrases")
        return existing_data, False  # Skipped

    # Paraphrase backgrounds
    paraphrased, llm_call_infos = await paraphrase_backgrounds(
        backgrounds,
        num_paraphrases,
        semaphore,
        existing_data,
        model,
        temperature,
    )

    # Get existing LLM call infos if any
    existing_llm_calls = existing_data.get("metadata", {}).get("llm_calls", []) if existing_data else []

    # Calculate total usage from new calls
    total_prompt_tokens = sum(info["usage"]["prompt_tokens"] for info in llm_call_infos)
    total_completion_tokens = sum(info["usage"]["completion_tokens"] for info in llm_call_infos)
    total_tokens = sum(info["usage"]["total_tokens"] for info in llm_call_infos)

    # Build result
    result = {
        "original": scenario_data,
        "paraphrased": paraphrased,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": model,
            "temperature": temperature,
            "num_paraphrases": num_paraphrases,
            "source_file": str(input_path),
            "llm_calls": existing_llm_calls + llm_call_infos,
            "usage_summary": {
                "new_calls": len(llm_call_infos),
                "total_calls": len(existing_llm_calls) + len(llm_call_infos),
                "new_prompt_tokens": total_prompt_tokens,
                "new_completion_tokens": total_completion_tokens,
                "new_total_tokens": total_tokens,
            },
        },
    }

    # Save result
    save_paraphrased_result(result, output_path)

    return result, True  # Processed


async def process_scenario_pair(
    scenario_dir: Path,
    input_files: list[Path],
    output_dir: Path,
    num_paraphrases: int,
    semaphore: asyncio.Semaphore,
    llm_call_log_file: Path,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[int, int]:
    """
    Process a scenario with multiple files (0_xxx.json, 1_xxx.json).
    Since backgrounds are the same for both files, only call LLM once and copy to both.
    LLM call info is saved to a separate log file.

    Returns:
        Tuple of (processed_count, skipped_count).
    """
    if not input_files:
        return 0, 0

    # Use the first file as the primary source for paraphrasing
    primary_file = input_files[0]
    primary_output = output_dir / primary_file.name

    # Check if we already have enough paraphrases in ANY of the output files
    existing_count = check_existing_paraphrases(primary_output, num_paraphrases)
    if existing_count >= num_paraphrases:
        # Already have enough, skip all files
        logger.debug(f"Skipping scenario {scenario_dir.name}: already has {existing_count} paraphrases")
        return 0, len(input_files)

    # Load and process the primary file
    scenario_data = load_scenario_file(primary_file)
    backgrounds = extract_backgrounds(scenario_data)

    if not backgrounds:
        logger.warning(f"No background fields found in {primary_file}")
        return 0, len(input_files)

    # Check existing data for incremental storage
    existing_data = load_existing_paraphrases(primary_output)

    # Paraphrase backgrounds (only once for the scenario)
    paraphrased, llm_call_infos = await paraphrase_backgrounds(
        backgrounds,
        num_paraphrases,
        semaphore,
        existing_data,
        model,
        temperature,
    )

    # Save LLM call info to separate log file
    if llm_call_infos:
        append_llm_call_log(
            llm_call_infos,
            llm_call_log_file,
            scenario_id=scenario_dir.name,
            source_file=str(primary_file),
        )

    # Now save results for ALL files in the scenario (without LLM call details)
    for input_file in input_files:
        file_scenario_data = load_scenario_file(input_file)
        output_path = output_dir / input_file.name

        result = {
            "original": file_scenario_data,
            "paraphrased": paraphrased,  # Same paraphrases for all files
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": model,
                "temperature": temperature,
                "num_paraphrases": num_paraphrases,
                "source_file": str(input_file),
                "primary_source": str(primary_file),
            },
        }

        save_paraphrased_result(result, output_path)

    return len(input_files), 0  # All files processed


def collect_scenarios(input_dir: Path) -> list[tuple[Path, list[Path]]]:
    """
    Collect all scenarios from input directory.
    Returns list of (scenario_dir, [file_paths]) tuples.
    Each scenario may have multiple files (0_xxx.json, 1_xxx.json).
    """
    scenarios = []
    for scenario_dir in sorted(input_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        files = sorted(scenario_dir.glob("*.json"))
        if files:
            scenarios.append((scenario_dir, files))
    return scenarios


def collect_all_scenario_files(input_dir: Path) -> list[tuple[Path, Path, Path]]:
    """
    Collect all scenario JSON files from input directory.
    Returns list of (scenario_dir, file_path, relative_path) tuples.

    NOTE: This is kept for backward compatibility but prefer using collect_scenarios().
    """
    files = []
    for scenario_dir in sorted(input_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for json_file in sorted(scenario_dir.glob("*.json")):
            relative_path = json_file.relative_to(input_dir)
            files.append((scenario_dir, json_file, relative_path))
    return files


def copy_hard_subset(all_output_dir: Path, hard_input_dir: Path, hard_output_dir: Path) -> int:
    """
    Copy relevant files from 'all' output to 'hard' output based on what exists in hard_input_dir.
    Returns number of files copied.
    """
    copied_count = 0

    for scenario_dir in sorted(hard_input_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue

        scenario_name = scenario_dir.name
        source_scenario_dir = all_output_dir / scenario_name
        target_scenario_dir = hard_output_dir / scenario_name

        if not source_scenario_dir.exists():
            logger.warning(f"Source scenario not found: {source_scenario_dir}")
            continue

        # Copy all JSON files from this scenario
        target_scenario_dir.mkdir(parents=True, exist_ok=True)
        for json_file in source_scenario_dir.glob("*.json"):
            target_file = target_scenario_dir / json_file.name
            shutil.copy2(json_file, target_file)
            copied_count += 1

    logger.info(f"Copied {copied_count} files to hard directory | {__file__}")
    return copied_count

