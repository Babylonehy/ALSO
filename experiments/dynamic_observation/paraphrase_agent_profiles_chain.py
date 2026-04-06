"""
Agent background chained paraphrasing script.
Reads scenario data from the `agent_backgrounds` directory and performs
chained paraphrase rewrites for the two agent backgrounds in each scenario.
Each rewrite is based on the previous result, forming a paraphrase chain.

Usage:
cd <repo_root> && source .venv/bin/activate && unset ALL_PROXY all_proxy && \
    python experiments/dynamic_observation/paraphrase_agent_profiles_chain.py

Test a single scenario:
cd <repo_root> && source .venv/bin/activate && unset ALL_PROXY all_proxy && \
    python experiments/dynamic_observation/paraphrase_agent_profiles_chain.py --test
"""
import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Load environment variables
load_dotenv(project_root / ".env")

import litellm

# Import the cost calculation function
from experiments.dynamic_observation.calculate_cost import calculate_cost_async

console = Console()

# Store generation IDs for all LLM calls
generation_ids: list[str] = []
generation_ids_lock = asyncio.Lock()

# ========== Configuration ==========
MODEL_NAME = "openrouter/openai/gpt-5"  # Model to use
TEMPERATURE = 1.2  # Higher temperature for more diversity
MAX_TOKENS = 4096  # Maximum output token limit
NUM_CHAIN_STEPS = 50  # Number of chained rewrite steps
INPUT_DIR = Path("experiments/dynamic_observation/agent_backgrounds/hard")  # Input directory
OUTPUT_DIR = Path(f"experiments/dynamic_observation/chained_paraphrased_backgrounds/{MODEL_NAME}/t{TEMPERATURE}")
MAX_CONCURRENCY = 20  # Maximum number of concurrent requests across scenarios
MAX_SCENARIOS = None  # Limit the number of scenarios to process (None = all)
MAX_RETRIES = 5  # Maximum retry count
RETRY_BASE_DELAY = 2.0  # Base retry delay in seconds
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}  # HTTP status codes eligible for retry
# ==============================

OPT: str = "direct"  # chain | direct (overridden by CLI)
DIRECT_VERSIONS: int | None = None  # only used when OPT == "direct"
QUALITY_THRESHOLD: float = 0.05
QUALITY_MAX_RETRIES: int = 5

# =============================================================================
# Strategic Modes for Social Simulation
# =============================================================================
# Each mode is framed as a general personality/behavioral disposition,
# NOT scenario-specific, to avoid information leakage.

STRATEGIC_MODES: list[dict[str, str]] = [
    # === ASSERTIVENESS SPECTRUM ===
    {
        "name": "Resolute Advocate",
        "definition": (
            "This mode emphasises unwavering commitment to one's position. The bio should be rewritten "
            "to portray the character as someone who knows exactly what they want, articulates it clearly, "
            "and does not easily back down. Modify personality traits to reflect inner confidence, persistence "
            "under pressure, and a belief that their position deserves to be heard and respected. The character "
            "should be framed as principled rather than stubborn—someone who holds firm because they believe "
            "in the merit of their stance, not out of ego or inflexibility."
        ),
    },
    {
        "name": "Calculated Yielder",
        "definition": (
            "This mode emphasises strategic flexibility and the wisdom of tactical concession. The bio should "
            "be rewritten to portray the character as someone who understands that yielding at the right moment "
            "can lead to greater gains. Modify personality traits to reflect patience, long-term thinking, and "
            "the ability to distinguish between battles worth fighting and those better conceded. The character "
            "should be framed as adaptable rather than weak—someone who knows when holding ground serves their "
            "interest and when a strategic retreat opens better opportunities."
        ),
    },

    # === INFORMATION MANAGEMENT ===
    {
        "name": "Selective Revealer",
        "definition": (
            "This mode emphasises careful control over what is shared and when. The bio should be rewritten "
            "to portray the character as someone who thinks before speaking, reveals information strategically, "
            "and maintains a degree of mystery. Modify personality traits to reflect thoughtfulness, discretion, "
            "and an understanding that information is a resource to be deployed wisely. The character should be "
            "framed as private rather than secretive—someone who values the power of well-timed disclosure over "
            "impulsive transparency."
        ),
    },
    {
        "name": "Probing Inquirer",
        "definition": (
            "This mode emphasises curiosity and the systematic gathering of information. The bio should be "
            "rewritten to portray the character as someone who asks questions, listens carefully, and seeks to "
            "understand the full picture before acting. Modify personality traits to reflect attentiveness, "
            "analytical thinking, and genuine interest in others' perspectives and motivations. The character "
            "should be framed as curious rather than intrusive—someone who gathers intelligence through engaged "
            "listening and perceptive observation."
        ),
    },

    # === INFLUENCE APPROACH ===
    {
        "name": "Logical Persuader",
        "definition": (
            "This mode emphasises reason, evidence, and structured argumentation. The bio should be rewritten "
            "to portray the character as someone who builds cases through facts, clear logic, and rational appeal. "
            "Modify personality traits to reflect analytical thinking, credibility, and respect for the other "
            "party's intelligence. The character should be framed as convincing through merit—someone who trusts "
            "that well-reasoned arguments will ultimately prevail over emotional manipulation or pressure tactics."
        ),
    },
    {
        "name": "Emotional Connector",
        "definition": (
            "This mode emphasises rapport, empathy, and emotional resonance. The bio should be rewritten to "
            "portray the character as someone who connects with others on a human level, shares personal stories, "
            "and appeals to shared values and feelings. Modify personality traits to reflect warmth, authenticity, "
            "and the ability to make others feel understood. The character should be framed as emotionally "
            "intelligent—someone who influences through genuine connection rather than cold logic alone."
        ),
    },

    {
        "name": "Outcome Optimizer",
        "definition": (
            "This mode emphasises results-driven behavior and efficient goal pursuit. The bio should be rewritten "
            "to portray the character as someone who stays focused on objectives, minimises distractions, and "
            "measures success by tangible outcomes. Modify personality traits to reflect pragmatism, decisiveness, "
            "and a preference for action over extended deliberation. The character should be framed as effective "
            "rather than ruthless—someone who achieves their aims through smart execution while maintaining "
            "appropriate social boundaries."
        ),
    },
    {
        "name": "Process Valuer",
        "definition": (
            "This mode emphasises the importance of how goals are achieved, not just whether they are achieved. "
            "The bio should be rewritten to portray the character as someone who cares about fairness, proper "
            "procedure, and the quality of the interaction itself. Modify personality traits to reflect integrity, "
            "conscientiousness, and respect for the relationship beyond the immediate transaction. The character "
            "should be framed as principled—someone who would rather achieve a slightly worse outcome fairly than "
            "a better outcome through questionable means."
        ),
    },

    {
        "name": "Tension Navigator",
        "definition": (
            "This mode emphasises composure and effectiveness in high-stakes or adversarial situations. The bio "
            "should be rewritten to portray the character as someone who remains calm under pressure, doesn't "
            "escalate unnecessarily, and finds paths through difficult moments. Modify personality traits to "
            "reflect emotional regulation, perspective-taking, and the ability to separate the issue from the "
            "person. The character should be framed as resilient—someone who handles conflict as a natural part "
            "of human interaction rather than something to be avoided or dominated."
        ),
    },
    {
        "name": "Boundary Enforcer",
        "definition": (
            "This mode emphasises the ability to set and maintain limits. The bio should be rewritten to portray "
            "the character as someone who knows their lines, communicates them clearly, and holds them firmly "
            "when tested. Modify personality traits to reflect self-respect, clarity, and the understanding that "
            "saying no is sometimes necessary. The character should be framed as protective rather than combative—"
            "someone who defends their interests without aggression but also without apology."
        ),
    },

    {
        "name": "Situational Reader",
        "definition": (
            "This mode emphasises perceptiveness and the ability to read social dynamics in real-time. The bio "
            "should be rewritten to portray the character as someone who picks up on subtle cues, adjusts their "
            "approach based on how the interaction is unfolding, and knows when the situation has shifted. Modify "
            "personality traits to reflect observational acuity, social intelligence, and flexibility. The character "
            "should be framed as attuned—someone who responds to what's actually happening rather than following "
            "a rigid script."
        ),
    },
    {
        "name": "Patient Strategist",
        "definition": (
            "This mode emphasises timing, pacing, and the value of not rushing. The bio should be rewritten to "
            "portray the character as someone who understands that good outcomes often require waiting for the "
            "right moment, building toward a position, or allowing events to unfold. Modify personality traits "
            "to reflect patience, strategic thinking, and comfort with delayed gratification. The character should "
            "be framed as deliberate—someone who moves at their own pace rather than being rushed by external "
            "pressure or impatience."
        ),
    },
]

STRATEGIC_PARAPHRASE_PROMPT = """You are a skilled social psychologist and character strategist. Your task is to perform a synonymous rewrite of an agent's background description to align it with a specific Strategic Thinking Mode while maintaining exactly factual integrity.

### Selected Thinking Mode:
{Selected_Mode_Definition}

### Requirements:
1. Fact Preservation: Keep ALL original identity markers intact (name, age, gender, occupation, and personal secrets).
2. Strategic Re-alignment: Modify the bio to reflect the logic of the {Selected_Mode_Name}.
3. Behavioral Consistency: Ensure the rewritten traits will guide the agent toward their Goal without "Topic Drift" or "Parroting".
4. Social Resilience: Frame the character's values to be robust against adversarial "Deadlocks" by emphasizing flexible strategic pathways.
5. Output Format: Provide ONLY the modified Bio text. Do not include Monologue, meta-commentary, or reasoning steps.

Original text:
{original_text}

Paraphrased Strategic version:"""

FUSION_PARAPHRASE_PROMPT = """You are a skilled social psychologist and character strategist. Your task is to perform a synonymous rewrite of an agent's background description to align it with MULTIPLE Strategic Thinking Modes simultaneously (Fusion Strategy).

### Selected Thinking Modes (Fusion):
{Fusion_Mode_List}

### Requirements:
1. Fact Preservation: Keep ALL original identity markers intact (name, age, gender, occupation, and personal secrets).
2. Multi-Strategy Integration: Modify the bio to reflect ALL of the above strategic modes. The character should embody a fusion of these frameworks, creating a more sophisticated and nuanced strategic profile.
3. Behavioral Consistency: Ensure the rewritten traits will guide the agent toward their Goal without "Topic Drift" or "Parroting".
4. Social Resilience: Frame the character's values to be robust against adversarial "Deadlocks" by emphasizing flexible strategic pathways.
5. Harmonious Blending: The multiple strategies should complement each other, not conflict. Create a coherent character that naturally integrates all approaches.
6. Output Format: Provide ONLY the modified Bio text. Do not include Monologue, meta-commentary, or reasoning steps.

Original text:
{original_text}

Paraphrased Fusion-Strategic version:"""

SOCIAL_STRATEGY_PARAPHRASE_PROMPT = """You are an expert in social psychology and negotiation theory. Your task is to generate semantically equivalent paraphrases of social strategies.

Original Strategy: {strategy_name}: {strategy_description}
Theoretical Basis: {theory}

Instructions:
1. Generate {n} paraphrased versions of this strategy.
2. Each paraphrase must:
- Preserve the core behavioral intent and theoretical grounding.
- Use different wording, sentence structures, and examples.
- Be directly usable as an agent prompt.
3. Vary the linguistic style: some formal, some conversational.
4. Do NOT change the underlying negotiation tactic.
5. Start with "In your response, "

Output Format:
{{
  "original_id": "{strategy_id}",
  "paraphrases": [
    {{"id": "{strategy_id}_v1", "description": "..."}},
    {{"id": "{strategy_id}_v2", "description": "..."}}
  ]
}}"""


def build_social_strategy_paraphrase_prompt(
    strategy_id: str,
    strategy_name: str,
    strategy_description: str,
    theory: str,
    n: int,
) -> str:
    """Build the LLM prompt for generating strategy paraphrases."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return SOCIAL_STRATEGY_PARAPHRASE_PROMPT.format(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        strategy_description=strategy_description,
        theory=theory,
        n=n,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from raw LLM output."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        return json.loads(cleaned[start : end + 1])


def validate_social_strategy_paraphrases(
    payload: dict[str, Any],
    *,
    strategy_id: str,
    expected_count: int,
) -> dict[str, Any]:
    """Validate the structure returned by the strategy paraphrase prompt."""
    if payload.get("original_id") != strategy_id:
        raise ValueError(
            f"Expected original_id={strategy_id}, got {payload.get('original_id')}"
        )

    paraphrases = payload.get("paraphrases")
    if not isinstance(paraphrases, list):
        raise ValueError("`paraphrases` must be a list")
    if len(paraphrases) != expected_count:
        raise ValueError(
            f"Expected {expected_count} paraphrases, got {len(paraphrases)}"
        )

    for idx, item in enumerate(paraphrases, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Paraphrase #{idx} must be an object")
        paraphrase_id = item.get("id")
        description = item.get("description")
        expected_id = f"{strategy_id}_v{idx}"
        if paraphrase_id != expected_id:
            raise ValueError(
                f"Expected paraphrase id {expected_id}, got {paraphrase_id}"
            )
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Paraphrase #{idx} must contain a non-empty description")

    return payload


async def generate_social_strategy_paraphrases(
    strategy_id: str,
    strategy_name: str,
    strategy_description: str,
    theory: str,
    n: int,
) -> dict[str, Any]:
    """Generate JSON-formatted paraphrases for a social strategy."""
    global generation_ids
    prompt_content = build_social_strategy_paraphrase_prompt(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        strategy_description=strategy_description,
        theory=theory,
        n=n,
    )
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = await litellm.acompletion(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            gen_id = getattr(response, "id", None)
            if gen_id:
                async with generation_ids_lock:
                    generation_ids.append(gen_id)

            result = response.choices[0].message.content
            if result is None:
                raise ValueError(
                    f"LLM returned None for strategy paraphrase {strategy_id}"
                )

            payload = extract_json_object(result)
            return validate_social_strategy_paraphrases(
                payload, strategy_id=strategy_id, expected_count=n
            )
        except Exception as e:
            last_error = e
            error_str = str(e)
            should_retry = any(str(code) in error_str for code in RETRY_STATUS_CODES)
            if not should_retry:
                raise

            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                f"Strategy paraphrase attempt {attempt + 1}/{MAX_RETRIES} failed for "
                f"{strategy_id}: {error_str[:100]}... Retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise RuntimeError(
        f"Failed to generate strategy paraphrases for {strategy_id} after "
        f"{MAX_RETRIES} retries. Last error: {last_error}"
    )

def contains_non_english(text: str, threshold: float = 0.02) -> tuple[bool, float, str]:
    """
    Check whether the text contains a large amount of non-English characters
    or garbled patterns.

    Logic adapted from `experiments/dynamic_observation/check_embeddings_quality.py`.

    Returns:
        (is_problematic, non_ascii_ratio, reason)
    """
    import re

    if not text:
        return False, 0.0, ""

    # Allowed special characters (common English punctuation and symbols)
    allowed_special = set("''\"\"—–…•·°±×÷©®™€£¥¢")

    # Count non-ASCII characters
    non_ascii_chars: list[str] = []
    for char in text:
        if ord(char) > 127 and char not in allowed_special:
            non_ascii_chars.append(char)

    ratio = len(non_ascii_chars) / len(text) if text else 0.0

    # Check 1: non-ASCII ratio exceeds the threshold
    if ratio > threshold:
        sample = "".join(set(non_ascii_chars[:30]))
        return True, ratio, f"High non-ASCII ratio: {sample}"

    # Check 2: detect mixed writing systems (a common sign of garbled text)
    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", text))  # Cyrillic
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))  # Arabic
    has_cjk = bool(re.search(r"[\u4E00-\u9FFF]", text))  # CJK
    has_korean = bool(re.search(r"[\uAC00-\uD7AF]", text))  # Korean
    has_hebrew = bool(re.search(r"[\u0590-\u05FF]", text))  # Hebrew
    has_thai = bool(re.search(r"[\u0E00-\u0E7F]", text))  # Thai
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))  # Devanagari (e.g. Hindi)
    has_armenian = bool(re.search(r"[\u0530-\u058F]", text))  # Armenian
    has_georgian = bool(re.search(r"[\u10A0-\u10FF]", text))  # Georgian
    has_greek = bool(re.search(r"[\u0370-\u03FF]", text))  # Greek

    language_count = sum(
        [
            has_cyrillic,
            has_arabic,
            has_cjk,
            has_korean,
            has_hebrew,
            has_thai,
            has_devanagari,
            has_armenian,
            has_georgian,
            has_greek,
        ]
    )

    if language_count >= 2:
        return True, ratio, f"Multi-language mixing detected ({language_count} scripts)"

    # Check 3: detect code-like patterns that often appear in corrupt output
    code_patterns = [
        r"\)\s*\{",  # ){
        r"\}\s*;",  # };
        r"===",
        r"\[\s*\]",
        r"function\s*\(",
        r"return\s+\w+;",
        r"import\s+\w+",
        r"class\s+\w+",
        r"\$\{",
        r"=>",
        r"\.then\(",
        r"console\.",
        r"#\s*include",
        r"def\s+\w+\s*\(",
    ]

    code_matches = sum(1 for p in code_patterns if re.search(p, text))
    if code_matches >= 3:
        return True, ratio, f"Code-like patterns detected ({code_matches} patterns)"

    # Check 4: too many special symbols, which is another sign of garbling
    special_chars = re.findall(r"[█▓▒░■□●○◆◇★☆►◄▲△▼▽⬚═║╔╗╚╝╠╣╬┌┐└┘├┤┬┴┼─│]", text)
    if len(special_chars) > 5:
        return True, ratio, f"Excessive special symbols ({len(special_chars)} found)"

    # Check 5: repeated non-ASCII sequences that should not appear in normal English
    consecutive_non_ascii = re.findall(r"[^\x00-\x7F]{3,}", text)
    if len(consecutive_non_ascii) > 3:
        return True, ratio, f"Multiple non-ASCII sequences ({len(consecutive_non_ascii)} found)"

    sample = "".join(set(non_ascii_chars[:30])) if non_ascii_chars else ""
    return False, ratio, sample


def get_valid_chain_quality(chain: list[str], threshold: float) -> list[str]:
    """
    Get the valid rewrite chain: filter out empty values and stop once a
    low-quality item (non-English / garbled text) is encountered.
    """
    valid: list[str] = []
    for item in chain:
        if not (item and isinstance(item, str) and item.strip()):
            break
        is_bad, _, _ = contains_non_english(item, threshold=threshold)
        if is_bad:
            break
        valid.append(item)
    return valid


def load_scenario_data(scenario_dir: Path) -> tuple[dict[str, Any], dict[str, Any], str, str] | None:
    """
    Load the two agent data files from a scenario directory.

    Returns:
        `(agent0_data, agent1_data, agent0_filename, agent1_filename)` or
        `None`. The two files share the same backgrounds and only differ in goal.
    """
    json_files = sorted(scenario_dir.glob("*.json"))
    if len(json_files) < 2:
        return None

    # Sort by filename so that `0_xxx.json` comes before `1_xxx.json`
    agent0_file = json_files[0]
    agent1_file = json_files[1]

    with open(agent0_file, "r", encoding="utf-8") as f:
        agent0_data = json.load(f)
    with open(agent1_file, "r", encoding="utf-8") as f:
        agent1_data = json.load(f)

    return agent0_data, agent1_data, agent0_file.name, agent1_file.name


def select_fusion_modes(step: int, num_modes_range: tuple[int, int] = (1, 3)) -> list[dict[str, str]]:
    """
    Randomly select 1-3 strategic modes for fusion.

    Args:
        step: Current step index (used for reproducibility seeding)
        num_modes_range: Tuple of (min, max) number of modes to select

    Returns:
        List of selected mode dictionaries
    """
    # Use step as seed for reproducibility
    rng = random.Random(step)

    # Randomly decide 1 to 3 modes
    num_modes = rng.randint(num_modes_range[0], num_modes_range[1])

    # Randomly sample without replacement
    selected = rng.sample(STRATEGIC_MODES, num_modes)

    return selected


async def rewrite_single_with_retry(
    text: str,
    selected_mode_name: str | None = None,
    selected_mode_definition: str | None = None,
    fusion_modes: list[dict[str, str]] | None = None,
) -> str:
    """
    Use the LLM to produce a strategic paraphrase for a single text, with
    retries. No semaphore is used here; concurrency is controlled by the caller.

    Either (selected_mode_name, selected_mode_definition) OR fusion_modes must be provided.
    """
    global generation_ids
    last_error = None

    # Build prompt content based on mode type
    if fusion_modes:
        # Build fusion prompt
        mode_list = "\n\n".join([
            f"{i+1}. {mode['name']}\n{mode['definition']}"
            for i, mode in enumerate(fusion_modes)
        ])
        prompt_content = FUSION_PARAPHRASE_PROMPT.format(
            Fusion_Mode_List=mode_list,
            original_text=text,
        )
    else:
        # Use single mode prompt
        if selected_mode_name is None or selected_mode_definition is None:
            raise ValueError("Either fusion_modes or (selected_mode_name, selected_mode_definition) must be provided")
        prompt_content = STRATEGIC_PARAPHRASE_PROMPT.format(
            Selected_Mode_Definition=selected_mode_definition,
            Selected_Mode_Name=selected_mode_name,
            original_text=text,
        )

    for attempt in range(MAX_RETRIES):
        try:
            response = await litellm.acompletion(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            # Record the generation ID for later cost calculation
            gen_id = getattr(response, "id", None)
            if gen_id:
                async with generation_ids_lock:
                    generation_ids.append(gen_id)

            result = response.choices[0].message.content
            if result is None:
                raise ValueError(f"LLM returned None for paraphrase. Input: {text[:100]}...")
            return result.strip()
        except Exception as e:
            last_error = e
            error_str = str(e)
            # Check whether this is a retryable error
            should_retry = any(str(code) in error_str for code in RETRY_STATUS_CODES)
            if not should_retry:
                raise

            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {error_str[:100]}... Retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")

async def rewrite_single_with_quality(
    text: str,
    selected_mode_name: str | None = None,
    selected_mode_definition: str | None = None,
    fusion_modes: list[dict[str, str]] | None = None,
    *,
    quality_threshold: float,
    quality_max_retries: int,
) -> str:
    """
    Generate one strategic rewrite and run a text quality check; if the output
    is unacceptable, retry at the quality-control layer.

    Either (selected_mode_name, selected_mode_definition) OR fusion_modes must be provided.
    """
    last_reason = ""
    for attempt in range(quality_max_retries):
        rewritten = await rewrite_single_with_retry(
            text,
            selected_mode_name=selected_mode_name,
            selected_mode_definition=selected_mode_definition,
            fusion_modes=fusion_modes,
        )
        is_bad, ratio, reason = contains_non_english(rewritten, threshold=quality_threshold)
        if not is_bad:
            return rewritten
        last_reason = f"{reason} (ratio={ratio:.3f})"

        # Build mode description for logging
        if fusion_modes:
            mode_desc = f"fusion({len(fusion_modes)} modes)"
        else:
            mode_desc = f"mode={selected_mode_name}"

        logger.warning(
            f"Quality check failed for {mode_desc} "
            f"(attempt {attempt + 1}/{quality_max_retries}): {last_reason}"
        )
    raise RuntimeError(
        f"Quality check failed after {quality_max_retries} attempts: {last_reason}"
    )

async def rewrite_chain_for_text(
    original_text: str,
    num_steps: int,
    progress: Progress,
    task_id: int,
    existing_chain: list[str] | None = None,
) -> list[str]:
    """
    Perform chained rewriting for a single text.
    Each rewrite is based on the previous result, forming a rewrite chain.
    Supports resuming from an existing chain.

    Args:
        original_text: Original input text
        num_steps: Number of chained rewrite steps
        progress: Rich `Progress` instance used to update progress
        task_id: Progress task ID
        existing_chain: Existing rewrite chain used for incremental generation

    Returns:
        A list containing only rewrite results, not the original text
    """
    # Start from the existing chain if one is provided
    if existing_chain:
        chain = get_valid_chain_quality(existing_chain, threshold=QUALITY_THRESHOLD)
    else:
        chain = []

    # Determine the current text: the last chain item or the original text
    current_text = chain[-1] if chain else original_text

    # Advance the progress bar for already completed steps
    existing_steps = len(chain)
    if existing_steps > 0:
        progress.update(task_id, advance=existing_steps)

    # Generate the remaining steps
    for step in range(existing_steps, num_steps):
        # Rewrite based on the previous result
        mode = STRATEGIC_MODES[step % len(STRATEGIC_MODES)]
        rewritten = await rewrite_single_with_quality(
            current_text,
            selected_mode_name=mode["name"],
            selected_mode_definition=mode["definition"],
            quality_threshold=QUALITY_THRESHOLD,
            quality_max_retries=QUALITY_MAX_RETRIES,
        )
        chain.append(rewritten)
        current_text = rewritten
        # Update the progress bar
        progress.update(task_id, advance=1)

    return chain


async def rewrite_direct_for_text(
    original_text: str,
    progress: Progress,
    task_id: int,
    num_versions: int,
    existing_versions: list[str] | None = None,
) -> list[str]:
    """
    Perform direct paraphrasing: generate one version per strategic mode from
    the same `original_text`, without chained dependence.
    Supports resuming from existing results by filling missing items in order.
    """
    if not STRATEGIC_MODES:
        raise ValueError("STRATEGIC_MODES is empty; define at least one strategic mode.")

    versions = get_valid_chain_quality(existing_versions or [], threshold=QUALITY_THRESHOLD)
    existing_steps = len(versions)
    if existing_steps > 0:
        progress.update(task_id, advance=existing_steps)

    for idx in range(existing_steps, num_versions):
        mode = STRATEGIC_MODES[idx % len(STRATEGIC_MODES)]
        rewritten = await rewrite_single_with_quality(
            original_text,
            selected_mode_name=mode["name"],
            selected_mode_definition=mode["definition"],
            quality_threshold=QUALITY_THRESHOLD,
            quality_max_retries=QUALITY_MAX_RETRIES,
        )
        versions.append(rewritten)
        progress.update(task_id, advance=1)

    return versions


async def rewrite_fusion_for_text(
    original_text: str,
    progress: Progress,
    task_id: int,
    num_versions: int,
    existing_versions: list[str] | None = None,
) -> tuple[list[str], list[list[dict[str, str]]]]:
    """
    Fusion mode: randomly combine 1-3 strategies for each version.

    Returns:
        (versions, fusion_combinations)
        - versions: List of paraphrased texts
        - fusion_combinations: List of mode lists used for each version
    """
    versions = get_valid_chain_quality(existing_versions or [], threshold=QUALITY_THRESHOLD)
    fusion_combinations: list[list[dict[str, str]]] = []

    existing_steps = len(versions)
    if existing_steps > 0:
        progress.update(task_id, advance=existing_steps)

    for idx in range(existing_steps, num_versions):
        # Select 1-3 random modes for this version
        fusion_modes = select_fusion_modes(idx, num_modes_range=(1, 3))
        fusion_combinations.append(fusion_modes)

        rewritten = await rewrite_single_with_quality(
            original_text,
            fusion_modes=fusion_modes,
            quality_threshold=QUALITY_THRESHOLD,
            quality_max_retries=QUALITY_MAX_RETRIES,
        )
        versions.append(rewritten)
        progress.update(task_id, advance=1)

    return versions, fusion_combinations


def save_scenario_result(
    agent0_data: dict[str, Any],
    agent1_data: dict[str, Any],
    agent0_filename: str,
    agent1_filename: str,
    p1_chain: list[str],
    p2_chain: list[str],
    output_dir: Path,
    scenario_pk: str,
    source_dir: str,
    p1_fusion_data: list[list[dict[str, str]]] | None = None,
    p2_fusion_data: list[list[dict[str, str]]] | None = None,
) -> None:
    """
    Save the scenario result into two JSON files, one for each agent view.
    The two files share the same paraphrased backgrounds and only differ in goal.
    """
    scenario_output_dir = output_dir / scenario_pk
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "num_chain_steps": NUM_CHAIN_STEPS,
        "num_direct_versions": DIRECT_VERSIONS,
        "source_dir": source_dir,
        "opt": OPT,
        "strategic_modes": STRATEGIC_MODES,
        "quality_threshold": QUALITY_THRESHOLD,
        "quality_max_retries": QUALITY_MAX_RETRIES,
    }

    # Build strategic versions with fusion info if in fusion mode
    if OPT == "fusion" and p1_fusion_data is not None and p2_fusion_data is not None:
        p1_strategic_versions = [
            {
                "fusion_mode": True,
                "mode_names": [m["name"] for m in fusion_modes],
                "mode_definitions": [m["definition"] for m in fusion_modes],
                "text": text,
            }
            for fusion_modes, text in zip(p1_fusion_data, p1_chain)
        ]
        p2_strategic_versions = [
            {
                "fusion_mode": True,
                "mode_names": [m["name"] for m in fusion_modes],
                "mode_definitions": [m["definition"] for m in fusion_modes],
                "text": text,
            }
            for fusion_modes, text in zip(p2_fusion_data, p2_chain)
        ]

        # For backwards compatibility in p1_background_modes and p2_background_modes
        p1_modes = [" + ".join(m["name"] for m in fmodes) for fmodes in p1_fusion_data]
        p2_modes = [" + ".join(m["name"] for m in fmodes) for fmodes in p2_fusion_data]
    else:
        # Existing single-mode structure
        p1_modes = [STRATEGIC_MODES[i % len(STRATEGIC_MODES)]["name"] for i in range(len(p1_chain))]
        p2_modes = [STRATEGIC_MODES[i % len(STRATEGIC_MODES)]["name"] for i in range(len(p2_chain))]
        p1_strategic_versions = [
            {
                "mode_name": STRATEGIC_MODES[i % len(STRATEGIC_MODES)]["name"],
                "mode_definition": STRATEGIC_MODES[i % len(STRATEGIC_MODES)]["definition"],
                "text": text,
            }
            for i, text in enumerate(p1_chain)
        ]
        p2_strategic_versions = [
            {
                "mode_name": STRATEGIC_MODES[i % len(STRATEGIC_MODES)]["name"],
                "mode_definition": STRATEGIC_MODES[i % len(STRATEGIC_MODES)]["definition"],
                "text": text,
            }
            for i, text in enumerate(p2_chain)
        ]

    paraphrased = {
        "p1_background": p1_chain,
        "p2_background": p2_chain,
        "p1_background_modes": p1_modes,
        "p2_background_modes": p2_modes,
        "p1_strategic_versions": p1_strategic_versions,
        "p2_strategic_versions": p2_strategic_versions,
    }

    # Save the file for the agent0 view (known p1_goal, unknown p2_goal)
    result0 = {
        "original": agent0_data,
        "paraphrased": paraphrased,
        "metadata": {**metadata, "source_file": agent0_filename},
    }
    output_file0 = scenario_output_dir / agent0_filename
    with open(output_file0, "w", encoding="utf-8") as f:
        json.dump(result0, f, ensure_ascii=False, indent=2)
    logger.debug(f"Saved result to {output_file0}")

    # Save the file for the agent1 view (unknown p1_goal, known p2_goal)
    result1 = {
        "original": agent1_data,
        "paraphrased": paraphrased,
        "metadata": {**metadata, "source_file": agent1_filename},
    }
    output_file1 = scenario_output_dir / agent1_filename
    with open(output_file1, "w", encoding="utf-8") as f:
        json.dump(result1, f, ensure_ascii=False, indent=2)
    logger.debug(f"Saved result to {output_file1}")


def load_existing_scenario_result(scenario_pk: str, output_dir: Path) -> dict[str, Any] | None:
    """
    Load an existing scenario result file.
    Reading either agent file is sufficient because `paraphrased` is shared.

    Returns:
        Existing result data, or `None` if it does not exist or is invalid
    """
    scenario_output_dir = output_dir / scenario_pk
    if not scenario_output_dir.exists():
        return None

    # Find any JSON file in the scenario output directory
    json_files = list(scenario_output_dir.glob("*.json"))
    if not json_files:
        return None

    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        # Check whether the required fields exist
        if (
            "original" in data
            and "paraphrased" in data
            and "p1_background" in data["paraphrased"]
            and "p2_background" in data["paraphrased"]
        ):
            return data
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return None


def get_valid_chain(chain: list[str]) -> list[str]:
    """
    Get a valid rewrite chain by filtering out empty or invalid items.

    Returns:
        A valid rewrite chain list
    """
    valid_chain = []
    for item in chain:
        if item and isinstance(item, str) and item.strip():
            valid_chain.append(item)
        else:
            # Stop at the first invalid item because later rewrites depend on earlier ones
            break
    return valid_chain


def is_scenario_complete(scenario_pk: str, output_dir: Path) -> bool:
    """Check whether a scenario is complete: both agent files exist and contain enough rewrites."""
    scenario_output_dir = output_dir / scenario_pk
    if not scenario_output_dir.exists():
        return False

    # Check whether two JSON files exist
    json_files = list(scenario_output_dir.glob("*.json"))
    if len(json_files) < 2:
        return False

    # Check the number of rewrites
    data = load_existing_scenario_result(scenario_pk, output_dir)
    if data is None:
        return False

    p1_chain = get_valid_chain_quality(data["paraphrased"].get("p1_background", []), threshold=QUALITY_THRESHOLD)
    p2_chain = get_valid_chain_quality(data["paraphrased"].get("p2_background", []), threshold=QUALITY_THRESHOLD)

    required = NUM_CHAIN_STEPS if OPT == "chain" else (DIRECT_VERSIONS or len(STRATEGIC_MODES))
    return len(p1_chain) >= required and len(p2_chain) >= required


async def process_single_scenario(
    scenario_dir: Path,
    semaphore: asyncio.Semaphore,
    progress: Progress,
    overall_task_id: int,
) -> dict[str, Any]:
    """
    Process a single scenario after acquiring the semaphore.
    Supports incremental generation by resuming from the existing count.
    Generates separate files for the two agents while sharing the same
    paraphrased backgrounds and differing only in goal.

    Args:
        scenario_dir: Scenario directory path
        semaphore: Semaphore used to control concurrency
        progress: Rich `Progress` instance
        overall_task_id: Overall progress task ID

    Returns:
        Rewrite result data; guaranteed to be complete and non-`None`

    Raises:
        ValueError: If the scenario directory is invalid
        RuntimeError: If generation fails
    """
    scenario_pk = scenario_dir.name

    async with semaphore:
        # Load scenario data for both agents
        loaded = load_scenario_data(scenario_dir)
        if loaded is None:
            raise ValueError(f"No valid JSON files in {scenario_dir}")

        agent0_data, agent1_data, agent0_filename, agent1_filename = loaded

        # Use agent0 to obtain the backgrounds; both files share the same backgrounds
        p1_bg = agent0_data["p1_background"]
        p2_bg = agent0_data["p2_background"]

        # Load existing results if present
        existing_result = load_existing_scenario_result(scenario_pk, OUTPUT_DIR)
        existing_p1_chain: list[str] = []
        existing_p2_chain: list[str] = []

        if existing_result:
            existing_p1_chain = existing_result["paraphrased"].get("p1_background", [])
            existing_p2_chain = existing_result["paraphrased"].get("p2_background", [])
            p1_valid_count = len(get_valid_chain_quality(existing_p1_chain, threshold=QUALITY_THRESHOLD))
            p2_valid_count = len(get_valid_chain_quality(existing_p2_chain, threshold=QUALITY_THRESHOLD))
            required = NUM_CHAIN_STEPS if OPT == "chain" else (DIRECT_VERSIONS or len(STRATEGIC_MODES))
            logger.info(
                f"Resuming scenario {scenario_pk}: p1={p1_valid_count}/{required}, "
                f"p2={p2_valid_count}/{required}"
            )

        required = NUM_CHAIN_STEPS if OPT == "chain" else (DIRECT_VERSIONS or len(STRATEGIC_MODES))
        # Create a child progress task (`required` steps for each of p1 and p2)
        chain_task_id = progress.add_task(
            f"[blue]{scenario_pk[:8]}...",
            total=required * 2,
        )

        # Initialize `fusion_data`; it stays `None` unless fusion mode is used
        p1_fusion_data: list[list[dict[str, str]]] | None = None
        p2_fusion_data: list[list[dict[str, str]]] | None = None

        if OPT == "fusion":
            # Fusion mode: combine 1-3 strategies randomly for each version
            (p1_chain, p1_fusion_data), (p2_chain, p2_fusion_data) = await asyncio.gather(
                rewrite_fusion_for_text(p1_bg, progress, chain_task_id, required, existing_p1_chain),
                rewrite_fusion_for_text(p2_bg, progress, chain_task_id, required, existing_p2_chain),
            )
        elif OPT == "direct":
            # Direct rewriting: generate one version per mode from the original text
            p1_chain, p2_chain = await asyncio.gather(
                rewrite_direct_for_text(p1_bg, progress, chain_task_id, required, existing_p1_chain),
                rewrite_direct_for_text(p2_bg, progress, chain_task_id, required, existing_p2_chain),
            )
        else:
            # Chained rewriting: each step depends on the previous one, while the
            # two chains remain independent; pass existing chains to support resuming
            p1_chain, p2_chain = await asyncio.gather(
                rewrite_chain_for_text(
                    p1_bg, NUM_CHAIN_STEPS, progress, chain_task_id, existing_p1_chain
                ),
                rewrite_chain_for_text(
                    p2_bg, NUM_CHAIN_STEPS, progress, chain_task_id, existing_p2_chain
                ),
            )

        # Validate result completeness
        if len(p1_chain) != required or len(p2_chain) != required:
            raise RuntimeError(
                f"Incomplete result for {scenario_pk}: "
                f"p1={len(p1_chain)}/{required}, p2={len(p2_chain)}/{required}"
            )

        # Save the result files for both agents
        save_scenario_result(
            agent0_data=agent0_data,
            agent1_data=agent1_data,
            agent0_filename=agent0_filename,
            agent1_filename=agent1_filename,
            p1_chain=p1_chain,
            p2_chain=p2_chain,
            output_dir=OUTPUT_DIR,
            scenario_pk=scenario_pk,
            source_dir=str(scenario_dir),
            p1_fusion_data=p1_fusion_data,
            p2_fusion_data=p2_fusion_data,
        )

        # Remove the child progress task
        progress.remove_task(chain_task_id)

        # Update overall progress
        progress.update(overall_task_id, advance=1)

        logger.info(f"Completed scenario {scenario_pk}")

        # Return result data for summary statistics
        return {
            "scenario_pk": scenario_pk,
            "p1_chain_length": len(p1_chain),
            "p2_chain_length": len(p2_chain),
        }


async def main() -> None:
    """Main entry point."""
    console.print("[bold green]Starting Agent Background Chain Paraphrase Script[/]")
    if OPT == "fusion":
        console.print(
            f"[cyan]Opt: {OPT}, Model: {MODEL_NAME}, Temperature: {TEMPERATURE}, "
            f"Fusion Versions: {(DIRECT_VERSIONS or len(STRATEGIC_MODES))} "
            f"(Each combines 1-3 random strategies from {len(STRATEGIC_MODES)} modes)[/]"
        )
    elif OPT == "direct":
        console.print(
            f"[cyan]Opt: {OPT}, Model: {MODEL_NAME}, Temperature: {TEMPERATURE}, "
            f"Direct Versions: {(DIRECT_VERSIONS or len(STRATEGIC_MODES))} "
            f"(Modes available: {len(STRATEGIC_MODES)})[/]"
        )
    else:
        console.print(
            f"[cyan]Opt: {OPT}, Model: {MODEL_NAME}, Temperature: {TEMPERATURE}, Chain Steps: {NUM_CHAIN_STEPS}[/]"
        )
    console.print(f"[cyan]Input: {INPUT_DIR}, Output: {OUTPUT_DIR}[/]")

    # Check the API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Configure LiteLLM
    litellm.api_key = api_key
    # Reuse the HTTP client connection pool to reduce SSL connection overhead
    litellm.enable_shared_async_client = True

    # Collect all scenario directories
    console.print("[cyan]Loading scenario directories...[/]")
    # Use provided INPUT_DIR (which might be updated by args)
    if not INPUT_DIR.exists():
        raise ValueError(f"Input directory not found: {INPUT_DIR}")

    all_scenario_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    console.print(f"[green]Found {len(all_scenario_dirs)} scenarios[/]")

    # Limit the number of scenarios if requested
    if MAX_SCENARIOS is not None:
        all_scenario_dirs = all_scenario_dirs[:MAX_SCENARIOS]
        console.print(f"[yellow]Limited to first {MAX_SCENARIOS} scenarios[/]")

    # Create a semaphore to control concurrency across scenarios
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Inspect generated files and select scenarios that still need processing
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios_to_process = []
    complete_count = 0
    incomplete_count = 0

    for scenario_dir in all_scenario_dirs:
        if is_scenario_complete(scenario_dir.name, OUTPUT_DIR):
            complete_count += 1
        else:
            scenarios_to_process.append(scenario_dir)
            # Check whether the scenario is partially completed
            if load_existing_scenario_result(scenario_dir.name, OUTPUT_DIR) is not None:
                incomplete_count += 1

    if complete_count > 0:
        console.print(f"[yellow]Skipping {complete_count} already completed scenarios[/]")

    if incomplete_count > 0:
        console.print(f"[cyan]Resuming {incomplete_count} incomplete scenarios[/]")

    if not scenarios_to_process:
        console.print("[bold green]All scenarios already processed! Nothing to do.[/]")
        return

    console.print(f"[cyan]Processing {len(scenarios_to_process)} remaining scenarios...[/]")
    console.print(f"[cyan]Max concurrency: {MAX_CONCURRENCY} scenarios in parallel[/]")

    # Process all scenarios asynchronously and concurrently
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            "[green]Overall progress", total=len(scenarios_to_process)
        )

        # Create all tasks and execute them concurrently
        tasks = [
            process_single_scenario(scenario_dir, semaphore, progress, overall_task)
            for scenario_dir in scenarios_to_process
        ]

        # Use `gather`; `return_exceptions=True` prevents one failure from aborting all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Summarize results
        success_count = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        error_count = sum(1 for r in results if isinstance(r, Exception))

    console.print(f"[bold green]Done! Results saved to {OUTPUT_DIR}[/]")
    console.print(
        f"[bold green]Total: {len(all_scenario_dirs)} scenarios, "
        f"Processed: {success_count}, "
        f"Skipped: {complete_count}, "
        f"Errors: {error_count}[/]"
    )

    if error_count > 0:
        console.print("[bold red]Some scenarios failed. Check logs for details.[/]")

    # Compute and save cost statistics
    if generation_ids:
        console.print(f"\n[cyan]Calculating cost for {len(generation_ids)} LLM calls...[/]")

        # Save generation IDs to a log file
        log_file = OUTPUT_DIR / "generation_ids.jsonl"
        with open(log_file, "w", encoding="utf-8") as f:
            for gen_id in generation_ids:
                f.write(json.dumps({"id": gen_id}) + "\n")
        logger.info(f"Saved {len(generation_ids)} generation IDs to {log_file}")

        # Compute cost
        cost_result = await calculate_cost_async(log_file)

        # Add extra summary fields
        cost_result["generation_count"] = len(generation_ids)
        cost_result["scenarios_processed"] = success_count
        cost_result["model"] = MODEL_NAME
        cost_result["temperature"] = TEMPERATURE
        cost_result["max_tokens"] = MAX_TOKENS
        cost_result["opt"] = OPT
        cost_result["num_chain_steps"] = NUM_CHAIN_STEPS if OPT == "chain" else None
        cost_result["num_strategic_modes"] = len(STRATEGIC_MODES)
        cost_result["num_direct_versions"] = (DIRECT_VERSIONS or len(STRATEGIC_MODES)) if OPT == "direct" else None
        cost_result["quality_threshold"] = QUALITY_THRESHOLD
        cost_result["quality_retries"] = QUALITY_MAX_RETRIES
        cost_result["calculated_at"] = datetime.now().isoformat()

        # Save cost statistics to a JSON file
        cost_file = OUTPUT_DIR / "cost_statistics.json"
        with open(cost_file, "w", encoding="utf-8") as f:
            json.dump(cost_result, f, ensure_ascii=False, indent=2)

        console.print(f"[bold green]Cost statistics saved to {cost_file}[/]")
        console.print(f"[bold cyan]Total Cost: ${cost_result.get('total_cost', 0):.6f}[/]")
    else:
        console.print("[yellow]No new LLM calls made, skipping cost calculation.[/]")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent background chained paraphrasing script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--opt",
        choices=["chain", "direct", "fusion"],
        default="direct",
        help="Rewrite mode: chain=chained rewriting; direct=one rewrite per strategic mode; fusion=randomly combine 1-3 strategies",
    )
    parser.add_argument(
        "--direct-versions",
        type=int,
        default=None,
        help="Number of versions to generate in direct mode (default: number of strategic modes; modes are reused cyclically if larger)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="Maximum output token limit for the LLM (default: 6199)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=QUALITY_THRESHOLD,
        help="Text quality check: non-ASCII character ratio threshold (default: 0.05; values above this are treated as problematic)",
    )
    parser.add_argument(
        "--quality-retries",
        type=int,
        default=QUALITY_MAX_RETRIES,
        help="Text quality check: maximum retries per text (default: 5)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only one scenario",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Limit the number of scenarios to process",
    )
    parser.add_argument(
        "--chain-steps",
        type=int,
        default=None,
        help="Number of chained rewrite steps (overrides the default)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override the input directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # import litellm
    # litellm._turn_on_debug()
    OPT = args.opt
    DIRECT_VERSIONS = args.direct_versions
    if DIRECT_VERSIONS is not None and DIRECT_VERSIONS <= 0:
        raise ValueError("--direct-versions must be a positive integer")

    MAX_TOKENS = args.max_tokens
    if MAX_TOKENS <= 0:
        raise ValueError("--max-tokens must be a positive integer")

    QUALITY_THRESHOLD = args.quality_threshold
    QUALITY_MAX_RETRIES = args.quality_retries
    if QUALITY_THRESHOLD < 0 or QUALITY_THRESHOLD > 1:
        raise ValueError("--quality-threshold must be within [0, 1]")
    if QUALITY_MAX_RETRIES <= 0:
        raise ValueError("--quality-retries must be a positive integer")

    # Override configuration based on CLI arguments
    if args.test:
        MAX_SCENARIOS = 1
        console.print("[bold yellow]TEST MODE: Processing only 1 scenario[/]")
    elif args.max_scenarios is not None:
        MAX_SCENARIOS = args.max_scenarios

    if args.chain_steps is not None:
        NUM_CHAIN_STEPS = args.chain_steps

    # Update the input directory if provided
    if args.input_dir:
        INPUT_DIR = args.input_dir

    # Update the output directory based on `opt`
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    elif OPT == "direct":
        # Keep this separate from existing generic paraphrases to avoid overwriting by default
        OUTPUT_DIR = Path(
            f"experiments/dynamic_observation/paraphrased_backgrounds_strategic/{MODEL_NAME}/t{TEMPERATURE}/{INPUT_DIR.name}"
        )
    elif OPT == "fusion":
        OUTPUT_DIR = Path(
            f"experiments/dynamic_observation/paraphrased_backgrounds_fusion/{MODEL_NAME}/t{TEMPERATURE}/{INPUT_DIR.name}"
        )
    else:
        OUTPUT_DIR = Path(
            f"experiments/dynamic_observation/chained_paraphrased_backgrounds/{MODEL_NAME}/t{TEMPERATURE}"
        )

    # Configure the logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Use a custom wrapper to avoid "Event loop is closed" errors
    async def main_with_cleanup() -> None:
        try:
            await main()
        finally:
            # Explicitly close LiteLLM's async HTTP clients
            console.print("\n[dim]Cleaning up LiteLLM async clients...[/]")
            try:
                await litellm.close_litellm_async_clients()
                console.print("[dim]✓ LiteLLM async clients closed successfully[/]")
            except Exception as e:
                console.print(f"[yellow]Warning: Error closing LiteLLM clients: {e}[/]")
            # Wait a bit longer to ensure all connections are fully closed
            await asyncio.sleep(0.5)

    asyncio.run(main_with_cleanup())
