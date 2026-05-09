"""
OPRO-style meta prompt templates for Sotopia bio optimization.

This module provides templates and utilities for generating meta-prompts
that guide LLM optimizers to generate better agent bio descriptions.
"""

from typing import Literal

# Meta prompt template for Sotopia agent bio optimization
SOTOPIA_BIO_META_PROMPT = """Your task is to generate an agent bio description that helps the agent achieve better social interaction outcomes.

Below are some previous bio descriptions with their scores. The scores range from 0 to 1, where higher scores indicate better social performance:

{instruction_score_pairs}

Generate a new bio description that is different from all the descriptions above and has a higher score than all of them.

Requirements for the new bio:
1. Be concise and actionable (under 200 words)
2. Be distinct from existing descriptions - do not simply rephrase

Write your new bio description in the following format:
<BIO>your bio here</BIO>"""

# Alternative template with scenario context
SOTOPIA_BIO_META_PROMPT_WITH_CONTEXT = """Your task is to generate an agent bio description that helps the agent achieve better social interaction outcomes.

Below are some previous bio descriptions with their scores. The scores range from 0 to 1, where higher scores indicate better social performance:

{instruction_score_pairs}

Generate a new bio description that is different from all the descriptions above and has a higher score than all of them.

Requirements for the new bio:
1. Be concise and actionable (under 200 words)
2. Be distinct from existing descriptions - do not simply rephrase

Write your new bio description in the following format:
<BIO>your bio here</BIO>"""


def gen_instruction_score_pairs_str(
    instructions_and_scores: list[tuple[str, float, int]],
    max_num_instructions: int = 20,
    score_threshold: float = 0.0,
    score_scale: float = 1.0,
) -> str:
    """
    Generate formatted string of instruction-score pairs for meta prompt.
    
    Args:
        instructions_and_scores: List of (instruction, score, step) tuples
        max_num_instructions: Maximum number of instructions to include
        score_threshold: Only include instructions with score >= threshold
        score_scale: Scale for displaying scores (default 10.0 for 0-10 range)
    
    Returns:
        Formatted string with instructions and scores in ascending order
    """
    # Filter by threshold
    filtered = [
        (ins, score, step) 
        for ins, score, step in instructions_and_scores
        if score >= score_threshold
    ]
    
    # Sort by score ascending (lower scores first, higher scores last)
    # This matches OPRO's approach where better instructions appear at the end
    sorted_pairs = sorted(filtered, key=lambda x: x[1])[-max_num_instructions:]
    
    # Format as string
    result_parts = []
    for instruction, score, step in sorted_pairs:
        # Scale score to display range
        display_score = round(score * score_scale, 2)
        result_parts.append(f"\ntext:\n{instruction}\nscore:\n{display_score}\n")
    
    return "".join(result_parts)


def gen_sotopia_meta_prompt(
    instructions_and_scores: list[tuple[str, float, int]],
    max_num_instructions: int = 20,
    score_threshold: float = 0.0,
    include_context: bool = False,
) -> str:
    """
    Generate complete meta prompt for Sotopia bio optimization.
    
    Args:
        instructions_and_scores: List of (instruction, score, step) tuples
        max_num_instructions: Maximum number of instructions to include
        score_threshold: Only include instructions with score >= threshold
        include_context: Whether to include scenario context in prompt
    
    Returns:
        Complete meta prompt string ready for LLM optimizer
    """
    pairs_str = gen_instruction_score_pairs_str(
        instructions_and_scores,
        max_num_instructions=max_num_instructions,
        score_threshold=score_threshold,
    )
    
    template = SOTOPIA_BIO_META_PROMPT_WITH_CONTEXT if include_context else SOTOPIA_BIO_META_PROMPT
    return template.format(instruction_score_pairs=pairs_str)


def parse_bio_from_response(response: str) -> str | None:
    """
    Extract bio description from LLM response.
    
    Args:
        response: Raw LLM response text
    
    Returns:
        Extracted bio string, or None if parsing fails
    """
    import re
    
    # Try to find <BIO>...</BIO> pattern
    pattern = r"<BIO>(.*?)</BIO>"
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Fallback: if no tags found, return the whole response if it's reasonable length
    cleaned = response.strip()
    if 10 < len(cleaned) < 1000:
        return cleaned
    
    return None


def get_thinking_style_prompt(style: Literal["analytical", "creative", "empathetic", "strategic"]) -> str:
    """
    Get a thinking style modifier for the optimizer prompt.
    
    Args:
        style: Type of thinking style to apply
    
    Returns:
        Prompt modifier string
    """
    styles = {
        "analytical": "Think step by step about what makes an effective agent bio.",
        "creative": "Be creative and explore unconventional approaches to agent personality.",
        "empathetic": "Focus on emotional intelligence and interpersonal connection.",
        "strategic": "Consider strategic communication patterns that achieve goals.",
    }
    return styles.get(style, "")
