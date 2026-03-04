"""
EvoPrompt-style templates for GA and DE evolution.

Adapted from EvoPrompt (https://github.com/beeevita/EvoPrompt) for agent bio
optimization in Sotopia social simulation.
"""

# =============================================================================
# GA (Genetic Algorithm) Templates
# =============================================================================

GA_CROSSOVER_MUTATION_TEMPLATE = """Please follow the instruction step-by-step to generate a better agent bio description.
1. Crossover the following agent bios and generate a new bio:
Bio 1: <bio1>
Bio 2: <bio2>
2. Mutate the bio generated in Step 1 and generate a final bio bracketed with <BIO> and </BIO>.

Example:
1. Crossover Bio: Combining the social nature from Bio 1 with the professional traits from Bio 2.
2. <BIO>A sociable professional who balances work ethics with interpersonal warmth.</BIO>

Please follow the instruction step-by-step to generate a better agent bio description.
1. Crossover the following agent bios and generate a new bio:
Bio 1: <bio1>
Bio 2: <bio2>
2. Mutate the bio generated in Step 1 and generate a final bio bracketed with <BIO> and </BIO>.

1. """


# =============================================================================
# DE (Differential Evolution) Templates
# =============================================================================

DE_DIFFERENTIAL_MUTATION_TEMPLATE = """Please follow the instruction step-by-step to generate a better agent bio description.
1. Identify the different parts between Bio 1 and Bio 2:
Bio 1: <bio1>
Bio 2: <bio2>
2. Randomly mutate the different parts
3. Combine the mutated parts with Bio 3, and generate a new bio.
Bio 3: <bio3>
4. Crossover with the base bio and generate a final bio bracketed with <BIO> and </BIO>:
Base Bio: <bio0>

Example:
1. Identifying differences between Bio 1 and Bio 2:
Bio 1: A friendly person who enjoys helping others.
Bio 2: An assertive leader who takes charge in difficult situations.
Differences: "friendly" vs "assertive", "helping others" vs "takes charge"

2. Randomly mutate: "friendly" -> "warm", "assertive" -> "confident"

3. Combine with Bio 3: Merge the warm confidence with Bio 3's traits.

4. <BIO>A confident and warm individual who can both lead and support others effectively.</BIO>

Please follow the instruction step-by-step to generate a better agent bio description.
1. Identify the different parts between Bio 1 and Bio 2:
Bio 1: <bio1>
Bio 2: <bio2>
2. Randomly mutate the different parts
3. Combine the mutated parts with Bio 3, and generate a new bio.
Bio 3: <bio3>
4. Crossover with the base bio and generate a final bio bracketed with <BIO> and </BIO>:
Base Bio: <bio0>

1. """


# =============================================================================
# Utility Functions
# =============================================================================

def build_ga_prompt(bio1: str, bio2: str) -> str:
    """Build GA crossover+mutation prompt from two parent bios."""
    return GA_CROSSOVER_MUTATION_TEMPLATE.replace("<bio1>", bio1).replace("<bio2>", bio2)


def build_de_prompt(bio0: str, bio1: str, bio2: str, bio3: str) -> str:
    """Build DE differential mutation prompt.
    
    Args:
        bio0: Base bio (the one being evolved)
        bio1: Random donor 1
        bio2: Random donor 2  
        bio3: Best donor (or random donor 3)
    """
    return (
        DE_DIFFERENTIAL_MUTATION_TEMPLATE
        .replace("<bio0>", bio0)
        .replace("<bio1>", bio1)
        .replace("<bio2>", bio2)
        .replace("<bio3>", bio3)
    )


def parse_bio_from_response(response: str) -> str | None:
    """Parse bio from LLM response.
    
    Looks for content between <BIO> and </BIO> tags.
    Falls back to cleaning up the raw response if tags not found.
    """
    import re
    
    # Try to find tagged bio
    match = re.search(r"<BIO>(.*?)</BIO>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: clean up response
    # Remove common prefixes
    cleaned = response.strip()
    prefixes_to_remove = [
        "Here is the improved bio:",
        "Final bio:",
        "Generated bio:",
        "New bio:",
    ]
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    
    # If response is too short, return None
    if len(cleaned) < 10:
        return None
    
    return cleaned
