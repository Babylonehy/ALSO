"""
Prompt templates for paraphrasing agent backgrounds.

This module contains all the prompts used for LLM-based paraphrasing.
"""

# Prompt template for paraphrasing agent background descriptions
PARAPHRASE_BACKGROUND_PROMPT = """You are a skilled rewriter. Your task is to paraphrase the following agent background description while preserving ALL the original information.

Requirements:
1. Keep ALL factual information intact:
   - Names (first name, last name)
   - Ages
   - Genders and pronouns
   - Occupations
   - Personality traits
   - Values and beliefs
   - Secrets
   - Any other specific details
2. Change the sentence structure and word choices
3. Use different expressions and phrasings
4. Maintain the same meaning and tone
5. Keep the description as a coherent paragraph
6. Output ONLY the paraphrased text, nothing else (no explanations, no prefixes)

Original text:
{original_text}

Paraphrased version:"""

# System prompt for better instruction following
SYSTEM_PROMPT = """You are a professional text paraphraser. You excel at rewriting text while preserving all factual information. You always output only the paraphrased content without any explanations or additional text."""

# Prompt for generating variations with specific styles (optional, for future use)
STYLE_VARIATION_PROMPTS = {
    "formal": """Paraphrase the following text in a more formal, professional tone while keeping all information:

{original_text}

Formal version:""",
    "casual": """Paraphrase the following text in a more casual, conversational tone while keeping all information:

{original_text}

Casual version:""",
    "concise": """Paraphrase the following text more concisely while keeping all important information:

{original_text}

Concise version:""",
}

