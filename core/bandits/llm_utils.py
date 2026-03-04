"""
LLM utility functions for bandit operations.

This module provides common utilities for LLM API calls with retry logic,
error handling, and other shared functionality across different bandits.
"""

import asyncio
from typing import Any

from litellm import acompletion
from loguru import logger


# Default retry configuration
DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# Non-retryable error keywords (auth, permission, invalid request)
NON_RETRYABLE_KEYWORDS = [
    "authentication",
    "unauthorized",
    "invalid_api_key",
    "api_key",
    "permission",
    "forbidden",
    "not found",
    "invalid request",
    "invalid_request",
]

# Retryable error keywords (network, server errors)
RETRYABLE_KEYWORDS = [
    "disconnect",
    "timeout",
    "connection",
    "internal server error",
    "server error",
    "apierror",
    "openrouterexception",
    "rate",
    "overload",
    "unavailable",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
]


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should be retried.
    
    Uses a conservative approach: retry unless it's clearly a non-retryable error
    (authentication, permission, invalid request).
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error should be retried, False otherwise
    """
    error_str = str(error)
    error_str_lower = error_str.lower()
    
    # Check if it's a non-retryable error
    is_non_retryable = any(kw in error_str_lower for kw in NON_RETRYABLE_KEYWORDS)
    
    # Check if it's explicitly retryable
    is_explicitly_retryable = (
        any(str(code) in error_str for code in DEFAULT_RETRY_STATUS_CODES)
        or any(kw in error_str_lower for kw in RETRYABLE_KEYWORDS)
    )
    
    # Retry if explicitly retryable OR not explicitly non-retryable
    return is_explicitly_retryable or not is_non_retryable


async def acompletion_with_retry(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    n: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    caller_name: str = "LLM",
    **kwargs: Any,
) -> Any:
    """
    Call LiteLLM acompletion with automatic retry logic.
    
    Args:
        model: The model name to use
        messages: The messages to send
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        n: Number of completions to generate
        max_retries: Maximum number of retry attempts
        retry_base_delay: Base delay for exponential backoff (seconds)
        caller_name: Name of the caller for logging
        **kwargs: Additional arguments to pass to acompletion
        
    Returns:
        The LiteLLM response object
        
    Raises:
        RuntimeError: If all retries are exhausted
        Exception: If a non-retryable error occurs
    """
    last_error: Exception | None = None
    
    for attempt in range(max_retries):
        try:
            response = await acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                **kwargs,
            )
            return response
        except Exception as e:
            last_error = e
            
            if not is_retryable_error(e):
                logger.error(f"{caller_name} failed with non-retryable error: {e}")
                raise
            
            delay = retry_base_delay * (2 ** attempt)
            error_str = str(e)
            logger.warning(
                f"{caller_name} call failed (attempt {attempt + 1}/{max_retries}): "
                f"{error_str[:150]}... Retrying in {delay:.1f}s for {model}"
            )
            await asyncio.sleep(delay)
    
    # All retries exhausted
    raise RuntimeError(
        f"{caller_name} failed after {max_retries} retries. Last error: {last_error}"
    )

