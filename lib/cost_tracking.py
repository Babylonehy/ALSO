"""Cost calculation utilities for OpenRouter API usage tracking."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger


async def calculate_cost_by_model_async(
    log_file: Path,
    *,
    batch_size: int = 50,
    max_workers: int = 100,
) -> dict[str, Any]:
    """
    Calculate OpenRouter costs grouped by model using Sotopia's LLM call log JSONL.

    Expected log entry keys: {"id": ..., "model": ...}

    Args:
        log_file: Path to LLM call log JSONL file
        batch_size: Number of concurrent API requests per batch
        max_workers: Maximum concurrent connections

    Returns:
        dict with keys:
            - total_cost: Total API cost
            - total_prompt_tokens: Total prompt tokens
            - total_completion_tokens: Total completion tokens
            - total_tokens: Total tokens (prompt + completion)
            - processed_count: Number of successfully processed entries
            - error_count: Number of failed entries
            - by_model: Dict of per-model statistics
    """

    async def fetch_usage(session: aiohttp.ClientSession, gen_id: str) -> dict[str, Any] | None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        url = f"https://openrouter.ai/api/v1/generation?id={gen_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {}) or {}
                logger.warning(f"Error querying ID {gen_id}: Status {response.status}")
                return None
        except Exception as e:
            logger.warning(f"Exception querying ID {gen_id}: {e}")
            return None

    if not log_file.exists():
        logger.warning(f"LLM call log not found at {log_file}")
        return {}

    entries: list[dict[str, Any]] = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            gen_id = entry.get("id")
            if not gen_id:
                continue
            entries.append(
                {
                    "id": gen_id,
                    "model": entry.get("model", "unknown"),
                    "caller": entry.get("caller", ""),
                }
            )

    if not entries:
        return {}

    # Deduplicate by id
    seen: set[str] = set()
    unique_entries: list[dict[str, Any]] = []
    for e in entries:
        gen_id = e["id"]
        if gen_id in seen:
            continue
        seen.add(gen_id)
        unique_entries.append(e)

    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    processed_count = 0
    error_count = 0

    by_model: dict[str, dict[str, Any]] = {}

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=max_workers, limit_per_host=30, force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch_usage(session, e["id"]) for e in unique_entries]

        results: list[dict[str, Any] | None] = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            # Convert exceptions to None but keep position correspondence
            for r in batch_results:
                if isinstance(r, (Exception, BaseException)) or r is None:
                    results.append(None)
                elif isinstance(r, dict):
                    results.append(r)
                else:
                    results.append(None)

    # Give more time for SSL transports and connections to close gracefully
    await asyncio.sleep(0.3)  # Increased from 0.1s - handles 100+ concurrent connections

    for e, usage in zip(unique_entries, results):
        if not usage:
            error_count += 1
            continue

        cost = float(usage.get("total_cost", 0) or 0)
        p_tokens = int(usage.get("native_tokens_prompt", 0) or 0)
        c_tokens = int(usage.get("native_tokens_completion", 0) or 0)
        if not p_tokens:
            p_tokens = int(usage.get("tokens_prompt", 0) or 0)
        if not c_tokens:
            c_tokens = int(usage.get("tokens_completion", 0) or 0)

        total_cost += cost
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens
        processed_count += 1

        model = str(e.get("model") or "unknown")
        model_bucket = by_model.setdefault(
            model,
            {
                "total_cost": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "processed_count": 0,
                "error_count": 0,
            },
        )
        model_bucket["total_cost"] += cost
        model_bucket["total_prompt_tokens"] += p_tokens
        model_bucket["total_completion_tokens"] += c_tokens
        model_bucket["total_tokens"] += p_tokens + c_tokens
        model_bucket["processed_count"] += 1

    return {
        "total_cost": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "processed_count": processed_count,
        "error_count": error_count,
        "by_model": dict(
            sorted(by_model.items(), key=lambda kv: kv[1].get("total_cost", 0), reverse=True)
        ),
    }
