import json
import os
import asyncio
import argparse
import aiohttp
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent

# Load environment variables
load_dotenv(project_root / ".env")
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("Error: OPENROUTER_API_KEY not found in .env")
    exit(1)

async def fetch_usage(session, gen_id):
    url = f"https://openrouter.ai/api/v1/generation?id={gen_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("data", {})
            else:
                print(f"Error querying ID {gen_id}: Status {response.status}")
                return None
    except Exception as e:
        print(f"Exception querying ID {gen_id}: {e}")
        return None

async def calculate_cost_async(
    log_file: Path,
    session: aiohttp.ClientSession | None = None,
) -> dict:
    """
    Calculate LLM API costs from log file.

    Args:
        log_file: Path to the JSONL log file
        session: Optional aiohttp session. If provided, caller is responsible for closing it.
                 If None, a new session will be created and closed within this function.
    """
    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        return {}

    print(f"Reading log file: {log_file}")

    with open(log_file, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} log entries. Querying OpenRouter API concurrently...")

    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    processed_count = 0
    error_count = 0

    # Determine if we need to manage the session ourselves
    should_close_session = session is None
    if session is None:
        connector = aiohttp.TCPConnector(force_close=True)  # Force close connections to avoid SSL issues
        session = aiohttp.ClientSession(connector=connector)

    try:
        tasks = []
        for line in lines:
            try:
                entry = json.loads(line)
                gen_id = entry.get("id")
                if gen_id:
                    tasks.append(fetch_usage(session, gen_id))
            except json.JSONDecodeError:
                pass

        # Process in batches to avoid rate limits if necessary
        batch_size = 50
        results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            print(f"Processed {min(i+batch_size, len(tasks))}/{len(tasks)}...")
    finally:
        if should_close_session:
            await session.close()
            # Give connections time to close gracefully
            await asyncio.sleep(0.25)

    for data in results:
        if data:
            cost = data.get("total_cost", 0) or 0
            p_tokens = data.get("native_tokens_prompt", 0) or 0
            c_tokens = data.get("native_tokens_completion", 0) or 0

            if not p_tokens:
                p_tokens = data.get("tokens_prompt", 0) or 0
            if not c_tokens:
                c_tokens = data.get("tokens_completion", 0) or 0

            total_cost += cost
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            processed_count += 1
        else:
            error_count += 1

    print("\n" + "="*30)
    print("Calculation Results")
    print("="*30)
    print(f"Entries Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print("-" * 30)
    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Total Completion Tokens: {total_completion_tokens}")
    print(f"Total Tokens: {total_prompt_tokens + total_completion_tokens}")
    print("-" * 30)
    print(f"Total Cost: ${total_cost:.6f}")
    print("="*30)

    return {
        "total_cost": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "processed_count": processed_count,
        "error_count": error_count
    }

async def main_async(log_file: Path) -> None:
    """Async main with proper cleanup."""
    try:
        await calculate_cost_async(log_file)
    finally:
        # Extra sleep to ensure all connections are closed
        await asyncio.sleep(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate LLM API costs from log file")
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the JSONL log file containing LLM call records",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args.log_file))


if __name__ == "__main__":
    main()
