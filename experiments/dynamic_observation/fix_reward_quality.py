#!/usr/bin/env python3
"""
Fix Reward Quality Script

This script checks and fixes reward data quality issues for a given experiment tag.
It identifies episodes with float rewards (missing breakdown data) and re-evaluates them
using the TerminalEvaluator to generate complete reward breakdowns.

IMPORTANT: To protect original data, this script copies all episodes to a new tag before
making any modifications. The original tag's data remains unchanged.

Usage:
    # Check only (no fixes)
    python fix_reward_quality.py --tag <tag> --check-only

    # Dry run (evaluate but don't update database)
    python fix_reward_quality.py --tag <tag> --dry-run

    # Fix and update database (creates new tag with "_fixed" suffix by default)
    python fix_reward_quality.py --tag <tag> --update-db

    # Fix with custom new tag name
    python fix_reward_quality.py --tag <tag> --new-tag <custom_new_tag> --update-db
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
import os
import sys
import traceback

# Unset proxy if causing issues (copied from evaluate_episodes_multi_run.py)
if os.environ.get("ALL_PROXY") or os.environ.get("all_proxy"):
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)

from dotenv import load_dotenv
# Load .env from project root
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from sotopia.database.logs import EpisodeLog
from sotopia.database import EnvironmentProfile, AgentProfile
from experiments.dynamic_observation.core.evaluator_reward_in_trun import TerminalEvaluator
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    SotopiaDimensions,
    unweighted_aggregate_evaluate
)
from sotopia.envs.parallel import get_bio
from sotopia.messages import AgentAction, Observation, ScriptBackground
from sotopia.renderers import RenderContext, XMLRenderer

console = Console()

# TerminalEvaluator is now imported from evaluator_reward_in_trun


class RewardQualityChecker:
    """Check reward data quality for a given tag."""
    
    def check_tag(self, tag: str) -> dict[str, Any]:
        """
        Check all episodes for a given tag.
        
        Returns:
            Dict with:
                - total: total episode count
                - float_episodes: list of episodes with float rewards
                - tuple_episodes: list of episodes with tuple rewards
                - needs_fix: whether fixes are needed
                - stats: detailed statistics
        """
        try:
            episodes = list(EpisodeLog.find(EpisodeLog.tag == tag).all())
        except Exception as e:
            logger.error(f"Failed to query database for tag {tag}: {e}")
            return {"error": str(e)}
        
        if not episodes:
            return {"error": f"No episodes found for tag: {tag}"}
        
        float_episodes = []
        tuple_episodes = []
        
        for episode in episodes:
            if len(episode.rewards) >= 2:
                r1 = episode.rewards[0]
                r2 = episode.rewards[1]
                
                # Check if both rewards are floats (missing breakdown)
                if isinstance(r1, (int, float)) or isinstance(r2, (int, float)):
                    float_episodes.append(episode)
                else:
                    tuple_episodes.append(episode)
        
        # Calculate statistics
        total = len(episodes)
        float_count = len(float_episodes)
        tuple_count = len(tuple_episodes)
        float_pct = (float_count / total * 100) if total > 0 else 0
        
        return {
            "total": total,
            "float_episodes": float_episodes,
            "tuple_episodes": tuple_episodes,
            "float_count": float_count,
            "tuple_count": tuple_count,
            "float_pct": float_pct,
            "needs_fix": float_count > 0,
            "stats": {
                "tag": tag,
                "total_episodes": total,
                "healthy_episodes": tuple_count,
                "degraded_episodes": float_count,
                "degradation_pct": float_pct,
            }
        }


class EpisodeReEvaluator:
    """Re-evaluate episodes with missing reward breakdowns."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the evaluator with a specific model."""
        self.model_name = model_name
        self.evaluator = TerminalEvaluator(
            model_name=model_name,
            response_format_class=EvaluationForTwoAgents[SotopiaDimensions]
        )
        self.evaluation_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    async def re_evaluate_episode(
        self,
        episode: EpisodeLog,
        max_retries: int = 3
    ) -> tuple[tuple[float, dict] | None, tuple[float, dict] | None]:
        """
        Re-evaluate a single episode.

        Returns:
            (p1_rating, p2_rating) where each rating is (score, breakdown) or None on failure
        """
        self.evaluation_count += 1

        # IMPORTANT: Use rewards_prompt to get the EXACT history used during simulation
        #
        # The episode.messages stores DynamicObservation for each agent (agent-specific perspective),
        # where one agent's goal is "Unknown". But during simulation, the TerminalEvaluator uses
        # self.background (global perspective) which contains BOTH agents' complete goals.
        #
        # The rewards_prompt stores the history that was actually sent to the API during simulation,
        # in the format: "Terminal evaluation for history: <history>"
        #
        # We extract this history to ensure consistency with the original evaluation.
        history = ""

        # Primary: Extract history from rewards_prompt (this is the exact history used during simulation)
        if episode.rewards_prompt:
            try:
                # Format: "Terminal evaluation for history: <history>"
                if "Terminal evaluation for history:" in episode.rewards_prompt:
                    history = episode.rewards_prompt.replace("Terminal evaluation for history:", "").strip()
                    logger.debug(f"Extracted history from rewards_prompt (Terminal format)")
                # Alternative format: "Prompt after formatting: <history>,\nBased on previous interactions..."
                elif "Prompt after formatting:" in episode.rewards_prompt:
                    history = episode.rewards_prompt.replace("Prompt after formatting:", "").split(
                        ",\nBased on previous interactions"
                    )[0].strip()
                    logger.debug(f"Extracted history from rewards_prompt (Prompt format)")
                else:
                    # Direct use if no known prefix
                    history = episode.rewards_prompt.split(",\nBased on previous interactions")[0].strip()
                    logger.debug(f"Extracted history from rewards_prompt (direct)")
            except Exception as e:
                logger.warning(f"Failed to extract history from rewards_prompt: {e}")

        # Fallback: Build history from database (EnvironmentProfile + AgentProfile) + episode.messages
        # This reconstructs the global background that was used by TerminalEvaluator during simulation
        if not history and episode.messages:
            logger.info(
                f"Building history from database profiles (no rewards_prompt available)"
            )
            history_lines = []

            try:
                # Get EnvironmentProfile and AgentProfiles from database
                env_profile = EnvironmentProfile.get(episode.environment)
                agent_profiles = [AgentProfile.get(agent_pk) for agent_pk in episode.agents]

                # Reconstruct the global ScriptBackground (same as self.background in simulation)
                # Use render_text_for_environment to get complete information
                def render_for_env(text: str) -> str:
                    return XMLRenderer()(text, RenderContext(viewer="environment"))

                # Get agent names from profiles
                p1_name = agent_profiles[0].first_name + " " + agent_profiles[0].last_name
                p2_name = agent_profiles[1].first_name + " " + agent_profiles[1].last_name

                # Get bios using the same logic as simulation
                p1_bio = get_bio(env_profile.relationship, agent_profiles[0], agent_id=0)
                p2_bio = get_bio(env_profile.relationship, agent_profiles[1], agent_id=1)

                # Get goals
                p1_goal = env_profile.agent_goals[0] if len(env_profile.agent_goals) > 0 else ""
                p2_goal = env_profile.agent_goals[1] if len(env_profile.agent_goals) > 1 else ""

                # Render for environment (complete view)
                background = ScriptBackground(
                    scenario=render_for_env(env_profile.scenario),
                    p1_name=p1_name,
                    p2_name=p2_name,
                    p1_background=render_for_env(p1_bio),
                    p2_background=render_for_env(p2_bio),
                    p1_goal=render_for_env(p1_goal),
                    p2_goal=render_for_env(p2_goal),
                )

                # First line is the complete background
                history_lines.append(background.to_natural_language())

                # Then add the conversation turns (skip turn 0 which is the initial observation)
                #
                # IMPORTANT: episode.messages format differs from inbox format!
                # - inbox: ("Environment", SimpleMessage("Turn #X")) + (agent, AgentAction)
                # - episode.messages: ("Environment", receiver, "Turn #X: Agent said: ...") + (agent, "Environment", "said: ...")
                #
                # We need to reconstruct the inbox format:
                # - "Turn #X" (from Environment)
                # - "Agent said: ..." (from agent actions)
                #
                for turn_idx, turn in enumerate(episode.messages):
                    if turn_idx == 0:
                        # Skip initial observations (we already have the background)
                        continue

                    # Add turn marker
                    history_lines.append(f"Turn #{turn_idx}")

                    # Add messages（与 evaluate_episodes_multi_run.py:558-569 保持一致）
                    for sender, receiver, message in turn:
                        # 只处理发给 Environment 的消息，避免重复添加 Environment 广播的消息
                        if receiver == "Environment" and sender != "Environment":
                            if "did nothing" not in message:
                                if "said:" in message:
                                    history_lines.append(f"{sender} {message}")
                                else:
                                    history_lines.append(f"{sender}: {message}")

                logger.info(f"Successfully reconstructed history from database profiles")

            except Exception as e:
                logger.warning(f"Failed to reconstruct from database: {e}. Using raw messages.")
                # Fallback to raw messages (may have incomplete goals)
                # Use the same format: "Turn #X" + agent actions
                for turn_idx, turn in enumerate(episode.messages):
                    if turn_idx == 0:
                        # For turn 0, include the first Environment message as background
                        for sender, receiver, message in turn:
                            if sender == "Environment":
                                history_lines.append(message)
                                break  # Only include one
                        continue

                    # Add turn marker
                    history_lines.append(f"Turn #{turn_idx}")

                    # Add messages（与 evaluate_episodes_multi_run.py:558-569 保持一致）
                    for sender, receiver, message in turn:
                        # 只处理发给 Environment 的消息，避免重复添加 Environment 广播的消息
                        if receiver == "Environment" and sender != "Environment":
                            if "did nothing" not in message:
                                if "said:" in message:
                                    history_lines.append(f"{sender} {message}")
                                else:
                                    history_lines.append(f"{sender}: {message}")

            history = "\n".join(history_lines)

        if not history:
            logger.warning(f"No history available for episode {episode.pk[:16]}")
            self.failure_count += 1
            return None, None

        # Log the constructed history for verification
        logger.info(f"\n{'='*80}\n[Episode {episode.pk[:16]}] Constructed History:\n{'-'*80}\n{history}\n{'='*80}")

        # 执行评估（参考 evaluate_episodes_multi_run.py 的简洁写法）
        for attempt in range(max_retries):
            try:
                response_list = await self.evaluator.__acall__(
                    turn_number=-1,
                    history=history,
                    messages=None,
                    temperature=0.0,
                )

                if not response_list:
                    logger.warning(f"Attempt {attempt + 1}: Empty response for episode {episode.pk[:16]}")
                    continue

                # 聚合评估结果
                response = unweighted_aggregate_evaluate(response_list)

                if response.p1_rate and response.p2_rate:
                    self.success_count += 1
                    logger.info(f"✓ Successfully re-evaluated episode {episode.pk[:16]}")
                    return response.p1_rate, response.p2_rate
                else:
                    logger.warning(f"Attempt {attempt + 1}: Incomplete ratings for {episode.pk[:16]}")

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for episode {episode.pk[:16]}: {e}")
                if attempt == max_retries - 1:
                    traceback.print_exc()
                await asyncio.sleep(1)  # Wait before retry

        self.failure_count += 1
        logger.error(f"All {max_retries} retries failed for episode {episode.pk[:16]}")
        return None, None
    
    def get_stats(self) -> dict[str, int]:
        """Get evaluation statistics."""
        return {
            "total_evaluated": self.evaluation_count,
            "successful": self.success_count,
            "failed": self.failure_count,
        }


class DatabaseUpdater:
    """Update episode rewards in the database."""
    
    def __init__(self):
        self.update_count = 0
        self.updates = []
    
    def update_episode_rewards(
        self,
        episode: EpisodeLog,
        p1_rating: tuple[float, dict],
        p2_rating: tuple[float, dict],
        dry_run: bool = True
    ) -> bool:
        """
        Update episode rewards in the database.
        
        Args:
            episode: The episode to update
            p1_rating: New P1 rating (score, breakdown)
            p2_rating: New P2 rating (score, breakdown)
            dry_run: If True, don't actually update (just log)
        
        Returns:
            True if update was successful
        """
        # Store update information
        update_info = {
            "pk": episode.pk,
            "old_p1": episode.rewards[0] if len(episode.rewards) > 0 else None,
            "old_p2": episode.rewards[1] if len(episode.rewards) > 1 else None,
            "new_p1": p1_rating,
            "new_p2": p2_rating,
        }
        self.updates.append(update_info)
        
        if dry_run:
            logger.info(f"[DRY RUN] Would update episode {episode.pk[:16]}")
            return True
        
        try:
            # Update the episode rewards
            episode.rewards = [p1_rating, p2_rating]
            episode.save()
            self.update_count += 1
            logger.info(f"✓ Updated episode {episode.pk[:16]}")
            return True
        except Exception as e:
            logger.error(f"Failed to update episode {episode.pk[:16]}: {e}")
            return False
    
    def save_update_log(self, output_path: Path):
        """Save update log to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.updates, f, indent=2)
        logger.info(f"Update log saved to: {output_path}")


def update_csv_tags(
    csv_path: Path,
    tag_updates: dict[str, str],
    dry_run: bool = False
) -> bool:
    """
    Update tags in a CSV file.

    For each tag in the CSV that exists in tag_updates, replace it with the new tag.
    Tags not in tag_updates are left unchanged.

    Args:
        csv_path: Path to the CSV file
        tag_updates: Mapping from old_tag -> new_tag
        dry_run: If True, don't actually write (just show what would be done)

    Returns:
        True if successful, False otherwise
    """
    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return False

    if not tag_updates:
        logger.info("No tag updates to apply to CSV")
        return True

    try:
        # Read original content
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        # Update tags
        updated_lines = []
        changes_made = []
        for line in lines:
            original_line = line.strip()
            if original_line and original_line in tag_updates:
                new_tag = tag_updates[original_line]
                updated_lines.append(new_tag + '\n')
                changes_made.append((original_line, new_tag))
            else:
                updated_lines.append(line)

        if not changes_made:
            console.print(f"[yellow]No tags in CSV needed updating[/]")
            return True

        # Show changes
        console.print(f"\n[bold cyan]CSV Tag Updates ({len(changes_made)} changes):[/]")
        for old_tag, new_tag in changes_made:
            console.print(f"  [red]{old_tag}[/] → [green]{new_tag}[/]")

        if dry_run:
            console.print(f"[yellow][DRY RUN] Would update {len(changes_made)} tags in {csv_path}[/]")
            return True

        # Write updated content
        with open(csv_path, 'w') as f:
            f.writelines(updated_lines)

        console.print(f"[green]✓ Updated {len(changes_made)} tags in {csv_path}[/]")
        logger.info(f"Updated {len(changes_made)} tags in CSV: {csv_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to update CSV file: {e}")
        console.print(f"[red]✗ Failed to update CSV: {e}[/]")
        return False


def copy_output_directory(
    original_tag: str,
    new_tag: str,
    dry_run: bool = False
) -> bool:
    """
    Copy the experiment output directory from original_tag to new_tag.

    Args:
        original_tag: The source tag
        new_tag: The destination tag
        dry_run: If True, don't actually copy (just simulate)

    Returns:
        True if successful, False otherwise
    """
    import shutil

    script_dir = Path(__file__).parent
    outputs_dir = script_dir / "outputs"

    source_dir = outputs_dir / original_tag
    target_dir = outputs_dir / new_tag

    if not source_dir.exists():
        logger.warning(f"Source output directory not found: {source_dir}")
        console.print(f"[yellow]⚠ No output directory found for '{original_tag}'[/]")
        return False

    if target_dir.exists():
        logger.info(f"Target output directory already exists: {target_dir}")
        console.print(f"[yellow]⚠ Output directory '{new_tag}' already exists, skipping copy[/]")
        return True

    if dry_run:
        # Calculate size for display
        total_size = sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        console.print(f"[yellow][DRY RUN] Would copy output directory ({size_mb:.1f} MB): {source_dir} → {target_dir}[/]")
        return True

    try:
        console.print(f"[cyan]Copying output directory: {original_tag} → {new_tag}[/]")
        shutil.copytree(source_dir, target_dir)

        # Calculate size for display
        total_size = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        console.print(f"[green]✓ Copied output directory ({size_mb:.1f} MB)[/]")
        logger.info(f"Copied output directory from '{original_tag}' to '{new_tag}'")
        return True

    except Exception as e:
        logger.error(f"Failed to copy output directory: {e}")
        console.print(f"[red]✗ Failed to copy output directory: {e}[/]")
        return False


def copy_episodes_to_new_tag(
    original_tag: str,
    new_tag: str,
    dry_run: bool = False
) -> tuple[list[EpisodeLog], dict[str, str]]:
    """
    Copy all episodes from original_tag to a new_tag, including output directory.

    This creates a copy of each episode with the new tag, preserving all data.
    The new episodes get new PKs but maintain references to the original episodes.
    Also copies the entire output directory (logs, results, tensorboard, etc.).

    Args:
        original_tag: The source tag to copy from
        new_tag: The destination tag to copy to
        dry_run: If True, don't actually save (just simulate)

    Returns:
        Tuple of (list of new episodes, mapping from old PK to new PK)
    """
    # Check if new tag already exists
    existing = list(EpisodeLog.find(EpisodeLog.tag == new_tag).all())
    if existing:
        logger.warning(f"Tag '{new_tag}' already exists with {len(existing)} episodes!")
        console.print(f"[yellow]⚠ Tag '{new_tag}' already has {len(existing)} episodes. Using existing data.[/]")
        # Return existing episodes with identity mapping
        pk_mapping = {ep.pk: ep.pk for ep in existing}
        return existing, pk_mapping

    # Get all episodes from original tag
    original_episodes = list(EpisodeLog.find(EpisodeLog.tag == original_tag).all())
    if not original_episodes:
        logger.error(f"No episodes found for original tag: {original_tag}")
        return [], {}

    console.print(f"\n[bold cyan]Copying {len(original_episodes)} episodes from '{original_tag}' to '{new_tag}'...[/]")

    # Step 1: Copy output directory first
    console.print(f"\n[bold]1. Copying output directory...[/]")
    copy_output_directory(original_tag, new_tag, dry_run=dry_run)

    # Step 2: Copy database episodes
    console.print(f"\n[bold]2. Copying database episodes...[/]")
    new_episodes = []
    pk_mapping = {}  # old_pk -> new_pk

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Copying episodes...", total=len(original_episodes))

        for original_ep in original_episodes:
            # Create a new episode with the same data but new tag
            # We need to copy all fields except pk (which will be auto-generated)
            # Note: render_for_humans is a method, not a field, so we don't copy it
            new_ep = EpisodeLog(
                environment=original_ep.environment,
                agents=original_ep.agents,
                tag=new_tag,  # Use the new tag
                models=original_ep.models if hasattr(original_ep, 'models') else None,
                messages=original_ep.messages,
                reasoning=original_ep.reasoning if hasattr(original_ep, 'reasoning') else "",
                rewards=original_ep.rewards,  # Copy original rewards (will be updated later)
                rewards_prompt=original_ep.rewards_prompt if hasattr(original_ep, 'rewards_prompt') else "",
            )

            if not dry_run:
                new_ep.save()
                pk_mapping[original_ep.pk] = new_ep.pk
                new_episodes.append(new_ep)
            else:
                # For dry run, just create a fake mapping
                pk_mapping[original_ep.pk] = f"DRY_RUN_{original_ep.pk[:16]}"
                new_episodes.append(new_ep)

            progress.advance(task)

    if dry_run:
        console.print(f"[yellow][DRY RUN] Would copy {len(original_episodes)} episodes to tag '{new_tag}'[/]")
    else:
        console.print(f"[green]✓ Successfully copied {len(new_episodes)} episodes to tag '{new_tag}'[/]")

    logger.info(f"Copied {len(new_episodes)} episodes from '{original_tag}' to '{new_tag}'")

    return new_episodes, pk_mapping


def display_quality_report(status: dict[str, Any]):
    """Display reward quality report."""
    if "error" in status:
        console.print(f"[red]Error: {status['error']}[/]")
        return
    
    stats = status["stats"]
    
    # Create report table
    table = Table(title="Reward Quality Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="yellow")
    
    table.add_row("Tag", stats["tag"])
    table.add_row("Total Episodes", str(stats["total_episodes"]))
    table.add_row("Healthy (Tuple)", f"{stats['healthy_episodes']} ({100 - stats['degradation_pct']:.1f}%)", style="green")
    table.add_row("Degraded (Float)", f"{stats['degraded_episodes']} ({stats['degradation_pct']:.1f}%)", 
                 style="red" if stats['degradation_pct'] > 20 else "yellow" if stats['degradation_pct'] > 0 else "green")
    
    console.print(table)
    
    # Show status
    if status["needs_fix"]:
        console.print(f"\n[yellow]⚠ Found {status['float_count']} episodes that need fixing[/]")
    else:
        console.print("\n[green]✓ All episodes have complete reward data![/]")


import aiohttp
from sotopia.generation_utils.generate import enable_llm_call_logging, disable_llm_call_logging

async def calculate_cost_from_log(log_file: Path) -> dict[str, Any]:
    """
    Calculate cost from LLM call log file by querying OpenRouter API.
    
    Returns dict with total_cost, total_tokens, etc.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not found, skipping cost calculation")
        return {}
    
    if not log_file.exists():
        logger.warning(f"Log file not found: {log_file}")
        return {}
    
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    if not lines:
        return {}
    
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    processed_count = 0
    
    async def fetch_usage(session: aiohttp.ClientSession, gen_id: str) -> dict | None:
        url = f"https://openrouter.ai/api/v1/generation?id={gen_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {})
        except Exception:
            pass
        return None
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for line in lines:
            try:
                entry = json.loads(line)
                gen_id = entry.get("id")
                if gen_id:
                    tasks.append(fetch_usage(session, gen_id))
            except json.JSONDecodeError:
                pass
        
        # Process in batches
        batch_size = 50
        results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
    
    for data in results:
        if data:
            cost = data.get("total_cost", 0) or 0
            p_tokens = data.get("native_tokens_prompt", 0) or data.get("tokens_prompt", 0) or 0
            c_tokens = data.get("native_tokens_completion", 0) or data.get("tokens_completion", 0) or 0
            
            total_cost += cost
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            processed_count += 1
    
    return {
        "total_cost": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "processed_count": processed_count,
    }


async def process_single_tag(
    original_tag: str,
    new_tag: str,
    args: argparse.Namespace,
    evaluator: EpisodeReEvaluator,
    updater: DatabaseUpdater,
    semaphore: asyncio.Semaphore = None
) -> dict[str, Any]:
    """
    Process a single tag.

    This function:
    1. Checks quality of the original tag
    2. Copies ALL episodes from original_tag to new_tag (preserving original data)
    3. Only updates episodes in the new_tag
    """
    # Check quality of original tag
    checker = RewardQualityChecker()
    status = checker.check_tag(original_tag)

    if "error" in status:
        console.print(f"[red]Error for tag {original_tag}: {status['error']}[/]")
        return {"tag": original_tag, "new_tag": new_tag, "error": status["error"], "needs_fix": False}

    display_quality_report(status)

    # If check-only mode or no fix needed, return early
    if args.check_only or not status["needs_fix"]:
        return {
            "tag": original_tag,
            "new_tag": new_tag,
            "needs_fix": status["needs_fix"],
            "stats": status["stats"],
            "fixed": 0
        }

    # For dry-run mode, we don't copy episodes, just use the original ones for evaluation
    # For update-db mode, we copy ALL episodes to new tag first
    if args.dry_run:
        console.print(f"\n[yellow][DRY RUN] Skipping episode copy - will evaluate original episodes[/]")
        episodes_to_fix = status["float_episodes"]
        pk_mapping = {}  # Empty mapping for dry-run
    else:
        # Step 1: Copy ALL episodes from original_tag to new_tag
        # This preserves the original data and creates a working copy
        console.print(f"\n[bold cyan]Step 1: Copying episodes to new tag '{new_tag}'...[/]")
        new_episodes, pk_mapping = copy_episodes_to_new_tag(
            original_tag=original_tag,
            new_tag=new_tag,
            dry_run=False
        )

        if not new_episodes:
            console.print(f"[red]Failed to copy episodes to new tag[/]")
            return {
                "tag": original_tag,
                "new_tag": new_tag,
                "error": "Failed to copy episodes",
                "needs_fix": True,
                "stats": status["stats"],
                "fixed": 0
            }

        # Step 2: Find episodes that need fixing in the new tag
        # Map original float episodes to their new copies
        original_float_pks = {ep.pk for ep in status["float_episodes"]}

        # pk_mapping maps old_pk -> new_pk, so we need to find new episodes whose source was in float_episodes
        # Reverse the mapping: new_pk -> old_pk
        reverse_mapping = {v: k for k, v in pk_mapping.items()}
        episodes_to_fix = [ep for ep in new_episodes if reverse_mapping.get(ep.pk, ep.pk) in original_float_pks]

        console.print(f"\n[bold]Found {len(episodes_to_fix)} episodes needing re-evaluation in new tag '{new_tag}'[/]")

    # Apply filters if specified
    if args.episode_pks:
        # Filter by original PKs (user specifies original episode PKs)
        target_pks = set(args.episode_pks.split(','))
        if args.dry_run:
            # In dry-run mode, episodes_to_fix are already the original episodes
            episodes_to_fix = [ep for ep in episodes_to_fix if ep.pk in target_pks]
        else:
            # Find which new episodes correspond to these original PKs
            reverse_mapping = {v: k for k, v in pk_mapping.items()}
            episodes_to_fix = [ep for ep in episodes_to_fix if reverse_mapping.get(ep.pk, ep.pk) in target_pks]
        console.print(f"[yellow]Filtering to {len(episodes_to_fix)} specified episodes[/]")

    # Limit if max_episodes specified
    if args.max_episodes and len(episodes_to_fix) > args.max_episodes:
        episodes_to_fix = episodes_to_fix[:args.max_episodes]
        console.print(f"[yellow]Limiting to {args.max_episodes} episodes[/]")

    if not episodes_to_fix:
        return {
            "tag": original_tag,
            "new_tag": new_tag,
            "needs_fix": False,
            "stats": status["stats"],
            "fixed": 0
        }

    # Setup LLM logging (use new_tag for output directory)
    script_dir = Path(__file__).parent
    outputs_dir = script_dir / "outputs"
    tag_output_dir = outputs_dir / new_tag / "evaluation"
    if outputs_dir.exists():
        tag_output_dir.mkdir(parents=True, exist_ok=True)
        llm_log_file = tag_output_dir / "fix_llm_calls.jsonl"
    else:
        llm_log_file = Path(f"{new_tag}_fix_llm_calls.jsonl")

    enable_llm_call_logging(llm_log_file, experiment_tag=new_tag)

    # Re-evaluate episodes
    if args.dry_run:
        console.print(f"\n[bold cyan]Re-evaluating {len(episodes_to_fix)} episodes (dry-run, no DB updates)...[/]")
    else:
        console.print(f"\n[bold cyan]Step 2: Re-evaluating {len(episodes_to_fix)} episodes in '{new_tag}'...[/]")

    async def process_episode(episode, progress, task):
        """Process a single episode with semaphore control."""
        async with semaphore if semaphore else asyncio.Semaphore(1):
            progress.update(task, description=f"[cyan]Evaluating {episode.pk[:16]}...")

            p1_rating, p2_rating = await evaluator.re_evaluate_episode(episode)

            success = False
            if p1_rating and p2_rating:
                # Update database if requested (always update the new tag's episode)
                if args.update_db:
                    success = updater.update_episode_rewards(episode, p1_rating, p2_rating, dry_run=args.dry_run)
                elif args.dry_run:
                    success = updater.update_episode_rewards(episode, p1_rating, p2_rating, dry_run=True)

            progress.advance(task)
            return success
    
    # Process episodes with progress bar
    fixed_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Processing {new_tag}...", total=len(episodes_to_fix))

        # Process episodes concurrently
        try:
            results = await asyncio.gather(
                *[process_episode(ep, progress, task) for ep in episodes_to_fix],
                return_exceptions=True
            )

            # Count successes
            for result in results:
                if result is True:
                    fixed_count += 1
                elif isinstance(result, Exception):
                    logger.error(f"Episode processing failed with exception: {result}")
        finally:
            # Disable logging
            disable_llm_call_logging()

    # Calculate costs
    console.print("\n[bold cyan]Calculating API costs...[/]")
    cost_info = await calculate_cost_from_log(llm_log_file)
    if cost_info:
        console.print(Panel(f"""[bold green]Cost Summary[/]
Total Cost: [yellow]${cost_info['total_cost']:.6f}[/]
Prompt Tokens: {cost_info['total_prompt_tokens']:,}
Completion Tokens: {cost_info['total_completion_tokens']:,}
Total Tokens: {cost_info['total_tokens']:,}
API Calls: {cost_info['processed_count']}"""))

        # Save cost info
        if tag_output_dir.exists():
            cost_file = tag_output_dir / "fix_cost_info.json"
            with open(cost_file, "w") as f:
                json.dump(cost_info, f, indent=2)
            console.print(f"[bold green]Cost info saved to:[/] {cost_file}")

    # Save update log
    if updater.updates:
        # Determine output path based on new_tag
        # Default layout: experiments/dynamic_observation/outputs/<new_tag>/evaluation/reward_fix_log.json
        script_dir = Path(__file__).parent
        outputs_dir = script_dir / "outputs"

        if outputs_dir.exists():
            tag_output_dir = outputs_dir / new_tag / "evaluation"
            tag_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = tag_output_dir / "reward_fix_log.json"
        else:
            # Fallback to current dir or args.output
            output_path = Path(args.output)

        updater.save_update_log(output_path)

    # Log summary
    if args.dry_run:
        console.print(f"\n[bold yellow]Summary (DRY RUN):[/]")
        console.print(f"  Original tag: [cyan]{original_tag}[/]")
        console.print(f"  Target new tag: [green]{new_tag}[/] (not created in dry-run)")
        console.print(f"  Episodes evaluated: {fixed_count}/{len(episodes_to_fix)}")
    else:
        console.print(f"\n[bold green]Summary for '{original_tag}' → '{new_tag}':[/]")
        console.print(f"  Original tag: [cyan]{original_tag}[/] (preserved, unchanged)")
        console.print(f"  New tag: [green]{new_tag}[/] (contains fixed data)")
        console.print(f"  Episodes copied: {len(pk_mapping)}")
        console.print(f"  Episodes fixed: {fixed_count}/{len(episodes_to_fix)}")

    return {
        "tag": original_tag,
        "new_tag": new_tag,
        "needs_fix": True,
        "stats": status["stats"],
        "fixed": fixed_count,
        "total_to_fix": len(episodes_to_fix)
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Fix reward quality issues for experiment episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check reward quality (no fixes)
    python fix_reward_quality.py --tag bandit_exp_tag --check-only

    # Batch check multiple tags
    python fix_reward_quality.py --tags tag1 tag2 tag3 --check-only

    # Dry run (evaluate but don't update)
    python fix_reward_quality.py --tag bandit_exp_tag --dry-run

    # Fix and update database (creates new tag with "_fixed" suffix)
    python fix_reward_quality.py --tag bandit_exp_tag --update-db

    # Fix with custom new tag name
    python fix_reward_quality.py --tag bandit_exp_tag --new-tag my_custom_fixed_tag --update-db

    # Batch fix multiple tags (each gets "_fixed" suffix)
    python fix_reward_quality.py --tags tag1 tag2 --update-db

    # Fix specific episodes
    python fix_reward_quality.py --tag bandit_exp_tag --episode-pks pk1,pk2 --update-db
"""
    )
    parser.add_argument("--tag", help="Single experiment tag to check/fix")
    parser.add_argument("--tags", nargs="+", help="Multiple experiment tags to check/fix (batch mode)")
    parser.add_argument("--csv", help="CSV file containing experiment tags (one per line)")
    parser.add_argument("--new-tag", help="Custom new tag name for fixed data (default: original_tag + '_fixed'). Only works with --tag, not --tags")
    parser.add_argument("--check-only", action="store_true", help="Only check quality, don't fix")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate but don't update database")
    parser.add_argument("--update-db", action="store_true", help="Update database with new ratings")
    parser.add_argument("--model", default="openrouter/openai/gpt-4o", help="Model to use for re-evaluation (default: openrouter/openai/gpt-4o)")
    parser.add_argument("--episode-pks", help="Comma-separated list of episode PKs to fix (optional)")
    parser.add_argument("--output", default="reward_fix_log.json", help="Output file for update log")
    parser.add_argument("--max-episodes", type=int, help="Maximum number of episodes to fix per tag (for testing)")
    parser.add_argument("--concurrency", type=int, default=30, help="Number of concurrent episodes to evaluate (default: 1)")

    args = parser.parse_args()

    # Handle --csv option
    if args.csv:
        import pandas as pd
        csv_path = Path(args.csv)
        if not csv_path.exists():
            console.print(f"[red]Error: CSV file not found: {args.csv}[/]")
            return
        try:
            csv_df = pd.read_csv(csv_path, header=None)
            args.tags = csv_df.iloc[:, 0].dropna().tolist()
            console.print(f"[green]Loaded {len(args.tags)} tags from CSV: {args.csv}[/]\n")
        except Exception as e:
            console.print(f"[red]Error reading CSV file: {e}[/]")
            return

    # Must have either --tag, --tags, or --csv
    if not args.tag and not args.tags:
        console.print("[red]Error: Must specify either --tag, --tags, or --csv[/]")
        parser.print_help()
        return

    # Cannot have both --tag and --tags
    if args.tag and args.tags:
        console.print("[red]Error: Cannot use both --tag and --tags[/]")
        return

    # Cannot have both --check-only and --update-db
    if args.check_only and args.update_db:
        console.print("[red]Error: --check-only and --update-db are mutually exclusive[/]")
        return

    # --new-tag only works with --tag (single tag mode)
    if args.new_tag and args.tags:
        console.print("[red]Error: --new-tag can only be used with --tag, not --tags[/]")
        return

    # Get list of tags to process
    tags_to_process = [args.tag] if args.tag else args.tags

    # Generate new tag names for each original tag
    # For batch mode (--tags), each tag gets "_fixed" suffix
    # For single tag mode, use --new-tag if provided, otherwise add "_fixed" suffix
    new_tag_mapping = {}  # original_tag -> new_tag
    for tag in tags_to_process:
        if args.new_tag and args.tag:
            new_tag_mapping[tag] = args.new_tag
        else:
            new_tag_mapping[tag] = f"{tag}_fixed"

    console.print(Panel(
        f"[bold cyan]Reward Quality Fixer[/]\n"
        f"[dim]Processing {len(tags_to_process)} tag(s)[/]\n"
        f"[dim]Concurrency: {args.concurrency}[/]\n"
        f"[dim]Original data will be preserved (copies made to new tags)[/]",
        expand=False
    ))

    # Display tag mapping
    if not args.check_only:
        console.print("\n[bold]Tag Mapping (Original → New):[/]")
        for orig_tag, new_tag in new_tag_mapping.items():
            console.print(f"  [cyan]{orig_tag}[/] → [green]{new_tag}[/]")

    # Initialize evaluator and updater (shared across all tags)
    evaluator = EpisodeReEvaluator(model_name=args.model) if not args.check_only else None
    updater = DatabaseUpdater() if not args.check_only else None

    # Initialize semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.concurrency) if not args.check_only else None

    # Process all tags
    results = []
    for i, tag in enumerate(tags_to_process, 1):
        new_tag = new_tag_mapping[tag]
        console.print(f"\n[bold cyan]═══ Processing tag {i}/{len(tags_to_process)}: {tag} → {new_tag} ═══[/]\n")

        result = await process_single_tag(tag, new_tag, args, evaluator, updater, semaphore)
        results.append(result)
    
    # Display final summary for batch mode
    if len(tags_to_process) > 1:
        console.print("\n[bold cyan]═══ Batch Summary ═══[/]\n")

        summary_table = Table(title="Batch Processing Summary")
        summary_table.add_column("Original Tag", style="cyan", no_wrap=False)
        if not args.check_only:
            summary_table.add_column("New Tag", style="green", no_wrap=False)
        summary_table.add_column("Episodes", justify="right")
        summary_table.add_column("Degraded", justify="right")
        summary_table.add_column("Status", justify="center")
        if not args.check_only:
            summary_table.add_column("Fixed", justify="right", style="green")

        total_episodes = 0
        total_degraded = 0
        total_fixed = 0

        for result in results:
            if "error" in result:
                row_data = [result["tag"][:40]]
                if not args.check_only:
                    row_data.append(result.get("new_tag", "-")[:40])
                row_data.extend(["-", "-", "[red]Error[/]"])
                summary_table.add_row(*row_data)
                continue

            stats = result["stats"]
            total_episodes += stats["total_episodes"]
            total_degraded += stats["degraded_episodes"]

            status = "✓" if not result["needs_fix"] else f"⚠ {stats['degradation_pct']:.1f}%"
            style = "green" if not result["needs_fix"] else "yellow" if stats['degradation_pct'] < 20 else "red"

            row_data = [result["tag"][:40]]
            if not args.check_only:
                row_data.append(result.get("new_tag", "-")[:40])
            row_data.extend([
                str(stats["total_episodes"]),
                str(stats["degraded_episodes"]),
                f"[{style}]{status}[/]"
            ])

            if not args.check_only:
                total_fixed += result.get("fixed", 0)
                row_data.append(str(result.get("fixed", 0)))

            summary_table.add_row(*row_data)
        
        console.print(summary_table)
        
        # Overall stats
        console.print(f"\n[bold]Overall Statistics:[/]")
        console.print(f"  Total Tags: {len(results)}")
        console.print(f"  Total Episodes: {total_episodes}")
        console.print(f"  Total Degraded: {total_degraded} ({total_degraded/total_episodes*100:.1f}%)" if total_episodes > 0 else "  Total Degraded: 0")
        
        if not args.check_only and evaluator:
            eval_stats = evaluator.get_stats()
            console.print(f"  Successfully Re-evaluated: {eval_stats['successful']}")
            console.print(f"  Failed Re-evaluations: {eval_stats['failed']}")
            if args.update_db or args.dry_run:
                console.print(f"  Total Fixed: {total_fixed}")
    
    # Save update log if any updates were made
    if updater and len(updater.updates) > 0:
        output_path = Path(args.output)
        updater.save_update_log(output_path)
    
    # Display evaluation stats for single tag mode
    if len(tags_to_process) == 1 and evaluator:
        stats = evaluator.get_stats()
        result_table = Table(title="Re-evaluation Results")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Count", justify="right", style="yellow")
        
        result_table.add_row("Total Evaluated", str(stats["total_evaluated"]))
        result_table.add_row("Successful", str(stats["successful"]), style="green")
        result_table.add_row("Failed", str(stats["failed"]), style="red" if stats["failed"] > 0 else "green")
        
        if args.update_db and updater:
            result_table.add_row("Database Updates", str(updater.update_count), style="green")
        
        console.print(f"\n{result_table}")
    
    # Update CSV file if --csv was used and --update-db was specified
    # Only update tags that were successfully fixed (needs_fix=True and fixed > 0)
    if args.csv and not args.check_only:
        csv_path = Path(args.csv)
        tag_updates = {}
        for result in results:
            if "error" not in result and result.get("needs_fix", False):
                # Only update if the tag needed fixing (had degraded episodes)
                original_tag = result["tag"]
                new_tag = result["new_tag"]
                # Include in updates if fix was attempted (even if not all succeeded)
                if result.get("fixed", 0) > 0 or args.update_db:
                    tag_updates[original_tag] = new_tag

        if tag_updates:
            console.print(f"\n[bold cyan]═══ Updating CSV File ═══[/]")
            update_csv_tags(csv_path, tag_updates, dry_run=args.dry_run)
        else:
            console.print(f"\n[green]✓ No CSV updates needed - all tags already satisfy quality requirements[/]")

    # Final message
    if args.dry_run:
        console.print("\n[yellow]DRY RUN: No database updates were made[/]")
        console.print("[dim]Re-run with --update-db to apply changes[/]")
    elif args.update_db and updater:
        console.print(f"\n[green]✓ Successfully updated {updater.update_count} episodes![/]")
    elif not args.check_only:
        console.print("\n[yellow]Evaluation complete. Use --update-db to save changes.[/]")


if __name__ == "__main__":
    asyncio.run(main())
