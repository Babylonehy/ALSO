"""Display and output formatting utilities."""

import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# Score ranges for reward normalization
SCORE_RANGES = [
    (0, 10),    # believability
    (-5, 5),    # relationship
    (0, 10),    # knowledge
    (-10, 0),   # secret
    (-10, 0),   # social_rules
    (-5, 5),    # financial_and_material_benefits
    (0, 10),    # goal
]
MIN_OVERALL = sum(r[0] for r in SCORE_RANGES) / len(SCORE_RANGES)  # -30/7 ≈ -4.29
MAX_OVERALL = sum(r[1] for r in SCORE_RANGES) / len(SCORE_RANGES)  # 30/7 ≈ 4.29


def format_duration(seconds: float) -> str:
    """
    Format time duration to HH:MM:SS or MM:SS.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def denormalize_reward(normalized: float) -> float:
    """将归一化分数转回原始范围"""
    return normalized * (MAX_OVERALL - MIN_OVERALL) + MIN_OVERALL


def normalize_reward(original: float) -> float:
    """将原始分数归一化到 0-1 范围"""
    return (original - MIN_OVERALL) / (MAX_OVERALL - MIN_OVERALL)


def display_summary(summary: dict, console: Console) -> None:
    """
    Display a nice summary of the simulation results.

    Args:
        summary: Simulation summary dictionary
        console: Rich console for output
    """
    console.print("\n")
    console.print(Panel("[bold]Simulation Summary[/]", expand=False))

    # Basic stats
    console.print(f"[cyan]Scenario ID:[/] {summary['scenario_id']}")
    console.print(f"[cyan]Optimization Mode:[/] {summary['optimize_mode']}")
    alt_updates = summary.get("alternate_optimization", False)
    console.print(f"[cyan]Alternate Updates:[/] {'Yes' if alt_updates else 'No'}")
    console.print(f"[cyan]Total Turns:[/] {summary['total_turns']}")
    console.print(f"[cyan]Duration:[/] {summary.get('duration_formatted', 'N/A')}")

    # Average rewards - 混合了归一化和原始分数，直接显示
    p1_avg = summary['p1_avg_reward']
    p2_avg = summary['p2_avg_reward']

    console.print(f"\n[bold]Average Rewards:[/]")
    console.print(f"  {summary.get('p1_name', 'P1')}: {p1_avg:.2f}")
    console.print(f"  {summary.get('p2_name', 'P2')}: {p2_avg:.2f}")

    # Determine score column name based on bandit type
    bandit_type = summary.get("bandit_type", "exp3")
    score_col_name = {
        "linucb": "UCB Value",
        "neural_ucb": "UCB Value",
        "exp3": "Cumulative Score",
        "adversarial": "Cumulative Score",
        "prompt_breeder": "Fitness",
    }.get(bandit_type, "Score")

    # 获取最后一轮的 turn 号
    final_turn = summary.get("total_turns", 0)

    # P1 selection history
    if summary.get("p1_bandit"):
        table_p1 = Table(title=f"P1 ({summary.get('p1_name', 'Agent1')}) Selection History")
        table_p1.add_column("Turn", style="cyan")
        table_p1.add_column("Arm", style="green")
        table_p1.add_column("Reward (orig|norm)", style="yellow")
        table_p1.add_column(score_col_name, style="blue")

        for sel in summary["p1_bandit"].get("selections", []):
            reward = sel.get('reward', 0.0)
            turn = sel["turn"]

            if turn == final_turn:
                # 最后一轮是原始分数
                norm = normalize_reward(reward)
                reward_display = f"{reward:.2f} | {norm:.2f}"
            else:
                # 其他轮是归一化分数
                orig = denormalize_reward(reward)
                reward_display = f"{orig:.2f} | {reward:.2f}"

            table_p1.add_row(
                str(turn),
                str(sel["arm_index"]),
                reward_display,
                f"{sel.get('cumulative_score', 0.0):.4f}",
            )
        console.print(table_p1)
    else:
        console.print(f"\n[yellow]P1 ({summary.get('p1_name', 'Agent1')}):[/] No optimization (baseline)")

    # P2 selection history
    if summary.get("p2_bandit"):
        table_p2 = Table(title=f"P2 ({summary.get('p2_name', 'Agent2')}) Selection History")
        table_p2.add_column("Turn", style="cyan")
        table_p2.add_column("Arm", style="green")
        table_p2.add_column("Reward (orig|norm)", style="yellow")
        table_p2.add_column(score_col_name, style="blue")

        for sel in summary["p2_bandit"].get("selections", []):
            reward = sel.get('reward', 0.0)
            turn = sel["turn"]

            if turn == final_turn:
                # 最后一轮是原始分数
                norm = normalize_reward(reward)
                reward_display = f"{reward:.2f} | {norm:.2f}"
            else:
                # 其他轮是归一化分数
                orig = denormalize_reward(reward)
                reward_display = f"{orig:.2f} | {reward:.2f}"

            table_p2.add_row(
                str(turn),
                str(sel["arm_index"]),
                reward_display,
                f"{sel.get('cumulative_score', 0.0):.4f}",
            )
        console.print(table_p2)
    else:
        console.print(f"\n[yellow]P2 ({summary.get('p2_name', 'Agent2')}):[/] No optimization (baseline)")

    # Turn-by-turn rewards table
    table_turns = Table(title="Turn-by-Turn Rewards")
    table_turns.add_column("Turn", style="cyan")
    table_turns.add_column("P1 Arm", style="magenta")
    table_turns.add_column("P1 Reward (orig|norm)", style="yellow")
    table_turns.add_column("P2 Arm", style="magenta")
    table_turns.add_column("P2 Reward (orig|norm)", style="yellow")

    for turn_info in summary.get("turn_rewards", []):
        p1_reward = turn_info['p1_reward']
        p2_reward = turn_info['p2_reward']
        is_final_turn = turn_info.get("terminated", False)

        if is_final_turn:
            # 最后一轮使用 TerminalEvaluator，返回原始分数
            p1_norm = normalize_reward(p1_reward)
            p2_norm = normalize_reward(p2_reward)
            p1_display = f"{p1_reward:.2f} | {p1_norm:.2f}"
            p2_display = f"{p2_reward:.2f} | {p2_norm:.2f}"
        else:
            # 其他轮使用 RewardInTurnEvaluator，返回归一化分数
            p1_orig = denormalize_reward(p1_reward)
            p2_orig = denormalize_reward(p2_reward)
            p1_display = f"{p1_orig:.2f} | {p1_reward:.2f}"
            p2_display = f"{p2_orig:.2f} | {p2_reward:.2f}"

        table_turns.add_row(
            str(turn_info["turn"]),
            str(turn_info.get("p1_arm", 0)),
            p1_display,
            str(turn_info.get("p2_arm", 0)),
            p2_display,
        )
    console.print(table_turns)


def get_output_path(args: argparse.Namespace, experiment_tag: str) -> Path | None:
    """
    Get the output file path based on arguments.

    Args:
        args: Parsed command line arguments
        experiment_tag: Experiment tag for auto-generated filename

    Returns:
        Path to output file, or None if output is disabled
    """
    if args.no_output:
        return None

    # Determine results directory relative to this module's location
    # Assuming lib/ is in experiments/dynamic_observation/
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output == "auto":
        # Auto-generate filename based on experiment tag
        return results_dir / f"{experiment_tag}.json"
    else:
        # Use specified path
        output_path = Path(args.output)
        # If it's just a filename (no directory), put it in results/
        if not output_path.parent.exists() and output_path.parent == Path("."):
            return results_dir / output_path
        return output_path


def print_cost_breakdown(cost_info: dict, console: Console) -> None:
    """
    Print formatted cost breakdown table.

    Args:
        cost_info: Cost information dictionary from calculate_cost_by_model_async
        console: Rich console for output
    """
    by_model = cost_info.get("by_model")
    if not by_model or not isinstance(by_model, dict):
        return

    table = Table(title="Cost Breakdown by Model")
    table.add_column("Model", overflow="fold")
    table.add_column("Calls", justify="right")
    table.add_column("Prompt", justify="right")
    table.add_column("Completion", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Cost", justify="right")

    for model, stats in by_model.items():
        table.add_row(
            str(model),
            str(stats.get("processed_count", 0)),
            str(stats.get("total_prompt_tokens", 0)),
            str(stats.get("total_completion_tokens", 0)),
            str(stats.get("total_tokens", 0)),
            f"${float(stats.get('total_cost', 0) or 0):.6f}",
        )

    console.print(table)
