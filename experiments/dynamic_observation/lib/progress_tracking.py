"""Progress tracking utilities for batch simulation runs."""


class BatchProgressTracker:
    """Track progress of multiple episodes running in parallel."""

    def __init__(self) -> None:
        self.episodes: dict[str, dict] = {}  # scenario_id -> {turn, max_turn, agent_names}
        self.completed = 0
        self.total = 0
        self.successes = 0
        self.errors = 0

    def update(self, scenario_id: str, turn: int, max_turn: int) -> None:
        if scenario_id in self.episodes:
            self.episodes[scenario_id]["turn"] = turn
            self.episodes[scenario_id]["max_turn"] = max_turn

    def register(self, scenario_id: str, max_turn: int, agent_names: str) -> None:
        self.episodes[scenario_id] = {"turn": 0, "max_turn": max_turn, "agent_names": agent_names}

    def complete(self, scenario_id: str, success: bool = True) -> None:
        if scenario_id in self.episodes:
            del self.episodes[scenario_id]
        self.completed += 1
        if success:
            self.successes += 1
        else:
            self.errors += 1

    def get_status(self) -> str:
        """Get current status description for progress bar."""
        if not self.episodes:
            return f"[cyan]Completed {self.completed}/{self.total} episodes"

        # Show running episodes (max 3)
        running = []
        for scenario_id, info in list(self.episodes.items())[:3]:
            name = info.get('agent_names', scenario_id[:10])
            running.append(f"{name[:15]}:T{info['turn']}/{info['max_turn']}")

        status = " | ".join(running)
        if len(self.episodes) > 3:
            status += f" (+{len(self.episodes) - 3} more)"

        return f"[cyan]{self.completed}/{self.total}[/] | {status}"
