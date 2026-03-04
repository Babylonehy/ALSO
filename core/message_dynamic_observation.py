import re
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, Field

from sotopia.utils import format_docstring
from sotopia.utils import format_docstring
from sotopia.messages import Observation, ScriptBackground

ActionType = Literal["none", "speak", "non-verbal communication", "action", "leave"]


@dataclass
class UpdateRecord:
    """记录单次更新"""
    turn_number: int
    old_value: str
    new_value: str



class DynamicObservation(Observation):
    """
    支持动态更新场景信息的 Observation，保留更新历史（含轮次）。
    仅用于 turn 0，存储完整的场景组件。
    """
    # 覆盖父类的必需字段，提供默认值
    agent_bio: str = Field(default="", description="the agent's bio")
    social_goal: str = Field(default="", description="the agent's social goal")

    # 标识当前 agent 是 p1 还是 p2
    agent_id: Literal[1, 2] = Field(description="which participant this observation belongs to")

    # 场景组件
    scenario: str = Field(default="", description="the scenario context")
    p1_name: str = Field(default="", description="name of participant 1")
    p2_name: str = Field(default="", description="name of participant 2")
    p1_background: str = Field(default="", description="background of participant 1")
    p2_background: str = Field(default="", description="background of participant 2")
    p1_goal: str = Field(default="", description="goal of participant 1")
    p2_goal: str = Field(default="", description="goal of participant 2")

    # 更新历史（含轮次信息）
    agent_bio_history: list[UpdateRecord] = Field(default_factory=list)
    social_goal_history: list[UpdateRecord] = Field(default_factory=list)
    scenario_history: list[UpdateRecord] = Field(default_factory=list)

    def update_agent_bio(self, new_bio: str, at_turn: int) -> None:
        """更新当前 agent 的 bio 并记录历史"""
        if self.agent_id == 1:
            self.agent_bio_history.append(
                UpdateRecord(turn_number=at_turn, old_value=self.p1_background, new_value=new_bio)
            )
            self.p1_background = new_bio
        else:
            self.agent_bio_history.append(
                UpdateRecord(turn_number=at_turn, old_value=self.p2_background, new_value=new_bio)
            )
            self.p2_background = new_bio
        self._rebuild_last_turn()

    def update_social_goal(self, new_goal: str, at_turn: int) -> None:
        """更新当前 agent 的 goal 并记录历史"""
        if self.agent_id == 1:
            self.social_goal_history.append(
                UpdateRecord(turn_number=at_turn, old_value=self.p1_goal, new_value=new_goal)
            )
            self.p1_goal = new_goal
        else:
            self.social_goal_history.append(
                UpdateRecord(turn_number=at_turn, old_value=self.p2_goal, new_value=new_goal)
            )
            self.p2_goal = new_goal
        self._rebuild_last_turn()

    def update_scenario(self, new_scenario: str, at_turn: int) -> None:
        """更新 scenario 并记录历史"""
        self.scenario_history.append(
            UpdateRecord(turn_number=at_turn, old_value=self.scenario, new_value=new_scenario)
        )
        self.scenario = new_scenario
        self._rebuild_last_turn()

    def _rebuild_last_turn(self) -> None:
        """根据当前组件值重建 last_turn"""
        if self.turn_number == 0:
            self.last_turn = self._build_context()

    def _build_context(self) -> str:
        """复用 ScriptBackground 的格式"""
        background = ScriptBackground(
            scenario=self.scenario,
            p1_name=self.p1_name,
            p2_name=self.p2_name,
            p1_background=self.p1_background,
            p2_background=self.p2_background,
            p1_goal=self.p1_goal,
            p2_goal=self.p2_goal,
        )
        return background.to_natural_language()

    def to_natural_language(self) -> str:
        """保持原始接口"""
        if self.turn_number == 0:
            return f"\n{self.last_turn}\nConversation Starts:\n"
        else:
            return f"Turn #{self.turn_number-1}: {self.last_turn}\n"
