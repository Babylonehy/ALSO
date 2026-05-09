import traceback
import abc
import logging
from collections import defaultdict
from typing import Any, TypeVar

import gin
from pydantic import BaseModel

from sotopia.generation_utils import PydanticOutputParser, agenerate
from sotopia.messages import AgentAction, Message
from sotopia.envs.evaluators import EvaluationForTwoAgents, Evaluator

log = logging.getLogger("evaluators")

T_eval_dim = TypeVar("T_eval_dim", bound=BaseModel)


class TerminalEvaluator(Evaluator):
    """
    终止评估器：在对话结束时进行最终评估。

    与 EpisodeLLMEvaluator 类似，但专门用于作为 terminal_evaluator 使用。
    """

    def __init__(
        self,
        model_name: str,
        response_format_class: type[Any],
    ) -> None:
        self.model_name = model_name
        self.response_format_class = response_format_class
        self.prompt = ""

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError(
            "TerminalEvaluator is not implemented for synchronous evaluation"
        )

    @gin.configurable
    async def __acall__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]] | None,
        history: str = "",
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # filter did nothing
        if not history and messages:
            messages_filtered = [
                (x, y)
                for x, y in messages
                if "did nothing" not in y.to_natural_language()
            ]
            history = "\n".join(
                [
                    (
                        f"{x} {y.to_natural_language()}"
                        if x != "Environment"
                        else y.to_natural_language()
                    )
                    for x, y in messages_filtered
                ]
            )

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response: EvaluationForTwoAgents[Any] = await agenerate(
                    model_name=self.model_name,
                    template="""{history},
                        Based on previous interactions, evaluate how well participants achieve their goals.
                        Please following the format:
                        {format_instructions}
                        IMPORTANT:Do not Reapet the format instructions. Output ONLY the JSON object. Do not include any other text, markdown formatting.
                    """,
                    input_values=dict(history=history),
                    output_parser=PydanticOutputParser[self.response_format_class](  # type: ignore[name-defined]
                        pydantic_object=self.response_format_class
                    ),
                    temperature=temperature,
                    structured_output=self.model_name.startswith("custom/structured"),
                )
                self.prompt = f"Terminal evaluation for history: {history}"

                response_list = []
                for dimension in response.agent_1_evaluation.dict().keys():
                    response_list.append(
                        (
                            "agent_1",
                            (
                                (
                                    dimension,
                                    response.agent_1_evaluation.dict()[dimension][1],
                                ),
                                response.agent_1_evaluation.dict()[dimension][0],
                            ),
                        )
                    )
                    response_list.append(
                        (
                            "agent_2",
                            (
                                (
                                    dimension,
                                    response.agent_2_evaluation.dict()[dimension][1],
                                ),
                                response.agent_2_evaluation.dict()[dimension][0],
                            ),
                        )
                    )
                return response_list
            except Exception as e:
                last_error = e
                log.warning(f"[yellow] Terminal evaluation attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(1.0 * (attempt + 1))  # 指数退避

        # 所有重试都失败
        log.error(f"[red] Failed to generate terminal evaluation after {max_retries} attempts: {last_error}")
        traceback.print_exc()
        return []


# 用于跟踪当前 episode 进度的全局状态
from typing import Callable

# 进度回调类型：(combo_pk, current_turn, max_turn)
ProgressCallback = Callable[[str, int, int], None]


class RewardInTurnEvaluator(Evaluator):
    """
    结合 RuleBasedTerminatedEvaluator 和 EpisodeLLMEvaluator 的功能：
    1. 基于规则判断是否终止
    2. 每轮返回两个 agent 的奖励（分数归一化到 0-1）
    """

    # 各维度的原始分数范围，用于归一化到 0-1
    # 基于 SotopiaDimensions 定义
    SCORE_RANGES: dict[str, tuple[int, int]] = {
        "believability": (0, 10),           # zero_to_ten
        "relationship": (-5, 5),            # minus_five_to_five
        "knowledge": (0, 10),               # zero_to_ten
        "secret": (-10, 0),                 # minus_ten_to_zero
        "social_rules": (-10, 0),           # minus_ten_to_zero
        "financial_and_material_benefits": (-5, 5),  # minus_five_to_five
        "goal": (0, 10),                    # zero_to_ten
    }

    def __init__(
        self,
        model_name: str,
        response_format_class: type[Any],  # EvaluationForTwoAgents[SotopiaDimensions] 等
        max_turn_number: int = 20,
        max_stale_turn: int = 2,
        progress_callback: ProgressCallback | None = None,
        episode_id: str = "",
    ) -> None:
        self.model_name = model_name
        self.response_format_class = response_format_class
        self.max_turn_number = max_turn_number
        self.max_stale_turn = max_stale_turn
        self.prompt = ""
        self.progress_callback = progress_callback
        self.episode_id = episode_id
        self.current_turn = 0  # 追踪当前轮次

    def _normalize_score(self, dimension: str, score: int | float) -> float:
        """
        将分数归一化到 0-1 范围。
        
        Args:
            dimension: 评估维度名称
            score: 原始分数
            
        Returns:
            归一化后的分数 (0-1)
        """
        if dimension not in self.SCORE_RANGES:
            # 未知维度，假设 0-10 范围
            log.warning(f"Unknown dimension '{dimension}', assuming 0-10 range")
            return float(score) / 10.0
        
        min_score, max_score = self.SCORE_RANGES[dimension]
        # 归一化: (score - min) / (max - min)
        normalized = (float(score) - min_score) / (max_score - min_score)
        # 确保在 0-1 范围内
        return max(0.0, min(1.0, normalized))

    def _check_termination(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> tuple[bool, str]:
        """检查是否应该终止（复用 RuleBasedTerminatedEvaluator 的逻辑）"""
        # Rule 1: If the conversation is too long, terminate the conversation
        conversation_too_long = turn_number >= self.max_turn_number

        # Rule 2: If one of the players leaves, terminate the conversation
        p1_leaving = (
            len(messages) > 1
            and isinstance(messages[-2][1], AgentAction)
            and messages[-2][1].action_type == "leave"
        )
        p2_leaving = (
            bool(len(messages))
            and isinstance(messages[-1][1], AgentAction)
            and messages[-1][1].action_type == "leave"
        )

        # Rule 3: If the conversation is stale for too long, terminate the conversation
        stale_count = 0
        for message in messages[::-1]:
            if message[0] == "Environment":
                continue
            assert isinstance(message[1], AgentAction)
            if message[1].action_type == "none":
                stale_count += 1
            else:
                break
            if stale_count > self.max_stale_turn:
                break
        stale_too_long = stale_count > self.max_stale_turn

        terminated = conversation_too_long or p1_leaving or p2_leaving or stale_too_long
        reasons_for_termination = (
            f"{'The conversation is too long; ' if conversation_too_long else ''}"
            f"{'Agent 1 is leaving; ' if p1_leaving else ''}"
            f"{'Agent 2 is leaving; ' if p2_leaving else ''}"
            f"{'The conversation stales for too long; ' if stale_too_long else ''}"
        )
        return terminated, reasons_for_termination

    async def _evaluate_rewards(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]],
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """评估当前轮次两个 agent 的奖励，带重试机制"""
        # 过滤 did nothing
        messages_filtered = [
            (x, y)
            for x, y in messages
            if "did nothing" not in y.to_natural_language()
        ]
        history = "\n".join(
            [
                (
                    f"{x} {y.to_natural_language()}"
                    if x != "Environment"
                    else y.to_natural_language()
                )
                for x, y in messages_filtered
            ]
        )

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response: EvaluationForTwoAgents[T_eval_dim] = await agenerate(
                    model_name=self.model_name,
                    template="""{history}
                    Based on previous interactions, evaluate how well participants achieve their goals.
                    Output format:
                    {format_instructions}
                    CRITICAL INSTRUCTIONS:
                    1. Output ONLY a valid JSON object
                    2. DO NOT repeat or copy the schema definition above
                    3. Replace the placeholder values with actual evaluation scores and reasoning
                    """,
                    input_values=dict(history=history),
                    output_parser=PydanticOutputParser[self.response_format_class](
                        pydantic_object=self.response_format_class
                    ),
                    temperature=temperature,
                    structured_output=self.model_name.startswith("custom/structured"),
                )

                response_list = []
                for dimension in response.agent_1_evaluation.dict().keys():
                    # 获取原始分数并归一化到 0-1 范围
                    agent1_raw_score = response.agent_1_evaluation.dict()[dimension][1]
                    agent2_raw_score = response.agent_2_evaluation.dict()[dimension][1]
                    agent1_normalized = self._normalize_score(dimension, agent1_raw_score)
                    agent2_normalized = self._normalize_score(dimension, agent2_raw_score)

                    response_list.append(
                        (
                            "agent_1",
                            (
                                (dimension, agent1_normalized),
                                response.agent_1_evaluation.dict()[dimension][0],
                            ),
                        )
                    )
                    response_list.append(
                        (
                            "agent_2",
                            (
                                (dimension, agent2_normalized),
                                response.agent_2_evaluation.dict()[dimension][0],
                            ),
                        )
                    )
                return response_list
            except Exception as e:
                last_error = e
                log.warning(f"[yellow] Reward evaluation attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(1.0 * (attempt + 1))  # 指数退避

        # 所有重试都失败
        log.error(f"[red] Failed to generate reward evaluation after {max_retries} attempts: {last_error}")
        traceback.print_exc()
        return []

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """同步调用：仅返回终止判断"""
        terminated, reasons = self._check_termination(turn_number, messages)
        return [
            (
                "environment",
                (("terminated", terminated), reasons),
            )
        ]

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """异步调用：返回终止判断 + 两个 agent 的奖励"""
        # 更新当前轮次并调用回调
        self.current_turn = turn_number
        if self.progress_callback:
            self.progress_callback(self.episode_id, turn_number, self.max_turn_number)

        # 1. 检查终止条件
        terminated, reasons = self._check_termination(turn_number, messages)
        result: list[tuple[str, tuple[tuple[str, int | float | bool], str]]] = [
            (
                "environment",
                (("terminated", terminated), reasons),
            )
        ]
        # 如果判断为终止，则不评估奖励
        if terminated:
            return result

        # 2. 评估当前轮次的奖励
        reward_results = await self._evaluate_rewards(turn_number, messages)
        result.extend(reward_results)

        return result
