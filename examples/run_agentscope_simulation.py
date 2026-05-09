"""
使用 Agentscope 运行 Sotopia EnvAgentComboStorage 仿真

从 Redis 读取 EnvAgentComboStorage 数据，利用 agentscope 框架进行社会交互仿真。
支持 OpenRouter 作为 LLM 后端。

Usage:
    # 设置环境变量
    export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
    export OPENAI_API_KEY="sk-or-v1-xxx"
    
    # 运行仿真
    python examples/run_agentscope_simulation.py --num-combos 1 --model openai/gpt-4o-mini
"""

import asyncio
import os
import sys
import traceback
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

# 加载环境变量
load_dotenv()

# Sotopia imports
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
)

# Agentscope imports
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub, sequential_pipeline

console = Console()

# 配置 loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


def build_agent_system_prompt(
    agent_profile: AgentProfile, 
    env_profile: EnvironmentProfile,
    agent_index: int
) -> str:
    """
    根据 AgentProfile 和 EnvironmentProfile 构建 agent 的 system prompt
    
    Args:
        agent_profile: Agent 的个人资料
        env_profile: 环境配置，包含场景和目标
        agent_index: Agent 索引 (0 或 1)
    
    Returns:
        构建好的 system prompt
    """
    # 基础人物设定
    name = f"{agent_profile.first_name} {agent_profile.last_name}"
    age = agent_profile.age
    occupation = agent_profile.occupation
    gender = agent_profile.gender
    personality = agent_profile.personality_and_values or agent_profile.big_five
    
    # 获取 agent 的目标
    goal = ""
    if env_profile.agent_goals and len(env_profile.agent_goals) > agent_index:
        goal = env_profile.agent_goals[agent_index]
    
    # 场景描述
    scenario = env_profile.scenario
    
    prompt = f"""You are {name}, a {age}-year-old {gender} who works as a {occupation}.

Personality: {personality}

{f"Public info: {agent_profile.public_info}" if agent_profile.public_info else ""}

Scenario: {scenario}

Your goal: {goal}

Instructions:
- Stay in character as {name} throughout the conversation
- Pursue your goal while maintaining believable social behavior
- Respond naturally based on your personality and the situation
- Do not reveal your internal goals explicitly unless it serves your purpose
"""
    return prompt.strip()


def create_agent(
    agent_profile: AgentProfile,
    env_profile: EnvironmentProfile,
    agent_index: int,
    model_name: str = "openai/gpt-4o-mini",
) -> ReActAgent:
    """
    创建 agentscope 的 ReActAgent
    
    Args:
        agent_profile: Sotopia AgentProfile
        env_profile: Sotopia EnvironmentProfile  
        agent_index: Agent 索引
        model_name: OpenRouter 模型名称
    
    Returns:
        配置好的 ReActAgent
    """
    name = f"{agent_profile.first_name} {agent_profile.last_name}"
    sys_prompt = build_agent_system_prompt(agent_profile, env_profile, agent_index)
    
    logger.info(f"Creating agent: {name}")
    
    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=OpenAIChatModel(
            model_name=model_name,
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
            client_kwargs={"base_url": os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")},
            stream=True,
        ),
        formatter=OpenAIMultiAgentFormatter(),
    )


async def run_simulation(
    combo: EnvAgentComboStorage,
    model_name: str = "openai/gpt-4o-mini",
    max_turns: int = 10,
) -> list[Msg]:
    """
    运行单个 EnvAgentCombo 的仿真
    
    Args:
        combo: EnvAgentComboStorage 实例
        model_name: 模型名称
        max_turns: 最大对话轮数
    
    Returns:
        对话消息列表
    """
    # 获取环境和 agent 配置
    env_profile = EnvironmentProfile.get(combo.env_id)
    agent_profiles = [AgentProfile.get(agent_id) for agent_id in combo.agent_ids]
    
    console.print(Panel(
        f"[bold blue]Scenario:[/bold blue] {env_profile.scenario[:200]}...\n\n"
        f"[bold green]Agent 1:[/bold green] {agent_profiles[0].first_name} {agent_profiles[0].last_name} ({agent_profiles[0].occupation})\n"
        f"[bold green]Agent 2:[/bold green] {agent_profiles[1].first_name} {agent_profiles[1].last_name} ({agent_profiles[1].occupation})",
        title="[bold]Simulation Setup[/bold]",
    ))
    
    # 创建 agents
    agents = [
        create_agent(profile, env_profile, idx, model_name)
        for idx, profile in enumerate(agent_profiles)
    ]
    
    # 创建初始消息
    initial_msg = Msg(
        name="Narrator",
        content=f"You are in the following situation: {env_profile.scenario}\n\nPlease begin the conversation.",
        role="user",
    )
    
    messages = [initial_msg]
    
    # 简单的轮流对话循环
    for turn in range(max_turns):
        console.print(f"\n[bold cyan]--- Turn {turn + 1} ---[/bold cyan]")
        
        # 每个 agent 轮流发言
        for agent in agents:
            try:
                # 获取最近的消息作为输入
                last_msg = messages[-1] if messages else initial_msg
                response = await agent(last_msg)
                if response:
                    messages.append(response)
                    content_preview = response.content[:500] if response.content else "[No content]"
                    console.print(f"[bold]{agent.name}:[/bold] {content_preview}")
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")
                traceback.print_exc()
                raise
        
        # 检查对话是否自然结束（简单的启发式规则）
        if messages and len(messages) > 1:
            last_msg_content = messages[-1].content
            # 处理 content 可能是 list 或 str 的情况
            if isinstance(last_msg_content, list):
                last_content = str(last_msg_content).lower()
            else:
                last_content = (last_msg_content or "").lower()
            if any(keyword in last_content for keyword in ["goodbye", "bye", "see you", "take care", "farewell"]):
                logger.info("Conversation ended naturally")
                break
    
    return messages


async def main(
    num_combos: int = 1,
    model_name: str = "openai/gpt-4o-mini",
    max_turns: int = 10,
    tag: Optional[str] = None,
):
    """
    主函数：从数据库读取 combos 并运行仿真
    
    Args:
        num_combos: 要运行的 combo 数量
        model_name: 模型名称
        max_turns: 每个仿真的最大轮数
        tag: 可选的 tag 过滤
    """
    console.print("[bold]Loading EnvAgentComboStorage from database...[/bold]")
    
    # 获取所有 combos
    try:
        all_combos = list(EnvAgentComboStorage.find().all())
        logger.info(f"Found {len(all_combos)} combos in database")
    except Exception as e:
        logger.error(f"Failed to load combos: {e}")
        traceback.print_exc()
        raise
    
    if not all_combos:
        console.print("[red]No combos found in database. Please ensure Redis is running and data is loaded.[/red]")
        return
    
    # 选择要运行的 combos
    combos_to_run = all_combos[:num_combos]
    console.print(f"[green]Running {len(combos_to_run)} simulations...[/green]")
    
    results = []
    for i, combo in enumerate(tqdm(combos_to_run, desc="Simulations")):
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold]Simulation {i + 1}/{len(combos_to_run)}[/bold]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")
        
        try:
            messages = await run_simulation(
                combo=combo,
                model_name=model_name,
                max_turns=max_turns,
            )
            results.append({
                "combo_pk": str(combo.pk),
                "env_id": combo.env_id,
                "agent_ids": combo.agent_ids,
                "messages": [{"name": m.name, "content": m.content} for m in messages],
                "success": True,
            })
            console.print(f"[green]✓ Simulation completed with {len(messages)} messages[/green]")
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            traceback.print_exc()
            results.append({
                "combo_pk": str(combo.pk),
                "error": str(e),
                "success": False,
            })
    
    # 打印总结
    successful = sum(1 for r in results if r.get("success"))
    console.print(f"\n[bold]Summary: {successful}/{len(results)} simulations completed successfully[/bold]")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Sotopia simulations with Agentscope")
    parser.add_argument("--num-combos", type=int, default=1, help="Number of combos to run")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini", help="Model name for OpenRouter")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum turns per simulation")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag filter")
    
    args = parser.parse_args()
    
    # 检查环境变量
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
        console.print("[red]Error: Please set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable[/red]")
        sys.exit(1)
    
    asyncio.run(main(
        num_combos=args.num_combos,
        model_name=args.model,
        max_turns=args.max_turns,
        tag=args.tag,
    ))
