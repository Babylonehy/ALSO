"""
Bandit-Based Dynamic Prompt Optimization Simulation

This script runs Sotopia simulations with AdversarialBandit algorithm(s)
that dynamically optimize agent bios based on reward feedback.

Usage:
    # Optimize both agents with separate bandits
    python run_bandit_simulation.py --scenario-id 01H7VKQHT745XAP1A4DDV8H419 --optimize both

    # Optimize only P1 (Agent 1)
    python run_bandit_simulation.py --scenario-id 01H7VKQHT745XAP1A4DDV8H419 --optimize p1

    # Optimize only P2 (Agent 2)
    python run_bandit_simulation.py --scenario-id 01H7VKQHT745XAP1A4DDV8H419 --optimize p2

    # Baseline: no optimization (use original bios only)
    python run_bandit_simulation.py --scenario-id 01H7VKQHT745XAP1A4DDV8H419 --optimize none

Key features:
- Uses pre-generated paraphrased backgrounds and embeddings
- Supports separate bandits for each agent with independent optimization
- Supports baseline mode (no bandit optimization)
- Integrates AdversarialBandit with the DynamicPromptParallelSotopiaEnv
- Logs bandit decisions and reward feedback at each turn
- Tracks timing and cost information
- Outputs selection history and reward progression
"""

import argparse
import asyncio
import atexit
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import traceback
import aiohttp
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.table import Table

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Imports from sotopia
from sotopia.agents import LLMAgent
from sotopia.database import (
    AgentProfile,
    EnvironmentProfile,
    EnvAgentComboStorage,
    EpisodeLog,
)
from sotopia.envs.evaluators import EvaluationForTwoAgents, SotopiaDimensions
from sotopia.generation_utils import enable_llm_call_logging, get_llm_call_log_path

# Imports from the dynamic observation module
from experiments.also.core.envs_dynamic_parallel import (
    DynamicPromptParallelSotopiaEnv,
)
from experiments.also.core.evaluator_reward_in_trun import (
    RewardInTurnEvaluator,
    TerminalEvaluator,
)
from experiments.also.core.bandits import (
    BaseBandit,
    BanditConfig,
    PromptSpace,
    StrategySpace,
    LearnableStrategySpaceV2,
    create_bandit,
    BANDIT_TYPES,
)
from experiments.also.core.bandits.prompt_breeder_bandit import (
    PromptBreederConfig,
)
from experiments.also.core.bandits.progressive_prompt_breeder_bandit import (
    ProgressivePromptBreederConfig,
)
from experiments.also.core.bandits.neural_evolution_bandit import (
    NeuralEvolutionConfig,
)
from experiments.also.core.bandits.neural_adversarial_learnable_v2_bandit import (
    LearnableAdversarialV2Config,
)
from experiments.also.core.bandits.opro_bandit import OPROConfig
from experiments.also.core.bandits.evoprompt_bandit import (
    EvoPromptConfig,
)
from experiments.also.core.logging_utils import (
    configure_logger,
    setup_terminal_logging,
    cleanup_terminal_logging,
)
from experiments.also.calculate_cost import calculate_cost_async

# Type alias for bandit type
BanditType = Literal[
    "exp3",
    "linucb",
    "neural_ucb",
    "adversarial",
    "neural_adversarial",
    "adversarial_learnable_v2",
    "neural_evolution",
    "none",
]

console = Console()

# Type alias for optimization mode
OptimizeMode = Literal["p1", "p2", "both", "none"]

# Default embedding model for context (must match the model used for prompt space embeddings)
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

# Default mihomo paths (ordered by priority)
MIHOMO_CONFIG_PATHS = [
    Path("/opt/clash/runtime.yaml"),
    Path("/root/clashctl/resources/runtime.yaml"),
    Path.home() / ".config/mihomo/config.yaml",
]
MIHOMO_EXECUTABLE_PATHS = [
    Path("/opt/clash/bin/mihomo"),
    Path("/root/clashctl/bin/mihomo"),
    Path.home() / ".local/bin/mihomo",
    Path("/usr/local/bin/mihomo"),
]


def find_free_port(start_port: int = 7900, max_attempts: int = 100) -> int:
    """查找可用的端口号"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No free port found in range {start_port}-{start_port + max_attempts}"
    )


def find_mihomo_executable() -> str | None:
    """
    查找 mihomo 可执行文件路径

    Returns:
        mihomo 路径，如果找不到返回 None
    """
    candidates = [
        *MIHOMO_EXECUTABLE_PATHS,
        shutil.which("mihomo"),
        shutil.which("clash"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def find_mihomo_config() -> Path | None:
    """
    查找 mihomo 配置文件路径

    Returns:
        配置文件路径，如果找不到返回 None
    """
    for config_path in MIHOMO_CONFIG_PATHS:
        if config_path.exists():
            return config_path
    return None


class MihomoProxyManager:
    """自动管理 mihomo 代理进程，用于单个仿真实例"""

    _instance: "MihomoProxyManager | None" = None  # 单例，用于 atexit 清理

    def __init__(
        self,
        base_config: Path | None = None,
        base_port: int = 7900,
    ):
        # 如果没有指定配置文件，自动查找
        if base_config is None:
            self.base_config = find_mihomo_config()
        else:
            self.base_config = base_config
        self.base_port = base_port
        self.http_port: int | None = None
        self.socks_port: int | None = None
        self.api_port: int | None = None
        self.process: subprocess.Popen | None = None
        self.work_dir: Path | None = None
        self._started = False

    def _find_available_ports(self) -> tuple[int, int, int]:
        """查找三个连续可用的端口 (HTTP, SOCKS, API)"""
        for base in range(self.base_port, self.base_port + 1000, 10):
            try:
                http_port = find_free_port(base)
                socks_port = find_free_port(base + 1)
                api_port = find_free_port(base + 2)
                # 确保端口连续且可用
                if socks_port == http_port + 1 and api_port == http_port + 2:
                    return http_port, socks_port, api_port
            except RuntimeError:
                continue
        raise RuntimeError("Could not find 3 consecutive free ports for mihomo")

    def _wait_for_proxy_ready(self) -> bool:
        """
        等待代理启动并验证可用性（无超时，直到成功或进程退出）

        Returns:
            代理是否就绪
        """
        start_time = time.time()
        check_interval = 0.5
        log_interval = 5  # 每5秒打印一次等待信息
        last_log_time = start_time

        console.print(f"[cyan]Waiting for mihomo proxy on port {self.http_port}...[/]")

        while True:
            # 检查进程是否还在运行
            if self.process and self.process.poll() is not None:
                console.print("[red]mihomo process exited unexpectedly[/]")
                return False

            # 尝试连接代理端口
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(("127.0.0.1", self.http_port))  # type: ignore
                    if result == 0:
                        # 端口可连接，再等一小会确保完全就绪
                        time.sleep(0.5)
                        elapsed = time.time() - start_time
                        console.print(
                            f"[green]✓ Proxy port {self.http_port} is ready (took {elapsed:.1f}s)[/]"
                        )
                        logger.info(
                            f"Proxy port {self.http_port} is ready after {elapsed:.1f}s"
                        )
                        return True
            except (OSError, socket.error):
                pass

            # 定期打印等待信息
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                elapsed = current_time - start_time
                console.print(
                    f"[yellow]Still waiting for proxy... ({elapsed:.0f}s elapsed)[/]"
                )
                last_log_time = current_time

            time.sleep(check_interval)

    def _filter_proxies(self, config: dict) -> dict:
        """
        过滤代理节点：只保留日本、美国、新加坡节点，排除香港和免费节点
        """
        # 允许的国家/地区关键词
        allowed_keywords = [
            "日本",
            "Japan",
            "JP",
            "tokyo",
            "osaka",
            "美国",
            "USA",
            "US",
            "America",
            "Los Angeles",
            "Seattle",
            "New York",
            "San",
            "新加坡",
            "Singapore",
            "SG",
        ]
        # 排除的国家/地区关键词
        blocked_keywords = [
            "香港",
            "Hong Kong",
            "HK",
            "Hongkong",
        ]
        # 排除的免费节点关键词
        free_keywords = [
            "免费",
            "free",
            "试用",
            "trial",
            "体验",
            "过期",
            "expire",
            "剩余",
            "套餐",
            "流量",
            "到期",
            "官网",
            "官方",
            "群",
            "订阅",
            "更新",
            "网址",
            "地址",
        ]

        def is_allowed_proxy(name: str) -> bool:
            name_lower = name.lower()
            # 先检查是否是免费/信息节点
            for free_kw in free_keywords:
                if free_kw.lower() in name_lower:
                    return False
            # 检查是否被阻止的地区
            for blocked in blocked_keywords:
                if blocked.lower() in name_lower:
                    return False
            # 检查是否允许的地区
            for allowed in allowed_keywords:
                if allowed.lower() in name_lower:
                    return True
            return False

        if "proxies" in config:
            original_count = len(config["proxies"])
            filtered_proxies = [
                p for p in config["proxies"] if is_allowed_proxy(p.get("name", ""))
            ]
            config["proxies"] = filtered_proxies
            logger.info(
                f"Filtered proxies: {len(filtered_proxies)}/{original_count} (JP/US/SG only, no HK/free)"
            )

            # 更新 proxy-groups 中的代理列表
            filtered_names = {p["name"] for p in filtered_proxies}
            if "proxy-groups" in config:
                for group in config["proxy-groups"]:
                    if "proxies" in group:
                        group["proxies"] = [
                            p
                            for p in group["proxies"]
                            if p in filtered_names
                            or p in ("DIRECT", "REJECT", "PASS")
                            or any(
                                g["name"] == p for g in config.get("proxy-groups", [])
                            )
                        ]

        return config

    def _create_auto_select_group(self, config: dict) -> dict:
        """
        创建自动选择延时最低节点的代理组
        """
        if "proxies" not in config or not config["proxies"]:
            return config

        proxy_names = [p["name"] for p in config["proxies"]]

        # 创建 url-test 代理组，自动选择延时最低的节点
        auto_select_group = {
            "name": "AutoSelect-LowLatency",
            "type": "url-test",
            "proxies": proxy_names,
            "url": "http://www.gstatic.com/generate_204",
            "interval": 300,  # 每5分钟测试一次
            "tolerance": 50,  # 延时容差 50ms
        }

        # 确保 proxy-groups 存在
        if "proxy-groups" not in config:
            config["proxy-groups"] = []

        # 在最前面插入自动选择组
        config["proxy-groups"].insert(0, auto_select_group)

        # 更新其他组，将 PROXY 或第一个组指向我们的自动选择组
        for group in config["proxy-groups"][1:]:
            if "proxies" in group and len(group["proxies"]) > 0:
                # 在代理列表最前面添加我们的自动选择组
                if "AutoSelect-LowLatency" not in group["proxies"]:
                    group["proxies"].insert(0, "AutoSelect-LowLatency")

        # 更新规则，确保使用我们的自动选择组
        if "rules" in config:
            new_rules = []
            for rule in config["rules"]:
                # 将 PROXY 替换为我们的自动选择组
                if ",PROXY" in rule:
                    rule = rule.replace(",PROXY", ",AutoSelect-LowLatency")
                new_rules.append(rule)
            config["rules"] = new_rules

        logger.info(
            f"Created auto-select group with {len(proxy_names)} proxies (url-test, lowest latency)"
        )

        return config

    def _generate_config(self, config_path: Path) -> None:
        """基于模板生成新的 mihomo 配置（在 start() 中检查后调用）"""
        import yaml

        # self.base_config 在 start() 中已确认非 None 且存在
        assert self.base_config is not None and self.base_config.exists()

        with open(self.base_config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 修改端口
        config["port"] = self.http_port
        config["socks-port"] = self.socks_port
        config["external-controller"] = f"127.0.0.1:{self.api_port}"
        config["allow-lan"] = False

        # 过滤代理节点（只保留日本、美国、新加坡，排除香港和免费节点）
        config = self._filter_proxies(config)

        # 创建自动选择延时最低的代理组
        config = self._create_auto_select_group(config)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def start(self) -> int:
        """
        启动 mihomo 进程，返回 HTTP 代理端口。

        Returns:
            HTTP 代理端口号

        Raises:
            FileNotFoundError: mihomo 可执行文件或配置文件不存在
            RuntimeError: 启动失败
        """
        if self._started:
            return self.http_port  # type: ignore

        # 检查配置文件是否存在
        if self.base_config is None or not self.base_config.exists():
            searched_paths = "\n  - ".join(str(p) for p in MIHOMO_CONFIG_PATHS)
            raise FileNotFoundError(
                f"mihomo config not found. Searched:\n  - {searched_paths}\n"
                f"Please create a config file or specify --mihomo-config"
            )

        # 查找 mihomo 可执行文件
        mihomo_exe = find_mihomo_executable()
        if mihomo_exe is None:
            searched_paths = "\n  - ".join(str(p) for p in MIHOMO_EXECUTABLE_PATHS)
            raise FileNotFoundError(
                f"mihomo executable not found. Searched:\n  - {searched_paths}\n"
                f"  - PATH (mihomo/clash)\n"
                f"Install from: https://github.com/MetaCubeX/mihomo/releases"
            )

        # 查找可用端口
        self.http_port, self.socks_port, self.api_port = self._find_available_ports()
        logger.info(
            f"Found available ports: HTTP={self.http_port}, SOCKS={self.socks_port}, API={self.api_port}"
        )

        # 创建临时工作目录
        self.work_dir = Path(tempfile.mkdtemp(prefix=f"mihomo_{self.http_port}_"))

        # 生成配置
        config_path = self.work_dir / "config.yaml"
        self._generate_config(config_path)

        # 启动进程
        self.process = subprocess.Popen(
            [mihomo_exe, "-d", str(self.work_dir), "-f", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.work_dir,
        )

        # 等待启动并验证代理可用
        if not self._wait_for_proxy_ready():
            stderr = self.process.stderr.read().decode() if self.process.stderr else ""
            self.process.kill()
            raise RuntimeError(
                f"mihomo failed to start or proxy not responding: {stderr}"
            )

        self._started = True
        logger.info(
            f"Started mihomo proxy on port {self.http_port} (PID: {self.process.pid})"
        )

        # 注册单例以便 atexit 清理
        MihomoProxyManager._instance = self

        return self.http_port

    def stop(self) -> None:
        """停止 mihomo 进程"""
        if self.process:
            logger.info(f"Stopping mihomo proxy (PID: {self.process.pid})")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self._started = False

        # 清理临时目录
        if self.work_dir and self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up work dir: {e}")

    def is_running(self) -> bool:
        """检查 mihomo 是否运行中"""
        return self.process is not None and self.process.poll() is None

    @classmethod
    def cleanup_instance(cls) -> None:
        """清理单例实例（用于 atexit）"""
        if cls._instance:
            cls._instance.stop()
            cls._instance = None


# 注册 atexit 清理函数
atexit.register(MihomoProxyManager.cleanup_instance)


class EmbeddingClient:
    """Client for generating embeddings from dialogue history."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_base: str = DEFAULT_API_BASE,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the embedding client.

        Args:
            model: Embedding model name
            api_base: API base URL
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
        """
        self.model = model
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Please set OPENROUTER_API_KEY env var."
            )
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.gen_ids: list[str] = []  # Track gen_ids for cost calculation

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as numpy array
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.model)

        # Extract gen_id from response for cost tracking
        if hasattr(response, "id") and response.id:
            self.gen_ids.append(response.id)
            logger.debug(f"Embedding gen_id: {response.id}")

            # Log to LLM call log for unified cost calculation
            self._log_embedding_call(response)

        return np.array(response.data[0].embedding)

    def _log_embedding_call(self, response: Any) -> None:
        """Log embedding call to LLM call log for cost tracking."""
        log_path = get_llm_call_log_path()
        if not log_path:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "id": getattr(response, "id", None),
            "model": self.model,
            "caller": "embedding",
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0)
                if hasattr(response, "usage") and response.usage
                else 0,
                "completion_tokens": 0,  # Embeddings don't have completion tokens
                "total_tokens": getattr(response.usage, "total_tokens", 0)
                if hasattr(response, "usage") and response.usage
                else 0,
            },
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def extract_agent_history(messages: list[tuple[str, object]]) -> str:
    """
    Extract agent-only messages for context embedding.

    This excludes Environment messages and "did nothing" messages,
    returning only actual agent speech.

    Args:
        messages: List of (sender, message) tuples from the dialogue

    Returns:
        Formatted string of agent-only messages
    """
    agent_messages = []
    for sender, msg in messages:
        # Skip Environment messages
        if sender == "Environment":
            continue
        # Get natural language representation
        if hasattr(msg, "to_natural_language"):
            msg_text = msg.to_natural_language()
        else:
            msg_text = str(msg)
        # Skip "did nothing" messages
        if "did nothing" in msg_text:
            continue
        agent_messages.append(f"{sender} {msg_text}")

    return "\n".join(agent_messages)


def extract_recent_agent_history(
    messages_by_turn: list[list[tuple[str, str, object]]],
    max_turns: int,
) -> str:
    """Extract agent-only dialogue from the most recent N turns."""
    if max_turns <= 0:
        return ""

    recent_turns = messages_by_turn[-max_turns:]
    agent_messages = []
    for turn_msgs in recent_turns:
        for sender, _receiver, msg in turn_msgs:
            if sender == "Environment":
                continue
            if hasattr(msg, "to_natural_language"):
                msg_text = msg.to_natural_language()
            else:
                msg_text = str(msg)
            if "did nothing" in msg_text:
                continue
            agent_messages.append(f"{sender}: {msg_text}")
    return "\n".join(agent_messages)


async def calculate_cost_by_model_async(log_file: Path) -> dict[str, Any]:
    """
    Calculate OpenRouter costs grouped by model using Sotopia's LLM call log JSONL.

    Expected log entry keys: {"id": ..., "model": ...}
    """

    async def fetch_usage(
        session: aiohttp.ClientSession, gen_id: str
    ) -> dict[str, Any] | None:
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

    # Deduplicate by id (some runs may log duplicates if resumed/merged logs)
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
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30, force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch_usage(session, e["id"]) for e in unique_entries]

        batch_size = 50
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
    await asyncio.sleep(
        0.3
    )  # Increased from 0.1s - handles 100+ concurrent connections

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
            sorted(
                by_model.items(),
                key=lambda kv: kv[1].get("total_cost", 0),
                reverse=True,
            )
        ),
    }


def _print_cost_breakdown(cost_info: dict[str, Any]) -> None:
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


def find_combo_by_pk(combo_pk: str) -> EnvAgentComboStorage | None:
    """Find an EnvAgentComboStorage by its primary key."""
    try:
        combo = EnvAgentComboStorage.get(pk=combo_pk)
        return combo
    except Exception as e:
        logger.warning(f"Combo not found by pk: {e}")
        return None


def load_profiles(
    combo: EnvAgentComboStorage,
) -> tuple[EnvironmentProfile, list[AgentProfile]]:
    """Load environment and agent profiles from combo."""
    env_profile = EnvironmentProfile.get(pk=combo.env_id)
    agent_profiles = [AgentProfile.get(pk=agent_id) for agent_id in combo.agent_ids]
    return env_profile, agent_profiles


def format_duration(seconds: float) -> str:
    """Format time duration to HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_git_info() -> dict[str, Any]:
    """
    Get current git repository information.

    Returns:
        Dictionary containing:
            - commit_id: Current commit hash (or None if not in git repo)
            - branch: Current branch name (or None)
            - dirty: Whether there are uncommitted changes (bool)
            - error: Error message if git command failed (or None)
    """
    import subprocess

    git_info: dict[str, Any] = {
        "commit_id": None,
        "branch": None,
        "dirty": False,
        "error": None,
    }

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["commit_id"] = result.stdout.strip()

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["branch"] = result.stdout.strip()

        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["dirty"] = bool(result.stdout.strip())

    except subprocess.CalledProcessError as e:
        git_info["error"] = f"Git command failed: {e}"
        logger.warning(f"Failed to get git info: {e}")
    except subprocess.TimeoutExpired:
        git_info["error"] = "Git command timed out"
        logger.warning("Git command timed out")
    except FileNotFoundError:
        git_info["error"] = "Git not found in PATH"
        logger.warning("Git executable not found")
    except Exception as e:
        git_info["error"] = f"Unexpected error: {e}"
        logger.warning(f"Unexpected error getting git info: {e}")

    return git_info


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_to_jsonable(v) for v in obj)
    return obj


def save_run_config(
    experiment_dir: Path,
    experiment_tag: str,
    args: argparse.Namespace,
    bandit_config: BanditConfig,
    bandit_type: str,
    filename: str = "run_config.json",
) -> Path:
    """Save a JSON snapshot of runtime args and bandit config into experiment_dir."""
    payload = {
        "experiment_tag": experiment_tag,
        "bandit_type": bandit_type,
        "saved_at": datetime.now().isoformat(),
        "argv": list(sys.argv),
        "args": _to_jsonable(vars(args)),
        "bandit_config": {
            "__config_type__": type(bandit_config).__name__,
            **_to_jsonable(asdict(bandit_config)),
        },
        "git_info": get_git_info(),
    }
    path = experiment_dir / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# Config type registry for deserialization
CONFIG_TYPES = {
    "BanditConfig": BanditConfig,
    "LearnableAdversarialV2Config": LearnableAdversarialV2Config,
    "PromptBreederConfig": PromptBreederConfig,
    "ProgressivePromptBreederConfig": ProgressivePromptBreederConfig,
    "NeuralEvolutionConfig": NeuralEvolutionConfig,
    "OPROConfig": OPROConfig,
    "EvoPromptConfig": EvoPromptConfig,
}


def load_bandit_config(config_dict: dict) -> BanditConfig:
    """
    Reconstruct bandit config from saved JSON with type information.

    Args:
        config_dict: Dictionary loaded from run_config.json["bandit_config"]

    Returns:
        Appropriate config instance (BanditConfig, PromptBreederConfig, etc.)

    Raises:
        ValueError: If config type is unknown or invalid
    """
    if not config_dict:
        logger.warning("Empty config dict provided, returning default BanditConfig")
        return BanditConfig()

    # Extract config type metadata (if present)
    config_type_name = config_dict.get("__config_type__", "BanditConfig")

    # Create a clean copy without metadata fields
    clean_dict = {k: v for k, v in config_dict.items() if not k.startswith("__")}

    # Backward compatibility: if no type info, default to BanditConfig
    if "__config_type__" not in config_dict:
        logger.warning(
            "No config type found in saved config, defaulting to BanditConfig. "
            "This may be an old experiment save file."
        )
        config_type_name = "BanditConfig"

    # Look up config class
    config_class = CONFIG_TYPES.get(config_type_name)
    if config_class is None:
        raise ValueError(
            f"Unknown config type: {config_type_name}. "
            f"Valid types: {list(CONFIG_TYPES.keys())}"
        )

    # Instantiate config with saved parameters
    try:
        return config_class(**clean_dict)
    except TypeError as e:
        logger.error(f"Failed to instantiate {config_type_name}: {e}")
        logger.error(f"Config dict keys: {list(clean_dict.keys())}")
        raise ValueError(
            f"Failed to create {config_type_name} from saved config. "
            f"This may indicate a config schema change. Error: {e}"
        ) from e


def create_bandit_config_from_args(args: argparse.Namespace) -> BanditConfig:
    """
    Create the appropriate bandit config based on bandit_type and command line args.

    For EvoPrompt, OPRO, PromptBreeder, ProgressivePromptBreeder:
    - Uses their specialized config classes
    - Applies selection_strategy, epsilon, selection_temperature from args

    For other bandits (exp3, linucb, etc.):
    - Uses the base BanditConfig
    """
    bandit_type = args.bandit_type

    # Common base config parameters
    base_params = dict(
        eta=args.eta,
        alpha=args.alpha,
        beta=args.beta,
        depth=args.depth,
        update_interval=args.update_interval,
        evolution_interval=args.evolution_interval,
        mask_unselected_scores=args.mask_unselected_scores,
        importance_weighted_reward=args.importance_weighted_reward,
        gamma=args.gamma,
        score_decay=args.score_decay,
        cumulative_score_mode=args.cumulative_score_mode,
        nn_weight=args.nn_weight,
        adaptive_nn_weight=args.adaptive_nn_weight,
        adaptive_nn_weight_warmup=args.adaptive_nn_weight_warmup,
        adaptive_nn_weight_scale=args.adaptive_nn_weight_scale,
        failure_penalty_threshold=args.failure_penalty_threshold,
        failure_penalty_factor=args.failure_penalty_factor,
        multi_dim_prediction=args.multi_dim_prediction,
        enable_reflection=args.enable_reflection,
        reflection_interval=args.reflection_interval,
        reflection_lookback=args.reflection_lookback,
        greedy_selection=args.greedy_selection,
    )

    learnable_params = dict(
        proposal_interval=args.proposal_interval,
        proposal_warmup_turns=args.proposal_warmup_turns,
        proposal_context_turns=args.proposal_context_turns,
        proposal_model=args.proposal_model,
        proposal_temperature=args.proposal_temperature,
    )

    # Selection strategy parameters (for ablation studies)
    selection_params = {}
    if args.selection_strategy is not None:
        selection_params["selection_strategy"] = args.selection_strategy
    if hasattr(args, "selection_epsilon"):
        selection_params["selection_epsilon"] = args.selection_epsilon
    if hasattr(args, "selection_temperature"):
        selection_params["selection_temperature"] = args.selection_temperature

    # Population size parameter (for evolutionary bandits)
    population_params = {}
    if hasattr(args, "population_size") and args.population_size is not None:
        population_params["population_size"] = args.population_size

    if bandit_type in ("evoprompt_ga", "evoprompt_de"):
        config = EvoPromptConfig(**base_params, **selection_params, **population_params)
    elif bandit_type == "opro":
        config = OPROConfig(**base_params, **selection_params, **population_params)
    elif bandit_type == "prompt_breeder":
        config = PromptBreederConfig(
            **base_params, **selection_params, **population_params
        )
    elif bandit_type == "progressive_prompt_breeder":
        # Progressive PromptBreeder uses initial_population_size instead of population_size
        ppb_params = {}
        if population_params.get("population_size"):
            ppb_params["initial_population_size"] = population_params["population_size"]
            ppb_params["max_population_size"] = population_params["population_size"]
        config = ProgressivePromptBreederConfig(
            **base_params, **selection_params, **ppb_params
        )
    elif bandit_type == "neural_evolution":
        config = NeuralEvolutionConfig(**base_params, **population_params)
    elif bandit_type == "adversarial_learnable_v2":
        config = LearnableAdversarialV2Config(**base_params, **learnable_params)
    else:
        # For exp3, linucb, neural_ucb, adversarial, tpe, etc.
        config = BanditConfig(**base_params, **selection_params)

    return config


def get_available_scenarios(embeddings_dir: Path) -> list[str]:
    """Get list of available scenario IDs from the embeddings directory."""
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    scenarios = []
    for subdir in embeddings_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            scenarios.append(subdir.name)

    return sorted(scenarios)


def load_completed_scenarios_from_results(results_path: Path) -> set[str]:
    """
    Load successfully completed scenario IDs from a results.json file or experiment directory.

    Args:
        results_path: Path to results.json or an experiment directory containing results.json

    Returns:
        Set of scenario IDs that completed successfully
    """
    # Handle both file and directory paths
    if results_path.is_dir():
        results_file = results_path / "results.json"
    else:
        results_file = results_path

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file) as f:
        data = json.load(f)

    # Extract successfully completed scenario IDs
    completed = set()
    results_list = data.get("results", [])
    for result in results_list:
        if result.get("success", False):
            scenario_id = result.get("scenario_id")
            if scenario_id:
                completed.add(scenario_id)

    return completed


def load_previous_results(results_path: Path) -> tuple[list[dict], dict]:
    """
    Load all results from a previous experiment.

    Args:
        results_path: Path to results.json or an experiment directory

    Returns:
        Tuple of (results_list, full_results_dict)
    """
    if results_path.is_dir():
        results_file = results_path / "results.json"
    else:
        results_file = results_path

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file) as f:
        data = json.load(f)

    return data.get("results", []), data


def migrate_episode_logs_to_new_tag(source_tag: str, target_tag: str) -> int:
    """
    Migrate EpisodeLog records from source tag to target tag.

    Creates copies of all EpisodeLog records with the new tag.
    Original records are preserved.

    Args:
        source_tag: The source experiment tag to copy from
        target_tag: The target experiment tag to copy to

    Returns:
        Number of records migrated
    """
    # Query all episodes with source tag
    source_episodes = EpisodeLog.find(EpisodeLog.tag == source_tag).all()

    if not source_episodes:
        logger.warning(f"No EpisodeLog records found for tag: {source_tag}")
        return 0

    migrated_count = 0
    for ep in source_episodes:
        # Create a new EpisodeLog with the target tag
        new_episode = EpisodeLog(
            environment=ep.environment,
            agents=ep.agents,
            tag=target_tag,  # Use new tag
            models=ep.models,
            messages=ep.messages,
            reasoning=ep.reasoning,
            rewards=ep.rewards,
            rewards_prompt=ep.rewards_prompt,
        )
        new_episode.save()
        migrated_count += 1

    logger.info(
        f"Migrated {migrated_count} EpisodeLog records from '{source_tag}' to '{target_tag}'"
    )
    return migrated_count


def migrate_experiment_directory(
    source_dir: Path,
    target_dir: Path,
    update_tag_in_files: bool = True,
    new_tag: str | None = None,
) -> None:
    """
    Migrate experiment directory contents from source to target.

    Copies all files and subdirectories, optionally updating experiment tag in config files.

    Args:
        source_dir: Source experiment directory
        target_dir: Target experiment directory
        update_tag_in_files: Whether to update experiment_tag in JSON files
        new_tag: New experiment tag (required if update_tag_in_files is True)
    """
    import shutil

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy all contents
    for item in source_dir.iterdir():
        src_path = item
        dst_path = target_dir / item.name

        if item.is_dir():
            # Recursively copy directories
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            logger.debug(f"Copied directory: {item.name}")
        else:
            # Copy files
            shutil.copy2(src_path, dst_path)
            logger.debug(f"Copied file: {item.name}")

    # Update experiment_tag in JSON config files if requested
    if update_tag_in_files and new_tag:
        json_files = ["run_config.json", "results.json"]
        for json_file in json_files:
            json_path = target_dir / json_file
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    if "experiment_tag" in data:
                        old_tag = data["experiment_tag"]
                        data["experiment_tag"] = new_tag
                        # Also update subset if present
                        if "subset" in data:
                            data["subset"] = "all"
                        # Add migration info
                        data["extended_from"] = {
                            "original_tag": old_tag,
                            "original_subset": "hard",
                            "migration_time": datetime.now().isoformat(),
                        }
                        with open(json_path, "w") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        logger.info(
                            f"Updated experiment_tag in {json_file}: {old_tag} -> {new_tag}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to update {json_file}: {e}")

    logger.info(f"Migrated experiment directory: {source_dir} -> {target_dir}")


def resolve_experiment_source(source_tag_or_path: str) -> tuple[str, Path]:
    """
    Resolve experiment tag or path to (tag, directory).

    Args:
        source_tag_or_path: Either an experiment tag or path to experiment directory

    Returns:
        Tuple of (experiment_tag, experiment_directory)
    """
    outputs_dir = PROJECT_ROOT / "experiments/also/outputs"

    # Check if it's a path
    source_path = Path(source_tag_or_path)
    if source_path.exists() and source_path.is_dir():
        # It's a directory path
        source_dir = source_path
        # Try to get tag from run_config.json
        run_config = source_dir / "run_config.json"
        if run_config.exists():
            with open(run_config) as f:
                config = json.load(f)
            source_tag = config.get("experiment_tag", source_dir.name)
        else:
            source_tag = source_dir.name
        return source_tag, source_dir

    # It's a tag name - look in outputs directory
    source_dir = outputs_dir / source_tag_or_path
    if source_dir.exists():
        return source_tag_or_path, source_dir

    raise FileNotFoundError(
        f"Could not find experiment: {source_tag_or_path}\n"
        f"Checked: {source_path} and {source_dir}"
    )


def parse_env_ids(file_path: Path) -> dict[str, list[str]]:
    """Parse env_ids.txt file, return hard and all dataset env_id lists."""
    with open(file_path, "r") as f:
        content = f.read()

    sections = content.split("SOTOPIA-ALL:")
    hard_section = sections[0]
    all_section = sections[1] if len(sections) > 1 else ""

    hard_ids = re.findall(r'"([A-Z0-9]+)"', hard_section)
    all_ids = re.findall(r'"([A-Z0-9]+)"', all_section)

    return {"hard": list(set(hard_ids)), "all": list(set(all_ids))}


def get_combos_for_subset(subset: str) -> list[EnvAgentComboStorage]:
    """Get all combos for the specified dataset subset.

    Args:
        subset: Dataset subset name ("hard", "all", "hard_small", or "non_hard")
            - "hard": All combos from hard environments
            - "all": All combos from all environments
            - "hard_small": Only first combo per hard environment (deterministic)
            - "non_hard": All combos from non-hard environments (all - hard)
    """
    env_ids_path = PROJECT_ROOT / "data/env_ids.txt"
    if not env_ids_path.exists():
        raise FileNotFoundError(f"env_ids.txt not found at {env_ids_path}")

    ids_by_category = parse_env_ids(env_ids_path)

    # Handle special subsets
    is_hard_small = subset == "hard_small"
    is_non_hard = subset == "non_hard"

    if is_non_hard:
        # non_hard = all - hard
        all_env_ids = set(ids_by_category.get("all", []))
        hard_env_ids = set(ids_by_category.get("hard", []))
        env_ids = list(all_env_ids - hard_env_ids)
    elif is_hard_small:
        env_ids = ids_by_category.get("hard", [])
    else:
        env_ids = ids_by_category.get(subset, [])

    if not env_ids:
        raise ValueError(f"No env_ids found for subset '{subset}'")

    # Sort env_ids for deterministic ordering
    env_ids = sorted(env_ids)

    # Get all combos and filter manually (Pydantic v2 compatibility)
    all_combos = list(EnvAgentComboStorage.find().all())
    combos = []
    for env_id in env_ids:
        found_combos = [c for c in all_combos if c.env_id == env_id]
        if is_hard_small:
            # Take only first combo (sorted by pk for determinism)
            found_combos = sorted(found_combos, key=lambda c: c.pk)
            if found_combos:
                combos.append(found_combos[0])
        else:
            combos.extend(found_combos)

    return combos


class BatchProgressTracker:
    """Track progress of multiple episodes running in parallel."""

    def __init__(self) -> None:
        self.episodes: dict[
            str, dict
        ] = {}  # scenario_id -> {turn, max_turn, agent_names, step}
        self.completed = 0
        self.total = 0
        self.successes = 0
        self.errors = 0

    def update(
        self, scenario_id: str, turn: int, max_turn: int, step: str = ""
    ) -> None:
        """Update episode progress with optional step info."""
        if scenario_id in self.episodes:
            self.episodes[scenario_id]["turn"] = turn
            self.episodes[scenario_id]["max_turn"] = max_turn
            if step:
                self.episodes[scenario_id]["step"] = step

    def register(self, scenario_id: str, max_turn: int, agent_names: str) -> None:
        self.episodes[scenario_id] = {
            "turn": 0,
            "max_turn": max_turn,
            "agent_names": agent_names,
            "step": "init",
        }

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

        # Show running episodes (max 3) with step info
        running = []
        for scenario_id, info in list(self.episodes.items())[:3]:
            name = info.get("agent_names", scenario_id[:10])
            step = info.get("step", "")
            step_short = step[:8] if step else ""
            running.append(
                f"{name[:12]}:T{info['turn']}/{info['max_turn']}({step_short})"
            )

        status = " | ".join(running)
        if len(self.episodes) > 3:
            status += f" (+{len(self.episodes) - 3} more)"

        return f"[cyan]{self.completed}/{self.total}[/] | {status}"


class BanditSimulationRunner:
    """
    Runs a simulation with bandit-based dynamic prompt optimization.

    Supports:
    - Multiple bandit algorithms: EXP3, LinUCB, NeuralUCB
    - Two separate bandits for p1 and p2 (independent optimization)
    - Configurable optimization mode: 'p1', 'p2', 'both', or 'none' (baseline)
    - Timing and cost tracking
    """

    def __init__(
        self,
        scenario_id: str,
        model_name: str = "openrouter/openai/gpt-4o-mini",
        p1_model_name: str | None = None,
        p2_model_name: str | None = None,
        env_model_name: str = "openrouter/openai/gpt-4o",
        reward_eval_model_name: str | None = None,
        terminal_eval_model_name: str | None = None,
        max_turns: int = 10,
        max_tokens: int | None = None,
        embeddings_dir: Path | None = None,
        bandit_config: BanditConfig | None = None,
        bandit_type: BanditType = "exp3",
        optimize_mode: OptimizeMode = "both",
        push_to_db: bool = False,
        experiment_tag: str = "",
        progress_tracker: BatchProgressTracker | None = None,
        verbose: bool = True,
        experiment_dir: Path | None = None,
        tensorboard_dir: Path | None = None,
        use_context_embedding: bool = False,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        context_embedding_dim: int = 4096,
        alternate_optimization: bool = False,
        save_checkpoint: bool = False,
        selection_mode: str = "paraphrase",
        strategy_cache_dir: Path | None = None,
        strategy_version: str = "v1",
        static_strategy: bool = False,
    ) -> None:
        self.scenario_id = scenario_id
        self.model_name = model_name
        # P1 and P2 can have different models, fallback to model_name if not specified
        self.p1_model_name = p1_model_name or model_name
        self.p2_model_name = p2_model_name or model_name
        self.env_model_name = env_model_name
        # Evaluators can have different models, fallback to env_model_name if not specified
        self.reward_eval_model_name = reward_eval_model_name or env_model_name
        self.terminal_eval_model_name = terminal_eval_model_name or env_model_name
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.optimize_mode = optimize_mode
        self.bandit_type = "none" if optimize_mode == "none" else bandit_type
        self.push_to_db = push_to_db
        self.experiment_tag = experiment_tag
        self.progress_tracker = progress_tracker
        self.verbose = verbose
        self.experiment_dir = experiment_dir
        self.tensorboard_dir = tensorboard_dir
        self.use_context_embedding = use_context_embedding
        self.embedding_model = embedding_model
        self.context_embedding_dim = context_embedding_dim
        self.save_checkpoint = save_checkpoint
        self.selection_mode = selection_mode
        self.strategy_cache_dir = strategy_cache_dir
        self.strategy_version = strategy_version
        self.static_strategy = static_strategy
        self._static_selections: dict[
            str, tuple[int, str]
        ] = {}  # For static mode: {agent_key: (arm_idx, bio)}

        # Initialize EmbeddingClient if context embedding is enabled
        self.embedding_client: EmbeddingClient | None = None
        if use_context_embedding:
            logger.info(f"Context embedding enabled, model: {embedding_model}")
            self.embedding_client = EmbeddingClient(model=embedding_model)

        # Setup paths
        if embeddings_dir is None:
            embeddings_dir = Path(__file__).parent / "embeddings_backgrounds" / "hard"
        self.embeddings_dir = embeddings_dir

        # Determine which agents to optimize
        self.optimize_p1 = optimize_mode in ("p1", "both")
        self.optimize_p2 = optimize_mode in ("p2", "both")
        self.alternate_optimization = (
            alternate_optimization and self.optimize_p1 and self.optimize_p2
        )
        self._alternate_next_agent: Literal["p1", "p2"] = "p1"
        if self.alternate_optimization:
            logger.info(
                "Alternate optimization enabled: P1 and P2 bandits will update on alternating turns."
            )

        # Find combo by pk and load profiles FIRST (needed for strategy mode)
        combo = find_combo_by_pk(scenario_id)
        if combo is None:
            raise ValueError(
                f"No combo found with pk={scenario_id}. "
                f"Make sure this is a valid EnvAgentComboStorage pk."
            )

        self.combo = combo
        self.env_profile, self.agent_profiles = load_profiles(combo)

        # Initialize bandits only for agents being optimized
        self.bandit_p1: BaseBandit | None = None
        self.bandit_p2: BaseBandit | None = None
        self.prompt_space: PromptSpace | StrategySpace | None = None

        # Create bandit config with context embedding settings
        self.bandit_config = bandit_config or BanditConfig()
        if use_context_embedding:
            self.bandit_config.use_context_embedding = True
            self.bandit_config.embedding_model = embedding_model
            self.bandit_config.context_embedding_dim = context_embedding_dim

        if self.optimize_p1 or self.optimize_p2:
            # Initialize prompt space based on selection mode
            if self.selection_mode == "strategy":
                # Strategy mode: use social strategies appended to original bios
                from experiments.also.core.envs_dynamic_parallel import (
                    get_bio,
                    render_text_for_agent,
                )

                # Get raw bios with XML tags
                p1_background_raw = get_bio(
                    self.env_profile.relationship,
                    self.agent_profiles[0],
                    agent_id=0,
                )
                p2_background_raw = get_bio(
                    self.env_profile.relationship,
                    self.agent_profiles[1],
                    agent_id=1,
                )
                # Render to remove secrets (only show to correct agent)
                p1_background = render_text_for_agent(p1_background_raw, agent_id=0)
                p2_background = render_text_for_agent(p2_background_raw, agent_id=1)

                p1_name = f"{self.agent_profiles[0].first_name} {self.agent_profiles[0].last_name}"
                p2_name = f"{self.agent_profiles[1].first_name} {self.agent_profiles[1].last_name}"

                # Determine if we should skip embedding computation
                # OPRO, PromptBreeder, ProgressivePromptBreeder, EvoPrompt, and TPE use fitness-based selection, not embedding-based
                skip_embeddings = bandit_type in (
                    "opro",
                    "prompt_breeder",
                    "progressive_prompt_breeder",
                    "evoprompt_ga",
                    "evoprompt_de",
                    "tpe",
                )
                strategy_space_cls = (
                    LearnableStrategySpaceV2
                    if bandit_type == "adversarial_learnable_v2"
                    else StrategySpace
                )
                logger.info(
                    f"Initializing {strategy_space_cls.__name__} for strategy selection mode "
                    f"(skip_embeddings={skip_embeddings}, strategy_version={self.strategy_version})"
                )
                self.prompt_space = strategy_space_cls.from_scenario_backgrounds(
                    p1_background=p1_background,
                    p2_background=p2_background,
                    p1_name=p1_name,
                    p2_name=p2_name,
                    embedding_model=self.bandit_config.embedding_model,
                    embedding_dim=self.bandit_config.context_embedding_dim,
                    cache_dir=self.strategy_cache_dir,
                    skip_embeddings=skip_embeddings,
                    strategy_version=self.strategy_version,
                )
            else:
                # Paraphrase mode: load pre-generated bio paraphrases
                logger.info(
                    f"Loading prompt space (paraphrase mode) for scenario {scenario_id}"
                )
                self.prompt_space = PromptSpace(
                    scenario_id=scenario_id,
                    base_dir=embeddings_dir,
                )

            if self.optimize_p1:
                logger.info(f"Creating {bandit_type} bandit for P1 (Agent 1)")
                self.bandit_p1 = create_bandit(
                    bandit_type=bandit_type,
                    prompt_space=self.prompt_space,
                    config=self.bandit_config,
                    tensorboard_dir=self.tensorboard_dir,
                    output_dir=self.experiment_dir,  # For evolution logs
                )

            if self.optimize_p2:
                logger.info(f"Creating {bandit_type} bandit for P2 (Agent 2)")
                self.bandit_p2 = create_bandit(
                    bandit_type=bandit_type,
                    prompt_space=self.prompt_space,
                    config=self.bandit_config,
                    tensorboard_dir=self.tensorboard_dir,
                    output_dir=self.experiment_dir,  # For evolution logs
                )

        else:
            logger.info("Baseline mode: no bandit optimization")

        logger.info(
            f"Loaded profiles: "
            f"p1={self.agent_profiles[0].first_name} {self.agent_profiles[0].last_name}, p2={self.agent_profiles[1].first_name} {self.agent_profiles[1].last_name}"
        )
        logger.info(
            f"Optimization mode: {optimize_mode} "
            f"(p1={self.optimize_p1}, p2={self.optimize_p2})"
        )

        # Track simulation state
        self.current_turn = 0
        self.turn_rewards: list[dict] = []

        # Timing
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        # Register with progress tracker if provided
        if self.progress_tracker is not None:
            agent_names = f"{self.agent_profiles[0].first_name} vs {self.agent_profiles[1].first_name}"
            self.progress_tracker.register(self.scenario_id, max_turns, agent_names)

    def reset_for_next_episode(self) -> None:
        """
        Reset episode-specific state for running another episode.

        This preserves bandit state (selection_history, models) while
        resetting turn counters and episode-specific tracking.

        This allows running multiple episodes with the same runner instance
        to accumulate bandit data across episodes until a target turn count is reached.
        """
        # Reset turn tracking
        self.current_turn = 0

        # Clear turn rewards for the new episode
        # (Previous episode's rewards are already recorded in bandit history)
        self.turn_rewards = []

        # Note: self.bandit_p1 and self.bandit_p2 are NOT reset
        # Their selection_history, model parameters, etc. all persist
        # This allows continuous learning across episodes

    async def run_episode(self) -> dict:
        """
        Run a single episode with bandit-based bio optimization.

        Returns:
            Dictionary with episode results and bandit selection history
        """
        self.start_time = time.time()
        if self.verbose:
            logger.info(f"Starting episode for scenario {self.scenario_id}")
            logger.info(f"Mode: {self.optimize_mode}")

        # Create evaluators based on optimization mode
        # In baseline mode (none), use rule-based evaluator (no per-turn LLM eval)
        # In optimization mode, use RewardInTurnEvaluator for per-turn feedback
        if self.optimize_mode == "none":
            # Baseline: only rule-based termination + terminal evaluation
            from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator

            evaluator = RuleBasedTerminatedEvaluator(
                max_turn_number=self.max_turns,
                max_stale_turn=2,
            )
            logger.info(
                "Baseline mode: using RuleBasedTerminatedEvaluator (no per-turn LLM eval)"
            )
        else:
            # Optimization mode: per-turn reward evaluation for bandit training
            evaluator = RewardInTurnEvaluator(
                model_name=self.reward_eval_model_name,
                response_format_class=EvaluationForTwoAgents[SotopiaDimensions],
                max_turn_number=self.max_turns,
                max_stale_turn=2,
            )

        terminal_evaluator = TerminalEvaluator(
            model_name=self.terminal_eval_model_name,
            response_format_class=EvaluationForTwoAgents[SotopiaDimensions],
        )

        # Create environment
        env = DynamicPromptParallelSotopiaEnv(
            env_profile=self.env_profile,
            model_name=self.env_model_name,
            action_order="round-robin",
            evaluators=[evaluator],
            terminal_evaluators=[terminal_evaluator],
        )

        # Create agents with separate models for P1 and P2
        agents = [
            LLMAgent(
                agent_profile=self.agent_profiles[0],
                model_name=self.p1_model_name,
                max_tokens=self.max_tokens,
            ),
            LLMAgent(
                agent_profile=self.agent_profiles[1],
                model_name=self.p2_model_name,
                max_tokens=self.max_tokens,
            ),
        ]
        # print agents name and model
        logger.info(
            f"Environment Model: {env.model_name}, Reward Eval Model: {evaluator.model_name if hasattr(evaluator, 'model_name') else 'None'}, Terminal Eval Model: {terminal_evaluator.model_name}"
        )
        logger.info(f"P1 Agent: {agents[0].agent_name}, Model: {self.p1_model_name}")
        logger.info(f"P2 Agent: {agents[1].agent_name}, Model: {self.p2_model_name}")
        # Initialize agents dict
        from sotopia.agents.llm_agent import Agents

        agents_dict = Agents({agent.agent_name: agent for agent in agents})

        # Reset environment
        observations = env.reset(agents=agents_dict)
        p1_name = env.background.p1_name
        p2_name = env.background.p2_name

        logger.info(f"Environment reset. Agents: {p1_name} vs {p2_name}")

        # Log initial bio and mode
        mode_desc = {
            "both": "Optimizing both agents",
            "p1": f"Optimizing {p1_name} only",
            "p2": f"Optimizing {p2_name} only",
            "none": "Baseline (no optimization)",
        }.get(self.optimize_mode, "Unknown")
        if self.alternate_optimization:
            mode_desc += " (alternating updates)"

        if self.verbose:
            console.print(
                Panel(
                    f"[bold]Mode:[/] {mode_desc}\n\n"
                    f"[bold]Initial Bios[/bold]\n\n"
                    f"[cyan]{p1_name}:[/] {env.background.p1_background[:200]}...\n\n"
                    f"[cyan]{p2_name}:[/] {env.background.p2_background[:200]}...",
                    title="Episode Start",
                )
            )

        # Run the episode turn by turn
        terminated = {p1_name: False, p2_name: False}
        info: dict = {}  # Will hold final evaluation info

        # Track messages for EpisodeLog (format: list of turns, each turn is list of (sender, receiver, message))
        # First message is the initial observation from environment to agents
        messages: list[list[tuple[str, str, object]]] = [
            [
                ("Environment", p1_name, observations[p1_name]),
                ("Environment", p2_name, observations[p2_name]),
            ]
        ]

        # Use progress bar only in verbose mode
        progress_context = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[bold blue]Turn {task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console,
            disable=not self.verbose,  # Disable progress bar in batch mode
        )

        with progress_context as progress:
            task = (
                progress.add_task("Running episode...", total=self.max_turns)
                if self.verbose
                else None
            )

            while not all(terminated.values()) and self.current_turn < self.max_turns:
                self.current_turn += 1

                # Update progress bar (if verbose) or progress tracker (if batch mode)
                if self.verbose and task is not None:
                    progress.update(
                        task,
                        completed=self.current_turn,
                        description=f"Turn {self.current_turn}",
                    )
                if self.progress_tracker is not None:
                    self.progress_tracker.update(
                        self.scenario_id,
                        self.current_turn,
                        self.max_turns,
                        step="action",
                    )

                # Get actions from agents (parallel execution for speed)
                # Log that we're about to generate actions with current bios
                for agent in agents:
                    dynamic_obs = env.get_dynamic_observation(agent.agent_name)
                    logger.debug(
                        f"Agent {agent.agent_name} generating action at turn {self.current_turn}. "
                        f"Current bio (first 100 chars): {dynamic_obs.p1_background if agent.agent_name == p1_name else dynamic_obs.p2_background}"
                    )

                # Execute both agents' actions in parallel
                agent_obs_pairs = [
                    (agent, observations.get(agent.agent_name)) for agent in agents
                ]
                action_results = await asyncio.gather(
                    *[agent.aact(obs) for agent, obs in agent_obs_pairs]
                )
                actions = {
                    agent.agent_name: action
                    for (agent, _), action in zip(agent_obs_pairs, action_results)
                }

                for agent in agents:
                    action = actions[agent.agent_name]
                    logger.info(
                        f"[Turn {self.current_turn}] {agent.agent_name} action: {action.action_type} - {action.argument if action.argument else 'N/A'}"
                    )

                # Record agent actions to messages (agents -> Environment)
                messages[-1].extend(
                    [
                        (agent_name, "Environment", action)
                        for agent_name, action in actions.items()
                    ]
                )

                # Step environment (includes reward evaluation via LLM)
                if self.progress_tracker is not None:
                    self.progress_tracker.update(
                        self.scenario_id,
                        self.current_turn,
                        self.max_turns,
                        step="env_step",
                    )
                observations, rewards, terminated, truncated, info = await env.astep(
                    actions
                )

                # Record environment responses to messages (Environment -> agents)
                messages.append(
                    [
                        ("Environment", agent_name, observations[agent_name])
                        for agent_name in env.agents
                    ]
                )

                # Extract opponent response text for reflection
                # Key insight: In turn T, agents generate actions based on observations from turn T-1
                # So P2's action in turn T is their response to seeing P1's action from turn T-1
                # We need to backfill: update turn T-1's records with turn T's opponent actions
                if self.current_turn > 1:
                    # Current turn's actions are responses to previous turn's actions
                    p1_opponent_response = actions.get(p2_name).argument if actions.get(p2_name) and actions.get(p2_name).argument else None
                    p2_opponent_response = actions.get(p1_name).argument if actions.get(p1_name) and actions.get(p1_name).argument else None

                    # Backfill previous turn's records with opponent responses
                    if self.bandit_p1 and hasattr(self.bandit_p1, 'reflection_buffer'):
                        if len(self.bandit_p1.reflection_buffer["p1"]) > 0:
                            self.bandit_p1.reflection_buffer["p1"][-1].opponent_response_text = p1_opponent_response

                    if self.bandit_p2 and hasattr(self.bandit_p2, 'reflection_buffer'):
                        if len(self.bandit_p2.reflection_buffer["p2"]) > 0:
                            self.bandit_p2.reflection_buffer["p2"][-1].opponent_response_text = p2_opponent_response

                # Extract rewards
                p1_reward = rewards.get(p1_name, 0.0)
                p2_reward = rewards.get(p2_name, 0.0)

                # Extract dimension rewards for multi-dim prediction mode
                # complete_rating is tuple[float, dict[str, float]] with dimension scores
                p1_dim_rewards: dict[str, float] | None = None
                p2_dim_rewards: dict[str, float] | None = None
                if info:
                    p1_info = info.get(p1_name, {})
                    p2_info = info.get(p2_name, {})
                    p1_complete = (
                        p1_info.get("complete_rating")
                        if isinstance(p1_info, dict)
                        else None
                    )
                    p2_complete = (
                        p2_info.get("complete_rating")
                        if isinstance(p2_info, dict)
                        else None
                    )
                    # complete_rating is (overall, {dim: score, ...})
                    if isinstance(p1_complete, tuple) and len(p1_complete) >= 2:
                        dim_dict = p1_complete[1]
                        if isinstance(dim_dict, dict):
                            p1_dim_rewards = {
                                k: v
                                for k, v in dim_dict.items()
                                if k != "overall_score"
                            }
                    if isinstance(p2_complete, tuple) and len(p2_complete) >= 2:
                        dim_dict = p2_complete[1]
                        if isinstance(dim_dict, dict):
                            p2_dim_rewards = {
                                k: v
                                for k, v in dim_dict.items()
                                if k != "overall_score"
                            }

                turn_info = {
                    "turn": self.current_turn,
                    "p1_reward": p1_reward,
                    "p2_reward": p2_reward,
                    "p1_action": actions[p1_name].action_type,
                    "p2_action": actions[p2_name].action_type,
                    "p1_arm": self.bandit_p1.current_selections["p1"]
                    if self.bandit_p1
                    else 0,
                    "p2_arm": self.bandit_p2.current_selections["p2"]
                    if self.bandit_p2
                    else 0,
                    "terminated": all(terminated.values()),
                }
                self.turn_rewards.append(turn_info)

                # Log rewards
                if self.verbose:
                    console.print(
                        f"[bold green]Turn {self.current_turn}:[/] "
                        f"{p1_name} reward={p1_reward:.2f}, {p2_name} reward={p2_reward:.2f}"
                    )

                # Update bandits with rewards (only for optimized agents)
                # Generate context embedding if enabled
                # Skip embedding generation if optimize_mode is "none" (baseline mode)
                context_embedding = None
                recent_dialogue_text = ""
                context_turns = getattr(
                    self.bandit_config, "proposal_context_turns", 0
                )
                if context_turns:
                    recent_dialogue_text = extract_recent_agent_history(
                        messages, context_turns
                    )
                if (
                    self.use_context_embedding
                    and self.embedding_client
                    and self.optimize_mode != "none"
                ):
                    # Flatten messages for extraction
                    flat_messages = []
                    for turn_msgs in messages:
                        for sender, receiver, msg in turn_msgs:
                            flat_messages.append((sender, msg))

                    agent_history = extract_agent_history(flat_messages)
                    if agent_history:  # Only embed if there's content
                        context_embedding = self.embedding_client.get_embedding(
                            agent_history
                        )
                        logger.debug(
                            f"Generated context embedding for {agent_history}, shape: {context_embedding.shape}"
                        )

                # Update progress step for bandit update
                if self.progress_tracker is not None:
                    self.progress_tracker.update(
                        self.scenario_id,
                        self.current_turn,
                        self.max_turns,
                        step="update",
                    )

                if self.bandit_p1:
                    # Skip update if reward is 0 (likely API error or missing evaluation)
                    if p1_reward == 0.0:
                        logger.warning(
                            f"Skipping P1 bandit update: reward=0.00 at turn {self.current_turn} (likely API error)"
                        )
                    else:
                        logger.info(
                            f"Updating P1 bandit with reward={p1_reward:.2f} at turn {self.current_turn}"
                        )
                        p1_idx = self.bandit_p1.current_selections["p1"]
                        # Use update_with_context if bandit supports it and context is available
                        if self.use_context_embedding and hasattr(
                            self.bandit_p1, "update_with_context"
                        ):
                            self.bandit_p1.update_with_context(
                                "p1",
                                p1_idx,
                                p1_reward,
                                self.current_turn,
                                context_embedding,
                                dimension_rewards=p1_dim_rewards,
                                opponent_response_text=None,  # Will be backfilled next turn
                            )
                        else:
                            self.bandit_p1.update(
                                "p1", p1_idx, p1_reward, self.current_turn
                            )

                if self.bandit_p2:
                    # Skip update if reward is 0 (likely API error or missing evaluation)
                    if p2_reward == 0.0:
                        logger.warning(
                            f"Skipping P2 bandit update: reward=0.00 at turn {self.current_turn} (likely API error)"
                        )
                    else:
                        logger.info(
                            f"Updating P2 bandit with reward={p2_reward:.2f} at turn {self.current_turn}"
                        )
                        p2_idx = self.bandit_p2.current_selections["p2"]
                        # Use update_with_context if bandit supports it and context is available
                        if self.use_context_embedding and hasattr(
                            self.bandit_p2, "update_with_context"
                        ):
                            self.bandit_p2.update_with_context(
                                "p2",
                                p2_idx,
                                p2_reward,
                                self.current_turn,
                                context_embedding,
                                dimension_rewards=p2_dim_rewards,
                                opponent_response_text=None,  # Will be backfilled next turn
                            )
                        else:
                            self.bandit_p2.update(
                                "p2", p2_idx, p2_reward, self.current_turn
                            )

                # If not terminated, train and select new bios for optimized agents
                if not all(terminated.values()):
                    update_interval = self.bandit_config.update_interval
                    bandit_map = {
                        "p1": (self.bandit_p1, p1_name),
                        "p2": (self.bandit_p2, p2_name),
                    }
                    agents_to_refresh = [
                        agent_key
                        for agent_key, (bandit, _) in bandit_map.items()
                        if bandit is not None
                    ]

                    if self.alternate_optimization and len(agents_to_refresh) > 1:
                        target_agent = self._alternate_next_agent
                        if target_agent not in agents_to_refresh:
                            target_agent = "p2" if target_agent == "p1" else "p1"
                        agents_to_refresh = [target_agent]
                        self._alternate_next_agent = (
                            "p2" if target_agent == "p1" else "p1"
                        )
                        logger.debug(
                            f"Alternating optimization: updating {target_agent.upper()} on turn {self.current_turn}"
                        )

                    for agent_key in agents_to_refresh:
                        bandit, agent_name = bandit_map[agent_key]
                        if bandit is None:
                            continue

                        if (
                            update_interval > 0
                            and self.current_turn % update_interval == 0
                        ):
                            logger.info(
                                f"Training {agent_key.upper()} bandit at turn {self.current_turn} (interval={update_interval})"
                            )
                            bandit.train_model(verbose=True)

                        if bandit.is_stopped():
                            continue

                        # Static strategy mode: use cached selection after first turn
                        if (
                            self.static_strategy
                            and agent_key in self._static_selections
                        ):
                            arm_idx, new_bio = self._static_selections[agent_key]
                            logger.info(
                                f"[Turn {self.current_turn}] Static mode: reusing arm {arm_idx} for {agent_key}:{agent_name}"
                            )
                            env.update_agent_context(
                                agent_name=p1_name if agent_key == "p1" else p2_name,
                                at_turn=self.current_turn,
                                new_bio=new_bio,
                            )
                            continue

                        if self.use_context_embedding and context_embedding is not None:
                            if hasattr(bandit, "set_context_embedding"):
                                bandit.set_context_embedding(context_embedding)
                        if hasattr(bandit, "set_generation_context"):
                            bandit.set_generation_context(
                                recent_dialogue_text,
                                context_embedding=context_embedding,
                            )

                        # Update progress step for bandit select
                        if self.progress_tracker is not None:
                            self.progress_tracker.update(
                                self.scenario_id,
                                self.current_turn,
                                self.max_turns,
                                step="select",
                            )

                        # Use async select if available (e.g., NeuralAdversarialEvolutionBandit)
                        if hasattr(bandit, "select_async"):
                            arm_idx, new_bio, _ = await bandit.select_async(
                                agent_key, self.current_turn
                            )
                        else:
                            arm_idx, new_bio, _ = bandit.select(
                                agent_key, self.current_turn
                            )
                        # Cache selection for static strategy mode
                        if self.static_strategy:
                            self._static_selections[agent_key] = (arm_idx, new_bio)
                        env.update_agent_context(
                            agent_name=p1_name if agent_key == "p1" else p2_name,
                            at_turn=self.current_turn,
                            new_bio=new_bio,
                        )
                        logger.info(
                            f"[Turn {self.current_turn}] Updated {agent_key}:{agent_name} bio to arm {arm_idx}"
                        )

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        # Episode complete
        if self.verbose:
            console.print(
                Panel(
                    f"[bold green]Episode Complete![/]\n"
                    f"Duration: {format_duration(duration)} ({duration:.1f}s)",
                    title="Done",
                )
            )

        # Build and optionally save EpisodeLog
        (
            episode_pk,
            p1_final_reward,
            p2_final_reward,
        ) = await self._build_and_save_episode_log(
            env=env,
            agents=agents,
            messages=messages,
            info=info,
        )

        # Build final summary (includes final_rewards with dimension breakdowns)
        summary = self._build_summary(
            p1_name,
            p2_name,
            p1_final_reward=p1_final_reward,
            p2_final_reward=p2_final_reward,
        )
        summary["duration_seconds"] = duration
        summary["duration_formatted"] = format_duration(duration)
        summary["episode_pk"] = episode_pk

        # Mark completion in progress tracker
        if self.progress_tracker is not None:
            self.progress_tracker.complete(self.scenario_id, success=True)

        # Save final models if directory is set and save_checkpoint is enabled
        if self.experiment_dir and self.save_checkpoint:
            models_dir = self.experiment_dir / "models"
            models_dir.mkdir(exist_ok=True)
            if self.bandit_p1:
                self.bandit_p1.save_model(models_dir / "p1_bandit_final.pt")
            if self.bandit_p2:
                self.bandit_p2.save_model(models_dir / "p2_bandit_final.pt")

        return summary

    async def _build_and_save_episode_log(
        self,
        env: DynamicPromptParallelSotopiaEnv,
        agents: list[LLMAgent],
        messages: list[list[tuple[str, str, object]]],
        info: dict,
    ) -> tuple[
        str | None,
        float | tuple[float, dict[str, float]],
        float | tuple[float, dict[str, float]],
    ]:
        """
        Build EpisodeLog from episode data and optionally save to database.

        Args:
            env: The environment instance
            agents: List of LLM agents
            messages: List of message turns, each containing (sender, receiver, message) tuples
            info: Final evaluation info from terminal evaluator

        Returns:
            Tuple of (episode_pk, p1_complete_rating, p2_complete_rating)
            - episode_pk: Episode primary key if saved, None otherwise
            - p1_complete_rating: P1's final reward (float or tuple with breakdown)
            - p2_complete_rating: P2's final reward (float or tuple with breakdown)
        """
        # Convert messages to EpisodeLog format: list[list[tuple[str, str, str]]]
        # Each message needs to be converted to its natural language representation
        formatted_messages: list[list[tuple[str, str, str]]] = []
        for turn_messages in messages:
            turn_formatted = []
            for sender, receiver, msg in turn_messages:
                # Convert message object to natural language string
                if hasattr(msg, "to_natural_language"):
                    msg_str = msg.to_natural_language()
                else:
                    msg_str = str(msg)
                turn_formatted.append((sender, receiver, msg_str))
            formatted_messages.append(turn_formatted)

        # Extract rewards from info (complete_rating from terminal evaluator)
        p1_name = env.background.p1_name
        p2_name = env.background.p2_name

        # Get complete ratings from info if available
        p1_complete_rating: float | tuple[float, dict[str, float]] = 0.0
        p2_complete_rating: float | tuple[float, dict[str, float]] = 0.0
        reasoning = ""
        rewards_prompt = ""

        if info:
            if p1_name in info and "complete_rating" in info[p1_name]:
                p1_complete_rating = info[p1_name]["complete_rating"]
            if p2_name in info and "complete_rating" in info[p2_name]:
                p2_complete_rating = info[p2_name]["complete_rating"]
            if p1_name in info and "comments" in info[p1_name]:
                reasoning = str(info[p1_name].get("comments", ""))
            if "rewards_prompt" in info and "overall_prompt" in info["rewards_prompt"]:
                rewards_prompt = info["rewards_prompt"]["overall_prompt"]

        # Build EpisodeLog
        epilog = EpisodeLog(
            environment=self.env_profile.pk,
            agents=[agent.profile.pk for agent in agents],
            tag=self.experiment_tag,
            models=[env.model_name, agents[0].model_name, agents[1].model_name],
            messages=formatted_messages,
            reasoning=reasoning,
            rewards=[p1_complete_rating, p2_complete_rating],
            rewards_prompt=rewards_prompt,
        )

        logger.info(
            f"EpisodeLog built with {len(formatted_messages)} turns, "
            f"rewards: P1={p1_complete_rating}, P2={p2_complete_rating}"
        )
        # 如果奖励是tuple类型，说明是多维度奖励
        if isinstance(p1_complete_rating, tuple) or isinstance(
            p2_complete_rating, tuple
        ):
            logger.info("Multi-dimensional rewards detected")
        else:
            logger.warning(
                "pk: {epilog.pk} rewards: P1={p1_complete_rating}, P2={p2_complete_rating} is not tuple!"
            )
        # Log rewards details
        # Log rewards details as a table
        if isinstance(p1_complete_rating, tuple) or isinstance(
            p2_complete_rating, tuple
        ):
            table = Table(title="Final Scenario Rewards")
            table.add_column("Dimension", style="cyan")
            table.add_column(f"P1 ({p1_name})", style="green")
            table.add_column(f"P2 ({p2_name})", style="magenta")

            # Extract dimensions from P1 or P2
            dims = []
            if isinstance(p1_complete_rating, tuple):
                dims = list(p1_complete_rating[1].keys())
            elif isinstance(p2_complete_rating, tuple):
                dims = list(p2_complete_rating[1].keys())

            # Ensure overall_score is at the bottom
            if "overall_score" in dims:
                dims.remove("overall_score")
                dims.append("overall_score")

            for dim in dims:
                p1_val = "N/A"
                if isinstance(p1_complete_rating, tuple):
                    val = p1_complete_rating[1].get(dim, "N/A")
                    p1_val = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)

                p2_val = "N/A"
                if isinstance(p2_complete_rating, tuple):
                    val = p2_complete_rating[1].get(dim, "N/A")
                    p2_val = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)

                table.add_row(dim, p1_val, p2_val)

            console.print(table)

        # Save to database if requested
        episode_pk: str | None = None
        if self.push_to_db:
            try:
                # Use asyncio.to_thread to run synchronous save() in a thread
                # This prevents blocking the event loop and SSL socket issues
                await asyncio.to_thread(epilog.save)
                episode_pk = epilog.pk
                if self.verbose:
                    console.print(
                        f"[green]Episode saved to database with pk:[/] {episode_pk}"
                    )
                logger.info(f"Episode saved with pk: {episode_pk}")
            except Exception as e:
                logger.error(f"Failed to save episode: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                if self.verbose:
                    console.print(f"[red]Failed to save episode to database:[/] {e}")
                # Don't re-raise - allow experiment to continue even if DB save fails
                # The results are still saved to JSON files
                logger.warning(f"Continuing experiment despite database save failure")
        else:
            if self.verbose:
                console.print(
                    "[yellow]push_to_db=False, episode not saved to database[/]"
                )

        return episode_pk, p1_complete_rating, p2_complete_rating

    def _build_summary(
        self,
        p1_name: str,
        p2_name: str,
        p1_final_reward: float | tuple[float, dict[str, float]] | None = None,
        p2_final_reward: float | tuple[float, dict[str, float]] | None = None,
    ) -> dict:
        """Build the summary dictionary from both bandits."""
        summary: dict = {
            "scenario_id": self.scenario_id,
            "optimize_mode": self.optimize_mode,
            "alternate_optimization": self.alternate_optimization,
            "total_turns": self.current_turn,
            "turn_rewards": self.turn_rewards,
            "p1_name": p1_name,
            "p2_name": p2_name,
            "bandit_type": getattr(self, "bandit_type", "exp3"),
        }

        # Calculate average rewards
        p1_rewards = [t["p1_reward"] for t in self.turn_rewards]
        p2_rewards = [t["p2_reward"] for t in self.turn_rewards]
        summary["p1_avg_reward"] = (
            sum(p1_rewards) / len(p1_rewards) if p1_rewards else 0.0
        )
        summary["p2_avg_reward"] = (
            sum(p2_rewards) / len(p2_rewards) if p2_rewards else 0.0
        )

        # Add bandit-specific summaries
        if self.bandit_p1:
            p1_summary = self.bandit_p1.get_selection_summary()
            summary["p1_bandit"] = {
                "total_selections": p1_summary.get("total_selections", 0),
                "selections": p1_summary.get("p1_selections", []),
                "reward_progression": [
                    r
                    for r in p1_summary.get("reward_progression", [])
                    if r["agent"] == "p1"
                ],
            }
        else:
            summary["p1_bandit"] = None

        if self.bandit_p2:
            p2_summary = self.bandit_p2.get_selection_summary()
            summary["p2_bandit"] = {
                "total_selections": p2_summary.get("total_selections", 0),
                "selections": p2_summary.get("p2_selections", []),
                "reward_progression": [
                    r
                    for r in p2_summary.get("reward_progression", [])
                    if r["agent"] == "p2"
                ],
            }
        else:
            summary["p2_bandit"] = None

        # Add final rewards with dimension breakdowns (for local file evaluation)
        # This allows evaluate_by_tag.py to read goal and other dimensions from results.json
        # without requiring --push-to-db
        if p1_final_reward is not None or p2_final_reward is not None:
            summary["final_rewards"] = {
                "p1": self._serialize_reward(p1_final_reward),
                "p2": self._serialize_reward(p2_final_reward),
            }

        # Save prediction records for error analysis
        if self.p1_bandit is not None:
            self.p1_bandit.save_prediction_records()
        if self.p2_bandit is not None:
            self.p2_bandit.save_prediction_records()

        return summary

    def _serialize_reward(
        self, reward: float | tuple[float, dict[str, float]] | None
    ) -> dict[str, Any] | None:
        """Serialize reward to a JSON-compatible format."""
        if reward is None:
            return None
        if isinstance(reward, tuple) and len(reward) == 2:
            return {
                "overall": float(reward[0]),
                "breakdown": reward[1],
            }
        return {
            "overall": float(reward),
            "breakdown": None,
        }


def display_summary(summary: dict) -> None:
    """Display a nice summary of the simulation results."""
    console.print("\n")
    console.print(Panel("[bold]Simulation Summary[/]", expand=False))

    # 分数范围定义
    SCORE_RANGES = [
        (0, 10),  # believability
        (-5, 5),  # relationship
        (0, 10),  # knowledge
        (-10, 0),  # secret
        (-10, 0),  # social_rules
        (-5, 5),  # financial_and_material_benefits
        (0, 10),  # goal
    ]
    min_overall = sum(r[0] for r in SCORE_RANGES) / len(SCORE_RANGES)  # -30/7 ≈ -4.29
    max_overall = sum(r[1] for r in SCORE_RANGES) / len(SCORE_RANGES)  # 30/7 ≈ 4.29

    def denormalize_reward(normalized: float) -> float:
        """将归一化分数转回原始范围"""
        return normalized * (max_overall - min_overall) + min_overall

    def normalize_reward(original: float) -> float:
        """将原始分数归一化到 0-1 范围"""
        return (original - min_overall) / (max_overall - min_overall)

    # Basic stats
    console.print(f"[cyan]Scenario ID:[/] {summary['scenario_id']}")
    console.print(f"[cyan]Optimization Mode:[/] {summary['optimize_mode']}")
    alt_updates = summary.get("alternate_optimization", False)
    console.print(f"[cyan]Alternate Updates:[/] {'Yes' if alt_updates else 'No'}")
    console.print(f"[cyan]Total Turns:[/] {summary['total_turns']}")
    console.print(f"[cyan]Duration:[/] {summary.get('duration_formatted', 'N/A')}")

    # Average rewards - 混合了归一化和原始分数，直接显示
    p1_avg = summary["p1_avg_reward"]
    p2_avg = summary["p2_avg_reward"]

    console.print(f"\n[bold]Average Rewards:[/]")
    console.print(f"  {summary.get('p1_name', 'P1')}: {p1_avg:.2f}")
    console.print(f"  {summary.get('p2_name', 'P2')}: {p2_avg:.2f}")

    # Selection history tables for each bandit
    # Determine score column name based on bandit type
    bandit_type = summary.get("bandit_type", "exp3")
    score_col_name = {
        "linucb": "UCB Value",
        "neural_ucb": "UCB Value",
        "exp3": "Cumulative Score",
        "adversarial": "Cumulative Score",
        "prompt_breeder": "Fitness",
        "progressive_prompt_breeder": "Fitness",
        "tpe": "TPE Score",
    }.get(bandit_type, "Score")

    # 获取最后一轮的 turn 号
    final_turn = summary.get("total_turns", 0)

    if summary.get("p1_bandit"):
        table_p1 = Table(
            title=f"P1 ({summary.get('p1_name', 'Agent1')}) Selection History"
        )
        table_p1.add_column("Turn", style="cyan")
        table_p1.add_column("Arm", style="green")
        table_p1.add_column("Reward (orig|norm)", style="yellow")
        table_p1.add_column(score_col_name, style="blue")

        for sel in summary["p1_bandit"].get("selections", []):
            reward = sel.get("reward", 0.0)
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
        console.print(
            f"\n[yellow]P1 ({summary.get('p1_name', 'Agent1')}):[/] No optimization (baseline)"
        )

    if summary.get("p2_bandit"):
        table_p2 = Table(
            title=f"P2 ({summary.get('p2_name', 'Agent2')}) Selection History"
        )
        table_p2.add_column("Turn", style="cyan")
        table_p2.add_column("Arm", style="green")
        table_p2.add_column("Reward (orig|norm)", style="yellow")
        table_p2.add_column(score_col_name, style="blue")

        for sel in summary["p2_bandit"].get("selections", []):
            reward = sel.get("reward", 0.0)
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
        console.print(
            f"\n[yellow]P2 ({summary.get('p2_name', 'Agent2')}):[/] No optimization (baseline)"
        )

    # Turn-by-turn rewards table
    table_turns = Table(title="Turn-by-Turn Rewards")
    table_turns.add_column("Turn", style="cyan")
    table_turns.add_column("P1 Arm", style="magenta")
    table_turns.add_column("P1 Reward (orig|norm)", style="yellow")
    table_turns.add_column("P2 Arm", style="magenta")
    table_turns.add_column("P2 Reward (orig|norm)", style="yellow")

    for turn_info in summary.get("turn_rewards", []):
        p1_reward = turn_info["p1_reward"]
        p2_reward = turn_info["p2_reward"]
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

    results_dir = Path(__file__).parent / "results"
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run bandit-based dynamic prompt optimization simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scenario: Optimize both agents with separate bandits
  python run_bandit_simulation.py --scenario-id 01H7VKQHT745XAP1A4DDV8H419 --optimize both

  # Single scenario: Baseline (no optimization)
  python run_bandit_simulation.py --scenario-id 01H7VKQHT745XAP1A4DDV8H419 --optimize none

  # Batch mode: Run all hard scenarios with parallelism
  python run_bandit_simulation.py --batch --subset hard --batch-size 5 --optimize both

  # Batch mode: Run first 10 scenarios
  python run_bandit_simulation.py --batch --subset hard --max-episodes 10 --batch-size 3
""",
    )
    # Single scenario mode
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="01H7VKQHT745XAP1A4DDV8H419",
        help="Scenario (EnvAgentComboStorage pk) to run (single mode)",
    )
    parser.add_argument(
        "--scenario-ids",
        type=str,
        default=None,
        help=(
            "Multiple scenario IDs to run sequentially. "
            "Can be: comma-separated IDs, path to JSON file with combo list, "
            "or path to summary.json (will extract combos from 'scenarios'). "
            "Example: --scenario-ids 'ID1,ID2,ID3' or --scenario-ids path/to/summary.json"
        ),
    )

    # Batch mode arguments
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode (multiple scenarios in parallel)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["hard", "all", "hard_small", "non_hard"],
        default="hard",
        help="Dataset subset to use in batch mode: hard (hard scenarios), all (all scenarios), hard_small (first combo per hard env), non_hard (all - hard)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of concurrent episodes in batch mode (default: 5)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to run in batch mode (default: all)",
    )

    # Common arguments
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/openai/gpt-4o-mini",
        help="Default model for agents (used if p1-model/p2-model not specified)",
    )
    parser.add_argument(
        "--env-model",
        type=str,
        default="openrouter/openai/gpt-4o",
        help="Model for environment/evaluator calls (default: openrouter/openai/gpt-4o)",
    )
    parser.add_argument(
        "--p1-model",
        type=str,
        default=None,
        help="Model for P1 agent (overrides --model for P1)",
    )
    parser.add_argument(
        "--p2-model",
        type=str,
        default=None,
        help="Model for P2 agent (overrides --model for P2)",
    )
    parser.add_argument(
        "--reward-eval-model",
        type=str,
        default="openrouter/deepseek/deepseek-v3.2",
        help="Model for RewardInTurnEvaluator (default: deepseek-v3.2)",
    )
    parser.add_argument(
        "--terminal-eval-model",
        type=str,
        default=None,
        help="Model for TerminalEvaluator (overrides --env-model for terminal evaluation)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per episode",
    )
    parser.add_argument(
        "--bandit-type",
        type=str,
        choices=BANDIT_TYPES,
        default="adversarial",
        help=(
            "Type of bandit algorithm: 'adversarial', 'neural_adversarial', "
            "'adversarial_learnable_v2', 'linucb', 'neural_ucb', "
            "'prompt_breeder', etc. Default: adversarial"
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Number of hidden layers in the neural surrogate (default: 2)",
    )
    parser.add_argument(
        "--cumulative-score-mode",
        type=str,
        choices=["nn", "actual", "mean"],
        default="nn",
        help=(
            "Cumulative score mode for arm selection: "
            "'nn' = use neural network predicted scores, "
            "'actual' = use observed importance-weighted scores (pure EXP3), "
            "'mean' = average of both. Default: nn"
        ),
    )
    parser.add_argument(
        "--static-strategy",
        action="store_true",
        help=(
            "Static strategy mode: sample a strategy at the start of each episode "
            "and hold it fixed for all turns (no per-turn adaptation). "
            "Used for ablation to compare with dynamic ALSO."
        ),
    )
    parser.add_argument(
        "--greedy-selection",
        action="store_true",
        help=(
            "Greedy selection mode: directly select arm with highest NN prediction "
            "(no EXP3 softmax, no exploration). Used for ablation to test pure NN prediction."
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=5.0,
        help="EXP3 exploration parameter (default: 5.0 for stability)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="LinUCB exploration parameter (default: 1.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="NeuralUCB exploration parameter (default: 1.0)",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=1,
        help="Train bandit every N turns (default: 1)",
    )
    parser.add_argument(
        "--evolution-interval",
        type=int,
        default=5,
        help="Evolve population every N turns for evolutionary bandits (default: 5)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help=(
            "Population size for evolutionary bandits (OPRO, EvoPrompt, PromptBreeder, etc.). "
            "If not specified, uses algorithm defaults: OPRO/EvoPrompt=5, PromptBreeder/NeuralEvo=10. "
            "Set to match strategy count (e.g., 16 for v3) to use all strategies."
        ),
    )
    parser.add_argument(
        "--proposal-interval",
        type=int,
        default=5,
        help="For adversarial_learnable_v2: propose one new strategy every N turns (default: 5)",
    )
    parser.add_argument(
        "--proposal-warmup-turns",
        type=int,
        default=5,
        help="For adversarial_learnable_v2: do not propose new strategies before this turn (default: 5)",
    )
    parser.add_argument(
        "--proposal-context-turns",
        type=int,
        default=3,
        help="For adversarial_learnable_v2: include the most recent N dialogue turns in the proposer prompt (default: 3)",
    )
    parser.add_argument(
        "--proposal-model",
        type=str,
        default="openrouter/qwen/qwen-2.5-72b-instruct",
        help="For adversarial_learnable_v2: model used to propose new strategies",
    )
    parser.add_argument(
        "--proposal-temperature",
        type=float,
        default=0.3,
        help="For adversarial_learnable_v2: proposer sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--mask-unselected-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Neural adversarial bandit only: if enabled, only the selected arm's predicted score "
            "is recorded each turn (others are masked to 0). Default: disabled."
        ),
    )
    parser.add_argument(
        "--importance-weighted-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Neural adversarial bandit only: if enabled, training uses importance weighting "
            "(1 - reward) / p_selected; otherwise uses (1 - reward). Default: enabled."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help=(
            "Gamma mixing for exploration guarantee: probs = (1 - gamma) * softmax + gamma / n_arms. "
            "0.0 = disabled, 0.1 = recommended (default)"
        ),
    )
    parser.add_argument(
        "--enable-reflection",
        action="store_true",
        help="Enable textual gradient optimization for learnable strategies",
    )
    parser.add_argument(
        "--reflection-interval",
        type=int,
        default=10,
        help="Minimum turns between reflections (default: 10)",
    )
    parser.add_argument(
        "--reflection-lookback",
        type=int,
        default=5,
        help="Number of recent turns to check for variance (default: 5)",
    )
    parser.add_argument(
        "--score-decay",
        type=float,
        default=0.9,
        help=(
            "Score decay factor for cumulative scores: cumulative = decay * old + new. "
            "1.0 = disabled, 0.9 = recommended (default)"
        ),
    )
    parser.add_argument(
        "--dynamic-eta",
        action="store_true",
        default=False,
        help=(
            "Enable dynamic eta scheduling: eta_t = eta * sqrt(turn + 1). "
            "Early turns have smaller eta (more exploration), "
            "later turns have larger eta (more exploitation)."
        ),
    )
    parser.add_argument(
        "--nn-weight",
        type=float,
        default=0.5,
        help="Weight for NN predicted scores vs actual observed scores in cumulative score (default: 0.5)",
    )
    parser.add_argument(
        "--adaptive-nn-weight",
        action="store_true",
        default=False,
        help="Enable adaptive nn_weight that increases from 0 to 1 as turns progress",
    )
    parser.add_argument(
        "--adaptive-nn-weight-warmup",
        type=int,
        default=5,
        help="Midpoint turn where adaptive nn_weight ≈ 0.5 (default: 5)",
    )
    parser.add_argument(
        "--adaptive-nn-weight-scale",
        type=float,
        default=2.0,
        help="Transition sharpness for adaptive nn_weight (larger = smoother, default: 2.0)",
    )
    parser.add_argument(
        "--failure-penalty-threshold",
        type=float,
        default=0.3,
        help="Reward threshold below which failure penalty is applied (default: 0.3)",
    )
    parser.add_argument(
        "--failure-penalty-factor",
        type=float,
        default=1.5,
        help=(
            "Failure penalty multiplier for selected arm's score when reward < threshold. "
            "1.0 = disabled, 1.5 = recommended (default)"
        ),
    )
    # Selection strategy parameters for EvoPrompt/OPRO/PromptBreeder
    parser.add_argument(
        "--selection-strategy",
        type=str,
        choices=[
            "greedy",
            "epsilon_greedy",
            "softmax",
            "round_robin",
            "fitness_weighted",
        ],
        default=None,
        help=(
            "Selection strategy after initial exploration phase. "
            "Options: greedy, epsilon_greedy, softmax, round_robin, fitness_weighted. "
            "Default varies by bandit type (see config)."
        ),
    )
    parser.add_argument(
        "--selection-epsilon",
        type=float,
        default=0.1,
        help="Epsilon for epsilon-greedy selection strategy (default: 0.1)",
    )
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax selection strategy (default: 1.0)",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        choices=["p1", "p2", "both", "none"],
        default="both",
        help="Which agent(s) to optimize: 'p1', 'p2', 'both', or 'none' (baseline)",
    )
    parser.add_argument(
        "--alternate-optimization",
        action="store_true",
        help="When optimizing both agents, alternate bandit updates between P1 and P2 each turn",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["paraphrase", "strategy"],
        default="paraphrase",
        help="Selection mode: 'paraphrase' uses pre-generated bio paraphrases, 'strategy' appends social strategies to original bios",
    )
    parser.add_argument(
        "--strategy-cache-dir",
        type=Path,
        default=None,
        help="Cache directory for strategy embeddings (only used with --selection-mode strategy)",
    )
    parser.add_argument(
        "--strategy-version",
        type=str,
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v3_size3", "v3_size6", "v3_size12", "v3_diverse2_12", "v3_diverse4_12", "v4_size24", "v4_size48", "s6", "s8", "s24", "s48"],
        default="v1",
        help="Strategy version for social strategies: 'v1' (13 strategies), 'v2' (10 quadrant-based strategies), 'v3' (13 extended strategies), 'v3_size3' (4 strategies), 'v3_size6' (7 strategies), 'v3_size12' (13 strategies), 'v3_diverse2_12' (13 strategies: baseline + 12 active from 2 V3 parent families), 'v3_diverse4_12' (13 strategies: baseline + 12 active from 4 V3 parent families), 'v4' (49 semantic variants), 'v4_size24' (25 strategies), 'v4_size48' (49 strategies), 'v5' (10 adversarial-optimized), 'v6' (10 hard-optimized), or legacy 's6/s8/s24/s48'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="auto",
        help="Output file for results (JSON). Use 'auto' to auto-generate filename in results/ folder, or specify a path",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable saving output to file",
    )
    parser.add_argument(
        "--push-to-db",
        action="store_true",
        help="Push episode results to database for comparison",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom experiment tag (overrides auto-generated tag)",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default=None,
        help="Prefix to add to auto-generated tag",
    )
    parser.add_argument(
        "--tag-suffix",
        type=str,
        default=None,
        help="Suffix to add to auto-generated tag",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Directory containing pre-generated embeddings and paraphrased backgrounds",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for LLM output (default: 4096)",
    )
    parser.add_argument(
        "--context-embedding",
        action="store_true",
        help="Enable context embedding for bandit training (uses dialogue history)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Model for context embedding (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--context-embedding-dim",
        type=int,
        default=4096,
        help="Dimension of context embedding (default: 4096 for qwen3-embedding-8b, use 2048 for qwen3-embedding-4b)",
    )
    parser.add_argument(
        "--multi-dim-prediction",
        action="store_true",
        help=(
            "Enable multi-dimensional prediction mode. "
            "If enabled, the neural network predicts 7 dimension scores separately "
            "(believability, relationship, knowledge, secret, social_rules, financial_and_material_benefits, goal), "
            "computes per-dimension loss, then averages for final reward. "
            "Applicable to: neural_ucb, neural_adversarial, neural_evolution bandits."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a previous experiment directory. Only failed scenarios will be re-run and merged into the same results.",
    )
    parser.add_argument(
        "--exclude-from",
        type=Path,
        default=None,
        help=(
            "Path to a previous experiment's results.json or directory. "
            "Successfully completed scenarios from that experiment will be excluded. "
            "Useful for incremental runs, e.g., run 'all' subset while excluding already-completed 'hard' scenarios."
        ),
    )
    parser.add_argument(
        "--continue-from",
        type=Path,
        default=None,
        help=(
            "Continue from a previous experiment directory. "
            "Copies all previous results to the new experiment directory and only runs remaining scenarios. "
            "The new experiment will have complete records including both previous and new results."
        ),
    )
    parser.add_argument(
        "--extend-from-hard",
        type=str,
        default=None,
        help=(
            "Extend from a completed 'hard' subset experiment to 'all' subset. "
            "Specify the experiment tag or directory of the completed hard experiment. "
            "This will: 1) Copy all data (DB records + files) to a new experiment, "
            "2) Automatically switch to 'all' subset, "
            "3) Exclude already-completed hard scenarios, "
            "4) Run only the remaining scenarios. "
            "Example: --extend-from-hard bandit_neural_evolution_both_hard_deepseek_v32_mus0_iw1_20260121_133647"
        ),
    )
    parser.add_argument(
        "--calculate-cost",
        action="store_true",
        help="Calculate API costs after experiment (requires OpenRouter API, may be slow)",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=None,
        help="HTTP proxy port for this simulation (e.g., 7891). Sets HTTP_PROXY and HTTPS_PROXY env vars.",
    )
    parser.add_argument(
        "--auto-proxy",
        action="store_true",
        help="Automatically start a mihomo proxy instance with an available port. Requires mihomo in PATH.",
    )
    parser.add_argument(
        "--mihomo-config",
        type=Path,
        default=None,
        help="mihomo base config file path for --auto-proxy (auto-detected if not specified)",
    )
    parser.add_argument(
        "--proxy-base-port",
        type=int,
        default=7900,
        help="Base port for auto-proxy. If occupied, will search upward. Use different values for parallel runs (default: 7900)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=500,
        help="Maximum number of automatic retries for failed scenarios (default: 10)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated list of CUDA device IDs to distribute NN models across (e.g., '0,1,2,3'). "
        "If not specified, uses CUDA_VISIBLE_DEVICES or defaults to 'cuda'. "
        "Useful when running many parallel simulations that can't all fit on one GPU.",
    )
    return parser.parse_args()


def parse_scenario_ids(
    scenario_ids_arg: str, embeddings_dir: str | Path | None = None
) -> list[str]:
    """
    Parse --scenario-ids argument to get list of combo IDs.

    Supports:
    - Comma-separated IDs: "ID1,ID2,ID3"
    - Path to JSON file with list: ["ID1", "ID2", ...]
    - Path to summary.json: extracts combos from scenarios[*].combos
    - Plain text file with one ID per line
    - **Env IDs**: automatically converted to combo IDs by querying the database

    If embeddings_dir is provided, only returns combo IDs that exist in the embeddings directory.

    Returns:
        List of scenario (combo) IDs
    """
    import json
    from pathlib import Path

    # Check if it's a file path
    path = Path(scenario_ids_arg)
    if path.exists() and path.is_file():
        with open(path) as f:
            content = f.read().strip()

        # Try JSON first
        try:
            data = json.loads(content)

            # Check if it's a summary.json format (has 'scenarios' key)
            if isinstance(data, dict) and "scenarios" in data:
                raw_ids = []
                for scenario in data["scenarios"]:
                    if "combos" in scenario:
                        raw_ids.extend(scenario["combos"])
                logger.info(f"Loaded {len(raw_ids)} IDs from summary file: {path}")
            elif isinstance(data, list):
                raw_ids = data
                logger.info(f"Loaded {len(raw_ids)} IDs from JSON file: {path}")
            else:
                raise ValueError(f"Unsupported JSON format in {path}")

        except json.JSONDecodeError:
            # Not JSON, treat as plain text (one ID per line)
            raw_ids = [line.strip() for line in content.split("\n") if line.strip()]
            logger.info(f"Loaded {len(raw_ids)} IDs from text file: {path}")
    else:
        # Treat as comma-separated IDs
        raw_ids = [s.strip() for s in scenario_ids_arg.split(",") if s.strip()]
        logger.info(f"Parsed {len(raw_ids)} comma-separated IDs")

    # Get available combo IDs from embeddings directory if provided
    available_combo_ids: set[str] | None = None
    if embeddings_dir:
        emb_path = Path(embeddings_dir)
        if emb_path.exists():
            available_combo_ids = {d.name for d in emb_path.iterdir() if d.is_dir()}
            logger.info(
                f"Found {len(available_combo_ids)} combo IDs in embeddings directory: {emb_path}"
            )

    # Resolve IDs: check if they are combo IDs or env IDs
    resolved_combo_ids: list[str] = []
    env_to_combo_cache: dict[str, list[str]] = {}

    for raw_id in raw_ids:
        # First check if it's directly a combo ID in embeddings
        if available_combo_ids and raw_id in available_combo_ids:
            resolved_combo_ids.append(raw_id)
            continue

        # Try as combo ID in database
        combo = find_combo_by_pk(raw_id)
        if combo:
            if available_combo_ids is None or raw_id in available_combo_ids:
                resolved_combo_ids.append(raw_id)
            else:
                logger.warning(
                    f"Combo ID {raw_id} exists in DB but not in embeddings directory"
                )
            continue

        # Treat as env_id: query database for matching combos
        if raw_id not in env_to_combo_cache:
            try:
                combos_for_env = list(
                    EnvAgentComboStorage.find(
                        EnvAgentComboStorage.env_id == raw_id
                    ).all()
                )
                env_to_combo_cache[raw_id] = [c.pk for c in combos_for_env if c.pk]
            except Exception as e:
                logger.warning(f"Failed to query combos for env_id {raw_id}: {e}")
                env_to_combo_cache[raw_id] = []

        combo_ids_for_env = env_to_combo_cache[raw_id]
        if combo_ids_for_env:
            # Filter by available embeddings if specified
            if available_combo_ids:
                matching = [
                    cid for cid in combo_ids_for_env if cid in available_combo_ids
                ]
                if matching:
                    logger.info(
                        f"Env ID {raw_id} -> {len(matching)} combo(s) with embeddings: {matching}"
                    )
                    resolved_combo_ids.extend(matching)
                else:
                    logger.warning(
                        f"Env ID {raw_id} has {len(combo_ids_for_env)} combos but none in embeddings directory"
                    )
            else:
                logger.info(f"Env ID {raw_id} -> {len(combo_ids_for_env)} combo(s)")
                resolved_combo_ids.extend(combo_ids_for_env)
        else:
            logger.warning(f"ID {raw_id} not found as combo or env in database")

    logger.info(
        f"Resolved {len(raw_ids)} input IDs to {len(resolved_combo_ids)} combo IDs"
    )
    return resolved_combo_ids


async def run_single_scenario(
    scenario_id: str,
    args: argparse.Namespace,
    experiment_tag: str,
    bandit_config: BanditConfig,
    bandit_type: str = "exp3",
    progress_tracker: BatchProgressTracker | None = None,
    verbose: bool = True,
    experiment_dir: Path | None = None,
    tensorboard_dir: Path | None = None,
) -> dict:
    """
    Run a single scenario with bandit optimization.

    Args:
        scenario_id: The scenario ID (EnvAgentComboStorage pk)
        args: Parsed command line arguments
        experiment_tag: Experiment tag for logging
        bandit_config: Bandit configuration
        bandit_type: Type of bandit algorithm to use
        progress_tracker: Optional progress tracker for batch mode
        verbose: Whether to show detailed output

    Returns:
        Dictionary with episode results
    """
    try:
        runner = BanditSimulationRunner(
            scenario_id=scenario_id,
            model_name=args.model,
            p1_model_name=args.p1_model,
            p2_model_name=args.p2_model,
            env_model_name=args.env_model,
            reward_eval_model_name=args.reward_eval_model,
            terminal_eval_model_name=args.terminal_eval_model,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            embeddings_dir=args.embeddings_dir,  # Pass custom embeddings dir
            bandit_config=bandit_config,
            bandit_type=bandit_type,  # type: ignore
            optimize_mode=args.optimize,  # type: ignore
            push_to_db=args.push_to_db,
            experiment_tag=experiment_tag,
            progress_tracker=progress_tracker,
            verbose=verbose,
            experiment_dir=experiment_dir,
            tensorboard_dir=tensorboard_dir,
            use_context_embedding=args.context_embedding,
            embedding_model=args.embedding_model,
            context_embedding_dim=args.context_embedding_dim,
            alternate_optimization=args.alternate_optimization,
            selection_mode=args.selection_mode,
            strategy_cache_dir=args.strategy_cache_dir,
            strategy_version=args.strategy_version,
            static_strategy=getattr(args, "static_strategy", False),
        )

        summary = await runner.run_episode()
        return {
            "success": True,
            "scenario_id": scenario_id,
            "summary": summary,
            "error": None,
        }
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in scenario {scenario_id}: {e}")
        if progress_tracker:
            progress_tracker.complete(scenario_id, success=False)
        return {
            "success": False,
            "scenario_id": scenario_id,
            "summary": None,
            "error": str(e),
        }


async def retry_failed_scenarios(
    failed_scenario_ids: list[str],
    args: argparse.Namespace,
    experiment_tag: str,
    bandit_config: BanditConfig,
    bandit_type: str,
    experiment_dir: Path,
    tensorboard_dir: Path,
    retry_attempt: int,
) -> dict:
    """
    Retry failed scenarios with parallel execution.

    Args:
        failed_scenario_ids: List of scenario IDs that failed
        args: Command line arguments
        experiment_tag: Experiment tag for logging
        bandit_config: Bandit configuration
        bandit_type: Type of bandit algorithm
        experiment_dir: Experiment directory path
        tensorboard_dir: TensorBoard directory path
        retry_attempt: Current retry attempt number (1-indexed)

    Returns:
        Dictionary containing:
            - "results": List of retry results
            - "success_count": Number of newly successful scenarios
            - "still_failed": Number of scenarios still failing
            - "failed_ids": List of scenario IDs still failing
    """
    console.print(
        Panel(
            f"[bold yellow]Retry Attempt {retry_attempt}[/]\n"
            f"Retrying {len(failed_scenario_ids)} failed scenarios...",
            title=f"Automatic Retry #{retry_attempt}",
        )
    )

    # Initialize progress tracker for retry
    progress_tracker = BatchProgressTracker()
    progress_tracker.total = len(failed_scenario_ids)

    # Use semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.batch_size)
    retry_results: list[dict] = []

    async def run_with_semaphore(scenario_id: str) -> dict:
        async with semaphore:
            return await run_single_scenario(
                scenario_id=scenario_id,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=bandit_config,
                bandit_type=bandit_type,
                progress_tracker=progress_tracker,
                verbose=False,
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
            )

    # Incremental results file for retry
    incremental_results_path = experiment_dir / f"results_retry_{retry_attempt}.jsonl"

    def save_incremental_result(result: dict) -> None:
        """Append a single result to incremental retry file."""
        try:
            with open(incremental_results_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save incremental retry result: {e}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(
            f"[yellow]Retrying {len(failed_scenario_ids)} scenarios...",
            total=len(failed_scenario_ids),
        )

        async def update_progress_description() -> None:
            while progress_tracker.completed < progress_tracker.total:
                status = progress_tracker.get_status()
                progress.update(task, description=status)
                await asyncio.sleep(0.5)

        tasks = [run_with_semaphore(sid) for sid in failed_scenario_ids]
        update_task = asyncio.create_task(update_progress_description())

        for coro in asyncio.as_completed(tasks):
            result = await coro
            retry_results.append(result)
            save_incremental_result(result)
            progress.update(task, advance=1)

        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass

    # Calculate retry statistics
    success_count = sum(1 for r in retry_results if r["success"])
    still_failed_count = len(retry_results) - success_count
    still_failed_ids = [r["scenario_id"] for r in retry_results if not r["success"]]

    console.print(f"[green]Newly Succeeded: {success_count}[/]")
    console.print(f"[red]Still Failed: {still_failed_count}[/]")

    if still_failed_ids:
        console.print(f"\n[bold red]Still Failed Scenario IDs:[/]")
        for sid in still_failed_ids[:5]:
            console.print(f"  • {sid}")
        if len(still_failed_ids) > 5:
            console.print(f"  ... and {len(still_failed_ids) - 5} more")

    return {
        "results": retry_results,
        "success_count": success_count,
        "still_failed": still_failed_count,
        "failed_ids": still_failed_ids,
    }


async def run_batch_episodes(args: argparse.Namespace) -> None:
    """Run episodes in batch mode with parallel execution."""
    batch_start_time = time.time()

    # Resolve model names for P1 and P2 first (needed for tag generation)
    p1_model = args.p1_model or args.model
    p2_model = args.p2_model or args.model

    if args.tag:
        experiment_tag = args.tag
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_tag = f"mus{int(args.mask_unselected_scores)}"
        iw_tag = f"iw{int(args.importance_weighted_reward)}"

        # Extract short model name for unique tag (e.g., "qwen-2.5-72b-instruct" -> "qwen25_72b")
        def get_model_short(model: str) -> str:
            name = model.split("/")[-1]  # Get last part after /
            return name.replace(".", "").replace("-", "_")[:20]  # Shorten and sanitize

        p1_short = get_model_short(p1_model)
        p2_short = get_model_short(p2_model)
        # If same model, use single tag; if different, use "p1model_vs_p2model"
        if p1_short == p2_short:
            model_tag = p1_short
        else:
            model_tag = f"{p1_short}_vs_{p2_short}"
        experiment_tag = f"bandit_{args.bandit_type}_{args.optimize}_{args.subset}_{model_tag}_{mask_tag}_{iw_tag}_{timestamp}"

        # Apply prefix/suffix if provided
        if args.tag_prefix:
            experiment_tag = f"{args.tag_prefix}_{experiment_tag}"
        if args.tag_suffix:
            experiment_tag = f"{experiment_tag}_{args.tag_suffix}"

    # Mode description
    mode_desc = {
        "both": "Optimizing BOTH agents",
        "p1": "Optimizing P1 only",
        "p2": "Optimizing P2 only",
        "none": "BASELINE (no optimization)",
    }.get(args.optimize, "Unknown")

    console.print(
        Panel(
            f"[bold green]Batch Bandit Simulation[/]\n\n"
            f"Subset: {args.subset}\n"
            f"Batch Size (concurrency): {args.batch_size}\n"
            f"Max Episodes: {args.max_episodes or 'all'}\n"
            f"P1 Model: {p1_model}\n"
            f"P2 Model: {p2_model}\n"
            f"Env Model: {args.env_model}\n"
            f"Max Turns: {args.max_turns}\n"
            f"[bold cyan]Bandit Type: {args.bandit_type}[/]\n"
            f"ETA (EXP3): {args.eta}, Alpha (LinUCB): {args.alpha}, Beta (NeuralUCB): {args.beta}\n"
            f"[bold yellow]Mode: {mode_desc}[/]\n"
            f"Push to DB: {args.push_to_db}",
            title="Batch Configuration",
        )
    )

    # Create experiment directory structure
    experiment_dir = (
        PROJECT_ROOT / "experiments/also/outputs" / experiment_tag
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = experiment_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    tensorboard_dir = experiment_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    models_dir = experiment_dir / "models"
    models_dir.mkdir(exist_ok=True)

    llm_logs_dir = experiment_dir / "llm_logs"
    llm_logs_dir.mkdir(exist_ok=True)

    # Enable LLM call logging
    log_file = llm_logs_dir / "calls.jsonl"
    enable_llm_call_logging(log_file, experiment_tag=experiment_tag)
    console.print(f"[cyan]LLM call log:[/] {log_file}")

    # Enable LiteLLM shared async client to reduce SSL connection overhead
    try:
        import litellm

        litellm.enable_shared_async_client = True
    except ImportError:
        pass

    # Setup terminal logging to file in experiment dir
    log_file_path, tee_stdout, tee_stderr = setup_terminal_logging(
        experiment_name=f"bandit_{args.bandit_type}_{args.optimize}",
        log_dir=logs_dir,
    )
    configure_logger(level="DEBUG", include_function=True)

    # Determine embeddings directory (not needed for strategy mode without --scenario-ids filtering)
    embeddings_dir: Path | None = None
    if args.embeddings_dir:
        embeddings_dir = Path(args.embeddings_dir)
    elif args.selection_mode != "strategy":
        # Only compute embeddings_dir for paraphrase mode
        # hard_small is a subset of hard, so use hard's embeddings
        subset_for_embeddings = "hard" if args.subset == "hard_small" else args.subset
        embeddings_dir = (
            Path(__file__).parent / "embeddings_backgrounds" / subset_for_embeddings
        )

    # Get combos - either from --scenario-ids or from subset
    if args.scenario_ids:
        # Use specific scenario IDs from --scenario-ids
        # parse_scenario_ids now handles env_id -> combo_id conversion and embeddings filtering
        # For strategy mode without embeddings_dir, skip embeddings filtering
        scenario_ids = parse_scenario_ids(
            args.scenario_ids, embeddings_dir=embeddings_dir
        )
        console.print(
            f"[cyan]Using {len(scenario_ids)} resolved combo IDs from --scenario-ids[/]"
        )

        if not scenario_ids:
            console.print("[red]Error: No valid combo IDs found. Check if:[/]")
            console.print(f"  - The input IDs exist in the database")
            if embeddings_dir:
                console.print(
                    f"  - The embeddings directory contains matching combos: {embeddings_dir}"
                )
            raise ValueError("No valid combo IDs found after resolution")

        # Create lightweight combo objects for iteration
        class ComboStub:
            def __init__(self, pk: str):
                self.pk = pk

        filtered_combos = [ComboStub(sid) for sid in scenario_ids]  # type: ignore
    elif args.selection_mode == "strategy":
        # Strategy mode: get combos directly from database, no embeddings filtering needed
        # Strategy embeddings are generated on-demand from strategy_cache_dir
        combos = get_combos_for_subset(args.subset)
        console.print(
            f"[green]Found {len(combos)} combos in database (strategy mode, no embeddings filtering)[/]"
        )
        filtered_combos = combos
    else:
        # Paraphrase mode: get combos from subset and filter by available embeddings
        if embeddings_dir is None:
            raise ValueError("embeddings_dir is required for paraphrase mode")
        available_scenarios = get_available_scenarios(embeddings_dir)
        console.print(
            f"[cyan]Available scenarios with embeddings:[/] {len(available_scenarios)}"
        )

        # Get combos from database
        combos = get_combos_for_subset(args.subset)
        console.print(f"[green]Found {len(combos)} combos in database[/]")

        # Filter combos to only those with embeddings
        filtered_combos = [c for c in combos if c.pk in available_scenarios]
        console.print(f"[cyan]Combos with embeddings:[/] {len(filtered_combos)}")

    # Handle --continue-from: load previous results and exclude completed scenarios
    previous_results: list[dict] = []
    continue_from_path: Path | None = None

    if args.continue_from:
        continue_from_path = args.continue_from
        try:
            previous_results, prev_data = load_previous_results(continue_from_path)
            completed_scenarios = {
                r["scenario_id"] for r in previous_results if r.get("success", False)
            }
            original_count = len(filtered_combos)
            filtered_combos = [
                c for c in filtered_combos if c.pk not in completed_scenarios
            ]
            excluded_count = original_count - len(filtered_combos)

            console.print(
                Panel(
                    f"[bold cyan]Continuing from previous experiment[/]\n\n"
                    f"Source: {continue_from_path}\n"
                    f"Previous results loaded: {len(previous_results)}\n"
                    f"Successfully completed: {len(completed_scenarios)}\n"
                    f"Scenarios to skip: {excluded_count}\n"
                    f"[bold green]Remaining to run: {len(filtered_combos)}[/]",
                    title="Continue Mode",
                    border_style="cyan",
                )
            )
        except FileNotFoundError as e:
            console.print(f"[red]Error loading continue-from file: {e}[/]")
            raise

    # Exclude scenarios from previous experiment if --exclude-from is specified
    elif args.exclude_from:
        exclude_path = args.exclude_from
        try:
            excluded_scenarios = load_completed_scenarios_from_results(exclude_path)
            original_count = len(filtered_combos)
            filtered_combos = [
                c for c in filtered_combos if c.pk not in excluded_scenarios
            ]
            excluded_count = original_count - len(filtered_combos)
            console.print(
                f"[yellow]Excluded {excluded_count} scenarios from previous experiment:[/] {exclude_path}"
            )
            console.print(
                f"[cyan]Remaining scenarios to run:[/] {len(filtered_combos)}"
            )
        except FileNotFoundError as e:
            console.print(f"[red]Error loading exclusion file: {e}[/]")
            raise

    # Handle extend mode: exclude completed scenarios from hard experiment
    if (
        hasattr(args, "_extend_completed_scenarios")
        and args._extend_completed_scenarios
    ):
        excluded_scenarios = args._extend_completed_scenarios
        original_count = len(filtered_combos)
        filtered_combos = [c for c in filtered_combos if c.pk not in excluded_scenarios]
        excluded_count = original_count - len(filtered_combos)
        console.print(
            f"[yellow]Excluded {excluded_count} scenarios from hard experiment (extend mode)[/]"
        )
        console.print(f"[cyan]Remaining scenarios to run:[/] {len(filtered_combos)}")

    # Limit if max_episodes specified
    if args.max_episodes and args.max_episodes < len(filtered_combos):
        filtered_combos = filtered_combos[: args.max_episodes]
        console.print(f"[yellow]Limited to {args.max_episodes} episodes[/]")

    total_episodes = len(filtered_combos)
    console.print(
        f"[cyan]Will run {total_episodes} episodes with concurrency={args.batch_size}[/]"
    )

    # Create bandit config based on bandit type (with selection strategy support)
    bandit_config = create_bandit_config_from_args(args)
    run_config_path = save_run_config(
        experiment_dir=experiment_dir,
        experiment_tag=experiment_tag,
        args=args,
        bandit_config=bandit_config,
        bandit_type=args.bandit_type,
    )
    console.print(f"[cyan]Run config:[/] {run_config_path}")

    # Initialize progress tracker
    progress_tracker = BatchProgressTracker()
    progress_tracker.total = total_episodes

    # Parse device list for multi-GPU distribution
    device_list: list[str] | None = None
    if args.devices:
        device_list = [f"cuda:{d.strip()}" for d in args.devices.split(",")]
        console.print(f"[cyan]Distributing NN models across devices:[/] {device_list}")

    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(args.batch_size)
    results: list[dict] = []

    # Counter for round-robin device assignment
    device_counter = [0]  # Use list to allow mutation in nested function

    async def run_with_semaphore(combo: EnvAgentComboStorage) -> dict:
        async with semaphore:
            # Create a copy of bandit_config with assigned device
            scenario_bandit_config = bandit_config
            if device_list:
                import copy

                scenario_bandit_config = copy.deepcopy(bandit_config)
                assigned_device = device_list[device_counter[0] % len(device_list)]
                scenario_bandit_config.device = assigned_device
                device_counter[0] += 1
                logger.debug(f"Scenario {combo.pk} assigned to {assigned_device}")

            return await run_single_scenario(
                scenario_id=combo.pk,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=scenario_bandit_config,
                bandit_type=args.bandit_type,
                progress_tracker=progress_tracker,
                verbose=False,  # Disable verbose output in batch mode
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
            )

    # Incremental results file for crash recovery
    incremental_results_path = experiment_dir / "results_incremental.jsonl"

    def save_incremental_result(result: dict) -> None:
        """Append a single result to incremental file (JSONL format)."""
        try:
            with open(incremental_results_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save incremental result: {e}")

    # If continuing from previous experiment, copy previous results to incremental file first
    if previous_results:
        console.print(
            f"[cyan]Copying {len(previous_results)} previous results to new experiment...[/]"
        )
        for prev_result in previous_results:
            save_incremental_result(prev_result)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Running {total_episodes} episodes...", total=total_episodes
        )

        # Background task to update progress description
        async def update_progress_description() -> None:
            while progress_tracker.completed < progress_tracker.total:
                status = progress_tracker.get_status()
                progress.update(task, description=status)
                await asyncio.sleep(0.5)

        # Create all tasks
        tasks = [run_with_semaphore(c) for c in filtered_combos]
        update_task = asyncio.create_task(update_progress_description())

        # Process tasks as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            # Save incrementally for crash recovery
            save_incremental_result(result)
            progress.update(task, advance=1)

        # Cancel update task
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass

    # Track retry history
    retry_history: list[dict] = []
    all_results = results  # Start with initial results

    # Auto-retry logic if enabled
    if args.max_retries > 0:
        current_retry = 0

        while current_retry < args.max_retries:
            # Find failed scenarios
            failed_scenario_ids = [
                r["scenario_id"] for r in all_results if not r["success"]
            ]

            # If no failures, exit retry loop
            if not failed_scenario_ids:
                console.print(
                    "\n[bold green]All scenarios succeeded! No retries needed.[/]"
                )
                break

            console.print(
                f"\n[yellow]Found {len(failed_scenario_ids)} failed scenarios.[/]"
            )
            console.print(
                f"[cyan]Starting retry {current_retry + 1}/{args.max_retries}...[/]"
            )

            # Retry failed scenarios
            retry_start_time = time.time()
            retry_result = await retry_failed_scenarios(
                failed_scenario_ids=failed_scenario_ids,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=bandit_config,
                bandit_type=args.bandit_type,
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
                retry_attempt=current_retry + 1,
            )
            retry_duration = time.time() - retry_start_time

            # Record retry history
            retry_history.append(
                {
                    "retry_attempt": current_retry + 1,
                    "failed_count": len(failed_scenario_ids),
                    "failed_ids": failed_scenario_ids,
                    "newly_succeeded": retry_result["success_count"],
                    "still_failed": retry_result["still_failed"],
                    "still_failed_ids": retry_result["failed_ids"],
                    "retry_duration_seconds": retry_duration,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Merge results: replace failed scenarios with retry results
            retry_results_map = {r["scenario_id"]: r for r in retry_result["results"]}

            merged_results = []
            for r in all_results:
                if r["scenario_id"] in retry_results_map:
                    # Replace with retry result
                    merged_results.append(retry_results_map[r["scenario_id"]])
                else:
                    # Keep original result
                    merged_results.append(r)

            all_results = merged_results

            # Check if we should continue retrying
            if retry_result["still_failed"] == 0:
                console.print(
                    f"\n[bold green]All scenarios succeeded after {current_retry + 1} retries![/]"
                )
                break

            current_retry += 1

            if current_retry < args.max_retries:
                console.print(
                    f"\n[yellow]Will retry {retry_result['still_failed']} scenarios in next attempt...[/]"
                )
            else:
                console.print(
                    f"\n[bold red]Max retries ({args.max_retries}) reached. {retry_result['still_failed']} scenarios still failed.[/]"
                )

    # Merge previous results with new results if continuing from previous experiment
    if previous_results:
        # Combine previous results with new results
        all_results = previous_results + all_results
        console.print(
            f"\n[cyan]Merged {len(previous_results)} previous + {len(results)} new = {len(all_results)} total results[/]"
        )

    # Calculate timing (include retry time if applicable)
    batch_end_time = time.time()
    total_duration = batch_end_time - batch_start_time
    # For continued experiments, total_episodes should reflect all scenarios
    total_episodes_combined = len(all_results)
    avg_duration = total_duration / total_episodes if total_episodes > 0 else 0

    # Compute statistics (use all_results after retries)
    success_count = sum(1 for r in all_results if r["success"])
    error_count = len(all_results) - success_count

    # Calculate average rewards for successful episodes
    p1_rewards = [
        r["summary"]["p1_avg_reward"]
        for r in all_results
        if r["success"] and r["summary"]
    ]
    p2_rewards = [
        r["summary"]["p2_avg_reward"]
        for r in all_results
        if r["success"] and r["summary"]
    ]

    avg_p1_reward = sum(p1_rewards) / len(p1_rewards) if p1_rewards else 0.0
    avg_p2_reward = sum(p2_rewards) / len(p2_rewards) if p2_rewards else 0.0

    console.print(f"\n[bold green]Batch completed![/]")
    console.print(f"[green]Success: {success_count}[/]")
    console.print(f"[red]Errors: {error_count}[/]")
    console.print(
        f"[cyan]Total time: {format_duration(total_duration)} ({total_duration:.1f}s)[/]"
    )
    console.print(
        f"[cyan]Average time per episode: {format_duration(avg_duration)} ({avg_duration:.1f}s)[/]"
    )
    console.print(f"\n[bold]Average Rewards (across all episodes):[/]")
    console.print(f"  P1 Average: {avg_p1_reward:.2f}")
    console.print(f"  P2 Average: {avg_p2_reward:.2f}")

    # Save results to output file FIRST (before cost calculation which may fail)
    output_path = experiment_dir / "results.json"

    if output_path:
        # Convert for JSON serialization
        def convert_to_serializable(obj: object) -> object:
            if hasattr(obj, "tolist"):
                return obj.tolist()  # type: ignore
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        batch_summary = {
            "experiment_tag": experiment_tag,
            "subset": args.subset,
            "optimize_mode": args.optimize,
            "total_episodes": total_episodes_combined,  # Use combined count for continued experiments
            "success_count": success_count,
            "error_count": error_count,
            "total_duration_seconds": total_duration,
            "avg_duration_per_episode": avg_duration,
            "avg_p1_reward": avg_p1_reward,
            "avg_p2_reward": avg_p2_reward,
            "results": convert_to_serializable(all_results),
        }

        # Add continue-from information if applicable
        if continue_from_path:
            batch_summary["continue_from"] = {
                "source_path": str(continue_from_path),
                "previous_results_count": len(previous_results),
                "new_results_count": len(results),
            }

        # Add retry information if retries were performed
        if args.max_retries > 0 and retry_history:
            batch_summary["retry_info"] = {
                "max_retries_configured": args.max_retries,
                "total_retry_attempts": len(retry_history),
                "retry_history": retry_history,
                "final_failed_count": error_count,
                "final_failed_ids": [
                    r["scenario_id"] for r in all_results if not r["success"]
                ],
            }

        with open(output_path, "w") as f:
            json.dump(batch_summary, f, indent=2)
        console.print(f"\n[green]Results saved to:[/] {output_path}")

    # Calculate API costs (after saving results) - only if requested
    if args.calculate_cost:
        log_path = get_llm_call_log_path()
        if log_path and log_path.exists():
            console.print(f"\n[cyan]LLM call log saved to:[/] {log_path}")
            console.print("\n[bold cyan]Calculating API costs...[/]")
            try:
                cost_info = await calculate_cost_by_model_async(log_path)
                if not cost_info:
                    cost_info = await calculate_cost_async(log_path)
                else:
                    _print_cost_breakdown(cost_info)

                # Save cost info
                cost_info_path = experiment_dir / "cost_info.json"
                with open(cost_info_path, "w") as f:
                    json.dump(cost_info, f, indent=2)
                console.print(f"[green]Cost info saved to:[/] {cost_info_path}")
            except Exception as e:
                logger.warning(f"Failed to calculate cost: {e}")
    else:
        console.print(
            "\n[dim]Skipping cost calculation (use --calculate-cost to enable)[/]"
        )

    # Prepare summary text
    summary_text = (
        f"[bold green]Batch Simulation Completed![/]\n"
        f"Total: {total_episodes} episodes\n"
        f"Success: {success_count}, Errors: {error_count}\n"
        f"Time: {format_duration(total_duration)}\n"
        f"Avg P1 Reward: {avg_p1_reward:.2f}, Avg P2 Reward: {avg_p2_reward:.2f}"
    )

    # Add retry summary if retries were performed
    if args.max_retries > 0 and retry_history:
        summary_text += (
            f"\n\n[bold yellow]Retry Summary:[/]\n"
            f"Total Retries: {len(retry_history)}/{args.max_retries}\n"
            f"Initially Failed: {retry_history[0]['failed_count']}\n"
            f"Finally Succeeded: {retry_history[0]['failed_count'] - error_count}\n"
            f"Still Failed: {error_count}"
        )

    console.print(Panel(summary_text, title="Batch Summary"))

    # Cleanup logging
    cleanup_terminal_logging(tee_stdout, tee_stderr)

    # Allow pending aiohttp connections to close gracefully
    await asyncio.sleep(0.5)


async def run_multiple_scenarios(args: argparse.Namespace) -> None:
    """
    Run multiple scenarios sequentially from --scenario-ids.

    This is a lightweight alternative to batch mode for quick testing of specific combos.
    Supports:
    - Comma-separated IDs: --scenario-ids "ID1,ID2,ID3"
    - JSON file with list: --scenario-ids path/to/combos.json
    - Summary file: --scenario-ids path/to/summary.json (extracts from scenarios[*].combos)
    """
    from datetime import datetime
    import time

    batch_start_time = time.time()
    scenario_ids = parse_scenario_ids(args.scenario_ids)

    if not scenario_ids:
        console.print("[red]Error: No scenario IDs found in --scenario-ids[/]")
        return

    console.print(
        Panel(
            f"[bold cyan]Running {len(scenario_ids)} Scenarios Sequentially[/]\n\n"
            f"Bandit Type: {args.bandit_type}\n"
            f"Optimize: {args.optimize}\n"
            f"Model: {args.model}\n"
            f"Max Turns: {args.max_turns}",
            title="Multiple Scenarios Mode",
        )
    )

    # Generate experiment tag
    if args.tag:
        experiment_tag = args.tag
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_tag = f"mus{int(args.mask_unselected_scores)}"
        iw_tag = f"iw{int(args.importance_weighted_reward)}"
        experiment_tag = f"bandit_{args.bandit_type}_{args.optimize}_multi_{mask_tag}_{iw_tag}_{timestamp}"

    # Create experiment directory
    experiment_dir = (
        PROJECT_ROOT / "experiments/also/outputs" / experiment_tag
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = experiment_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    tensorboard_dir = experiment_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    llm_logs_dir = experiment_dir / "llm_logs"
    llm_logs_dir.mkdir(exist_ok=True)

    # Enable LLM call logging
    log_file = llm_logs_dir / "calls.jsonl"
    enable_llm_call_logging(log_file, experiment_tag=experiment_tag)

    # Setup terminal logging
    log_file_path, tee_stdout, tee_stderr = setup_terminal_logging(
        experiment_name=f"multi_{args.bandit_type}_{args.optimize}",
        log_dir=logs_dir,
    )
    configure_logger(level="DEBUG", include_function=True)

    # Create bandit config based on bandit type (with selection strategy support)
    bandit_config = create_bandit_config_from_args(args)

    # Save run config
    save_run_config(
        experiment_dir=experiment_dir,
        experiment_tag=experiment_tag,
        args=args,
        bandit_config=bandit_config,
        bandit_type=args.bandit_type,
    )

    # Run scenarios sequentially
    results: list[dict] = []
    successful = 0
    failed = 0

    for i, scenario_id in enumerate(scenario_ids, 1):
        console.print(
            f"\n[bold cyan]━━━ Scenario {i}/{len(scenario_ids)}: {scenario_id} ━━━[/]"
        )

        try:
            summary = await run_single_scenario(
                scenario_id=scenario_id,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=bandit_config,
                bandit_type=args.bandit_type,
                verbose=True,
            )
            results.append(
                {"scenario_id": scenario_id, "status": "success", "summary": summary}
            )
            successful += 1

            # Show quick score
            if summary and "final_reward" in summary:
                reward = summary["final_reward"]
                console.print(f"[green]✓ Completed with reward: {reward:.3f}[/]")
        except Exception as e:
            logger.error(f"Failed scenario {scenario_id}: {e}")
            traceback.print_exc()
            results.append(
                {"scenario_id": scenario_id, "status": "failed", "error": str(e)}
            )
            failed += 1

    # Summary
    elapsed = time.time() - batch_start_time
    console.print(f"\n[bold green]{'═' * 60}[/]")
    console.print(f"[bold]Multiple Scenarios Complete[/]")
    console.print(
        f"  Total: {len(scenario_ids)}, Success: {successful}, Failed: {failed}"
    )
    console.print(f"  Time: {elapsed / 60:.1f} min")
    console.print(f"  Tag: {experiment_tag}")
    console.print(f"[bold green]{'═' * 60}[/]")

    # Save results summary
    results_path = experiment_dir / "multi_scenario_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "experiment_tag": experiment_tag,
                "scenario_ids": scenario_ids,
                "total": len(scenario_ids),
                "successful": successful,
                "failed": failed,
                "elapsed_seconds": elapsed,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    console.print(f"[green]Results saved to:[/] {results_path}")

    # Cleanup
    cleanup_terminal_logging(tee_stdout, tee_stderr)


async def run_extend_from_hard(args: argparse.Namespace) -> None:
    """
    Extend a completed 'hard' subset experiment to 'all' subset.

    This function:
    1. Resolves the source experiment (tag or directory)
    2. Creates a new experiment with extended tag
    3. Migrates all data (DB records + files) from hard to new experiment
    4. Runs only the remaining scenarios (all - hard)
    """
    extend_start_time = time.time()
    source_tag_or_path = args.extend_from_hard

    console.print(
        Panel(
            f"[bold cyan]Extending from Hard Experiment[/]\n\n"
            f"Source: {source_tag_or_path}\n"
            f"Target Subset: all",
            title="Extend Mode",
        )
    )

    # Step 1: Resolve source experiment
    try:
        source_tag, source_dir = resolve_experiment_source(source_tag_or_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/]")
        return

    console.print(f"[green]✓ Found source experiment:[/] {source_tag}")
    console.print(f"  Directory: {source_dir}")

    # Step 2: Generate new experiment tag
    # Replace "_hard_" with "_all_extended_" or append "_extended_all" if no "_hard_"
    if "_hard_" in source_tag:
        new_tag = source_tag.replace("_hard_", "_all_extended_")
    else:
        new_tag = f"{source_tag}_extended_all"

    # Override with user-provided tag if specified
    if args.tag:
        new_tag = args.tag

    console.print(f"[cyan]New experiment tag:[/] {new_tag}")

    # Step 3: Create new experiment directory
    outputs_dir = PROJECT_ROOT / "experiments/also/outputs"
    new_dir = outputs_dir / new_tag

    if new_dir.exists():
        console.print(f"[yellow]Warning: Target directory already exists: {new_dir}[/]")
        console.print("[yellow]Do you want to overwrite? (y/N): [/]", end="")
        try:
            response = input().strip().lower()
            if response not in ("y", "yes"):
                console.print("[red]Aborted by user.[/]")
                return
            import shutil

            shutil.rmtree(new_dir)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]Aborted.[/]")
            return

    # Step 4: Migrate experiment directory
    console.print("\n[cyan]Migrating experiment directory...[/]")
    migrate_experiment_directory(
        source_dir=source_dir,
        target_dir=new_dir,
        update_tag_in_files=True,
        new_tag=new_tag,
    )
    console.print(f"[green]✓ Directory migrated to:[/] {new_dir}")

    # Step 5: Migrate database records
    console.print("\n[cyan]Migrating database records...[/]")
    migrated_count = migrate_episode_logs_to_new_tag(source_tag, new_tag)
    console.print(f"[green]✓ Migrated {migrated_count} EpisodeLog records[/]")

    # Step 6: Load completed scenarios from source
    completed_scenarios = load_completed_scenarios_from_results(source_dir)
    console.print(
        f"[green]✓ Found {len(completed_scenarios)} completed scenarios to exclude[/]"
    )

    # Step 7: Modify args for batch mode
    # Override subset to "all"
    args.subset = "all"
    # Set the new tag
    args.tag = new_tag
    # Set exclude-from to use the source results for filtering
    # We use a temporary mechanism: store completed IDs in args
    args._extend_completed_scenarios = completed_scenarios
    # Mark as extend mode
    args._extend_mode = True
    args._extend_source_tag = source_tag
    args._extend_source_dir = source_dir
    args._extend_new_dir = new_dir

    console.print(f"\n[bold green]Starting extended batch run...[/]")
    console.print(f"  Subset: {args.subset}")
    console.print(
        f"  Excluding: {len(completed_scenarios)} already-completed scenarios"
    )

    # Step 8: Run batch episodes with modified args
    await run_batch_episodes(args)

    extend_elapsed = time.time() - extend_start_time
    console.print(
        f"\n[bold green]Extension completed in {extend_elapsed / 60:.1f} minutes[/]"
    )


async def run_resume_episodes(args: argparse.Namespace) -> None:
    """Resume a previous experiment by re-running only failed/incomplete scenarios."""
    batch_start_time = time.time()

    resume_dir = args.resume
    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    results_file = resume_dir / "results.json"
    incremental_file = resume_dir / "results_incremental.jsonl"

    # Try to load from results.json first, then fallback to incremental file
    prev_results: dict = {}
    prev_results_list: list[dict] = []

    if results_file.exists():
        # Normal case: results.json exists
        with open(results_file) as f:
            prev_results = json.load(f)
        prev_results_list = prev_results.get("results", [])
        experiment_tag = prev_results.get("experiment_tag", resume_dir.name)
    elif incremental_file.exists():
        # Crash recovery: only incremental file exists
        console.print(
            "[yellow]No results.json found, recovering from incremental file...[/]"
        )
        with open(incremental_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        prev_results_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed line in incremental file: {e}"
                        )
        # Try to get experiment tag from run_config.json
        run_config_file = resume_dir / "run_config.json"
        if run_config_file.exists():
            with open(run_config_file) as f:
                run_config = json.load(f)
            experiment_tag = run_config.get("experiment_tag", resume_dir.name)
        else:
            experiment_tag = resume_dir.name
        prev_results = {
            "experiment_tag": experiment_tag,
            "results": prev_results_list,
            "success_count": sum(
                1 for r in prev_results_list if r.get("success", False)
            ),
            "total_episodes": len(prev_results_list),
        }
        console.print(
            f"[green]Recovered {len(prev_results_list)} results from incremental file[/]"
        )
    else:
        raise FileNotFoundError(
            f"Neither results.json nor results_incremental.jsonl found in {resume_dir}"
        )

    # Get all scenario IDs that were supposed to run (from run_config or embeddings dir)
    run_config_file = resume_dir / "run_config.json"
    all_scenario_ids: set[str] = set()

    if run_config_file.exists():
        with open(run_config_file) as f:
            run_config = json.load(f)

        # Check git commit consistency
        original_git_info = run_config.get("git_info", {})
        current_git_info = get_git_info()

        # Compare git commits if both are available
        if original_git_info.get("commit_id") and current_git_info.get("commit_id"):
            original_commit = original_git_info["commit_id"]
            current_commit = current_git_info["commit_id"]
            original_dirty = original_git_info.get("dirty", False)
            current_dirty = current_git_info.get("dirty", False)

            if original_commit != current_commit or original_dirty != current_dirty:
                # Git state mismatch - warn user and ask for confirmation
                console.print("\n")
                console.print(
                    Panel(
                        f"[bold yellow]⚠️  Git State Mismatch Detected[/]\n\n"
                        f"[bold]Original run:[/]\n"
                        f"  Commit: {original_commit[:8]}\n"
                        f"  Branch: {original_git_info.get('branch', 'unknown')}\n"
                        f"  Dirty: {'Yes' if original_dirty else 'No'}\n\n"
                        f"[bold]Current state:[/]\n"
                        f"  Commit: {current_commit[:8]}\n"
                        f"  Branch: {current_git_info.get('branch', 'unknown')}\n"
                        f"  Dirty: {'Yes' if current_dirty else 'No'}\n\n"
                        f"[red]⚠️  Resuming with different code may lead to inconsistent results![/]",
                        title="⚠️  Warning: Code Version Mismatch",
                        border_style="yellow",
                    )
                )

                # Ask user for confirmation
                from rich.prompt import Confirm

                should_continue = Confirm.ask(
                    "\n[yellow]Do you want to continue anyway?[/]",
                    default=False,
                )

                if not should_continue:
                    console.print("\n[yellow]Resume cancelled by user.[/]")
                    logger.info("Resume cancelled due to git mismatch")
                    return
                else:
                    console.print(
                        "\n[yellow]Continuing resume with different code version...[/]"
                    )
                    logger.warning("User chose to continue resume despite git mismatch")

        # Try to get total expected from config
        original_args = run_config.get("args", {})

        # First try: get all scenarios from subset (works for strategy mode)
        original_subset = original_args.get("subset", "hard")
        original_selection_mode = original_args.get("selection_mode", "paraphrase")

        if original_selection_mode == "strategy":
            # Strategy mode: get combos directly from database
            try:
                combos = get_combos_for_subset(original_subset)
                all_scenario_ids = {c.pk for c in combos}
                console.print(
                    f"[cyan]Found {len(all_scenario_ids)} total scenarios for subset '{original_subset}' from database[/]"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to get combos for subset '{original_subset}': {e}"
                )

        # Fallback: try embeddings_dir for paraphrase mode
        if not all_scenario_ids:
            embeddings_dir = original_args.get("embeddings_dir")
            if embeddings_dir:
                embeddings_path = Path(embeddings_dir)
                if embeddings_path.exists():
                    all_scenario_ids = set(get_available_scenarios(embeddings_path))
                    console.print(
                        f"[cyan]Found {len(all_scenario_ids)} total scenarios from embeddings dir[/]"
                    )

    # Find completed scenarios (successful ones)
    completed_scenarios = {
        r["scenario_id"] for r in prev_results_list if r.get("success", False)
    }

    # Find failed scenarios (explicitly failed)
    failed_scenarios_set = {
        r["scenario_id"] for r in prev_results_list if not r.get("success", True)
    }

    # If we know all scenarios, find incomplete ones (never started)
    if all_scenario_ids:
        incomplete_scenarios = (
            all_scenario_ids - completed_scenarios - failed_scenarios_set
        )
    else:
        incomplete_scenarios = set()

    # Scenarios to resume = failed + incomplete
    scenarios_to_resume = list(failed_scenarios_set | incomplete_scenarios)

    if not scenarios_to_resume:
        console.print(
            Panel(
                "[bold green]No failed or incomplete scenarios to resume![/]\n"
                f"All {len(completed_scenarios)} episodes succeeded.",
                title="Resume Complete",
            )
        )
        return

    console.print(
        Panel(
            f"[bold yellow]Resume Mode[/]\n\n"
            f"Resume Directory: {resume_dir}\n"
            f"Experiment Tag: {experiment_tag}\n"
            f"[bold red]Failed Scenarios: {len(failed_scenarios_set)}[/]\n"
            f"[bold yellow]Incomplete Scenarios: {len(incomplete_scenarios)}[/]\n"
            f"[bold cyan]Total to Resume: {len(scenarios_to_resume)}[/]\n"
            f"Previous Success: {len(completed_scenarios)}\n"
            f"Batch Size: {args.batch_size}\n"
            f"Optimize Mode: {args.optimize}",
            title="Resume Configuration",
        )
    )

    if failed_scenarios_set:
        console.print("\n[bold red]Failed Scenario IDs:[/]")
        for sid in list(failed_scenarios_set)[:10]:
            console.print(f"  • {sid}")
        if len(failed_scenarios_set) > 10:
            console.print(f"  ... and {len(failed_scenarios_set) - 10} more")

    if incomplete_scenarios:
        console.print(
            f"\n[bold yellow]Incomplete Scenarios:[/] {len(incomplete_scenarios)} (not started)"
        )

    # Use existing experiment directory
    experiment_dir = resume_dir
    logs_dir = experiment_dir / "logs"
    tensorboard_dir = experiment_dir / "tensorboard"
    llm_logs_dir = experiment_dir / "llm_logs"

    # Enable LLM call logging (append mode)
    log_file = llm_logs_dir / "calls_resume.jsonl"
    enable_llm_call_logging(log_file, experiment_tag=f"{experiment_tag}_resume")
    console.print(f"[cyan]LLM call log:[/] {log_file}")

    # Enable LiteLLM shared async client to reduce SSL connection overhead
    try:
        import litellm

        litellm.enable_shared_async_client = True
    except ImportError:
        pass

    # Setup terminal logging
    log_file_path, tee_stdout, tee_stderr = setup_terminal_logging(
        experiment_name=f"resume_{experiment_tag}",
        log_dir=logs_dir,
    )
    configure_logger(level="DEBUG", include_function=True)

    # Load saved bandit config instead of creating from args
    run_config_file = resume_dir / "run_config.json"
    if not run_config_file.exists():
        raise FileNotFoundError(
            f"run_config.json not found in {resume_dir}. "
            f"Cannot resume without original configuration."
        )

    with open(run_config_file) as f:
        run_config = json.load(f)

    # Validate bandit type consistency
    saved_bandit_type = run_config.get("bandit_type")
    if saved_bandit_type and saved_bandit_type != args.bandit_type:
        logger.error(
            f"Bandit type mismatch: saved run used '{saved_bandit_type}', "
            f"but --bandit-type='{args.bandit_type}' was specified"
        )
        raise ValueError(
            f"Cannot resume {saved_bandit_type} run with {args.bandit_type}. "
            f"Use --bandit-type={saved_bandit_type} or start a new experiment."
        )

    # Load saved config
    saved_config = run_config.get("bandit_config", {})
    bandit_config = load_bandit_config(saved_config)

    logger.info(f"Loaded saved config: {type(bandit_config).__name__}")
    logger.info(f"Config details: {bandit_config}")

    # Save resume run config for tracking
    resume_config_name = (
        f"run_config_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if (experiment_dir / "run_config.json").exists()
        else "run_config.json"
    )
    save_run_config(
        experiment_dir=experiment_dir,
        experiment_tag=experiment_tag,
        args=args,
        bandit_config=bandit_config,
        bandit_type=args.bandit_type,
        filename=resume_config_name,
    )

    # Initialize progress tracker
    progress_tracker = BatchProgressTracker()
    progress_tracker.total = len(scenarios_to_resume)

    # Parse device list for multi-GPU distribution
    device_list: list[str] | None = None
    if args.devices:
        device_list = [f"cuda:{d.strip()}" for d in args.devices.split(",")]
        console.print(f"[cyan]Distributing NN models across devices:[/] {device_list}")

    # Use semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.batch_size)
    resume_results: list[dict] = []

    # Counter for round-robin device assignment
    device_counter = [0]  # Use list to allow mutation in nested function

    async def run_with_semaphore(scenario_id: str) -> dict:
        async with semaphore:
            # Create a copy of bandit_config with assigned device
            scenario_bandit_config = bandit_config
            if device_list:
                import copy

                scenario_bandit_config = copy.deepcopy(bandit_config)
                assigned_device = device_list[device_counter[0] % len(device_list)]
                scenario_bandit_config.device = assigned_device
                device_counter[0] += 1
                logger.debug(f"Scenario {scenario_id} assigned to {assigned_device}")

            return await run_single_scenario(
                scenario_id=scenario_id,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=scenario_bandit_config,
                bandit_type=args.bandit_type,
                progress_tracker=progress_tracker,
                verbose=False,
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
            )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Re-running {len(scenarios_to_resume)} scenarios...",
            total=len(scenarios_to_resume),
        )

        async def update_progress_description() -> None:
            while progress_tracker.completed < progress_tracker.total:
                status = progress_tracker.get_status()
                progress.update(task, description=status)
                await asyncio.sleep(0.5)

        tasks = [run_with_semaphore(sid) for sid in scenarios_to_resume]
        update_task = asyncio.create_task(update_progress_description())

        for coro in asyncio.as_completed(tasks):
            result = await coro
            resume_results.append(result)
            progress.update(task, advance=1)

        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass

    # Track retry history for resume mode
    retry_history: list[dict] = []
    all_resume_results = resume_results  # Start with initial resume results

    # Auto-retry logic if enabled
    if args.max_retries > 0:
        current_retry = 0

        while current_retry < args.max_retries:
            # Find failed scenarios from resume results
            failed_scenario_ids = [
                r["scenario_id"] for r in all_resume_results if not r["success"]
            ]

            # If no failures, exit retry loop
            if not failed_scenario_ids:
                console.print(
                    "\n[bold green]All resumed scenarios succeeded! No retries needed.[/]"
                )
                break

            console.print(
                f"\n[yellow]Found {len(failed_scenario_ids)} failed scenarios in resume.[/]"
            )
            console.print(
                f"[cyan]Starting retry {current_retry + 1}/{args.max_retries}...[/]"
            )

            # Retry failed scenarios
            retry_start_time = time.time()
            retry_result = await retry_failed_scenarios(
                failed_scenario_ids=failed_scenario_ids,
                args=args,
                experiment_tag=experiment_tag,
                bandit_config=bandit_config,
                bandit_type=args.bandit_type,
                experiment_dir=experiment_dir,
                tensorboard_dir=tensorboard_dir,
                retry_attempt=current_retry + 1,
            )
            retry_duration = time.time() - retry_start_time

            # Record retry history
            retry_history.append(
                {
                    "retry_attempt": current_retry + 1,
                    "failed_count": len(failed_scenario_ids),
                    "failed_ids": failed_scenario_ids,
                    "newly_succeeded": retry_result["success_count"],
                    "still_failed": retry_result["still_failed"],
                    "still_failed_ids": retry_result["failed_ids"],
                    "retry_duration_seconds": retry_duration,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Merge results: replace failed scenarios with retry results
            retry_results_map = {r["scenario_id"]: r for r in retry_result["results"]}

            merged_retry_results = []
            for r in all_resume_results:
                if r["scenario_id"] in retry_results_map:
                    # Replace with retry result
                    merged_retry_results.append(retry_results_map[r["scenario_id"]])
                else:
                    # Keep original result
                    merged_retry_results.append(r)

            all_resume_results = merged_retry_results

            # Check if we should continue retrying
            if retry_result["still_failed"] == 0:
                console.print(
                    f"\n[bold green]All resumed scenarios succeeded after {current_retry + 1} retries![/]"
                )
                break

            current_retry += 1

            if current_retry < args.max_retries:
                console.print(
                    f"\n[yellow]Will retry {retry_result['still_failed']} scenarios in next attempt...[/]"
                )
            else:
                console.print(
                    f"\n[bold red]Max retries ({args.max_retries}) reached. {retry_result['still_failed']} scenarios still failed.[/]"
                )

    # Calculate statistics for resumed runs (use all_resume_results after retries)
    batch_end_time = time.time()
    total_duration = batch_end_time - batch_start_time

    resume_success = sum(1 for r in all_resume_results if r["success"])
    resume_error = len(all_resume_results) - resume_success

    console.print(f"\n[bold green]Resume completed![/]")
    console.print(f"[green]Newly Succeeded: {resume_success}[/]")
    console.print(f"[red]Still Failed: {resume_error}[/]")
    console.print(f"[cyan]Resume Duration: {format_duration(total_duration)}[/]")

    # Add retry summary if retries were performed
    if args.max_retries > 0 and retry_history:
        console.print(f"\n[bold yellow]Retry Summary:[/]")
        console.print(f"Total Retries: {len(retry_history)}/{args.max_retries}")
        console.print(f"Initially Failed: {retry_history[0]['failed_count']}")
        console.print(
            f"Finally Succeeded: {retry_history[0]['failed_count'] - resume_error}"
        )
        console.print(f"Still Failed: {resume_error}")

    # Merge results: replace failed scenarios with new results
    prev_results_list = prev_results.get("results", [])

    # Keep successful results from previous run
    merged_results = [r for r in prev_results_list if r.get("success", True)]

    # Add new results (use all_resume_results which includes retries)
    merged_results.extend(all_resume_results)

    # Recalculate statistics
    merged_success = sum(1 for r in merged_results if r["success"])
    merged_error = len(merged_results) - merged_success

    # Calculate average rewards
    p1_rewards = [
        r["summary"]["p1_avg_reward"]
        for r in merged_results
        if r["success"] and r.get("summary")
    ]
    p2_rewards = [
        r["summary"]["p2_avg_reward"]
        for r in merged_results
        if r["success"] and r.get("summary")
    ]
    avg_p1_reward = sum(p1_rewards) / len(p1_rewards) if p1_rewards else 0.0
    avg_p2_reward = sum(p2_rewards) / len(p2_rewards) if p2_rewards else 0.0

    # Update and save merged results
    def convert_to_serializable(obj: object) -> object:
        if hasattr(obj, "tolist"):
            return obj.tolist()  # type: ignore
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    merged_summary = {
        "experiment_tag": experiment_tag,
        "subset": prev_results.get("subset", args.subset),
        "optimize_mode": prev_results.get("optimize_mode", args.optimize),
        "total_episodes": len(merged_results),
        "success_count": merged_success,
        "error_count": merged_error,
        "total_duration_seconds": prev_results.get("total_duration_seconds", 0)
        + total_duration,
        "avg_duration_per_episode": (
            prev_results.get("total_duration_seconds", 0) + total_duration
        )
        / len(merged_results),
        "avg_p1_reward": avg_p1_reward,
        "avg_p2_reward": avg_p2_reward,
        "resume_info": {
            "resumed_at": datetime.now().isoformat(),
            "previously_failed": len(failed_scenarios_set),
            "previously_incomplete": len(incomplete_scenarios),
            "total_resumed": len(scenarios_to_resume),
            "newly_succeeded": resume_success,
            "still_failed": resume_error,
            "resume_duration_seconds": total_duration,
        },
        "results": convert_to_serializable(merged_results),
    }

    with open(results_file, "w") as f:
        json.dump(merged_summary, f, indent=2)
    console.print(f"\n[green]Merged results saved to:[/] {results_file}")

    # Calculate API costs (after saving results) - only if requested
    if args.calculate_cost:
        log_path = get_llm_call_log_path()
        if log_path and log_path.exists():
            console.print(f"\n[cyan]LLM call log saved to:[/] {log_path}")
            console.print("\n[bold cyan]Calculating API costs...[/]")
            try:
                cost_info = await calculate_cost_by_model_async(log_path)
                if not cost_info:
                    cost_info = await calculate_cost_async(log_path)
                else:
                    _print_cost_breakdown(cost_info)

                # Save cost info
                cost_info_path = experiment_dir / "cost_info.json"
                with open(cost_info_path, "w") as f:
                    json.dump(cost_info, f, indent=2)
                console.print(f"[green]Cost info saved to:[/] {cost_info_path}")
            except Exception as e:
                logger.warning(f"Failed to calculate cost: {e}")
    else:
        console.print(
            "\n[dim]Skipping cost calculation (use --calculate-cost to enable)[/]"
        )

    console.print(
        Panel(
            f"[bold green]Resume Complete![/]\n"
            f"[bold]Final Statistics:[/]\n"
            f"Total Episodes: {len(merged_results)}\n"
            f"Success: {merged_success}, Errors: {merged_error}\n"
            f"[dim]Previously failed: {len(failed_scenarios_set)}, Now succeeded: {resume_success}[/]\n"
            f"Avg P1 Reward: {avg_p1_reward:.2f}, Avg P2 Reward: {avg_p2_reward:.2f}",
            title="Resume Summary",
        )
    )

    cleanup_terminal_logging(tee_stdout, tee_stderr)

    # Allow pending aiohttp connections to close gracefully
    await asyncio.sleep(0.5)


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Auto-start mihomo proxy if --auto-proxy is specified
    proxy_manager: MihomoProxyManager | None = None
    if args.auto_proxy:
        console.print(
            f"[cyan]Starting mihomo proxy automatically (base port: {args.proxy_base_port})...[/]"
        )
        proxy_manager = MihomoProxyManager(
            base_config=args.mihomo_config,
            base_port=args.proxy_base_port,
        )
        try:
            proxy_port = proxy_manager.start()
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
            os.environ["http_proxy"] = proxy_url
            os.environ["https_proxy"] = proxy_url
            console.print(f"[green]✓ Proxy started on port {proxy_port}[/]")
            logger.info(f"Auto-started mihomo proxy: {proxy_url}")
        except FileNotFoundError as e:
            # mihomo 或配置文件不存在，询问用户是否跳过
            console.print(f"[yellow]Warning: {e}[/]")
            console.print(
                "[yellow]Do you want to continue without proxy? (y/N): [/]", end=""
            )
            try:
                response = input().strip().lower()
                if response in ("y", "yes"):
                    console.print("[yellow]Continuing without proxy...[/]")
                    logger.warning("Proxy skipped by user, continuing without proxy")
                else:
                    console.print("[red]Aborted by user.[/]")
                    sys.exit(1)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[red]Aborted.[/]")
                sys.exit(1)
        except Exception as e:
            console.print(f"[red]Failed to start mihomo proxy: {e}[/]")
            traceback.print_exc()
            sys.exit(1)
    # Set proxy environment variables if --proxy-port is specified (manual mode)
    elif args.proxy_port:
        proxy_url = f"http://127.0.0.1:{args.proxy_port}"
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        logger.info(f"Using proxy: {proxy_url}")

    # Log max_tokens if specified
    if args.max_tokens:
        logger.info(f"Using max_tokens limit: {args.max_tokens}")

    # Check if running in extend-from-hard mode
    if args.extend_from_hard:
        await run_extend_from_hard(args)
        return

    # Check if running in resume mode
    if args.resume:
        await run_resume_episodes(args)
        return

    # Check if running in batch mode
    if args.batch:
        await run_batch_episodes(args)
        return

    # Check if running multiple scenarios mode
    if args.scenario_ids:
        await run_multiple_scenarios(args)
        return

    # Resolve model names for P1 and P2 first (needed for tag generation)
    p1_model = args.p1_model or args.model
    p2_model = args.p2_model or args.model

    # Generate experiment tag
    if args.tag:
        experiment_tag = args.tag
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_tag = f"mus{int(args.mask_unselected_scores)}"
        iw_tag = f"iw{int(args.importance_weighted_reward)}"

        # Extract short model name for unique tag (e.g., "qwen-2.5-72b-instruct" -> "qwen25_72b")
        def get_model_short(model: str) -> str:
            name = model.split("/")[-1]  # Get last part after /
            return name.replace(".", "").replace("-", "_")[:20]  # Shorten and sanitize

        p1_short = get_model_short(p1_model)
        p2_short = get_model_short(p2_model)
        # If same model, use single tag; if different, use "p1model_vs_p2model"
        if p1_short == p2_short:
            model_tag = p1_short
        else:
            model_tag = f"{p1_short}_vs_{p2_short}"
        experiment_tag = f"bandit_{args.bandit_type}_{args.optimize}_{model_tag}_{mask_tag}_{iw_tag}_{timestamp}"

    # Mode description
    mode_desc = {
        "both": "Optimizing BOTH agents",
        "p1": "Optimizing P1 only",
        "p2": "Optimizing P2 only",
        "none": "BASELINE (no optimization)",
    }.get(args.optimize, "Unknown")

    console.print(
        Panel(
            f"[bold green]Bandit-Based Dynamic Prompt Optimization[/]\n\n"
            f"Scenario: {args.scenario_id}\n"
            f"P1 Model: {p1_model}\n"
            f"P2 Model: {p2_model}\n"
            f"Env Model: {args.env_model}\n"
            f"Max Turns: {args.max_turns}\n"
            f"[bold cyan]Bandit Type: {args.bandit_type}[/]\n"
            f"ETA (EXP3): {args.eta}, Alpha (LinUCB): {args.alpha}, Beta (NeuralUCB): {args.beta}\n"
            f"[bold yellow]Mode: {mode_desc}[/]\n"
            f"Push to DB: {args.push_to_db}",
            title="Configuration",
        )
    )

    # Create experiment directory structure
    experiment_dir = (
        PROJECT_ROOT / "experiments/also/outputs" / experiment_tag
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = experiment_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    tensorboard_dir = experiment_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    models_dir = experiment_dir / "models"
    models_dir.mkdir(exist_ok=True)

    llm_logs_dir = experiment_dir / "llm_logs"
    llm_logs_dir.mkdir(exist_ok=True)

    # Enable LLM call logging for cost tracking
    log_file = llm_logs_dir / "calls.jsonl"
    enable_llm_call_logging(log_file, experiment_tag=experiment_tag)
    console.print(f"[cyan]LLM call log:[/] {log_file}")

    # Enable LiteLLM shared async client to reduce SSL connection overhead
    try:
        import litellm

        litellm.enable_shared_async_client = True
    except ImportError:
        pass

    # Setup terminal logging to file in experiment dir
    log_file_path, tee_stdout, tee_stderr = setup_terminal_logging(
        experiment_name=f"bandit_{args.bandit_type}_{args.optimize}",
        log_dir=logs_dir,
    )
    configure_logger(level="DEBUG", include_function=True)

    # Create bandit config based on bandit type (with selection strategy support)
    bandit_config = create_bandit_config_from_args(args)
    save_run_config(
        experiment_dir=experiment_dir,
        experiment_tag=experiment_tag,
        args=args,
        bandit_config=bandit_config,
        bandit_type=args.bandit_type,
    )

    # Create runner with optimization mode
    runner = BanditSimulationRunner(
        scenario_id=args.scenario_id,
        model_name=args.model,
        p1_model_name=args.p1_model,
        p2_model_name=args.p2_model,
        env_model_name=args.env_model,
        reward_eval_model_name=args.reward_eval_model,
        terminal_eval_model_name=args.terminal_eval_model,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        embeddings_dir=args.embeddings_dir,  # Pass custom embeddings dir
        bandit_config=bandit_config,
        bandit_type=args.bandit_type,  # type: ignore
        optimize_mode=args.optimize,  # type: ignore
        push_to_db=args.push_to_db,
        experiment_tag=experiment_tag,
        experiment_dir=experiment_dir,
        tensorboard_dir=tensorboard_dir,
        use_context_embedding=args.context_embedding,
        embedding_model=args.embedding_model,
        context_embedding_dim=args.context_embedding_dim,
        alternate_optimization=args.alternate_optimization,
        selection_mode=args.selection_mode,
        strategy_cache_dir=args.strategy_cache_dir,
        strategy_version=args.strategy_version,
        static_strategy=getattr(args, "static_strategy", False),
    )

    # Run episode
    summary = await runner.run_episode()

    # Display summary
    display_summary(summary)

    # Save results to output file FIRST (before cost calculation which may fail)
    output_path = experiment_dir / "results.json"

    if output_path:
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj: object) -> object:
            if hasattr(obj, "tolist"):
                return obj.tolist()  # type: ignore
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        serializable_summary = convert_to_serializable(summary)
        with open(output_path, "w") as f:
            json.dump(serializable_summary, f, indent=2)  # type: ignore
        console.print(f"\n[green]Results saved to:[/] {output_path}")

    # Calculate and display cost (after saving results) - only if requested
    if args.calculate_cost:
        log_path = get_llm_call_log_path()
        if log_path and log_path.exists():
            console.print(f"\n[cyan]LLM call log saved to:[/] {log_path}")
            console.print("\n[bold cyan]Calculating API costs...[/]")
            try:
                # Compute total + per-model breakdown for the logged gen_ids
                cost_info = await calculate_cost_by_model_async(log_path)
                if not cost_info:
                    cost_info = await calculate_cost_async(log_path)
                else:
                    _print_cost_breakdown(cost_info)

                # Save cost info
                cost_info_path = experiment_dir / "cost_info.json"
                with open(cost_info_path, "w") as f:
                    json.dump(cost_info, f, indent=2)
                console.print(f"[green]Cost info saved to:[/] {cost_info_path}")
            except Exception as e:
                logger.warning(f"Failed to calculate cost: {e}")
    else:
        console.print(
            "\n[dim]Skipping cost calculation (use --calculate-cost to enable)[/]"
        )

    # Cleanup: restore original streams
    cleanup_terminal_logging(tee_stdout, tee_stderr)

    # Explicit cleanup of LiteLLM async clients to prevent "Event loop is closed" errors
    console.print("\n[dim]Cleaning up LiteLLM async clients...[/]")
    try:
        import litellm

        await litellm.close_litellm_async_clients()
        # Note: logger is unavailable after cleanup_terminal_logging, so we use console
        console.print("[dim]✓ LiteLLM async clients closed successfully[/]")
    except Exception as e:
        # Note: logger is unavailable after cleanup_terminal_logging
        console.print(
            f"[yellow]Warning: Error closing LiteLLM clients (non-fatal): {e}[/]"
        )
        import traceback

        traceback.print_exc()

    # Allow pending async operations to complete with progress indication
    console.print("[dim]Finalizing async cleanup...[/]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=False,
    ) as progress:
        task = progress.add_task("Waiting for async cleanup...", total=100)
        for i in range(10):
            await asyncio.sleep(0.15)  # Total 1.5s (increased from 0.5s)
            progress.update(task, completed=(i + 1) * 10)


if __name__ == "__main__":
    # Warning suppression removed - proper async cleanup should eliminate these warnings
    # If warnings reappear after fix, they indicate incomplete resource cleanup
    import sys

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    # No finally block needed - cleanup happens inside main() before event loop closes
