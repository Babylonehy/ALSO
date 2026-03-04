"""Core simulation runner with bandit optimization."""

import time
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from sotopia.agents import LLMAgent
from sotopia.database import AgentProfile, EnvironmentProfile, EnvAgentComboStorage, EpisodeLog
from sotopia.envs.evaluators import EvaluationForTwoAgents, SotopiaDimensions

# Import from local modules
from .database_utils import find_combo_by_pk, load_profiles
from .display_utils import format_duration
from .embedding_client import EmbeddingClient, extract_agent_history, DEFAULT_EMBEDDING_MODEL
from .progress_tracking import BatchProgressTracker

# Import from core modules
from experiments.dynamic_observation.core.envs_dynamic_parallel import DynamicPromptParallelSotopiaEnv
from experiments.dynamic_observation.core.evaluator_reward_in_trun import RewardInTurnEvaluator, TerminalEvaluator
from experiments.dynamic_observation.core.bandits import (
    BaseBandit,
    BanditConfig,
    PromptSpace,
    StrategySpace,
    create_bandit,
)

console = Console()

# Get PROJECT_ROOT
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Type aliases
BanditType = Literal["exp3", "linucb", "neural_ucb", "none"]
OptimizeMode = Literal["p1", "p2", "both", "none"]


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
        context_embedding_dim: int = 2048,
        alternate_optimization: bool = False,
        save_checkpoint: bool = False,
        selection_mode: str = "paraphrase",
        strategy_cache_dir: Path | None = None,
        strategy_version: str = "v1",
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

        # Initialize EmbeddingClient if context embedding is enabled
        self.embedding_client: EmbeddingClient | None = None
        if use_context_embedding:
            logger.info(f"Context embedding enabled, model: {embedding_model}")
            self.embedding_client = EmbeddingClient(model=embedding_model)

        # Setup paths
        if embeddings_dir is None:
            embeddings_dir = PROJECT_ROOT / "experiments/dynamic_observation/embeddings_backgrounds" / "hard"
        self.embeddings_dir = embeddings_dir

        # Determine which agents to optimize
        self.optimize_p1 = optimize_mode in ("p1", "both")
        self.optimize_p2 = optimize_mode in ("p2", "both")
        self.alternate_optimization = alternate_optimization and self.optimize_p1 and self.optimize_p2
        self._alternate_next_agent: Literal["p1", "p2"] = "p1"
        if self.alternate_optimization:
            logger.info("Alternate optimization enabled: P1 and P2 bandits will update on alternating turns.")

        # Initialize bandits only for agents being optimized
        self.bandit_p1: BaseBandit | None = None
        self.bandit_p2: BaseBandit | None = None
        self.prompt_space: PromptSpace | StrategySpace | None = None

        if self.optimize_p1 or self.optimize_p2:
            # Create bandit config with context embedding settings
            self.bandit_config = bandit_config or BanditConfig()
            if use_context_embedding:
                self.bandit_config.use_context_embedding = True
                self.bandit_config.embedding_model = embedding_model
                self.bandit_config.context_embedding_dim = context_embedding_dim

            # Initialize prompt space based on selection mode
            if self.selection_mode == "strategy":
                # Strategy mode: need original bios from agent profiles
                # We'll initialize this after loading the combo/profiles
                logger.info(f"Using strategy selection mode for scenario {scenario_id}")
                self._strategy_mode_pending = True  # Flag for deferred init
            else:
                # Paraphrase mode: load pre-generated bio paraphrases
                logger.info(f"Loading prompt space (paraphrase mode) for scenario {scenario_id}")
                self.prompt_space = PromptSpace(
                    scenario_id=scenario_id,
                    base_dir=embeddings_dir,
                )
                self._strategy_mode_pending = False

                if self.optimize_p1:
                    logger.info(f"Creating {bandit_type} bandit for P1 (Agent 1)")
                    self.bandit_p1 = create_bandit(
                        bandit_type=bandit_type,
                        prompt_space=self.prompt_space,
                        config=self.bandit_config,
                        tensorboard_dir=self.tensorboard_dir,
                        output_dir=self.experiment_dir,
                    )

                if self.optimize_p2:
                    logger.info(f"Creating {bandit_type} bandit for P2 (Agent 2)")
                    self.bandit_p2 = create_bandit(
                        bandit_type=bandit_type,
                        prompt_space=self.prompt_space,
                        config=self.bandit_config,
                        tensorboard_dir=self.tensorboard_dir,
                        output_dir=self.experiment_dir,
                    )

        else:
            self.bandit_config = bandit_config or BanditConfig()
            logger.info("Baseline mode: no bandit optimization")

        # Find combo by pk and load profiles
        combo = find_combo_by_pk(scenario_id)
        if combo is None:
            raise ValueError(
                f"No combo found with pk={scenario_id}. "
                f"Make sure this is a valid EnvAgentComboStorage pk."
            )

        self.combo = combo
        self.env_profile, self.agent_profiles = load_profiles(combo)

        logger.info(
            f"Loaded profiles: "
            f"p1={self.agent_profiles[0].first_name} {self.agent_profiles[0].last_name}, p2={self.agent_profiles[1].first_name} {self.agent_profiles[1].last_name}"
        )
        logger.info(
            f"Optimization mode: {optimize_mode} "
            f"(p1={self.optimize_p1}, p2={self.optimize_p2})"
        )

        # Complete deferred initialization for strategy mode
        if hasattr(self, '_strategy_mode_pending') and self._strategy_mode_pending:
            self._initialize_strategy_mode(bandit_type)

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

    def _initialize_strategy_mode(self, bandit_type: str) -> None:
        """
        Initialize strategy mode after profiles are loaded.

        In strategy mode, we use social strategies appended to original bios
        instead of pre-generated bio paraphrases.
        """
        from experiments.dynamic_observation.core.envs_dynamic_parallel import get_bio

        # Get original bios based on relationship and agent profiles
        p1_background = get_bio(
            self.env_profile.relationship,
            self.agent_profiles[0],
            agent_id=0,
        )
        p2_background = get_bio(
            self.env_profile.relationship,
            self.agent_profiles[1],
            agent_id=1,
        )

        p1_name = f"{self.agent_profiles[0].first_name} {self.agent_profiles[0].last_name}"
        p2_name = f"{self.agent_profiles[1].first_name} {self.agent_profiles[1].last_name}"

        # Create StrategySpace with original bios
        logger.info(f"Initializing StrategySpace for strategy selection mode (strategy_version={self.strategy_version})")
        self.prompt_space = StrategySpace.from_scenario_backgrounds(
            p1_background=p1_background,
            p2_background=p2_background,
            p1_name=p1_name,
            p2_name=p2_name,
            embedding_model=self.bandit_config.embedding_model,
            embedding_dim=self.bandit_config.context_embedding_dim,
            cache_dir=self.strategy_cache_dir,
            strategy_version=self.strategy_version,
        )

        # Create bandits
        if self.optimize_p1:
            logger.info(f"Creating {bandit_type} bandit for P1 (strategy mode)")
            self.bandit_p1 = create_bandit(
                bandit_type=bandit_type,
                prompt_space=self.prompt_space,
                config=self.bandit_config,
                tensorboard_dir=self.tensorboard_dir,
                output_dir=self.experiment_dir,
            )

        if self.optimize_p2:
            logger.info(f"Creating {bandit_type} bandit for P2 (strategy mode)")
            self.bandit_p2 = create_bandit(
                bandit_type=bandit_type,
                prompt_space=self.prompt_space,
                config=self.bandit_config,
                tensorboard_dir=self.tensorboard_dir,
                output_dir=self.experiment_dir,
            )

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
            logger.info("Baseline mode: using RuleBasedTerminatedEvaluator (no per-turn LLM eval)")
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
            LLMAgent(agent_profile=self.agent_profiles[0], model_name=self.p1_model_name, max_tokens=self.max_tokens),
            LLMAgent(agent_profile=self.agent_profiles[1], model_name=self.p2_model_name, max_tokens=self.max_tokens),
        ]
        # print agents name and model
        logger.info(f"Environment Model: {env.model_name}, Reward Eval Model: {evaluator.model_name if hasattr(evaluator, 'model_name') else 'None'}, Terminal Eval Model: {terminal_evaluator.model_name}")
        logger.info(f"P1 Agent: {agents[0].agent_name}, Model: {self.p1_model_name}")
        logger.info(f"P2 Agent: {agents[1].agent_name}, Model: {self.p2_model_name}")
        # Initialize agents dict
        from sotopia.agents.llm_agent import Agents
        agents_dict = Agents({agent.agent_name: agent for agent in agents})

        # Reset environment
        observations = env.reset(agents=agents_dict)
        p1_name = env.background.p1_name
        p2_name = env.background.p2_name

        logger.info(
            f"Environment reset. "
            f"Agents: {p1_name} vs {p2_name}"
        )

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
            console.print(Panel(
                f"[bold]Mode:[/] {mode_desc}\n\n"
                f"[bold]Initial Bios[/bold]\n\n"
                f"[cyan]{p1_name}:[/] {env.background.p1_background[:200]}...\n\n"
                f"[cyan]{p2_name}:[/] {env.background.p2_background[:200]}...",
                title="Episode Start",
            ))

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
            task = progress.add_task("Running episode...", total=self.max_turns) if self.verbose else None

            while not all(terminated.values()) and self.current_turn < self.max_turns:
                self.current_turn += 1

                # Update progress bar (if verbose) or progress tracker (if batch mode)
                if self.verbose and task is not None:
                    progress.update(task, completed=self.current_turn, description=f"Turn {self.current_turn}")
                if self.progress_tracker is not None:
                    self.progress_tracker.update(self.scenario_id, self.current_turn, self.max_turns)

                # Get actions from agents
                actions = {}
                for agent in agents:
                    obs = observations.get(agent.agent_name)

                    # Log that we're about to generate action with current bio
                    dynamic_obs = env.get_dynamic_observation(agent.agent_name)
                    logger.debug(
                        f"Agent {agent.agent_name} generating action at turn {self.current_turn}. "
                        f"Current bio (first 100 chars): {dynamic_obs.p1_background if agent.agent_name == p1_name else dynamic_obs.p2_background}"
                    )

                    action = await agent.aact(obs)
                    actions[agent.agent_name] = action

                    logger.info(
                        f"[Turn {self.current_turn}] {agent.agent_name} action: {action.action_type} - {action.argument if action.argument else 'N/A'}"
                    )

                # Record agent actions to messages (agents -> Environment)
                messages[-1].extend([
                    (agent_name, "Environment", action)
                    for agent_name, action in actions.items()
                ])

                # Step environment
                observations, rewards, terminated, truncated, info = await env.astep(actions)

                # Record environment responses to messages (Environment -> agents)
                messages.append([
                    ("Environment", agent_name, observations[agent_name])
                    for agent_name in env.agents
                ])

                # Extract rewards
                p1_reward = rewards.get(p1_name, 0.0)
                p2_reward = rewards.get(p2_name, 0.0)

                turn_info = {
                    "turn": self.current_turn,
                    "p1_reward": p1_reward,
                    "p2_reward": p2_reward,
                    "p1_action": actions[p1_name].action_type,
                    "p2_action": actions[p2_name].action_type,
                    "p1_arm": self.bandit_p1.current_selections["p1"] if self.bandit_p1 else 0,
                    "p2_arm": self.bandit_p2.current_selections["p2"] if self.bandit_p2 else 0,
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
                if self.use_context_embedding and self.embedding_client and self.optimize_mode != "none":
                    # Flatten messages for extraction
                    flat_messages = []
                    for turn_msgs in messages:
                        for sender, receiver, msg in turn_msgs:
                            flat_messages.append((sender, msg))

                    agent_history = extract_agent_history(flat_messages)
                    if agent_history:  # Only embed if there's content
                        context_embedding = self.embedding_client.get_embedding(agent_history)
                        logger.debug(f"Generated context embedding for {agent_history}, shape: {context_embedding.shape}")

                if self.bandit_p1:
                    logger.info(
                        f"Updating P1 bandit with reward={p1_reward:.2f} at turn {self.current_turn}"
                    )
                    p1_idx = self.bandit_p1.current_selections["p1"]
                    # Use update_with_context if bandit supports it and context is available
                    if self.use_context_embedding and hasattr(self.bandit_p1, 'update_with_context'):
                        self.bandit_p1.update_with_context("p1", p1_idx, p1_reward, self.current_turn, context_embedding)
                    else:
                        self.bandit_p1.update("p1", p1_idx, p1_reward, self.current_turn)

                if self.bandit_p2:
                    logger.info(
                        f"Updating P2 bandit with reward={p2_reward:.2f} at turn {self.current_turn}"
                    )
                    p2_idx = self.bandit_p2.current_selections["p2"]
                    # Use update_with_context if bandit supports it and context is available
                    if self.use_context_embedding and hasattr(self.bandit_p2, 'update_with_context'):
                        self.bandit_p2.update_with_context("p2", p2_idx, p2_reward, self.current_turn, context_embedding)
                    else:
                        self.bandit_p2.update("p2", p2_idx, p2_reward, self.current_turn)

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
                        self._alternate_next_agent = "p2" if target_agent == "p1" else "p1"
                        logger.debug(f"Alternating optimization: updating {target_agent.upper()} on turn {self.current_turn}")

                    for agent_key in agents_to_refresh:
                        bandit, agent_name = bandit_map[agent_key]
                        if bandit is None:
                            continue

                        if update_interval > 0 and self.current_turn % update_interval == 0:
                            logger.info(
                                f"Training {agent_key.upper()} bandit at turn {self.current_turn} (interval={update_interval})"
                            )
                            bandit.train_model(verbose=True)

                        if bandit.is_stopped():
                            continue

                        if self.use_context_embedding and context_embedding is not None:
                            if hasattr(bandit, "set_context_embedding"):
                                bandit.set_context_embedding(context_embedding)

                        # Use async select if available (e.g., NeuralAdversarialEvolutionBandit)
                        if hasattr(bandit, "select_async"):
                            arm_idx, new_bio, _ = await bandit.select_async(agent_key, self.current_turn)
                        else:
                            arm_idx, new_bio, _ = bandit.select(agent_key, self.current_turn)
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
            console.print(Panel(
                f"[bold green]Episode Complete![/]\n"
                f"Duration: {format_duration(duration)} ({duration:.1f}s)",
                title="Done"
            ))

        # Build and optionally save EpisodeLog
        episode_pk, p1_final_reward, p2_final_reward = await self._build_and_save_episode_log(
            env=env,
            agents=agents,
            messages=messages,
            info=info,
        )

        # Build final summary (includes final_rewards with dimension breakdowns)
        summary = self._build_summary(
            p1_name, p2_name,
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
    ) -> tuple[str | None, float | tuple[float, dict[str, float]], float | tuple[float, dict[str, float]]]:
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
                if hasattr(msg, 'to_natural_language'):
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
        if isinstance(p1_complete_rating, tuple) or isinstance(p2_complete_rating, tuple):
            logger.info("Multi-dimensional rewards detected")
        else:
            logger.warning("pk: {epilog.pk} rewards: P1={p1_complete_rating}, P2={p2_complete_rating} is not tuple!")            
        # Log rewards details
        # Log rewards details as a table
        if isinstance(p1_complete_rating, tuple) or isinstance(p2_complete_rating, tuple):
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
                epilog.save()
                episode_pk = epilog.pk
                if self.verbose:
                    console.print(f"[green]Episode saved to database with pk:[/] {episode_pk}")
                logger.info(f"Episode saved with pk: {episode_pk}")
            except Exception as e:
                logger.error(f"Failed to save episode: {e}")
                if self.verbose:
                    console.print(f"[red]Failed to save episode to database:[/] {e}")
                raise  # Re-raise to ensure caller knows about failure
        else:
            if self.verbose:
                console.print("[yellow]push_to_db=False, episode not saved to database[/]")

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
        summary["p1_avg_reward"] = sum(p1_rewards) / len(p1_rewards) if p1_rewards else 0.0
        summary["p2_avg_reward"] = sum(p2_rewards) / len(p2_rewards) if p2_rewards else 0.0

        # Add bandit-specific summaries
        if self.bandit_p1:
            p1_summary = self.bandit_p1.get_selection_summary()
            summary["p1_bandit"] = {
                "total_selections": p1_summary.get("total_selections", 0),
                "selections": p1_summary.get("p1_selections", []),
                "reward_progression": [r for r in p1_summary.get("reward_progression", []) if r["agent"] == "p1"],
            }
        else:
            summary["p1_bandit"] = None

        if self.bandit_p2:
            p2_summary = self.bandit_p2.get_selection_summary()
            summary["p2_bandit"] = {
                "total_selections": p2_summary.get("total_selections", 0),
                "selections": p2_summary.get("p2_selections", []),
                "reward_progression": [r for r in p2_summary.get("reward_progression", []) if r["agent"] == "p2"],
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

