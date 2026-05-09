"""
Comprehensive tests for NeuralAdversarialBandit and NeuralAdversarialEvolutionBandit.

Tests cover:
1. Initialization and configuration
2. Selection mechanism (probability calculation, arm selection)
3. Update mechanism (importance weighting, score traces)
4. NN training (raw rewards, prediction range)
5. Fitness normalization
6. Evolution mechanism
"""

import json
import asyncio
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path for proper imports
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from experiments.also.core.bandits.base_bandit import BanditConfig
from experiments.also.core.bandits import BANDIT_TYPES, create_bandit
from experiments.also.core.bandits.learnable_strategy_space_v2 import (
    LearnableStrategySpaceV2,
)
from experiments.also.core.bandits.neural_adversarial_bandit import (
    NeuralAdversarialBandit,
    ValueNetwork,
)
from experiments.also.core.bandits.neural_adversarial_learnable_v2_bandit import (
    LearnableAdversarialV2Config,
    NeuralAdversarialLearnableV2Bandit,
)
from experiments.also.core.bandits.neural_evolution_bandit import (
    NeuralAdversarialEvolutionBandit,
    NeuralEvolutionConfig,
)
from experiments.also.core.bandits.prompt_breeder_bandit import (
    EvolutionUnit,
)
from experiments.also.core.bandits.prompt_space import PromptSpace


# ============================================================================
# Fixtures: Create mock PromptSpace with synthetic data
# ============================================================================


@pytest.fixture(autouse=True)
def offline_evolution_mutation(monkeypatch):
    """Keep evolution tests offline while still exercising replacement logic."""

    async def fake_mutate_unit_async(
        self,
        agent,
        unit_idx,
        turn,
        inherit_fitness=None,
        inherit_score=None,
    ):
        parent = self.populations[agent][unit_idx]
        new_prompt = f"{parent.task_prompt}\n[offline mutation turn {turn}]"
        embedding = np.array(parent.embedding, copy=True)

        return EvolutionUnit(
            thinking_style=parent.thinking_style,
            mutation_prompt=parent.mutation_prompt,
            task_prompt=new_prompt,
            fitness=float(inherit_fitness if inherit_fitness is not None else parent.fitness),
            embedding=embedding,
            history=parent.history + [new_prompt],
            is_original=False,
        )

    monkeypatch.setattr(
        NeuralAdversarialEvolutionBandit,
        "_mutate_unit_async",
        fake_mutate_unit_async,
    )


@pytest.fixture
def temp_prompt_space_dir():
    """Create a temporary directory with mock embeddings and texts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock embeddings (5 arms, 64-dim embeddings)
        n_arms = 5
        embedding_dim = 64
        scenario_id = "test_scenario"

        # Create scenario subdirectory
        scenario_dir = os.path.join(tmpdir, scenario_id)
        os.makedirs(scenario_dir, exist_ok=True)

        # Create p1 and p2 embeddings
        p1_embeddings = np.random.randn(n_arms, embedding_dim).astype(np.float32)
        p2_embeddings = np.random.randn(n_arms, embedding_dim).astype(np.float32)

        # Normalize embeddings
        p1_embeddings = p1_embeddings / np.linalg.norm(
            p1_embeddings, axis=1, keepdims=True
        )
        p2_embeddings = p2_embeddings / np.linalg.norm(
            p2_embeddings, axis=1, keepdims=True
        )

        # Save embeddings
        np.save(os.path.join(scenario_dir, "p1_embeddings.npy"), p1_embeddings)
        np.save(os.path.join(scenario_dir, "p2_embeddings.npy"), p2_embeddings)

        # Create mock texts (matching PromptSpace expected format)
        texts = {
            "original": {
                "p1_background": "Original background for agent 1",
                "p2_background": "Original background for agent 2",
            },
            "paraphrased": {
                "p1_background": [f"Paraphrased prompt {i} for agent 1" for i in range(n_arms - 1)],
                "p2_background": [f"Paraphrased prompt {i} for agent 2" for i in range(n_arms - 1)],
            },
        }
        with open(os.path.join(scenario_dir, "texts.json"), "w") as f:
            json.dump(texts, f)

        # Create mock statistics
        stats = {
            "p1_name": "Agent1",
            "p2_name": "Agent2",
        }
        with open(os.path.join(scenario_dir, "statistics.json"), "w") as f:
            json.dump(stats, f)

        yield tmpdir, scenario_id, n_arms, embedding_dim


@pytest.fixture
def prompt_space(temp_prompt_space_dir):
    """Create a PromptSpace from the temporary directory."""
    tmpdir, scenario_id, n_arms, embedding_dim = temp_prompt_space_dir
    return PromptSpace(scenario_id=scenario_id, base_dir=Path(tmpdir))


@pytest.fixture
def bandit_config():
    """Create a basic BanditConfig."""
    return BanditConfig(
        eta=0.4,
        gamma=0.1,
        score_decay=0.9,
        cumulative_score_mode="nn",
    )


@pytest.fixture
def evolution_config():
    """Create a NeuralEvolutionConfig."""
    return NeuralEvolutionConfig(
        eta=0.4,
        gamma=0.1,
        score_decay=0.9,
        cumulative_score_mode="nn",
        population_size=10,
        mutation_rate=0.1,
        elite_ratio=0.2,
        evolution_interval=5,
    )


@pytest.fixture
def learnable_strategy_space():
    """Create a LearnableStrategySpaceV2 with dummy embeddings."""
    return LearnableStrategySpaceV2.from_scenario_backgrounds(
        p1_background="Original background for agent 1",
        p2_background="Original background for agent 2",
        p1_name="Agent1",
        p2_name="Agent2",
        skip_embeddings=True,
        strategy_version="v3_size3",
        embedding_dim=64,
    )


# ============================================================================
# Test ValueNetwork
# ============================================================================


class TestValueNetwork:
    """Tests for the ValueNetwork neural network."""

    def test_initialization(self):
        """Test ValueNetwork initialization with different input dimensions."""
        # Test with bio embedding only
        net = ValueNetwork(input_dim=64)
        # Check layers list exists and has correct structure
        assert len(net.layers) >= 2  # At least input and output layer
        assert net.layers[0].in_features == 64
        assert net.layers[-1].out_features == 1

        # Test with bio + context embedding
        net_with_context = ValueNetwork(input_dim=128)
        assert net_with_context.layers[0].in_features == 128

    def test_forward_pass(self):
        """Test forward pass produces valid output."""
        net = ValueNetwork(input_dim=64)
        x = torch.randn(10, 64)  # batch of 10

        output = net(x)

        # Output shape should be (10,) after squeeze
        assert output.shape == (10,)

    def test_output_is_finite(self):
        """Test that output is always finite for various inputs."""
        net = ValueNetwork(input_dim=64)

        # Test with extreme inputs
        extreme_inputs = [
            torch.zeros(5, 64),
            torch.ones(5, 64) * 100,
            torch.ones(5, 64) * -100,
            torch.randn(5, 64) * 10,
        ]

        for x in extreme_inputs:
            output = net(x)
            assert torch.all(torch.isfinite(output)), f"Output not finite for input with mean {x.mean()}"


# ============================================================================
# Test NeuralAdversarialBandit
# ============================================================================


class TestNeuralAdversarialBandit:
    """Tests for NeuralAdversarialBandit."""

    def test_initialization(self, prompt_space, bandit_config):
        """Test bandit initialization."""
        bandit = NeuralAdversarialBandit(
            prompt_space=prompt_space,
            config=bandit_config,
        )

        # Check basic attributes
        assert bandit.config == bandit_config
        assert bandit.embedding_dim == prompt_space.get_embedding("p1", 0).shape[0]

        # Check model is created (single shared model)
        assert isinstance(bandit.model, ValueNetwork)
        assert bandit.input_dim == bandit.embedding_dim  # No context embedding by default

        # Check score traces initialized
        assert "p1" in bandit.score_traces
        assert "p2" in bandit.score_traces
        assert len(bandit.score_traces["p1"]) == 0
        assert len(bandit.score_traces["p2"]) == 0

    def test_select_returns_valid_arm(self, prompt_space, bandit_config):
        """Test that select() returns a valid arm index."""
        bandit = NeuralAdversarialBandit(
            prompt_space=prompt_space,
            config=bandit_config,
        )
        n_arms = prompt_space.get_num_arms("p1")

        for turn in range(5):
            arm, prompt_text, embedding = bandit.select("p1", turn=turn)

            # Arm should be valid index
            assert 0 <= arm < n_arms
            # Prompt text should be a string
            assert isinstance(prompt_text, str)
            # Embedding should be numpy array
            assert isinstance(embedding, np.ndarray)

    def test_select_updates_score_traces(self, prompt_space, bandit_config):
        """Test that select() properly updates score_traces."""
        bandit = NeuralAdversarialBandit(
            prompt_space=prompt_space,
            config=bandit_config,
        )
        n_arms = prompt_space.get_num_arms("p1")

        # Initially empty
        assert len(bandit.score_traces["p1"]) == 0

        # After select
        bandit.select("p1", turn=0)
        assert len(bandit.score_traces["p1"]) == 1
        assert len(bandit.score_traces["p1"][0]) == n_arms

        # After another select
        bandit.select("p1", turn=1)
        assert len(bandit.score_traces["p1"]) == 2

    def test_update_with_context_importance_weighting(self, prompt_space, bandit_config):
        """Test that update_with_context applies importance weighting correctly."""
        bandit = NeuralAdversarialBandit(
            prompt_space=prompt_space,
            config=bandit_config,
        )

        # Select an arm
        arm, prompt_text, embedding = bandit.select("p1", turn=0)
        raw_reward = 0.5
        selection_prob = bandit.last_selection_probs["p1"]

        # Update (signature: agent, arm_index, reward, turn, context_embedding)
        bandit.update_with_context(
            agent="p1",
            arm_index=arm,
            reward=raw_reward,
            turn=0,
            context_embedding=None,
        )

        # Check selection history
        assert len(bandit.selection_history) == 1
        record = bandit.selection_history[0]
        assert record.agent == "p1"
        assert record.arm_index == arm
        assert record.reward == raw_reward  # Raw reward stored

        # Check actual_score_traces has importance-weighted reward
        iw_reward = raw_reward / selection_prob
        assert bandit.actual_score_traces["p1"][0][arm] == pytest.approx(iw_reward, rel=1e-5)

    def test_cumulative_score_mode_nn(self, prompt_space):
        """Test cumulative_score_mode='nn' uses only NN predictions."""
        config = BanditConfig(cumulative_score_mode="nn")
        bandit = NeuralAdversarialBandit(prompt_space=prompt_space, config=config)

        # Run a few turns
        for turn in range(3):
            arm, prompt_text, embedding = bandit.select("p1", turn=turn)
            bandit.update_with_context("p1", arm, 0.5, turn, context_embedding=None)

        # The cumulative scores should be based on NN predictions
        # We can't directly test the internal calculation, but we can verify
        # the mode is set correctly
        assert bandit.config.cumulative_score_mode == "nn"

    def test_cumulative_score_mode_actual(self, prompt_space):
        """Test cumulative_score_mode='actual' uses only actual scores."""
        config = BanditConfig(cumulative_score_mode="actual")
        bandit = NeuralAdversarialBandit(prompt_space=prompt_space, config=config)

        assert bandit.config.cumulative_score_mode == "actual"

    def test_cumulative_score_mode_mean(self, prompt_space):
        """Test cumulative_score_mode='mean' uses average of NN and actual."""
        config = BanditConfig(cumulative_score_mode="mean")
        bandit = NeuralAdversarialBandit(prompt_space=prompt_space, config=config)

        assert bandit.config.cumulative_score_mode == "mean"

    def test_train_model_uses_raw_rewards(self, prompt_space, bandit_config):
        """Test that train_model uses raw rewards, not importance-weighted rewards."""
        bandit = NeuralAdversarialBandit(
            prompt_space=prompt_space,
            config=bandit_config,
        )

        # Simulate several turns with varying rewards
        rewards = [0.3, 0.5, 0.7, 0.4, 0.6]
        for turn, reward in enumerate(rewards):
            arm, prompt_text, embedding = bandit.select("p1", turn=turn)
            bandit.update_with_context("p1", arm, reward, turn, context_embedding=None)

        # Train the model
        loss = bandit.train_model(verbose=False)

        # Loss should be a valid number
        assert not np.isnan(loss)

        # After training, NN predictions should be in [0, 1] range
        # (because we train with raw rewards in [0, 1])
        embeddings = prompt_space.get_all_embeddings("p1")
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float64).to(bandit.config.device)

        with torch.no_grad():
            predictions = bandit.model(embeddings_tensor)

        # All predictions should be reasonable (not exploding)
        assert torch.all(predictions >= -10), f"Predictions too low: {predictions.min()}"
        assert torch.all(predictions <= 10), f"Predictions too high: {predictions.max()}"

    def test_multiple_agents(self, prompt_space, bandit_config):
        """Test that bandit works correctly with both p1 and p2 agents."""
        bandit = NeuralAdversarialBandit(
            prompt_space=prompt_space,
            config=bandit_config,
        )

        # Select for both agents
        arm_p1, _, _ = bandit.select("p1", turn=0)
        arm_p2, _, _ = bandit.select("p2", turn=0)

        # Update both agents
        bandit.update_with_context("p1", arm_p1, 0.5, 0, context_embedding=None)
        bandit.update_with_context("p2", arm_p2, 0.6, 0, context_embedding=None)

        # Check both have history
        p1_records = [r for r in bandit.selection_history if r.agent == "p1"]
        p2_records = [r for r in bandit.selection_history if r.agent == "p2"]

        assert len(p1_records) == 1
        assert len(p2_records) == 1
        assert p1_records[0].reward == 0.5
        assert p2_records[0].reward == 0.6


class TestLearnableStrategyBandit:
    """Tests for learnable strategy replacement flow."""

    def test_public_bandit_type_uses_v2_only(self):
        assert "adversarial_learnable_v2" in BANDIT_TYPES
        assert "adversarial_learnable" not in BANDIT_TYPES
        assert "neural_adversarial_learnable" not in BANDIT_TYPES

    def test_factory_creates_v2_bandit(self, learnable_strategy_space):
        bandit = create_bandit(
            "adversarial_learnable_v2",
            learnable_strategy_space,
            LearnableAdversarialV2Config(),
        )
        assert isinstance(bandit, NeuralAdversarialLearnableV2Bandit)

    def test_learnable_strategy_space_replaces_prompt(self, learnable_strategy_space):
        old_prompt = learnable_strategy_space.get_prompt("p1", 1)

        new_prompt, new_embedding = learnable_strategy_space.replace_strategy(
            "p1",
            1,
            "Adopt a calm probing strategy that surfaces interests before making offers.",
            source="unit_test",
            created_turn=3,
        )

        assert new_prompt != old_prompt
        assert "calm probing strategy" in learnable_strategy_space.get_prompt("p1", 1)
        assert new_embedding.shape == (64,)
        metadata = learnable_strategy_space.get_slot_metadata("p1", 1)
        assert metadata["source"] == "unit_test"
        assert metadata["created_turn"] == 3

    def test_select_async_replaces_worst_slot(
        self,
        learnable_strategy_space,
        monkeypatch,
    ):
        config = LearnableAdversarialV2Config(
            proposal_interval=1,
            proposal_warmup_turns=1,
            proposal_context_turns=2,
            depth=0,
            hidden_size=32,
        )
        bandit = NeuralAdversarialLearnableV2Bandit(
            prompt_space=learnable_strategy_space,
            config=config,
        )

        class _FakeMessage:
            content = (
                "Use tactful reciprocity and targeted questions to uncover leverage "
                "while keeping the tone cooperative."
            )

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        async def _fake_completion(*args, **kwargs):
            return _FakeResponse()

        monkeypatch.setattr(
            "experiments.also.core.bandits.neural_adversarial_learnable_v2_bandit.acompletion_with_retry",
            _fake_completion,
        )

        # Seed history so one non-baseline arm becomes the replacement target.
        bandit.select("p1", turn=0)
        bandit.score_traces["p1"][0][1] = 9.0
        bandit.update_with_context("p1", 1, 0.1, 0, context_embedding=None)
        bandit.set_generation_context("Agent1: We should slow down and ask more questions.")

        old_prompts = {
            idx: learnable_strategy_space.get_prompt("p1", idx)
            for idx in range(learnable_strategy_space.get_num_arms("p1"))
        }
        arm_index, prompt_text, _ = asyncio.run(
            bandit.select_async("p1", turn=1)
        )

        assert arm_index >= 0
        assert prompt_text == learnable_strategy_space.get_prompt("p1", arm_index)
        assert bandit.proposal_events["p1"]
        replaced_slot = bandit.proposal_events["p1"][0]["slot_index"]
        assert learnable_strategy_space.get_prompt("p1", replaced_slot) != old_prompts[replaced_slot]
        assert bandit.score_traces["p1"][0][replaced_slot] == 0.0
        assert bandit.actual_score_traces["p1"][0][replaced_slot] == 0.0
        assert "targeted questions" in learnable_strategy_space.get_prompt("p1", replaced_slot)


# ============================================================================
# Test NeuralAdversarialEvolutionBandit
# ============================================================================


class TestNeuralAdversarialEvolutionBandit:
    """Tests for NeuralAdversarialEvolutionBandit."""

    def test_initialization(self, prompt_space, evolution_config):
        """Test evolution bandit initialization."""
        bandit = NeuralAdversarialEvolutionBandit(
            prompt_space=prompt_space,
            config=evolution_config,
        )

        # Check populations are initialized
        assert "p1" in bandit.populations
        assert "p2" in bandit.populations

        # Check population size
        assert len(bandit.populations["p1"]) == evolution_config.population_size
        assert len(bandit.populations["p2"]) == evolution_config.population_size

        # Check each unit has correct attributes
        for unit in bandit.populations["p1"]:
            assert isinstance(unit, EvolutionUnit)
            assert unit.fitness == 1.0  # Initial fitness
            assert isinstance(unit.task_prompt, str)
            assert unit.embedding is not None

    def test_evolution_unit_structure(self, prompt_space, evolution_config):
        """Test EvolutionUnit has correct structure."""
        bandit = NeuralAdversarialEvolutionBandit(
            prompt_space=prompt_space,
            config=evolution_config,
        )

        unit = bandit.populations["p1"][0]

        # Check required attributes (EvolutionUnit structure)
        assert hasattr(unit, "thinking_style")
        assert hasattr(unit, "mutation_prompt")
        assert hasattr(unit, "task_prompt")
        assert hasattr(unit, "fitness")
        assert hasattr(unit, "embedding")
        assert hasattr(unit, "history")
        assert hasattr(unit, "is_original")

        # Check types
        assert isinstance(unit.thinking_style, str)
        assert isinstance(unit.mutation_prompt, str)
        assert isinstance(unit.task_prompt, str)
        assert isinstance(unit.fitness, float)
        assert isinstance(unit.embedding, np.ndarray)

    def test_select_returns_valid_arm(self, prompt_space, evolution_config):
        """Test that select() returns a valid arm from population."""
        bandit = NeuralAdversarialEvolutionBandit(
            prompt_space=prompt_space,
            config=evolution_config,
        )
        n_arms = prompt_space.get_num_arms("p1")

        for turn in range(5):
            arm, prompt_text, embedding = bandit.select("p1", turn=turn)

            # Arm should be valid index
            assert 0 <= arm < n_arms
            # Prompt text should be a string
            assert isinstance(prompt_text, str)
            # Embedding should be numpy array
            assert isinstance(embedding, np.ndarray)

    def test_fitness_update_dual_track(self, prompt_space, evolution_config):
        """Test dual-track fitness update: selected arm uses actual, others use NN."""
        bandit = NeuralAdversarialEvolutionBandit(
            prompt_space=prompt_space,
            config=evolution_config,
        )

        # Get initial fitness values
        initial_fitness = [unit.fitness for unit in bandit.populations["p1"]]

        # Select and update
        arm, prompt_text, embedding = bandit.select("p1", turn=0)
        bandit.update_with_context("p1", arm, 0.8, 0, context_embedding=None)

        # Check fitness values changed
        updated_fitness = [unit.fitness for unit in bandit.populations["p1"]]

        # At least some fitness values should have changed
        # (all units get updated, selected with actual reward, others with NN prediction)
        fitness_changed = any(
            abs(init - upd) > 1e-6
            for init, upd in zip(initial_fitness, updated_fitness)
        )
        assert fitness_changed, "Fitness values should change after update"

    def test_fitness_normalization_threshold(self, prompt_space, evolution_config):
        """Test that fitness normalization triggers when max exceeds threshold."""
        bandit = NeuralAdversarialEvolutionBandit(
            prompt_space=prompt_space,
            config=evolution_config,
        )

        # Manually set very high fitness to trigger normalization
        FITNESS_NORM_THRESHOLD = 1e6
        for unit in bandit.populations["p1"]:
            unit.fitness = FITNESS_NORM_THRESHOLD * 2  # Above threshold

        # Trigger update which should normalize
        arm, prompt_text, embedding = bandit.select("p1", turn=0)
        bandit.update_with_context("p1", arm, 0.5, 0, context_embedding=None)

        # After normalization, max fitness should be <= 1.0 (or close to it)
        max_fitness = max(unit.fitness for unit in bandit.populations["p1"])
        # The normalization divides by max, so max should be around 1.0 after update
        # But the update also applies exp(eta * reward), so it might be slightly higher
        assert max_fitness < FITNESS_NORM_THRESHOLD, f"Fitness should be normalized, got {max_fitness}"

    def test_fitness_stays_positive(self, prompt_space, evolution_config):
        """Test that fitness values always stay positive."""
        bandit = NeuralAdversarialEvolutionBandit(
            prompt_space=prompt_space,
            config=evolution_config,
        )

        # Run many turns with varying rewards
        for turn in range(20):
            arm, prompt_text, embedding = bandit.select("p1", turn=turn)
            reward = np.random.uniform(0, 1)
            bandit.update_with_context("p1", arm, reward, turn, context_embedding=None)

            # Check all fitness values are positive
            for unit in bandit.populations["p1"]:
                assert unit.fitness > 0, f"Fitness should be positive, got {unit.fitness}"

    def test_evolution_mechanism(self, prompt_space, evolution_config):
        """Test that evolution creates new population with valid units."""
        # Set evolution interval to 1 for testing
        evolution_config.evolution_interval = 1
        bandit = NeuralAdversarialEvolutionBandit(
            prompt_space=prompt_space,
            config=evolution_config,
        )

        # Run enough turns to trigger evolution
        for turn in range(3):
            arm, prompt_text, embedding = bandit.select("p1", turn=turn)
            bandit.update_with_context("p1", arm, 0.5, turn, context_embedding=None)

        # After evolution, population should still be valid
        assert len(bandit.populations["p1"]) == evolution_config.population_size

        for unit in bandit.populations["p1"]:
            assert isinstance(unit, EvolutionUnit)
            assert isinstance(unit.task_prompt, str)
            assert unit.fitness > 0

    def test_elite_preservation(self, prompt_space, evolution_config):
        """Test that elite units are preserved during evolution."""
        evolution_config.evolution_interval = 1
        evolution_config.elite_ratio = 0.5  # Keep top 50%
        bandit = NeuralAdversarialEvolutionBandit(
            prompt_space=prompt_space,
            config=evolution_config,
        )

        # Set distinct fitness values to identify elites
        for i, unit in enumerate(bandit.populations["p1"]):
            unit.fitness = float(i + 1)  # 1, 2, 3, ...

        # Trigger evolution
        arm, prompt_text, embedding = bandit.select("p1", turn=0)
        bandit.update_with_context("p1", arm, 0.5, 0, context_embedding=None)

        # Check that elite fitness values are still present (approximately)
        # Note: fitness values change due to update, so we just check population is valid
        assert len(bandit.populations["p1"]) == evolution_config.population_size


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the bandit algorithms."""

    def test_full_simulation_neural_adversarial(self, prompt_space, bandit_config):
        """Test a full simulation with NeuralAdversarialBandit."""
        bandit = NeuralAdversarialBandit(
            prompt_space=prompt_space,
            config=bandit_config,
        )

        n_turns = 10
        rewards_p1 = []
        rewards_p2 = []

        for turn in range(n_turns):
            # Select for both agents
            arm_p1, _, _ = bandit.select("p1", turn=turn)
            arm_p2, _, _ = bandit.select("p2", turn=turn)

            # Simulate rewards
            reward_p1 = np.random.uniform(0.3, 0.7)
            reward_p2 = np.random.uniform(0.3, 0.7)
            rewards_p1.append(reward_p1)
            rewards_p2.append(reward_p2)

            # Update
            bandit.update_with_context("p1", arm_p1, reward_p1, turn, context_embedding=None)
            bandit.update_with_context("p2", arm_p2, reward_p2, turn, context_embedding=None)

        # Train model
        loss = bandit.train_model(verbose=False)

        # Verify state
        assert len(bandit.selection_history) == n_turns * 2
        assert len(bandit.score_traces["p1"]) == n_turns
        assert len(bandit.score_traces["p2"]) == n_turns
        assert not np.isnan(loss)

    def test_full_simulation_evolution(self, prompt_space, evolution_config):
        """Test a full simulation with NeuralAdversarialEvolutionBandit."""
        bandit = NeuralAdversarialEvolutionBandit(
            prompt_space=prompt_space,
            config=evolution_config,
        )

        n_turns = 15  # Enough to trigger evolution (interval=5)

        for turn in range(n_turns):
            # Select for both agents
            arm_p1, _, _ = bandit.select("p1", turn=turn)
            arm_p2, _, _ = bandit.select("p2", turn=turn)

            # Simulate rewards
            reward_p1 = np.random.uniform(0.3, 0.7)
            reward_p2 = np.random.uniform(0.3, 0.7)

            # Update
            bandit.update_with_context("p1", arm_p1, reward_p1, turn, context_embedding=None)
            bandit.update_with_context("p2", arm_p2, reward_p2, turn, context_embedding=None)

        # Verify populations are still valid
        assert len(bandit.populations["p1"]) == evolution_config.population_size
        assert len(bandit.populations["p2"]) == evolution_config.population_size

        # All fitness values should be positive
        for unit in bandit.populations["p1"]:
            assert unit.fitness > 0
        for unit in bandit.populations["p2"]:
            assert unit.fitness > 0

    def test_numerical_stability_long_run(self, prompt_space, bandit_config):
        """Test numerical stability over many turns."""
        bandit = NeuralAdversarialBandit(
            prompt_space=prompt_space,
            config=bandit_config,
        )

        n_turns = 50

        for turn in range(n_turns):
            arm, _, _ = bandit.select("p1", turn=turn)
            reward = np.random.uniform(0, 1)
            bandit.update_with_context("p1", arm, reward, turn, context_embedding=None)

            # Check score traces don't explode
            if len(bandit.score_traces["p1"]) > 0:
                max_score = max(max(trace) for trace in bandit.score_traces["p1"])
                assert not np.isinf(max_score), f"Score exploded at turn {turn}"
                assert not np.isnan(max_score), f"Score is NaN at turn {turn}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
