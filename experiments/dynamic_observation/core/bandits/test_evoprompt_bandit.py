"""
Unit tests for EvoPromptBandit.

Tests the EvoPrompt-style GA and DE bandit implementations.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from experiments.dynamic_observation.core.bandits.evoprompt_templates import (
    build_ga_prompt,
    build_de_prompt,
    parse_bio_from_response,
    GA_CROSSOVER_MUTATION_TEMPLATE,
    DE_DIFFERENTIAL_MUTATION_TEMPLATE,
)
from experiments.dynamic_observation.core.bandits.evoprompt_bandit import (
    EvoPromptBandit,
    EvoPromptConfig,
    EvolutionUnit,
)
from experiments.dynamic_observation.core.bandits.prompt_space import PromptSpace


class TestEvoPromptTemplates:
    """Tests for EvoPrompt template utilities."""

    def test_build_ga_prompt(self):
        """Test GA crossover+mutation prompt building."""
        bio1 = "A friendly person who helps others."
        bio2 = "An assertive leader who takes charge."
        
        prompt = build_ga_prompt(bio1, bio2)
        
        assert bio1 in prompt
        assert bio2 in prompt
        assert "<BIO>" in prompt
        assert "</BIO>" in prompt

    def test_build_de_prompt(self):
        """Test DE differential mutation prompt building."""
        bio0 = "Original bio"
        bio1 = "Donor bio 1"
        bio2 = "Donor bio 2"
        bio3 = "Best donor bio"
        
        prompt = build_de_prompt(bio0, bio1, bio2, bio3)
        
        assert bio0 in prompt
        assert bio1 in prompt
        assert bio2 in prompt
        assert bio3 in prompt
        assert "<BIO>" in prompt

    def test_parse_bio_from_response_with_tags(self):
        """Test parsing bio from response with tags."""
        response = "Here is my suggestion:\n<BIO>Be a helpful assistant.</BIO>"
        
        result = parse_bio_from_response(response)
        
        assert result == "Be a helpful assistant."

    def test_parse_bio_from_response_without_tags(self):
        """Test parsing bio from response without tags (fallback)."""
        response = "Be a helpful assistant who listens carefully."
        
        result = parse_bio_from_response(response)
        
        assert result == response

    def test_parse_bio_from_response_multiline(self):
        """Test parsing multiline bio."""
        response = "<BIO>Be helpful.\nListen carefully.\nAct with empathy.</BIO>"
        
        result = parse_bio_from_response(response)
        
        assert "Be helpful." in result
        assert "Listen carefully." in result

    def test_parse_bio_from_response_short(self):
        """Test that short responses return None."""
        response = "Hi"
        
        result = parse_bio_from_response(response)
        
        assert result is None


class TestEvoPromptConfig:
    """Tests for EvoPromptConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvoPromptConfig()
        
        assert config.mode == "de"
        assert config.population_size == 10
        assert config.evolution_interval == 5
        assert config.ga_selection_mode == "tournament"
        assert config.de_use_best_donor is True
        assert config.selection_strategy == "softmax"
        assert config.score_update_method == "ema"

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvoPromptConfig(
            mode="ga",
            population_size=20,
            evolution_interval=10,
            ga_selection_mode="wheel",
        )
        
        assert config.mode == "ga"
        assert config.population_size == 20
        assert config.evolution_interval == 10
        assert config.ga_selection_mode == "wheel"


class TestEvoPromptBandit:
    """Tests for EvoPromptBandit."""

    @pytest.fixture
    def mock_prompt_space(self):
        """Create a mock prompt space for testing."""
        space = MagicMock(spec=PromptSpace)
        space.p1_prompts = [
            "Original P1 bio that is long enough",
            "Paraphrase P1 bio 1 that is long enough",
            "Paraphrase P1 bio 2 that is long enough",
        ]
        space.p2_prompts = [
            "Original P2 bio that is long enough",
            "Paraphrase P2 bio 1 that is long enough",
            "Paraphrase P2 bio 2 that is long enough",
        ]
        space.p1_embeddings = np.random.randn(3, 128).astype(np.float32)
        space.p2_embeddings = np.random.randn(3, 128).astype(np.float32)
        space.get_prompt = MagicMock(side_effect=lambda agent, idx: 
            space.p1_prompts[idx] if agent == "p1" else space.p2_prompts[idx]
        )
        space.get_embedding = MagicMock(side_effect=lambda agent, idx:
            space.p1_embeddings[idx] if agent == "p1" else space.p2_embeddings[idx]
        )
        return space

    @pytest.fixture
    def evoprompt_bandit_ga(self, mock_prompt_space):
        """Create EvoPromptBandit in GA mode."""
        config = EvoPromptConfig(
            mode="ga",
            evolution_interval=5,
            seed=42,
        )
        return EvoPromptBandit(
            prompt_space=mock_prompt_space,
            config=config,
        )

    @pytest.fixture
    def evoprompt_bandit_de(self, mock_prompt_space):
        """Create EvoPromptBandit in DE mode."""
        config = EvoPromptConfig(
            mode="de",
            evolution_interval=5,
            seed=42,
        )
        return EvoPromptBandit(
            prompt_space=mock_prompt_space,
            config=config,
        )

    def test_initialization_ga(self, evoprompt_bandit_ga):
        """Test GA mode initialization."""
        assert evoprompt_bandit_ga.evo_config.mode == "ga"
        assert len(evoprompt_bandit_ga.populations["p1"]) == 3
        assert len(evoprompt_bandit_ga.populations["p2"]) == 3
        
        # First prompt should be marked as original
        assert evoprompt_bandit_ga.populations["p1"][0].is_original is True
        assert evoprompt_bandit_ga.populations["p1"][1].is_original is False

    def test_initialization_de(self, evoprompt_bandit_de):
        """Test DE mode initialization."""
        assert evoprompt_bandit_de.evo_config.mode == "de"
        assert len(evoprompt_bandit_de.populations["p1"]) == 3

    def test_select_no_evolution(self, evoprompt_bandit_ga):
        """Test selection without triggering evolution."""
        # Turn 1 should not trigger evolution (interval=5)
        arm_idx, prompt, embedding = evoprompt_bandit_ga.select("p1", turn=1)
        
        assert arm_idx >= 0
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_update_ema(self, evoprompt_bandit_ga):
        """Test EMA fitness update."""
        initial_fitness = evoprompt_bandit_ga.populations["p1"][0].fitness
        
        evoprompt_bandit_ga.update("p1", arm_index=0, reward=0.9, turn=1)
        
        new_fitness = evoprompt_bandit_ga.populations["p1"][0].fitness
        alpha = evoprompt_bandit_ga.evo_config.ema_alpha
        expected = alpha * 0.9 + (1 - alpha) * initial_fitness
        
        assert abs(new_fitness - expected) < 1e-6

    def test_should_evolve(self, evoprompt_bandit_ga):
        """Test evolution trigger logic."""
        # Turn 0-4 should not trigger (interval=5)
        for turn in range(5):
            assert not evoprompt_bandit_ga._should_evolve("p1", turn)
        
        # Turn 5 should trigger
        assert evoprompt_bandit_ga._should_evolve("p1", turn=5)

    def test_get_hash(self, evoprompt_bandit_ga):
        """Test hash function for deduplication."""
        hash1 = evoprompt_bandit_ga._get_hash("Test prompt")
        hash2 = evoprompt_bandit_ga._get_hash("Test prompt")
        hash3 = evoprompt_bandit_ga._get_hash("Different prompt")
        
        assert hash1 == hash2
        assert hash1 != hash3

    def test_train_model_returns_zero(self, evoprompt_bandit_ga):
        """Test train_model returns 0.0 (EvoPrompt doesn't use NN)."""
        loss = evoprompt_bandit_ga.train_model(verbose=False)
        assert loss == 0.0

    def test_bandit_type_property_ga(self, evoprompt_bandit_ga):
        """Test bandit_type property for GA."""
        assert evoprompt_bandit_ga.bandit_type == "evoprompt_ga"

    def test_bandit_type_property_de(self, evoprompt_bandit_de):
        """Test bandit_type property for DE."""
        assert evoprompt_bandit_de.bandit_type == "evoprompt_de"

    def test_get_pool_summary(self, evoprompt_bandit_ga):
        """Test pool summary generation."""
        summary = evoprompt_bandit_ga.get_pool_summary("p1")
        
        assert "pool_size" in summary
        assert summary["pool_size"] == 3
        assert "mode" in summary
        assert summary["mode"] == "ga"
        assert "avg_fitness" in summary
        assert "top_3" in summary

    def test_is_valid_bio(self, evoprompt_bandit_ga):
        """Test bio validation."""
        assert evoprompt_bandit_ga._is_valid_bio("A sufficiently long bio description") is True
        assert evoprompt_bandit_ga._is_valid_bio("Short") is False
        # Empty string and None are falsy
        assert not evoprompt_bandit_ga._is_valid_bio("")
        assert not evoprompt_bandit_ga._is_valid_bio(None)

    def test_selection_strategy_greedy(self, mock_prompt_space):
        """Test greedy selection picks highest fitness."""
        config = EvoPromptConfig(
            mode="ga",
            selection_strategy="greedy",
            seed=42,
        )
        bandit = EvoPromptBandit(prompt_space=mock_prompt_space, config=config)
        
        # Set different fitness values
        bandit.populations["p1"][0].fitness = 0.3
        bandit.populations["p1"][1].fitness = 0.9
        bandit.populations["p1"][2].fitness = 0.5
        
        arm_idx, _, _ = bandit.select("p1", turn=1)
        
        # Should select highest fitness (idx=1)
        assert arm_idx == 1


class TestEvoPromptFactory:
    """Tests for factory function integration."""

    @pytest.fixture
    def mock_prompt_space(self):
        """Create a mock prompt space."""
        space = MagicMock(spec=PromptSpace)
        space.p1_prompts = ["Bio 1 that is long enough"] * 5
        space.p2_prompts = ["Bio 2 that is long enough"] * 5
        space.p1_embeddings = np.random.randn(5, 128).astype(np.float32)
        space.p2_embeddings = np.random.randn(5, 128).astype(np.float32)
        return space

    def test_create_evoprompt_ga(self, mock_prompt_space):
        """Test creating EvoPrompt GA via factory."""
        from experiments.dynamic_observation.core.bandits import create_bandit
        
        bandit = create_bandit("evoprompt_ga", mock_prompt_space)
        
        assert isinstance(bandit, EvoPromptBandit)
        assert bandit.evo_config.mode == "ga"

    def test_create_evoprompt_de(self, mock_prompt_space):
        """Test creating EvoPrompt DE via factory."""
        from experiments.dynamic_observation.core.bandits import create_bandit
        
        bandit = create_bandit("evoprompt_de", mock_prompt_space)
        
        assert isinstance(bandit, EvoPromptBandit)
        assert bandit.evo_config.mode == "de"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
