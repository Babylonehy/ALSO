"""
Unit tests for OPROBandit.

Tests the OPRO-style LLM optimization bandit implementation.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from experiments.dynamic_observation.core.bandits.opro_prompts import (
    gen_instruction_score_pairs_str,
    gen_sotopia_meta_prompt,
    parse_bio_from_response,
)
from experiments.dynamic_observation.core.bandits.opro_bandit import (
    OPROBandit,
    OPROConfig,
    InstructionRecord,
)
from experiments.dynamic_observation.core.bandits.prompt_space import PromptSpace


class TestOPROPrompts:
    """Tests for OPRO prompt utilities."""

    def test_gen_instruction_score_pairs_str_basic(self):
        """Test basic instruction-score pair formatting."""
        instructions_and_scores = [
            ("Be helpful and kind", 0.5, 0),
            ("Be assertive", 0.7, 1),
            ("Be empathetic", 0.3, 2),
        ]
        
        result = gen_instruction_score_pairs_str(
            instructions_and_scores,
            max_num_instructions=10,
            score_threshold=0.0,
        )
        
        # Should contain all instructions
        assert "Be helpful and kind" in result
        assert "Be assertive" in result
        assert "Be empathetic" in result
        
        # Scores should be scaled to 10
        assert "5.0" in result  # 0.5 * 10
        assert "7.0" in result  # 0.7 * 10
        assert "3.0" in result  # 0.3 * 10

    def test_gen_instruction_score_pairs_str_with_threshold(self):
        """Test score threshold filtering."""
        instructions_and_scores = [
            ("Low score", 0.2, 0),
            ("High score", 0.8, 1),
        ]
        
        result = gen_instruction_score_pairs_str(
            instructions_and_scores,
            max_num_instructions=10,
            score_threshold=0.5,
        )
        
        # Only high score should appear
        assert "Low score" not in result
        assert "High score" in result

    def test_gen_instruction_score_pairs_str_max_limit(self):
        """Test maximum instruction limit."""
        instructions_and_scores = [
            (f"Prompt number {i}", 0.1 * i, i) 
            for i in range(1, 11)  # 10 instructions
        ]
        
        result = gen_instruction_score_pairs_str(
            instructions_and_scores,
            max_num_instructions=3,
            score_threshold=0.0,
        )
        
        # Only top 3 by score should appear (items 8, 9, 10)
        assert "Prompt number 10" in result
        assert "Prompt number 9" in result
        assert "Prompt number 8" in result
        # Lower scored items should not appear
        assert "Prompt number 7" not in result
        assert "Prompt number 2" not in result

    def test_gen_sotopia_meta_prompt(self):
        """Test complete meta prompt generation."""
        instructions_and_scores = [
            ("Be helpful", 0.6, 0),
            ("Be kind", 0.8, 1),
        ]
        
        prompt = gen_sotopia_meta_prompt(
            instructions_and_scores,
            max_num_instructions=10,
            score_threshold=0.0,
            include_context=False,
        )
        
        # Should contain template parts
        assert "generate an agent bio description" in prompt
        assert "<BIO>" in prompt
        assert "Be helpful" in prompt
        assert "Be kind" in prompt

    def test_parse_bio_from_response_with_tags(self):
        """Test parsing bio from response with tags."""
        response = "Here is my suggestion:\n<BIO>Be a helpful assistant who listens carefully.</BIO>"
        
        result = parse_bio_from_response(response)
        
        assert result == "Be a helpful assistant who listens carefully."

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
        assert "Act with empathy." in result


class TestOPROConfig:
    """Tests for OPROConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OPROConfig()
        
        assert config.num_generated_per_step == 4
        assert config.max_num_instructions == 20
        assert config.score_threshold == 0.0
        assert config.evolution_interval == 5
        assert config.optimizer_temperature == 1.0
        assert config.selection_strategy == "greedy"
        assert config.score_update_method == "ema"

    def test_custom_config(self):
        """Test custom configuration."""
        config = OPROConfig(
            num_generated_per_step=8,
            evolution_interval=10,
            selection_strategy="epsilon_greedy",
            epsilon=0.2,
        )
        
        assert config.num_generated_per_step == 8
        assert config.evolution_interval == 10
        assert config.selection_strategy == "epsilon_greedy"
        assert config.epsilon == 0.2


class TestOPROBandit:
    """Tests for OPROBandit."""

    @pytest.fixture
    def mock_prompt_space(self):
        """Create a mock prompt space for testing."""
        space = MagicMock(spec=PromptSpace)
        space.p1_prompts = ["Original P1 bio", "Paraphrase P1 bio 1"]
        space.p2_prompts = ["Original P2 bio", "Paraphrase P2 bio 1"]
        space.p1_embeddings = np.random.randn(2, 128).astype(np.float32)
        space.p2_embeddings = np.random.randn(2, 128).astype(np.float32)
        space.get_prompt = MagicMock(side_effect=lambda agent, idx: 
            space.p1_prompts[idx] if agent == "p1" else space.p2_prompts[idx]
        )
        return space

    @pytest.fixture
    def opro_bandit(self, mock_prompt_space):
        """Create OPROBandit instance for testing."""
        config = OPROConfig(
            evolution_interval=5,
            seed=42,
        )
        return OPROBandit(
            prompt_space=mock_prompt_space,
            config=config,
        )

    def test_initialization(self, opro_bandit, mock_prompt_space):
        """Test OPROBandit initialization."""
        # Check pools initialized
        assert len(opro_bandit.instruction_pools["p1"]) == 2
        assert len(opro_bandit.instruction_pools["p2"]) == 2
        
        # Check initial scores
        for rec in opro_bandit.instruction_pools["p1"]:
            assert rec.score == 0.5  # Neutral initial score

    def test_select_greedy(self, opro_bandit):
        """Test greedy selection picks highest score."""
        # Set different scores
        opro_bandit.instruction_pools["p1"][0].score = 0.3
        opro_bandit.instruction_pools["p1"][1].score = 0.8
        
        arm_idx, prompt, _ = opro_bandit.select("p1", turn=1)
        
        # Should select the higher scoring arm
        assert arm_idx == 1

    def test_update_ema(self, opro_bandit):
        """Test EMA score update."""
        initial_score = opro_bandit.instruction_pools["p1"][0].score
        
        opro_bandit.update("p1", arm_index=0, reward=0.9, turn=1)
        
        new_score = opro_bandit.instruction_pools["p1"][0].score
        alpha = opro_bandit.opro_config.ema_alpha
        expected = alpha * 0.9 + (1 - alpha) * initial_score
        
        assert abs(new_score - expected) < 1e-6

    def test_should_evolve(self, opro_bandit):
        """Test evolution trigger logic."""
        # Turn 0-4 should not trigger (interval=5)
        for turn in range(5):
            assert not opro_bandit._should_evolve("p1", turn)
        
        # Turn 5 should trigger
        assert opro_bandit._should_evolve("p1", turn=5)

    def test_get_instruction_hash(self, opro_bandit):
        """Test instruction hashing for deduplication."""
        hash1 = opro_bandit._get_instruction_hash("Test instruction")
        hash2 = opro_bandit._get_instruction_hash("Test instruction")
        hash3 = opro_bandit._get_instruction_hash("Different instruction")
        
        assert hash1 == hash2  # Same input -> same hash
        assert hash1 != hash3  # Different input -> different hash

    def test_train_model_returns_zero(self, opro_bandit):
        """Test train_model returns 0.0 (OPRO doesn't use NN)."""
        loss = opro_bandit.train_model(verbose=False)
        assert loss == 0.0

    def test_bandit_type_property(self, opro_bandit):
        """Test bandit_type property."""
        assert opro_bandit.bandit_type == "opro"

    def test_get_pool_summary(self, opro_bandit):
        """Test pool summary generation."""
        summary = opro_bandit.get_pool_summary("p1")
        
        assert "pool_size" in summary
        assert summary["pool_size"] == 2
        assert "avg_score" in summary
        assert "max_score" in summary
        assert "top_3_instructions" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
