from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
from .prompt_space import PromptSpace
from .strategy_space import StrategySpace
from .exp3_bandit import EXP3Bandit
from .linucb_bandit import LinUCBBandit
from .neural_ucb_bandit import NeuralUCBBandit
from .adversarial_bandit import AdversarialBandit
from .neural_adversarial_bandit import NeuralAdversarialBandit
from .neural_evolution_bandit import NeuralAdversarialEvolutionBandit, NeuralEvolutionConfig
from .evoprompt_bandit import EvoPromptBandit, EvoPromptConfig
from .opro_bandit import OPROBandit, OPROConfig
from .tpe_bandit import TPEBandit, TPEConfig
from .neuro_adaptive_bandit import NeuroAdaptiveBandit, NeuroAdaptiveConfig
from .progressive_prompt_breeder_bandit import ProgressivePromptBreederBandit, ProgressivePromptBreederConfig
from .prompt_breeder_bandit import PromptBreederBandit, PromptBreederConfig

BANDIT_TYPES = [
    "exp3",
    "linucb",
    "neural_ucb",
    "adversarial",
    "neural_adversarial",
    "neural_evolution",
    "evoprompt_ga",
    "evoprompt_de",
    "opro",
    "tpe",
    "neuro_adaptive",
    "prompt_breeder",
    "progressive_prompt_breeder",
    "none",
]

def create_bandit(
    bandit_type: str,
    prompt_space: PromptSpace,
    config: BanditConfig | None = None,
    tensorboard_dir: str | None = None,
    output_dir: str | None = None,
) -> BaseBandit | None:
    """
    Factory function to create a bandit instance.
    
    Args:
        bandit_type: Type of bandit to create
        prompt_space: PromptSpace instance
        config: BanditConfig instance (optional)
        tensorboard_dir: Directory for tensorboard logs
        output_dir: Directory for output logs
        
    Returns:
        Instance of BaseBandit or None if type is "none"
    """
    if bandit_type == "none":
        return None
        
    if bandit_type == "exp3":
        return EXP3Bandit(prompt_space, config, tensorboard_dir)
    elif bandit_type == "linucb":
        return LinUCBBandit(prompt_space, config, tensorboard_dir)
    elif bandit_type == "neural_ucb":
        return NeuralUCBBandit(prompt_space, config, tensorboard_dir)
    elif bandit_type == "adversarial":
        return AdversarialBandit(prompt_space, config, tensorboard_dir)
    elif bandit_type == "neural_adversarial":
        return NeuralAdversarialBandit(prompt_space, config, tensorboard_dir)
    elif bandit_type == "neural_evolution":
        # Ensure config is compatible
        if config and not isinstance(config, NeuralEvolutionConfig):
            # Try to convert if it's a generic BanditConfig
            # Note: The constructor handles conversion, so we can pass it directly
            pass
        return NeuralAdversarialEvolutionBandit(prompt_space, config, tensorboard_dir, output_dir=output_dir)
    elif bandit_type in ["evoprompt_ga", "evoprompt_de"]:
        mode = "ga" if "ga" in bandit_type else "de"
        if config is None:
            config = EvoPromptConfig(mode=mode)
        elif isinstance(config, EvoPromptConfig):
            config.mode = mode
        else:
             # Create new config with correct mode if passed a generic BanditConfig
             # The constructor will handle this logic too, but setting mode is important
             pass 
             
        # Create config with correct mode if it's new
        if isinstance(config, EvoPromptConfig) and config.mode != mode:
             config.mode = mode
             
        return EvoPromptBandit(prompt_space, config, tensorboard_dir, output_dir=output_dir)
    elif bandit_type == "opro":
        return OPROBandit(prompt_space, config, tensorboard_dir, output_dir=output_dir)
    elif bandit_type == "tpe":
        return TPEBandit(prompt_space, config, tensorboard_dir)
    elif bandit_type == "neuro_adaptive":
        return NeuroAdaptiveBandit(prompt_space, config, tensorboard_dir)
    elif bandit_type == "prompt_breeder":
        return PromptBreederBandit(prompt_space, config, tensorboard_dir, output_dir=output_dir)
    elif bandit_type == "progressive_prompt_breeder":
        return ProgressivePromptBreederBandit(prompt_space, config, tensorboard_dir, output_dir=output_dir)
    else:
        raise ValueError(f"Unknown bandit type: {bandit_type}")
