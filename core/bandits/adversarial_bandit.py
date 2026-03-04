"""
Adversarial Bandit implementation for dynamic prompt optimization.

DEPRECATED: This module is kept for backward compatibility.
Please use the new modular imports instead:

    from .base_bandit import BaseBandit, BanditConfig, SelectionRecord
    from .exp3_bandit import EXP3Bandit
    from .neural_adversarial_bandit import NeuralAdversarialBandit, AdversarialBandit
    from .linucb_bandit import LinUCBBandit
    from .neural_ucb_bandit import NeuralUCBBandit
"""

import warnings

# Re-export from new modules for backward compatibility
from .base_bandit import BanditConfig, SelectionRecord
from .exp3_bandit import EXP3Bandit
from .neural_adversarial_bandit import AdversarialBandit, NeuralAdversarialBandit, ValueNetwork

warnings.warn(
    "Importing from adversarial_bandit.py is deprecated. "
    "Please import from the bandits package directly: "
    "from core.bandits import AdversarialBandit, BanditConfig, etc.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EXP3Bandit",
    "NeuralAdversarialBandit",
    "AdversarialBandit",
    "BanditConfig",
    "SelectionRecord",
    "ValueNetwork",
]

