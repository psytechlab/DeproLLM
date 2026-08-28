"""GRPO-reward для «депрессивного стиля»: текст -> скаляр.

Приватный пакет (содержит вендоренный код TITANIS) — не публиковать.
"""
import os

# должен быть выставлен ДО импорта isanlp (protobuf внутри старого isanlp)
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from .config import RewardConfig
from .grpo_adapter import make_reward_func
from .reward import DepressionReward

__all__ = ['RewardConfig', 'DepressionReward', 'make_reward_func']
