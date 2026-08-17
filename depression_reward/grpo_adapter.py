"""Адаптер под trl GRPOTrainer. trl/torch здесь НЕ импортируются:
reward_funcs у trl — просто callable(prompts, completions, **kwargs) -> list[float].
"""
import json
import re
import time

from .config import RewardConfig
from .reward import DepressionReward

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _strip_think(text: str) -> str:
    """Убирает <think>-блоки Qwen3; незакрытый <think> съедает всё до конца
    (осталась одна недописанная «мысль» -> пустой текст -> флор)."""
    text = _THINK_RE.sub('', text)
    if '<think>' in text:
        text = text[:text.index('<think>')]
    return text.strip()


def _completion_to_text(completion) -> str:
    # str — как есть; chat-формат trl: list[{'role','content'}] — последний ответ ассистента
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get('role') == 'assistant':
                return str(message.get('content', ''))
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get('content', ''))
    return ''


def make_reward_func(config: RewardConfig | None = None,
                     log_path: str | None = None):
    """Возвращает reward-функцию для trl GRPOTrainer(reward_funcs=[...]).

    log_path: JSONL-лог breakdown'ов по шагам — для мониторинга компонент
    (дрейф style/antisem/sentiment) во время обучения.
    """
    reward_model = DepressionReward(config)

    def depression_style_reward(prompts=None, completions=None, **kwargs):
        texts = [_strip_think(_completion_to_text(c)) for c in (completions or [])]
        breakdowns = reward_model.breakdown(texts)
        if log_path:
            with open(log_path, 'a', encoding='utf-8') as fp:
                for prompt, text, item in zip(prompts or [None] * len(texts),
                                              texts, breakdowns):
                    record = {'ts': time.time(), 'prompt': prompt,
                              'n_chars': len(text), **item}
                    fp.write(json.dumps(record, ensure_ascii=False) + '\n')
        return [item['reward'] for item in breakdowns]

    return depression_style_reward
