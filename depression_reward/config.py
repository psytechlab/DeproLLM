import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class RewardConfig:
    # веса компонент
    w_style: float = 1.0
    w_complete: float = 0.5
    w_quality: float = 0.25
    w_antisem: float = 1.0
    # запасной канал против reward-хакинга негативной лексикой мимо
    # antisem-лексикона; по умолчанию выключен
    w_sentiment_guard: float = 0.0

    # нормализация style: sigmoid((raw - center) / temp)
    # center по умолчанию — середина референсных скоров (-0.0558 / -0.1023);
    # после baseline-генерации выставить в средний raw baseline-текстов
    style_center: float = -0.08
    style_temp: float = 0.02
    # классификатор обучен на эссе ~150+ слов; короче — style гейтится линейно
    style_len_gate_words: int = 150

    # completeness: плато 1.0 на [target_min_words, target_max_words],
    # линейный спад до 0 на target_max_words + overlong_slack
    target_min_words: int = 300
    target_max_words: int = 500
    overlong_slack: int = 300
    # короче этого (или <40 кириллических символов) — флор без извлечения
    min_valid_words: int = 20

    # antisemantic-штраф: clip((rate - free) / scale, 0, 1),
    # rate = доля лемм из лексикона среди алфавитных лемм
    antisem_free_rate: float = 0.005
    antisem_scale: float = 0.015

    # sentiment_guard: штраф clip((-sentiment_abs - free) / scale, 0, 1),
    # sentiment_abs = sentiment_rate * word_count (норма по статье ~ +3.6)
    sentiment_guard_free: float = 2.0
    sentiment_guard_scale: float = 4.0

    # качество/дегенерация: rep_ratio = 1 - distinct n-грамм / всего n-грамм
    rep_ngram: int = 4
    rep_free: float = 0.10
    rep_scale: float = 0.30
    rep_hard_cap: float = 0.60

    floor_reward: float = -1.0
    model_dir: str | None = None  # None -> <пакет>/model

    @classmethod
    def from_json(cls, path: str | Path) -> 'RewardConfig':
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f'Unknown RewardConfig keys: {sorted(unknown)}')
        return cls(**data)
