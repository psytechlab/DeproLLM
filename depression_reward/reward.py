import math

from .classifier import StyleScorer
from .config import RewardConfig
from .features import FeaturePipeline
from .penalty import (antisem_penalty, antisem_rate, load_lexicon,
                      repetition_ratio)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class DepressionReward:
    """text -> reward для GRPO.

    reward = w_style * style_gated + w_complete * completeness
           + w_quality * quality - w_antisem * antisem_penalty
           - w_sentiment_guard * sentiment_guard

    Флор (floor_reward): извлечение не удалось / не-русский текст /
    меньше min_valid_words слов / rep_ratio > rep_hard_cap.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()
        self.pipeline = FeaturePipeline()
        self.scorer = StyleScorer(self.cfg.model_dir)
        self.lexicon = load_lexicon()

    def __call__(self, texts: list[str]) -> list[float]:
        return [item['reward'] for item in self.breakdown(texts)]

    def breakdown(self, texts: list[str]) -> list[dict]:
        cfg = self.cfg
        results = self.pipeline.extract_batch(texts, min_words=cfg.min_valid_words)
        out = []
        for text, res in zip(texts, results):
            rep = repetition_ratio(text, cfg.rep_ngram) if isinstance(text, str) else 1.0
            if res is None or rep > cfg.rep_hard_cap:
                out.append({
                    'reward': cfg.floor_reward, 'floored': True,
                    'raw_score': None, 'style': 0.0, 'style_gated': 0.0,
                    'completeness': 0.0, 'quality': 0.0,
                    'antisem_rate': None, 'antisem_penalty': 0.0,
                    'sentiment_guard': 0.0, 'rep_ratio': rep,
                    'word_count': res.word_count if res else 0,
                })
                continue

            raw = float(self.scorer.raw_scores(res.vector)[0])
            style = _sigmoid((raw - cfg.style_center) / cfg.style_temp)
            style_gated = style * min(1.0, res.word_count / cfg.style_len_gate_words)

            w = res.word_count
            completeness = _clip01(min(
                w / cfg.target_min_words,
                1.0,
                (cfg.target_max_words + cfg.overlong_slack - w) / cfg.overlong_slack,
            ))

            quality = 1.0 - _clip01((rep - cfg.rep_free) / cfg.rep_scale)

            a_rate = antisem_rate(res.lemmas, self.lexicon)
            a_pen = antisem_penalty(a_rate, cfg)

            # запасной канал: штраф за сильно негативный суммарный сентимент
            sentiment_abs = res.pdict['sentiment_rate'] * w
            s_guard = _clip01(
                (-sentiment_abs - cfg.sentiment_guard_free) / cfg.sentiment_guard_scale)

            reward = (cfg.w_style * style_gated
                      + cfg.w_complete * completeness
                      + cfg.w_quality * quality
                      - cfg.w_antisem * a_pen
                      - cfg.w_sentiment_guard * s_guard)

            out.append({
                'reward': reward, 'floored': False,
                'raw_score': raw, 'style': style, 'style_gated': style_gated,
                'completeness': completeness, 'quality': quality,
                'antisem_rate': a_rate, 'antisem_penalty': a_pen,
                'sentiment_guard': s_guard, 'rep_ratio': rep,
                'word_count': w,
            })
        return out
