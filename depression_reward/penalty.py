import json
from pathlib import Path

from .config import RewardConfig

_PKG = Path(__file__).parent


def _norm(lemma: str) -> str:
    return lemma.strip().lower().replace('ё', 'е')


def load_lexicon(curated_path: str | Path | None = None,
                 psydicts_path: str | Path | None = None) -> frozenset[str]:
    """Antisem-лексикон: курируемый список + be_sadness из psydicts (140 лемм).

    be_sadness — это ровно тот словарь «открытой грусти», которым модель могла
    бы хакать классификатор (be_sadness входит в его признаки).
    """
    curated_path = Path(curated_path) if curated_path else _PKG / 'curated_lexicon.txt'
    psydicts_path = Path(psydicts_path) if psydicts_path else _PKG / 'vendor' / 'data' / 'psydicts.json'

    lemmas = set()
    for line in curated_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            lemmas.add(_norm(line))

    psydicts = json.loads(psydicts_path.read_text())
    for lemma in psydicts['basic_emotions']['be_sadness']:
        lemmas.add(_norm(lemma))

    return frozenset(lemmas)


def antisem_rate(lemmas: list, lexicon: frozenset[str]) -> float:
    """Доля лемм из лексикона среди алфавитных лемм текста."""
    total = 0
    hits = 0
    for sentence in lemmas:
        for token in sentence:
            if token.isalpha():
                total += 1
                if _norm(token) in lexicon:
                    hits += 1
    return hits / total if total else 0.0


def antisem_penalty(rate: float, cfg: RewardConfig) -> float:
    return _clip01((rate - cfg.antisem_free_rate) / cfg.antisem_scale)


def repetition_ratio(text: str, n: int = 4) -> float:
    """1 - distinct n-грамм / всего n-грамм по whitespace-токенам; 0, если коротко."""
    tokens = text.lower().split()
    total = len(tokens) - n + 1
    if total < 1:
        return 0.0
    ngrams = {tuple(tokens[i:i + n]) for i in range(total)}
    return 1.0 - len(ngrams) / total


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
