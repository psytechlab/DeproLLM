import re
from typing import NamedTuple

import numpy as np

from .feature_names import FEATURE_NAMES

MIN_CYRILLIC_CHARS = 40


class ExtractionResult(NamedTuple):
    vector: np.ndarray        # (73,) float64
    cues: dict                # 39 именованных признаков PsyCues
    pdict: dict               # 34 именованных признака PsyDict
    lemmas: list              # list[list[str]] — леммы mystem по предложениям
    word_count: int
    sentence_count: int


class FeaturePipeline:
    """text -> 73-мерный вектор признаков классификатора (без Docker).

    Экстракторы создаются один раз: mystem — подпроцесс, его пересоздание
    на каждый вызов убило бы пропускную способность.
    """

    def __init__(self):
        # импорт здесь: vendor тянет isanlp/protobuf, env var уже выставлен в __init__
        from .vendor import (FeaturesMystem, FeaturesPsyCues, FeaturesPsyDict,
                             FeaturesRazdel)
        self._razdel = FeaturesRazdel()
        self._mystem = FeaturesMystem()
        self._cues = FeaturesPsyCues(psy_cues_normalization='words')
        self._pdict = FeaturesPsyDict(psy_dict_normalization='words')

    def _restart_mystem(self):
        from .vendor import FeaturesMystem
        self._mystem = FeaturesMystem()

    def _extract_once(self, text: str) -> ExtractionResult:
        rz = self._razdel(text)
        mys = self._mystem(rz['tokens'], rz['sentences'])
        cues = self._cues(text, mys['lemma_mys'], mys['postag_mys_unconverted'])
        pdict = self._pdict(mys['lemma_mys'])
        names = list(cues) + list(pdict)
        if names != FEATURE_NAMES:
            raise RuntimeError('feature order drift: extracted names != FEATURE_NAMES')
        vector = np.array(list(cues.values()) + list(pdict.values()), dtype=np.float64)
        return ExtractionResult(
            vector=vector,
            cues=cues,
            pdict=pdict,
            lemmas=mys['lemma_mys'],
            word_count=int(cues['word_count']),
            sentence_count=int(cues['sentence_count']),
        )

    def looks_valid(self, text: str, min_words: int = 20) -> bool:
        """Дешёвый пре-фильтр мусора до запуска mystem."""
        if not isinstance(text, str):
            return False
        if len(re.findall(r'[а-яА-ЯёЁ]', text)) < MIN_CYRILLIC_CHARS:
            return False
        if len(text.split()) < min_words:
            return False
        return True

    def extract(self, text: str) -> ExtractionResult:
        try:
            return self._extract_once(text)
        except Exception:
            # mystem-подпроцесс мог зависнуть/умереть — один рестарт и retry
            self._restart_mystem()
            return self._extract_once(text)

    def extract_batch(self, texts: list[str],
                      min_words: int = 20) -> list[ExtractionResult | None]:
        results = []
        for text in texts:
            if not self.looks_valid(text, min_words=min_words):
                results.append(None)
                continue
            try:
                results.append(self.extract(text))
            except Exception:
                results.append(None)
        return results
