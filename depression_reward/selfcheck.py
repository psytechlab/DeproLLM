"""Самопроверка пакета: python -m depression_reward.selfcheck [--timing]

Плоские assert-тесты (pytest не требуется). Ненулевой exit code при провале.
Использовать и локально, и в Colab после загрузки пакета.
"""
import sys
import time

from . import sample_texts as st
from .classifier import REFERENCE_SCORES, StyleScorer
from .config import RewardConfig
from .feature_names import FEATURE_NAMES
from .features import FeaturePipeline
from .grpo_adapter import make_reward_func
from .penalty import antisem_rate, load_lexicon, repetition_ratio
from .reward import DepressionReward

_failures = []


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    print(f'[{status}] {name}' + (f'  ({detail})' if detail else ''))
    if not condition:
        _failures.append(name)


def main():
    timing = '--timing' in sys.argv

    # 1. модель воспроизводит эталонные скоры на сохранённых 73-мерных векторах
    # (численная сверка зашита в init StyleScorer)
    try:
        scorer = StyleScorer()
        check('model-reference-vectors', True,
              ' '.join(f'{name}={score:.10f}'
                       for name, score in REFERENCE_SCORES.items()))
    except Exception as e:
        check('model-reference', False, str(e))
        print('дальнейшие проверки невозможны'); sys.exit(1)

    pipeline = FeaturePipeline()

    # 2. имена и порядок 73 признаков
    res = pipeline.extract(st.NEUTRAL_ESSAY)
    names = list(res.cues) + list(res.pdict)
    check('feature-names', names == FEATURE_NAMES and len(res.vector) == 73,
          f'{len(names)} имён')

    # 3. демонстрационный текст сдвинут выше нейтрального по raw-скору
    t0 = time.perf_counter()
    depr_res = pipeline.extract(st.DEPRESSIVE_ESSAY)
    extract_dt = time.perf_counter() - t0
    raw_depr = float(scorer.raw_scores(depr_res.vector)[0])
    raw_norm = float(scorer.raw_scores(res.vector)[0])
    check('sample-text-ordering', raw_depr - raw_norm > 0.005,
          f'sample_depr={raw_depr:.4f} sample_norm={raw_norm:.4f} '
          f'margin={raw_depr - raw_norm:.4f}')
    if timing:
        print(f'    извлечение признаков: {extract_dt * 1000:.0f} мс/текст')

    reward_model = DepressionReward()

    # 4. явная лексика включает штраф и снижает reward
    bd = reward_model.breakdown([st.NEUTRAL_ESSAY, st.SAD_ESSAY])
    clean, sad = bd
    check('antisem-monotonic',
          sad['antisem_penalty'] > 0 and sad['reward'] < clean['reward'],
          f"penalty={sad['antisem_penalty']:.2f} "
          f"reward {clean['reward']:.3f} -> {sad['reward']:.3f}")

    # 5. короткий текст: completeness низкая, reward ниже полного
    bd_short = reward_model.breakdown([st.SHORT_TEXT])[0]
    check('completeness-short',
          bd_short['completeness'] < 0.2 and bd_short['reward'] < clean['reward'],
          f"completeness={bd_short['completeness']:.2f} words={bd_short['word_count']}")

    # 6. дегенеративный повтор — флор
    bd_degen = reward_model.breakdown([st.DEGENERATE_TEXT])[0]
    check('degenerate-floor', bd_degen['floored'] and
          bd_degen['reward'] == reward_model.cfg.floor_reward,
          f"rep_ratio={bd_degen['rep_ratio']:.2f}")

    # 7. мусор не роняет и флорится
    garbage = ['', 'hello world, this is english text only', '<think>вечно думаю']
    rewards = reward_model(garbage)
    check('robustness', len(rewards) == 3 and
          all(r == reward_model.cfg.floor_reward for r in rewards),
          f'{rewards}')

    # 8. адаптер: str и chat-формат, <think>-блоки
    func = make_reward_func()
    out_str = func(prompts=['p1', 'p2'],
                   completions=[st.NEUTRAL_ESSAY,
                                '<think>рассуждения</think>' + st.DEPRESSIVE_ESSAY])
    out_chat = func(prompts=['p1'],
                    completions=[[{'role': 'assistant',
                                   'content': st.DEPRESSIVE_ESSAY}]])
    check('adapter',
          len(out_str) == 2 and len(out_chat) == 1 and
          all(isinstance(r, float) for r in out_str + out_chat) and
          out_str[1] > reward_model.cfg.floor_reward,
          f'str={[round(r, 3) for r in out_str]} chat={[round(r, 3) for r in out_chat]}')

    # 9. лексикон загружен и ловит явные слова
    lexicon = load_lexicon()
    sad_res = pipeline.extract(st.SAD_ESSAY)
    check('lexicon', len(lexicon) > 150 and antisem_rate(sad_res.lemmas, lexicon) > 0,
          f'{len(lexicon)} лемм, rate={antisem_rate(sad_res.lemmas, lexicon):.4f}')

    if timing:
        t0 = time.perf_counter()
        reward_model([st.NEUTRAL_ESSAY] * 8)
        dt = time.perf_counter() - t0
        print(f'    reward-батч из 8 текстов: {dt:.2f} с ({dt / 8 * 1000:.0f} мс/текст)')

    print()
    if _failures:
        print(f'ПРОВАЛ: {len(_failures)} проверок: {_failures}')
        sys.exit(1)
    print('Все проверки пройдены.')


if __name__ == '__main__':
    main()
