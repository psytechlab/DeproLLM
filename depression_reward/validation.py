"""Depression Markers сгенерированных текстов vs опубликованные средние
(Stankevich et al. 2019).

CLI:
  python -m depression_reward.validation texts.jsonl [--field completion]
  python -m depression_reward.validation texts.txt      # блоки через пустую строку
  python -m depression_reward.validation --demo         # на мок-текстах пакета

Каверзность sentiment: опубликованные 0.09/3.61 — абсолютные суммы по эссе
(per-word невозможен: веса linis-crowd ограничены 2.0), поэтому сравниваем
sentiment_abs = sentiment_rate * word_count. Для pro_1p_sing_rate принята
интерпретация «доля среди личных местоимений» (как в признаке классификатора);
если в статье была доля среди всех слов — сравнение остаётся направленным.
"""
import json
import statistics
import sys
from pathlib import Path

# маркер -> (среднее при депрессии, среднее в норме)
MARKERS_REF = {
    'pro_1p_sing_rate': (0.53, 0.27),
    'trager_coef': (1.27, 0.99),
    'mean_sentence_len': (13.4, 14.7),
    'sentiment_abs': (0.09, 3.61),
}


def depression_markers(texts: list[str], pipeline=None) -> list[dict]:
    if pipeline is None:
        from .features import FeaturePipeline
        pipeline = FeaturePipeline()
    rows = []
    for res in pipeline.extract_batch(texts):
        if res is None:
            rows.append(None)
            continue
        rows.append({
            'pro_1p_sing_rate': res.cues['pro_1p_sing_rate'],
            'trager_coef': res.cues['trager_coef'],
            'mean_sentence_len': res.cues['mean_sentence_len'],
            'sentiment_per_word': res.pdict['sentiment_rate'],
            'sentiment_abs': res.pdict['sentiment_rate'] * res.word_count,
            'word_count': res.word_count,
        })
    return rows


def markers_report(texts: list[str], pipeline=None) -> str:
    rows = [r for r in depression_markers(texts, pipeline) if r is not None]
    if not rows:
        return 'нет валидных текстов'
    lines = [f'Текстов: {len(rows)} (валидных из {len(texts)})',
             f"{'маркер':<22}{'среднее':>10}{'депр.':>10}{'норма':>10}  вердикт"]
    for marker, (ref_d, ref_n) in MARKERS_REF.items():
        mean = statistics.fmean(r[marker] for r in rows)
        # к какому референсу ближе
        leans = 'депрессия' if abs(mean - ref_d) < abs(mean - ref_n) else 'норма'
        lines.append(f'{marker:<22}{mean:>10.3f}{ref_d:>10.2f}{ref_n:>10.2f}  ~{leans}')
    mean_spw = statistics.fmean(r['sentiment_per_word'] for r in rows)
    lines.append(f'{"(sentiment_per_word)":<22}{mean_spw:>10.4f}{"—":>10}{"—":>10}')
    lines.append('Цель GRPO: грамматические маркеры сдвигаются к «депрессии», '
                 'sentiment остаётся нейтральным (не уходит в минус).')
    return '\n'.join(lines)


def _load_texts(path: Path, field: str) -> list[str]:
    if path.suffix == '.jsonl':
        texts = []
        for line in path.read_text().splitlines():
            if line.strip():
                texts.append(json.loads(line)[field])
        return texts
    # plain text: блоки, разделённые пустой строкой
    return [b.strip() for b in path.read_text().split('\n\n') if b.strip()]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--demo' in sys.argv:
        from . import sample_texts as st
        for name, text in [('DEPRESSIVE_ESSAY', st.DEPRESSIVE_ESSAY),
                           ('NEUTRAL_ESSAY', st.NEUTRAL_ESSAY)]:
            print(f'--- {name} ---')
            print(markers_report([text]))
            print()
        return
    if not args:
        print(__doc__)
        sys.exit(2)
    field = 'completion'
    if '--field' in sys.argv:
        field = sys.argv[sys.argv.index('--field') + 1]
    print(markers_report(_load_texts(Path(args[0]), field)))


if __name__ == '__main__':
    main()
