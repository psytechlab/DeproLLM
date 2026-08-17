"""topics.txt -> prompts.jsonl.

Шаблон промпта НЕ упоминает депрессию/стиль; каждая тема механически
проверяется: ни одна её mystem-лемма не должна входить в antisem-лексикон
(иначе промпт сам спровоцирует штрафуемую лексику).

Запуск из корня репозитория:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      .venv/bin/python -m depression_reward.prompts.make_prompts
"""
import json
from pathlib import Path

TEMPLATE = 'Напишите эссе объёмом 300–500 слов на тему: «{topic}».'

HERE = Path(__file__).parent


def main():
    from depression_reward.features import FeaturePipeline
    from depression_reward.penalty import _norm, load_lexicon

    lexicon = load_lexicon()
    pipeline = FeaturePipeline()

    topics = [line.strip() for line in (HERE / 'topics.txt').read_text().splitlines()
              if line.strip() and not line.strip().startswith('#')]
    assert len(set(topics)) == len(topics), 'дубликаты тем'

    collisions = []
    for topic in topics:
        # темы короткие — лемматизируем напрямую, без полного извлечения признаков
        rz = pipeline._razdel(topic)
        mys = pipeline._mystem(rz['tokens'], rz['sentences'])
        for sentence in mys['lemma_mys']:
            for lemma in sentence:
                if lemma.isalpha() and _norm(lemma) in lexicon:
                    collisions.append((topic, lemma))
    assert not collisions, f'темы содержат antisem-леммы: {collisions}'

    out_path = HERE / 'prompts.jsonl'
    with open(out_path, 'w', encoding='utf-8') as fp:
        for topic in topics:
            fp.write(json.dumps({'prompt': TEMPLATE.format(topic=topic)},
                                ensure_ascii=False) + '\n')
    print(f'{len(topics)} промптов -> {out_path}')


if __name__ == '__main__':
    main()
