# Topic framing v1

Этот файл нужен только для повторного запуска эксперимента. Для знакомства с
его смыслом и итогами достаточно прочитать [DESIGN.md](DESIGN.md), а затем
[RESULTS.md](RESULTS.md). Полное воспроизведение требует доступа к закрытому
reward runtime и двум обученным LoRA.

Краткая навигация: [дизайн](DESIGN.md), [результаты](RESULTS.md),
[точный протокол](protocol.json).

Inference-only follow-up, который попарно меняет личную и обобщённую форму
одних и тех же тем. Научный дизайн и ограничения полностью зафиксированы в
`DESIGN.md`; точные значения находятся в `protocol.json`. Повышение
`decision_function` означает только приближение к положительному классу SVC.

Эксперимент не обучает модель и не меняет reward, 73 признака, classifier,
calibration или приватный runtime v2. Base и два существующих GRPO-адаптера
генерируют все тексты заново в одном vLLM-окружении. Full: 360 текстов;
pilot: 12 служебных текстов, не входящих в анализ.

Локальные проверки:

```bash
.venv/bin/python -m unittest discover -s experiments/topic_framing_v1 -v
python3 check_grpo_notebook.py
```

Сборка пилота ничего не отправляет:

```bash
.venv/bin/python experiments/topic_framing_v1/prepare_kaggle.py \
  --mode pilot \
  --output downloads/kaggle/submissions/deprollm-topic-framing-v1-pilot-v1
```

Перед `kaggle kernels push` проверить `kernel-metadata.json`: private=true,
один приватный Dataset и два приватных kernel source. Отправлять только с
явным `--accelerator NvidiaTeslaT4`. Пилот успешен при COMPLETE, 12/12 строках,
нуле scoring errors, совпавших hashes, 73 признаках и geometry каждой строки.
Направление score в пилоте не является воротами.

После скачивания полного каталога `topic_framing_pilot` full-сборка требует
его неизменённый manifest и проверяет все pilot hashes:

```bash
.venv/bin/python experiments/topic_framing_v1/prepare_kaggle.py \
  --mode full \
  --pilot-evidence downloads/kaggle/topic-framing-pilot-v1/topic_framing_pilot/run_manifest.json \
  --output downloads/kaggle/submissions/deprollm-topic-framing-v1-full-v1
```

После COMPLETE скачать весь `topic_framing_full` и выполнить локальный анализ:

```bash
.venv/bin/python experiments/topic_framing_v1/analyze.py \
  --input downloads/kaggle/topic-framing-full-v1/topic_framing_full \
  --output downloads/kaggle/topic-framing-full-v1/analysis
```

Анализ сначала проверяет manifest, hashes, сетку и scores. Он создаёт JSON и
Markdown со всеми зафиксированными контрастами, geometry summaries,
`blind_review.csv` для человека и отдельный `blind_review_key.csv`. До
заполнения формы key не открывать.

Все submissions и outputs остаются в игнорируемом `downloads/`. Тексты,
LoRA, runtime, checkpoints и секреты в Git не добавляются.
