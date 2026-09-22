# Strong-model prompt v1

Этот файл нужен только для повторного запуска эксперимента. Для знакомства с
его смыслом и итогами достаточно прочитать [DESIGN.md](DESIGN.md), а затем
[RESULTS.md](RESULTS.md). Повторная оценка ответов требует доступа к закрытому
reward runtime; точное повторение генерации также зависит от внешней модели.

Краткая навигация: [дизайн](DESIGN.md), [результаты](RESULTS.md),
[точный протокол](protocol.json).

Зафиксированное inference-only сравнение GLM-5.3-Flash via ZCode Agent с условиями
последнего `personal-prompt-v1`. Генерационный kit намеренно не содержит
закрытый reward runtime или прежние результаты.

Локальные проверки и сборка нового kit:

```bash
.venv/bin/python -m unittest discover -s experiments/strong_model_prompt_v1 -v
.venv/bin/python experiments/strong_model_prompt_v1/prepare_zcode.py \
  --output downloads/zcode/strong-model-prompt-v1-flash-kit
```

Открывать в ZCode нужно отдельную папку конкретного батча, а не репозиторий и
не весь kit. Сначала выполнить `pilot/batch_00_pilot`; после успешной чисто
технической проверки — девять папок из `full/`, каждый раз в новой задаче.
Точный текст задачи уже находится в `TASK.md` каждого батча.

После генерации собрать ответы в новый локальный каталог:

```bash
.venv/bin/python experiments/strong_model_prompt_v1/collect_responses.py \
  --kit downloads/zcode/strong-model-prompt-v1-flash-kit --mode pilot \
  --output downloads/zcode/strong-model-prompt-v1-flash-pilot
```

Main собирается той же командой с `--mode full` и новым output. Pilot не
оценивается классификатором до фиксации и запуска main. Scoring и анализ
выполняются только локально закрытым runtime отдельными командами после
получения 180/180 ответов.

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  .venv/bin/python experiments/strong_model_prompt_v1/score_responses.py \
  --input downloads/zcode/strong-model-prompt-v1-flash-full \
  --reward-config downloads/kaggle/personal-prompt-full-v1/personal_prompt_full/reward_config.json \
  --output downloads/zcode/strong-model-prompt-v1-flash-scored

.venv/bin/python experiments/strong_model_prompt_v1/analyze.py \
  --input downloads/zcode/strong-model-prompt-v1-flash-scored \
  --qwen-scored downloads/kaggle/personal-prompt-full-v1/personal_prompt_full/scored.jsonl \
  --qwen-manifest downloads/kaggle/personal-prompt-full-v1/personal_prompt_full/run_manifest.json \
  --output downloads/zcode/strong-model-prompt-v1-flash-analysis
```

Kit и результаты остаются в игнорируемом `downloads/`.
