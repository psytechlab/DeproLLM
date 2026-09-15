# Personal prompt v1

Этот файл нужен только для повторного запуска эксперимента. Для знакомства с
его смыслом и итогами достаточно прочитать [DESIGN.md](DESIGN.md), а затем
[RESULTS.md](RESULTS.md). Полное воспроизведение требует доступа к закрытому
reward runtime и двум обученным LoRA.

Краткая навигация: [дизайн](DESIGN.md), [результаты](RESULTS.md),
[точный протокол](protocol.json).

Inference-only сравнение двух существующих GRPO-LoRA с исходной Qwen при
neutral, role и constrained prompting в одной личной форме. Научный вопрос,
точные инструкции и ограничения находятся в DESIGN.md и protocol.json.

Локальные проверки:

```bash
.venv/bin/python -m unittest discover -s experiments/personal_prompt_v1 -v
.venv/bin/python -m unittest discover -s experiments/topic_framing_v1 -v
.venv/bin/python -m unittest discover -s experiments/prompt_baseline_v1 -v
python3 check_grpo_notebook.py
```

Сборка пилота:

```bash
.venv/bin/python experiments/personal_prompt_v1/prepare_kaggle.py \
  --mode pilot \
  --output downloads/kaggle/submissions/deprollm-personal-prompt-v1-pilot-v1
```

Перед отправкой проверить в kernel-metadata.json: is_private=true, только
приватный runtime Dataset и два приватных kernel source. Отправлять с
`--accelerator NvidiaTeslaT4`. Пилот должен завершить 10/10 запросов без
scoring errors; prepare full повторно проверяет все его hashes и схему.

```bash
.venv/bin/python experiments/personal_prompt_v1/prepare_kaggle.py \
  --mode full \
  --pilot-evidence downloads/kaggle/personal-prompt-pilot-v1/personal_prompt_pilot/run_manifest.json \
  --output downloads/kaggle/submissions/deprollm-personal-prompt-v1-full-v1
```

После скачивания COMPLETE full:

```bash
.venv/bin/python experiments/personal_prompt_v1/analyze.py \
  --input downloads/kaggle/personal-prompt-full-v1/personal_prompt_full \
  --output downloads/kaggle/personal-prompt-full-v1/analysis
```

Outputs, submissions, LoRA и закрытый runtime остаются в игнорируемом
downloads/ или во временном /tmp. В Git их не добавлять.
