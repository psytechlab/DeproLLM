# Prompt baseline v1: техническое воспроизведение

Этот файл нужен только для повторного запуска эксперимента. Для знакомства с
его смыслом и итогами достаточно прочитать [DESIGN.md](DESIGN.md), а затем
[RESULTS.md](RESULTS.md). Полное воспроизведение требует доступа к закрытому
reward runtime и двум обученным LoRA.

Научный вопрос и условия описаны в [DESIGN.md](DESIGN.md), итоговые метрики -
в [RESULTS.md](RESULTS.md), а точные prompts, topics, seeds, версии и хэши -
в [protocol.json](protocol.json).

Эксперимент выполняет inference исходной Qwen и двух уже обученных GRPO-LoRA.
Нового обучения здесь нет. `prepare_kaggle.py` создаёт приватный Kaggle notebook
и metadata, `run_eval.py` выполняет генерацию и scoring внутри Kaggle,
`test_protocol.py` проверяет сетку условий, хэши и приватность сборки.

Используется один T4, vLLM 0.9.2, Transformers 4.53.2 и Triton 3.2.0. Два
приватных kernel с LoRA и приватный reward Dataset подключаются как inputs.
Их manifests, версии и SHA-256 проверяются до генерации; закрытые файлы
распаковываются в `/tmp` и не сохраняются в output.

## Проверка и пилот

```bash
.venv/bin/python -m unittest discover -s experiments/prompt_baseline_v1 -v
.venv/bin/python experiments/prompt_baseline_v1/prepare_kaggle.py \
  --mode pilot \
  --output downloads/kaggle/submissions/deprollm-prompt-baseline-v1-pilot
```

Сборщик ничего не отправляет в Kaggle. Перед отправкой выполнить
`python3 check_grpo_notebook.py`, проверить `is_private=true` в metadata и явно
указать `--accelerator NvidiaTeslaT4`. Пилот содержит две темы, пять условий и
один seed, всего 10 служебных текстов; они не входят в основной анализ.

После успешного пилота скачать весь output. Full-сборка принимает только
совпадающий manifest с 10/10 строками, 73 признаками и без scoring errors:

```bash
.venv/bin/python experiments/prompt_baseline_v1/prepare_kaggle.py --mode full \
  --pilot-evidence downloads/kaggle/prompt-baseline-pilot-v3/prompt_eval_pilot/run_manifest.json \
  --output downloads/kaggle/submissions/deprollm-prompt-baseline-v1-full
```

Полный output содержит 300 текстов, token IDs, 73 признака, reward breakdown,
manifests и логи. Он вместе с LoRA и закрытым runtime остаётся в игнорируемом
`downloads/` или в приватном Kaggle и в Git не добавляется.
