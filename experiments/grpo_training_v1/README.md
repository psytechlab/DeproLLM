# GRPO training v1: техническое воспроизведение

Этот файл нужен только для повторного запуска эксперимента. Для знакомства с
его смыслом и итогами достаточно прочитать [DESIGN.md](DESIGN.md), а затем
[RESULTS.md](RESULTS.md). Полное воспроизведение требует доступа к закрытому
reward runtime.

Научная последовательность запусков и параметры описаны в
[DESIGN.md](DESIGN.md), результаты — в [RESULTS.md](RESULTS.md), нормализованная
конфигурация — в [configs/common.json](configs/common.json), сведения о двух
воспроизводимых запусках — в [configs/runs.json](configs/runs.json).

Обучающий workflow находится в корневом `grpo_depression.ipynb`. Notebook
загружает закреплённый приватный reward runtime, проверяет manifest и SHA-256,
загружает Qwen3-4B-Instruct-2507, подключает LoRA и выполняет GRPO. Затем он
генерирует baseline и post-training eval на десяти отложенных темах.

## Локальные проверки

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  .venv/bin/python -m depression_reward.selfcheck
python3 check_grpo_notebook.py
```

В Git `SMOKE_RUN` должен оставаться `False`. Для двухшагового smoke run нужна
временная копия notebook с `SMOKE_RUN = True`; в Git она не добавляется. После
успешного smoke полный запуск отправляется только как приватный Kaggle kernel с
явным `--accelerator NvidiaTeslaT4`.

Reference run использовал training seed 3407. Для repeat менялись только
training seed на 4407 и идентификаторы версии запуска; модель, данные, reward,
learning rate, scheduler и остальные параметры оставались зафиксированными.

Полные Kaggle outputs, eval-тексты, reward logs, LoRA, checkpoints и optimizer
states остаются вне Git. В репозитории сохраняются только notebook, код reward,
конфигурации протокола, хэши и агрегированные результаты.
