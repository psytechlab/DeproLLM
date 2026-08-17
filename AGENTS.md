# Инструкции проекта DeproLLM

## Область проекта

Репозиторий содержит приватный reward-пакет и Kaggle-ноутбук для GRPO-обучения
`unsloth/Qwen3-4B-Instruct-2507` генерировать русские эссе с заданными
грамматическими и психолингвистическими характеристиками. Промпты не упоминают
депрессию, реальных эссе пациентов в репозитории нет.

## Приватность

- `depression_reward/vendor/`, `vendor/data/psydicts.json` и параметры
  классификатора происходят из приватного TITANIS. Не публиковать репозиторий
  или эти файлы без отдельного разрешения правообладателя.
- Не добавлять результаты обучения, checkpoints, optimizer states, LoRA-веса
  и полные Kaggle outputs без отдельного решения.
- Автономный `grpo_depression.ipynb` содержит скрытую base64-копию
  `depression_reward.zip`; это намеренный транспортный дубль только для запуска
  одного файла в Kaggle.

## Классификатор и reward

- Фактический классификатор: `StandardScaler -> PCA(46) -> SVC(rbf, C=1000)`.
- `decision_function` — не вероятность и не клиническая оценка. Большее
  значение означает лишь большую близость к положительному классу модели.
- Порядок 73 признаков в `depression_reward/feature_names.py` заморожен и не
  должен меняться.
- Единственный сохраняемый формат классификатора — `model/weights.npz` вместе
  с `params.json`; `model_repacked.joblib` и верхнеуровневый `repacked_model/`
  не добавлять.
- Эталонные scores: depr `-0.0557977018`, norm `-0.1023075880`.

## Проверка и упаковка

После изменения reward-пакета выполнить:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python .venv/bin/python -m depression_reward.selfcheck
.venv/bin/python embed_zip_in_notebook.py
```

Вторая команда пересобирает игнорируемый `depression_reward.zip` без кэшей и
joblib-дубля, затем обновляет встроенную копию в `grpo_depression.ipynb`.
Перед коммитом убедиться, что ноутбук не содержит outputs и execution counts.

Локально использовать `.venv/bin/python`, если окружение присутствует. GPU-
обучение выполнять в Kaggle; локальные проверки reward работают на CPU.
