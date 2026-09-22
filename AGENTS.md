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
- `grpo_depression.ipynb` загружает runtime из приватного Kaggle Dataset
  `deprollm-depression-reward-runtime`. Ноутбук закрепляет версию и SHA-256;
  Dataset нельзя делать публичным.

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
```

Новый runtime Dataset собирать только из чистого коммита и под новой версией:

```bash
.venv/bin/python make_kaggle_runtime.py build v3
.venv/bin/python make_kaggle_runtime.py pin \
  downloads/kaggle/deprollm-depression-reward-runtime-v3
.venv/bin/python make_kaggle_runtime.py check \
  downloads/kaggle/deprollm-depression-reward-runtime-v3
```

`build` создаёт игнорируемые каталог и upload ZIP без кэшей, README и
joblib-дубля. Внутренний ZIP хранится как `depression_reward.payload`, чтобы
Kaggle не распаковывал его и notebook мог проверить SHA-256 всего runtime.
`pin` обновляет в notebook ожидаемые версию и SHA-256; `check` проверяет Dataset,
отсутствие встроенного base64, outputs и execution counts. Существующую версию
не перезаписывать: создать следующую и загрузить её как новую версию приватного
Kaggle Dataset. Изменение только `depression_reward/README.md` не требует новой
версии runtime и не блокирует упаковку. Notebook распаковывает runtime в
`/tmp/deprollm_reward_runtime`, а не в `/kaggle/working`, чтобы закрытый пакет
не сохранялся повторно среди Kaggle outputs.

Локально использовать `.venv/bin/python`, если окружение присутствует. GPU-
обучение выполнять в Kaggle; локальные проверки reward работают на CPU.

Перед отправкой notebook в Kaggle выполнить:

```bash
python3 check_grpo_notebook.py
```

В Git `SMOKE_RUN` всегда должен оставаться `False`. Для технического smoke run
создавать временную копию notebook с `SMOKE_RUN = True` и не коммитить её.
Полный 300-шаговый прогон начинать только после успешного двухшагового smoke.
При запуске через Kaggle CLI явно указывать
`--accelerator NvidiaTeslaT4`: обычный `enable_gpu` может выдать P100, который
несовместим с текущим CUDA/Triton-окружением notebook.
