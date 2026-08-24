# depression_reward — GRPO-reward «депрессивного стиля»

Самодостаточный пакет: текст → скалярный reward. Внутри — переупакованный
классификатор депрессии TITANIS (StandardScaler → PCA(46) → SVC(rbf)) и
вендоренное извлечение его 73 признаков (razdel → mystem → PsyCues + PsyDict).
Docker не нужен, GPU не нужен, torch/trl пакет не импортирует.

**Приватно.** Содержит вендоренный код TITANIS (ФИЦ ИУ РАН) — не публиковать,
не выкладывать в публичные репозитории; после завершения проекта удалить.

## Установка

Локально: зависимости уже в `.venv` репозитория. В Colab:

```bash
pip install -r depression_reward/requirements-colab.txt
python -m depression_reward.selfcheck   # обязательный smoke-тест после загрузки
```

Пакет сам выставляет `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
(нужно старому isanlp) при условии, что `depression_reward` импортируется
раньше isanlp. mystem-бинарник pymystem3 докачает при первом вызове.

`StyleScorer` собирает модель в памяти из `model/weights.npz` (источник истины)
и сверяет её с эталоном. Альтернативная joblib-копия намеренно не хранится,
чтобы классификатор не дублировался и не зависел от версии сериализации sklearn.

## Использование

```python
from depression_reward import DepressionReward, RewardConfig, make_reward_func

rm = DepressionReward()                 # или DepressionReward(RewardConfig(w_antisem=1.5))
rewards = rm(["текст эссе...", ...])    # list[float]
detail = rm.breakdown(["текст..."])     # по-компонентная раскладка (см. ниже)

# для trl:
trainer = GRPOTrainer(..., reward_funcs=[make_reward_func(log_path='reward_log.jsonl')])
```

`breakdown` возвращает по тексту: `reward, floored, raw_score, style,
style_gated, completeness, quality, antisem_rate, antisem_penalty,
sentiment_guard, rep_ratio, word_count` — используйте для мониторинга
во время обучения (лог пишется в `log_path` как JSONL).

## Формула

```
reward = w_style·style_gated + w_complete·completeness + w_quality·quality
       − w_antisem·antisem_penalty − w_sentiment_guard·sentiment_guard
```

- `style` = sigmoid((decision_function − style_center)/style_temp); гейт ×w/150
  на коротких текстах (классификатор обучен на эссе ~150+ слов).
- `completeness`: плато 1.0 на 300–500 словах, линейные скаты по краям.
- `antisem_penalty`: доля лемм из лексикона (curated_lexicon.txt +
  be_sadness из psydicts, всего ~190 лемм) сверх 1/200 слов. Не даёт модели
  хакать классификатор явной грустной лексикой — стиль должен приходить
  из грамматики.
- `quality`: штраф за повторы 4-грамм; rep_ratio > 0.6 → флор.
- Флор −1.0: пустой/не-русский/<20 слов/дегенерация. GRPO работает на
  относительных внутригрупповых преимуществах, абсолютная шкала не важна.
- `sentiment_guard` (вес 0 по умолчанию) — запасной канал, если модель начнёт
  обходить лексикон несписочным негативом: следите за sentiment-маркером
  в `validation`, при дрейфе в минус включите `w_sentiment_guard=0.5`.

## Калибровка (обязательно перед GRPO)

Дефолтный `style_center=−0.08` — середина двух референсных векторов. После
baseline-генерации (~30 эссе instruct-моделью по prompts.jsonl):

```python
bd = rm.breakdown(baseline_texts)
center = float(np.mean([b['raw_score'] for b in bd if not b['floored']]))
cfg = RewardConfig(style_center=center)   # style baseline'а станет ~0.5
```

Иначе style может насытиться (все ~0 или ~1) и градиент преимуществ умрёт.

## Валидация «стиль, а не семантика»

```bash
python -m depression_reward.validation generated.jsonl --field completion
```

Сравнивает Depression Markers с опубликованными средними (Stankevich et al.
2019): доля местоимений 1 л. ед. ч. (0.53/0.27), коэффициент Трейгера
(1.27/0.99), длина предложения (13.4/14.7), sentiment (0.09/3.61 — абсолютные
суммы). Успех: грамматические маркеры уходят к «депрессии», sentiment остаётся
нейтральным.

## Файлы

- `reward.py`, `config.py` — ядро и все ручки; `grpo_adapter.py` — обёртка trl
  (вырезает `<think>` Qwen3, понимает chat-формат).
- `classifier.py` + `model/` — веса модели, сборка в памяти и эталонная сверка.
- `features.py` + `vendor/` — 73 признака без Docker (порядок заморожен
  в `feature_names.py` и проверяется на каждом извлечении).
- `penalty.py` + `curated_lexicon.txt` — antisem-штраф и дегенерация.
- `prompts/prompts.jsonl` — 80 тем эссе (8 категорий), темы механически
  проверены на отсутствие antisem-лемм (`make_prompts.py`).
- `selfcheck.py` — 9 проверок; `validation.py` — маркеры; `sample_texts.py` —
  мок-эссе.

## Закрытый runtime для Kaggle

```bash
python make_kaggle_runtime.py build v3
python make_kaggle_runtime.py pin \
  downloads/kaggle/deprollm-depression-reward-runtime-v3
python make_kaggle_runtime.py check \
  downloads/kaggle/deprollm-depression-reward-runtime-v3
```

Загрузите созданный `deprollm-depression-reward-runtime-v3.zip` как новую
версию **приватного** Kaggle Dataset и подключите его к notebook через Input.
Kaggle распакует upload ZIP в одноимённую папку. Внутренний ZIP намеренно имеет
имя `depression_reward.payload`, чтобы Kaggle не распаковал его второй раз.
Notebook рекурсивно найдёт `manifest.json`, сверит закреплённые версию и SHA-256,
проверит `SHA256SUMS` и только затем распакует payload в
`/tmp/deprollm_reward_runtime`. Этот путь не попадает в сохраняемые Kaggle
outputs; notebook явно настраивает `PYTHONPATH` и пути к requirements/prompts.

Упаковщик не перезаписывает существующие версии и отказывается собирать runtime
из незакоммиченных изменений кода или данных `depression_reward/`. Изменение
README не влияет на runtime и не блокирует сборку. Это связывает приватный
артефакт с конкретным Git-коммитом и не даёт случайно переиспользовать имя
старой версии для новых весов или кода.
