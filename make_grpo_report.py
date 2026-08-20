# -*- coding: utf-8 -*-
"""Генерирует grpo_report.ipynb с нуля (страховочная копия генератора).

Источник истины теперь — сам grpo_report.ipynb: точечные правки делать в нём.
Этот скрипт нужен только если ноутбук утрачен/испорчен; после перегенерации
ячейки пустые — выполнить:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python .venv/bin/python -m nbconvert --to notebook --execute --inplace grpo_report.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# GRPO-обучение депрессивного стиля: отчёт по прогонам №1–2

**Модель:** Qwen3-4B-Instruct-2507 + LoRA (r=32), GRPO (trl + Unsloth, Kaggle T4).
**Reward:** классификатор депрессии TITANIS (Stankevich et al., 2019), извлечённый из Docker в самодостаточный пакет `depression_reward`.
**Даты прогонов:** №1 — 10.07.2026, №2 — 11.07.2026.

**TL;DR.** Прогон №2 (300 шагов) значимо сдвинул скор классификатора на отложенных темах: Δraw = **+0.52** (Welch t = 4.06, p ≈ 1.5·10⁻⁴; консервативный парный тест по 10 темам: t = 3.32, **p ≈ 0.009**; Уилкоксон p ≈ 0.0098; сдвиг положителен на 9 из 10 тем). Разложение сдвига по 73 признакам показывает: он достигнут **грамматикой** (блок PsyCues), а не грустной лексикой — словари негативных эмоций не выросли, sentiment остался нейтральным. Это и был критерий успеха: *выучен стиль, а не семантика*. Прогон №1 (100 шагов) вышел на плато и сохранён как диагностический кейс — его разбор дал правки, которые сделали №2 успешным.

*Как читать:* все цифры ниже пересчитываются кодом из лёгких артефактов `grpo_results/` (№1) и `grpo_results_v3/` (№2); ноутбук выполняется локально на CPU (~1–2 мин). Полные архивы прогонов: `results.zip` (№1) и `kaggle_output_v3/` (№2, LoRA-веса и чекпоинты 100/200/300).""")

md("""## 1. Постановка

**Задача.** Обучить LLM писать эссе, *стилистически* похожие на эссе людей с клинической депрессией: воспроизводить грамматику, синтаксис и структуру текста, а не тематику («как пишут», а не «о чём»). База — Liu et al., 2025 (*Capturing Classic Authorial Style in Long-Form Story Generation with GRPO Fine-Tuning*): там GRPO учит стиль конкретных авторов, reward — парный style judge.

**Ключевые отличия от статьи:**
- reward — **одновходовый классификатор** вместо парного judge: пайплайн StandardScaler → PCA(46) → SVC(rbf) поверх 73 психолингвистических признаков (39 PsyCues + 34 PsyDict), обучен на 557 эссе «Я, другие, мир» (110 пациентов с клинической депрессией + 447 здоровых, F1 ≈ 73%). Используется `decision_function` (выше = «депрессивнее»); эталонные скоры: депрессивный текст −0.0558, нейтральный −0.1023. Reference-тексты не нужны — паттерн стиля зашит в весах классификатора;
- **промпт не упоминает депрессию** — стиль должен въесться в LoRA-веса, а не исполняться по инструкции. Baseline и GRPO-модель получают байт-в-байт одинаковые промпты, поэтому весь сдвиг скора — из обучения;
- SFT-этап пропущен: instruct-модель уже умеет писать эссе нужного формата.

**Reward-функция:**

r = 1.0·style + 0.5·completeness + 0.25·quality − 1.0·antisem, где style = σ((raw − center)/temp)

- `style` — сигмоида от raw-скора классификатора (center/temp калибруются по baseline);
- `antisem` — штраф за явную депрессивную лексику (лексикон 189 лемм): главный механизм против «семантического шортката» — накачки скора грустными словами вместо стиля;
- `completeness` — плато 1.0 на 300–500 словах (синхронизировано с текстом промпта);
- `quality` — штраф за вырождение (повторы 4-грамм).

**Промпты:** 80 рефлексивных тем от первого лица («Напишите эссе объёмом 300–500 слов на тему: …»), имитирующих домен классификатора; 70 — обучение, 10 — hold-out для eval. Темы проверены на отсутствие лемм antisem-лексикона (иначе reward штрафовал бы за раскрытие темы).""")

code("""import json, os, statistics
from pathlib import Path

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

R1, R2 = Path('grpo_results'), Path('grpo_results_v3')

# палитра и chrome (валидированы validate_palette.js, light mode)
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURF = '#e1e0d9', '#c3c2b7', '#fcfcfb'
C1, C2 = '#2a78d6', '#e34948'          # прогон №1 / baseline — синий, №2 / GRPO — красный
C_CUES, C_DICT = '#2a78d6', '#eda100'  # блоки признаков: PsyCues / PsyDict

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': AXIS, 'axes.linewidth': 1.0,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.frameon': False, 'legend.fontsize': 9,
    'font.family': 'sans-serif', 'figure.dpi': 110,
})

def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]

log1, log2 = read_jsonl(R1 / 'reward_log.jsonl'), read_jsonl(R2 / 'reward_log.jsonl')
ts1 = json.load(open(R1 / 'outputs/checkpoint-100/trainer_state.json'))['log_history']
ts2 = json.load(open(R2 / 'outputs/checkpoint-300/trainer_state.json'))['log_history']
base_eval = read_jsonl(R2 / 'baseline_eval_essays.jsonl')
grpo_eval = read_jsonl(R2 / 'grpo_eval_essays.jsonl')
baseline30 = read_jsonl(R2 / 'baseline_essays.jsonl')

REF_DEPR, REF_NORM = -0.0557977018, -0.1023075880  # эталоны классификатора

print(f'прогон №1: {len(ts1)} шагов, {len(log1)} генераций в reward_log')
print(f'прогон №2: {len(ts2)} шагов, {len(log2)} генераций в reward_log')
print(f'eval №2: baseline {len(base_eval)} эссе, GRPO {len(grpo_eval)} эссе')""")

md("""## 2. Прогон №1 (100 шагов): плато и диагноз

Параметры: 100 шагов, пик LR 5·10⁻⁶, style_temp = 0.02, style_center = медиана raw baseline (−0.661). Итог по ноутбуку выглядел позитивно (reward 1.22 → 1.44 на eval), но разбор артефактов показал, что **обучение фактически не пошло**:

- reward по логу плоский: первые 160 генераций 1.27 → последние 160 1.32 (шум);
- **KL к шагу 100 ≈ 0.001** — политика почти не отошла от базовой модели;
- eval-сдвиг незначим: Δraw +0.16, Welch t = 0.82 (baseline n=30 против GRPO n=10, по 1 сиду).

**Диагноз.** При style_temp = 0.02 и sd(raw) ≈ 0.48 сигмоида style — практически ступенька на медиане baseline: style бимодален (55% значений >0.8, 41% <0.2), advantage внутри GRPO-группы почти бинарный и не говорит, *насколько* один текст лучше другого. Completeness и quality всегда 1.0 (сигнала не дают), antisem ≈ 0 — газовать нечем, градиент слабый и зашумлённый.

**Правки для прогона №2:** (а) style_temp автокалибруется как std(baseline raw)/2 → 0.234 — лечит ступеньку; (б) LR 5·10⁻⁶ → 1·10⁻⁵ и 100 → 300 шагов — против KL ≈ 0.001; (в) честный eval: 10 hold-out тем × 3 сида на каждую сторону + статистика. Reward-пакет не менялся — сравнение прогонов корректно.""")

code("""# динамика обучения: reward и KL из trainer_state, raw — по 8 генераций на шаг из reward_log
def per_step_raw(log, per=8):
    return [statistics.fmean(r['raw_score'] for r in log[i:i+per]) for i in range(0, len(log), per)]

def rolling(xs, w=15):
    return [statistics.fmean(xs[max(0, i-w+1):i+1]) for i in range(len(xs))]

panels = [
    ('reward (среднее по группе)', [e['reward'] for e in ts1], [e['reward'] for e in ts2], None),
    ('raw-скор классификатора',    per_step_raw(log1),          per_step_raw(log2),          REF_DEPR),
    ('KL к базовой политике (log)', [max(e['kl'], 1e-5) for e in ts1], [max(e['kl'], 1e-5) for e in ts2], None),
]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
for ax, (title, y1, y2, ref) in zip(axes, panels):
    ax.plot(range(1, len(y1)+1), rolling(y1), color=C1, lw=2)
    ax.plot(range(1, len(y2)+1), rolling(y2), color=C2, lw=2)
    ax.text(len(y1)+4, rolling(y1)[-1], '№1', color=C1, fontsize=9, va='center', fontweight='bold')
    ax.text(len(y2)+4, rolling(y2)[-1], '№2', color=C2, fontsize=9, va='center', fontweight='bold')
    ax.set_title(title, loc='left')
    ax.set_xlabel('шаг GRPO')
    ax.set_xlim(0, 330)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls='--')
        ax.text(8, ref, 'эталон depr', color=MUTED, fontsize=8, ha='left', va='bottom')
axes[2].set_yscale('log')
fig.suptitle('Прогон №1 стоит на месте, №2 — учится (скользящее среднее, окно 15 шагов)',
             x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.93))
plt.show()

print(f"KL на последнем шаге: №1 {ts1[-1]['kl']:.4f}, №2 {ts2[-1]['kl']:.4f}")
print(f"raw, среднее по первым/последним 20 шагам: №1 {statistics.fmean(per_step_raw(log1)[:20]):+.3f} -> "
      f"{statistics.fmean(per_step_raw(log1)[-20:]):+.3f}; №2 {statistics.fmean(per_step_raw(log2)[:20]):+.3f} -> "
      f"{statistics.fmean(per_step_raw(log2)[-20:]):+.3f}")""")

code("""# иллюстрация диагноза: распределение style-компоненты (первые 800 генераций каждого прогона)
s1 = [r['style'] for r in log1[:800]]
s2 = [r['style'] for r in log2[:800]]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, s, c, name, temp in [(axes[0], s1, C1, 'прогон №1', 0.02), (axes[1], s2, C2, 'прогон №2', 0.234)]:
    ax.hist(s, bins=25, range=(0, 1), color=c, edgecolor=SURF, linewidth=2)
    lo, hi = statistics.fmean(v < 0.2 for v in s), statistics.fmean(v > 0.8 for v in s)
    ax.set_title(f'{name}: style_temp = {temp}  ({lo:.0%} <0.2, {hi:.0%} >0.8)', loc='left')
    ax.set_xlabel('style = σ((raw − center)/temp)')
axes[0].set_ylabel('генераций')
fig.suptitle('Ступенька сигмоиды в №1: style почти бинарен — advantage не различает «чуть лучше» и «сильно лучше»',
             x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.91))
plt.show()""")

md("""## 3. Eval прогона №2: значимость сдвига

Дизайн: 10 hold-out тем (модель их не видела в обучении) × 3 сида на каждую сторону → 30 эссе baseline против 30 эссе GRPO, скоринг тем же классификатором. Welch-тест по всем 60 эссе — первичная оценка, но 3 сида внутри одной темы не независимы (псевдорепликация), поэтому основной тест — **консервативный парный**: усредняем raw по сидам внутри темы и сравниваем 10 пар (парный t-тест + знаковый ранговый критерий Уилкоксона).""")

code("""by_prompt_base, by_prompt_grpo = {}, {}
for rec in base_eval:
    by_prompt_base.setdefault(rec['prompt'], []).append(rec['raw_score'])
for rec in grpo_eval:
    by_prompt_grpo.setdefault(rec['prompt'], []).append(rec['raw_score'])
assert set(by_prompt_base) == set(by_prompt_grpo)
prompts = sorted(by_prompt_base)

def short_topic(prompt):
    return prompt.split('тему:')[-1].strip(' «».')

b_all = [x for p in prompts for x in by_prompt_base[p]]
g_all = [x for p in prompts for x in by_prompt_grpo[p]]
welch = stats.ttest_ind(g_all, b_all, equal_var=False)
print(f'Все эссе (30 vs 30): baseline {statistics.fmean(b_all):+.4f}, GRPO {statistics.fmean(g_all):+.4f}, '
      f'Δ = {statistics.fmean(g_all)-statistics.fmean(b_all):+.4f}')
print(f'  Welch: t = {welch.statistic:.2f}, p = {welch.pvalue:.2e}')
print(f'  эссе выше эталона depr ({REF_DEPR:.4f}): baseline {sum(x > REF_DEPR for x in b_all)}/30, '
      f'GRPO {sum(x > REF_DEPR for x in g_all)}/30')

b_mean = [statistics.fmean(by_prompt_base[p]) for p in prompts]
g_mean = [statistics.fmean(by_prompt_grpo[p]) for p in prompts]
paired = stats.ttest_rel(g_mean, b_mean)
wilc = stats.wilcoxon(g_mean, b_mean)
diffs = [g - b for g, b in zip(g_mean, b_mean)]
print(f'\\nПарный тест по темам (n = 10): t = {paired.statistic:.2f}, p = {paired.pvalue:.4f}')
print(f'Уилкоксон: W = {wilc.statistic:.0f}, p = {wilc.pvalue:.4f}; '
      f'сдвиг положителен на {sum(d > 0 for d in diffs)}/10 темах')

order = np.argsort(diffs)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for row, i in enumerate(order):
    ax.plot([b_mean[i], g_mean[i]], [row, row], color=AXIS, lw=1.4, zorder=1)
ax.scatter([b_mean[i] for i in order], range(len(order)), s=70, color=C1, zorder=2, label='baseline')
ax.scatter([g_mean[i] for i in order], range(len(order)), s=70, color=C2, zorder=2, label='GRPO')
ax.axvline(REF_DEPR, color=MUTED, lw=1, ls='--')
ax.set_ylim(-1.3, len(order) - 0.4)
ax.text(REF_DEPR, -0.75, 'эталон depr', color=MUTED, fontsize=8, va='top', ha='center')
ax.set_yticks(range(len(order)))
ax.set_yticklabels([short_topic(prompts[i]) for i in order], fontsize=8.5, color=INK2)
ax.set_xlabel('raw-скор классификатора (среднее по 3 сидам)')
ax.set_title('Сдвиг по каждой из 10 отложенных тем: 9 из 10 — вправо, к «депрессивнее»',
             loc='left', fontweight='bold', fontsize=12)
ax.legend(loc='lower right')
ax.grid(axis='y', visible=False)
fig.tight_layout()
plt.show()""")

md("""## 4. Маркеры Станкевич: местоимения сдвинулись, сантимент не ушёл в минус

Четыре маркера с опубликованными средними (Stankevich et al., 2019). Ожидание «выучен стиль»: грамматические маркеры движутся к депрессивному полюсу, sentiment не уходит в минус (не куплен грустной лексикой). Признаки извлекаются тем же пайплайном, что видит классификатор (razdel → mystem → PsyCues/PsyDict); `sentiment_abs` — абсолютная сумма по эссе, как в статье. Эти 4 маркера — лишь малая часть из 73 признаков модели, поэтому они дают направленную проверку, а не полную картину; полная — в разложении §5.""")

code("""from depression_reward.features import FeaturePipeline

pipe = FeaturePipeline()
res_base = pipe.extract_batch([r['completion'] for r in base_eval])
res_grpo = pipe.extract_batch([r['completion'] for r in grpo_eval])
assert all(r is not None for r in res_base + res_grpo)

def marker_frame(results):
    return pd.DataFrame([{
        'pro_1p_sing_rate': r.cues['pro_1p_sing_rate'],
        'trager_coef': r.cues['trager_coef'],
        'mean_sentence_len': r.cues['mean_sentence_len'],
        'sentiment_abs': r.pdict['sentiment_rate'] * r.word_count,
        'word_count': r.word_count,
    } for r in results])

mf_b, mf_g = marker_frame(res_base), marker_frame(res_grpo)
refs = {'pro_1p_sing_rate': (0.53, 0.27), 'trager_coef': (1.27, 0.99),
        'mean_sentence_len': (13.4, 14.7), 'sentiment_abs': (0.09, 3.61)}
table = pd.DataFrame({
    'baseline': mf_b.mean().round(3), 'GRPO': mf_g.mean().round(3),
    'реф. депрессия': pd.Series({k: v[0] for k, v in refs.items()}),
    'реф. норма': pd.Series({k: v[1] for k, v in refs.items()}),
})
display(table)
print('word_count здесь — служебная строка (референсов нет); о срезании длины — §6.')
print('Итог: pro_1p_sing_rate сдвинулся к депрессивному референсу (0.47 -> 0.52 при реф. 0.53);')
print('Трейгер и длина предложения практически не изменились (при Δraw +0.52 это ожидаемо —')
print('классификатор опирается на десятки признаков, см. §5); sentiment_abs остался в')
print('положительной зоне — сдвиг скора НЕ куплен грустной лексикой.')""")

md("""## 5. За счёт чего сдвиг: разложение по 73 признакам

Два взаимодополняющих метода на средних векторах eval-групп:

1. **Групповая абляция** — в средний baseline-вектор подставляем блок PsyCues (первые 39 признаков) или PsyDict (последние 34) из GRPO-среднего и смотрим, как меняется `decision_function`;
2. **Линеаризация** — градиент `decision_function` в baseline-центре (центральные разности, шаг 0.1·std признака); вклад признака = градиент × сдвиг его среднего.

Если сдвиг тянут PsyCues (морфология, синтаксис, местоимения) — это стиль; если PsyDict (тематические словари эмоций/стресса) — семантика.""")

code("""from depression_reward.classifier import StyleScorer
from depression_reward.feature_names import FEATURE_NAMES

scorer = StyleScorer()
score = lambda v: float(scorer.raw_scores(np.asarray(v).reshape(1, -1))[0])

vecs_b = np.vstack([r.vector for r in res_base])
vecs_g = np.vstack([r.vector for r in res_grpo])
mb, mg = vecs_b.mean(axis=0), vecs_g.mean(axis=0)
s_b, s_g = score(mb), score(mg)
print(f'decision_function в центрах групп: baseline {s_b:+.4f} -> GRPO {s_g:+.4f} (Δ {s_g - s_b:+.4f})')

n_cues = 39
abl_cues = score(np.concatenate([mg[:n_cues], mb[n_cues:]])) - s_b
abl_dict = score(np.concatenate([mb[:n_cues], mg[n_cues:]])) - s_b
print(f'абляция: только PsyCues от GRPO -> Δ {abl_cues:+.4f}; только PsyDict от GRPO -> Δ {abl_dict:+.4f}')

std = np.vstack([vecs_b, vecs_g]).std(axis=0)
h = np.where(std > 0, 0.1 * std, 1e-8)
grad = np.empty(len(mb))
for i in range(len(mb)):
    e = np.zeros(len(mb)); e[i] = h[i]
    grad[i] = (score(mb + e) - score(mb - e)) / (2 * h[i])
contrib = grad * (mg - mb)
print(f'линеаризация: сумма вкладов {contrib.sum():+.4f} '
      f'(PsyCues {contrib[:n_cues].sum():+.4f}, PsyDict {contrib[n_cues:].sum():+.4f})')

top = np.argsort(-np.abs(contrib))[:12][::-1]
fig, ax = plt.subplots(figsize=(9.5, 4.8))
colors = [C_CUES if i < n_cues else C_DICT for i in top]
bars = ax.barh(range(len(top)), contrib[top], color=colors, height=0.62)
for row, i in enumerate(top):
    v = contrib[i]
    ax.text(v + (0.006 if v >= 0 else -0.006), row, f'{v:+.3f}',
            va='center', ha='left' if v >= 0 else 'right', fontsize=8, color=INK2)
ax.axvline(0, color=AXIS, lw=1)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([FEATURE_NAMES[i] for i in top], fontsize=8.5, color=INK2)
ax.set_xlabel('вклад в сдвиг decision_function (градиент × сдвиг среднего)')
ax.set_title('Топ-12 движителей сдвига: почти все — грамматический блок PsyCues',
             loc='left', fontweight='bold', fontsize=12)
ax.grid(axis='y', visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_CUES, label='PsyCues (грамматика)'),
                   plt.Rectangle((0, 0), 1, 1, color=C_DICT, label='PsyDict (словари/семантика)')],
          loc='lower right')
xlim = ax.get_xlim(); ax.set_xlim(xlim[0] - 0.02, xlim[1] + 0.04)
fig.tight_layout()
plt.show()

sad = [i for i, n in enumerate(FEATURE_NAMES) if 'sad' in n or 'negative_emo' in n]
for i in sad:
    print(f'{FEATURE_NAMES[i]}: baseline {mb[i]:.4f} -> GRPO {mg[i]:.4f} (вклад {contrib[i]:+.4f})')""")

md("""## 6. Побочный эффект: модель срезает длину

GRPO-модель пишет короче baseline (см. цифры ниже; в промпте и completeness-плато — 300–500 слов). Проверка на reward-хакинг: суммарный вклад признаков длины (`word_count`, `sentence_count` и связанных) в сдвиг скора **отрицательный** — короткие тексты классификатор сами по себе «депрессивнее» не делает, т.е. это не взлом классификатора. Причина экономическая: w_complete = 0.5 дёшев относительно выигрыша в style — модели выгодно пожертвовать частью completeness. Возможные фиксы (к обсуждению, §9): поднять w_complete до ~1.0 или target_min_words до ~320.""")

code("""wc_b, wc_g = mf_b['word_count'], mf_g['word_count']
print(f'eval: слов в эссе baseline {wc_b.mean():.0f} (медиана {wc_b.median():.0f}) -> '
      f'GRPO {wc_g.mean():.0f} (медиана {wc_g.median():.0f}); '
      f'короче 300 слов: baseline {(wc_b < 300).sum()}/30, GRPO {(wc_g < 300).sum()}/30')

len_feats = [i for i, n in enumerate(FEATURE_NAMES) if n in ('word_count', 'sentence_count', 'unique_words_count')]
print(f"вклад признаков длины в сдвиг: {contrib[len_feats].sum():+.4f}  "
      f"({', '.join(FEATURE_NAMES[i] for i in len_feats)})")

wc_steps = [statistics.fmean(r['word_count'] for r in log2[i:i+8]) for i in range(0, len(log2), 8)]
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(range(1, len(wc_steps)+1), rolling(wc_steps), color=C2, lw=2)
ax.axhline(300, color=MUTED, lw=1, ls='--')
ax.text(3, 302, 'нижняя граница задания (300 слов)', color=MUTED, fontsize=8, va='bottom')
ax.set_xlabel('шаг GRPO'); ax.set_ylabel('слов в генерации')
ax.set_title('Прогон №2: длина генераций дрейфует вниз по ходу обучения', loc='left', fontweight='bold', fontsize=12)
fig.tight_layout()
plt.show()""")

md("""## 7. Пример: одна тема, до и после

Тема с наибольшим сдвигом, один и тот же сид. На уровне группы GRPO-эссе отличают больше существительных и уникальных слов, больше 1-го лица единственного числа, прошедшего времени и отрицаний (топ-движители из §5); при этом тексты не «о депрессии» — тема раскрывается нормально.""")

code("""best = prompts[int(np.argmax(diffs))]
seed = sorted({r['seed'] for r in base_eval if r['prompt'] == best})[0]
pair_b = next(r for r in base_eval if r['prompt'] == best and r['seed'] == seed)
pair_g = next(r for r in grpo_eval if r['prompt'] == best and r['seed'] == seed)
print(f'Тема: «{short_topic(best)}», сид {seed}')
for name, rec in [('BASELINE', pair_b), ('GRPO', pair_g)]:
    print(f"\\n{'='*30} {name}: raw {rec['raw_score']:+.3f}, {len(rec['completion'].split())} слов {'='*30}")
    print(rec['completion'].strip())""")

md("""## 8. Ограничения и оговорки

1. **Малый eval:** 10 hold-out тем × 3 сида. Парный p ≈ 0.009 устойчив к псевдорепликации, но расширение eval (больше тем/сидов) сделало бы оценку точнее.
2. **Домен классификатора:** обучен на 557 студенческих эссе «Я, другие, мир» (F1 ≈ 73%); на сгенерированных LLM текстах его скор — экстраполяция. «Депрессивность» здесь всюду означает *скор классификатора*, а не клиническую оценку.
3. **Классификатор — SVC, не Random Forest** (уточнено при извлечении из Docker; в исходном брифе — RF). `predict_proba` отсутствует, использован `decision_function`; он монотонен по «уверенности», но не откалиброван как вероятность.
4. **Версии sklearn:** на Kaggle 1.6.1, локально 1.9.0 → InconsistentVersionWarning при загрузке; парируется пересборкой модели из голых весов и обязательной сверкой с эталонными скорами (self-check PASS в обеих средах).
5. **Срезание длины** (§6) — известный побочный эффект, фикс намечен.
6. **Один обучающий прогон:** воспроизводимость по сидам обучения не проверялась (дорого на бесплатном GPU-тире).
7. **Одна модель:** результат показан для Qwen3-4B-Instruct-2507 + LoRA r=32; переносимость на другие модели не проверялась.""")

md("""## 9. Вопросы к научному руководителю

1. **Длина.** Поднять w_complete 0.5 → 1.0 или target_min_words 300 → 320? Стоит ли отдельный прогон только ради этого, или совместить со следующим экспериментом?
2. **Персона-микс.** Дизайн третьего прогона готов: добавить в промпты персону рассказчика (вторая ось вариации помимо тем), eval с двойным hold-out (невиданная тема × невиданная персона) — измеримая «устойчивость стиля». Интересно ли это как следующий шаг?
3. **Достаточность eval.** 10 тем × 3 сида и парный тест — достаточно для вывода, или расширить (все 80 тем, больше сидов, доп. метрики)?
4. **Интерпретация скора.** Корректно ли в тексте работы трактовать `decision_function` SVC как непрерывную меру «депрессивности стиля» без вероятностной калибровки?
5. **Форма результата.** Курсовая / статья / доклад — от этого зависит, что дорабатывать в первую очередь (эксперименты vs текст).""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
}
out = str(Path(__file__).with_name('grpo_report.ipynb'))
nbf.write(nb, out)
print('written', out, len(cells), 'cells')
