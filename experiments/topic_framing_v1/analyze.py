"""Validate a completed full run and produce the frozen topic-level analysis."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from run_eval import grid, require, sha256, write_json


HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def mean(values):
    return float(np.mean(values)) if values else None


def median(values):
    return float(np.median(values)) if values else None


def ci(values, indices, percentiles):
    values = np.asarray(values, dtype=float)
    draws = values[indices].mean(axis=1)
    lo, hi = np.percentile(draws, percentiles)
    return [float(lo), float(hi)]


def fmt(value):
    return 'NA' if value is None else f'{value:+.3f}'


def validate_run(root, protocol):
    manifest = json.loads((root / 'run_manifest.json').read_text())
    require(manifest['mode'] == 'full' and manifest['status'] == 'complete',
            'Analysis requires a completed full run')
    require(manifest['n_saved'] == protocol['expected_full_texts'], 'Unexpected saved count')
    require(manifest['protocol_sha256'] == sha256(root / 'protocol.json'),
            'Result protocol hash mismatch')
    require(json.loads((root / 'protocol.json').read_text()) == protocol,
            'Result protocol differs from frozen local protocol')
    for name, digest in manifest['artifact_sha256'].items():
        require(Path(name).name == name, 'Unsafe artifact name in manifest')
        require(sha256(root / name) == digest, f'Artifact SHA256 mismatch: {name}')

    expected = grid(protocol, 'full')
    expected_ids = [row['id'] for row in expected]
    planned = json.loads((root / 'planned_requests.json').read_text())
    generations = read_jsonl(root / 'generations.jsonl')
    scored = read_jsonl(root / 'scored.jsonl')
    require([row['id'] for row in planned] == expected_ids, 'Planned grid/order mismatch')
    require([row['id'] for row in generations] == expected_ids, 'Generation grid/order mismatch')
    require([row['id'] for row in scored] == expected_ids, 'Scored grid/order mismatch')
    require(len(set(expected_ids)) == len(expected_ids), 'Duplicate result IDs')
    feature_names = json.loads((root / 'feature_names.json').read_text())
    require(len(feature_names) == 73 and len(set(feature_names)) == 73,
            'Unexpected feature schema')
    geometry_rows = read_jsonl(root / 'geometry.jsonl')
    require(len({row['id'] for row in geometry_rows}) == len(geometry_rows),
            'Duplicate geometry IDs')
    for row in scored:
        if isinstance(row.get('feature_score'), (int, float)):
            require(len(row.get('features', [])) == 73, f"{row['id']}: missing features")
            raw = row.get('metrics', {}).get('raw_score')
            if raw is not None:
                require(abs(raw - row['feature_score']) <= 1e-6,
                        f"{row['id']}: reward/feature score mismatch")
            require(row['id'] in {item['id'] for item in geometry_rows},
                    f"{row['id']}: missing geometry")
    return manifest, scored, feature_names


def cell_summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['condition'], row['topic_group'], row['frame'])].append(row)
    out = []
    for key in sorted(groups):
        members = groups[key]
        scored = [row for row in members if isinstance(row.get('feature_score'), (int, float))]
        metrics = [row.get('metrics', {}) for row in members]
        scores = [row['feature_score'] for row in scored]
        out.append({
            'condition': key[0], 'topic_group': key[1], 'frame': key[2],
            'n_total': len(members), 'n_valid_score': len(scored),
            'n_missing_score': len(members) - len(scored),
            'score_mean': mean(scores), 'score_median': median(scores),
            'n_svc_positive': sum(score > 0 for score in scores),
            'n_floor': sum(item.get('floored') is True for item in metrics),
            'n_300_500_words': sum(300 <= item.get('word_count', -1) <= 500 for item in metrics),
            'word_count_mean': mean([item['word_count'] for item in metrics
                                     if isinstance(item.get('word_count'), (int, float))]),
            'antisem_rate_mean': mean([item['antisem_rate'] for item in metrics
                                       if isinstance(item.get('antisem_rate'), (int, float))]),
            'n_antisem_penalty': sum((item.get('antisem_penalty') or 0) > 0 for item in metrics),
            'n_sentiment_guard': sum((item.get('sentiment_guard') or 0) > 0 for item in metrics),
            'rep_ratio_mean': mean([item['rep_ratio'] for item in metrics
                                    if isinstance(item.get('rep_ratio'), (int, float))]),
            'n_length_stops': sum(row.get('finish_reason') == 'length' for row in members),
            'n_scoring_errors': sum('scoring_error' in row for row in members),
        })
    return out


def primary_analysis(rows, protocol):
    base = next(item['id'] for item in protocol['conditions'] if item['adapter'] is None)
    adapters = [item['id'] for item in protocol['conditions'] if item['adapter'] is not None]
    scores = {(row['condition'], row['topic_id'], row['frame'], row['seed']): row['feature_score']
              for row in rows if isinstance(row.get('feature_score'), (int, float))}
    topics = protocol['topics']
    frames = [item['id'] for item in protocol['frames']]
    seeds = protocol['seeds']
    complete, excluded = [], []
    for topic in topics:
        keys = [(condition, topic['topic_id'], frame, seed)
                for condition in [base] + adapters for frame in frames for seed in seeds]
        (complete if all(key in scores for key in keys) else excluded).append(topic)
    require(complete, 'No complete topics for primary contrasts')

    deltas = {}
    shifts = {}
    for topic in complete:
        topic_id = topic['topic_id']
        for condition in adapters:
            for frame in frames:
                deltas[(condition, topic_id, frame)] = mean([
                    scores[(condition, topic_id, frame, seed)] -
                    scores[(base, topic_id, frame, seed)] for seed in seeds])
        for condition in [base] + adapters:
            shifts[(condition, topic_id)] = mean([
                scores[(condition, topic_id, 'personal', seed)] -
                scores[(condition, topic_id, 'general', seed)] for seed in seeds])

    groups = {'old': [t for t in complete if t['group'] == 'old'],
              'new': [t for t in complete if t['group'] == 'new']}
    require(groups['old'] and groups['new'], 'Need complete topics in both groups')
    rng = np.random.default_rng(protocol['analysis']['bootstrap_seed'])
    repetitions = protocol['analysis']['bootstrap_repetitions']
    bootstrap = {group: rng.integers(0, len(items), size=(repetitions, len(items)))
                 for group, items in groups.items()}
    percentiles = protocol['analysis']['bootstrap_ci_percentiles']

    def values(mapping, condition, group, frame=None):
        items = groups[group]
        if frame is None:
            return [mapping[(condition, item['topic_id'])] for item in items]
        return [mapping[(condition, item['topic_id'], frame)] for item in items]

    records = []
    for condition in adapters:
        for group in ['old', 'new']:
            for frame in frames:
                vals = values(deltas, condition, group, frame)
                records.append({'contrast': 'grpo_minus_base', 'condition': condition,
                                'group': group, 'frame': frame, 'n_topics': len(vals),
                                'estimate': mean(vals),
                                'ci95': ci(vals, bootstrap[group], percentiles),
                                'n_positive_topics': sum(value > 0 for value in vals)})
            interaction = [deltas[(condition, item['topic_id'], 'personal')] -
                           deltas[(condition, item['topic_id'], 'general')]
                           for item in groups[group]]
            records.append({'contrast': 'framing_interaction', 'condition': condition,
                            'group': group, 'frame': 'personal_minus_general',
                            'n_topics': len(interaction), 'estimate': mean(interaction),
                            'ci95': ci(interaction, bootstrap[group], percentiles),
                            'n_positive_topics': sum(value > 0 for value in interaction)})

        old_i = [deltas[(condition, item['topic_id'], 'personal')] -
                 deltas[(condition, item['topic_id'], 'general')] for item in groups['old']]
        new_i = [deltas[(condition, item['topic_id'], 'personal')] -
                 deltas[(condition, item['topic_id'], 'general')] for item in groups['new']]
        old_draw = np.asarray(old_i)[bootstrap['old']].mean(axis=1)
        new_draw = np.asarray(new_i)[bootstrap['new']].mean(axis=1)
        combined_draw = (old_draw + new_draw) / 2
        records.append({'contrast': 'framing_interaction', 'condition': condition,
                        'group': 'all_equal_group_weight', 'frame': 'personal_minus_general',
                        'n_topics': len(old_i) + len(new_i),
                        'estimate': (mean(old_i) + mean(new_i)) / 2,
                        'ci95': [float(x) for x in np.percentile(combined_draw, percentiles)],
                        'n_positive_topics': sum(x > 0 for x in old_i + new_i)})

        for frame in frames:
            old_vals = values(deltas, condition, 'old', frame)
            new_vals = values(deltas, condition, 'new', frame)
            draws = (np.asarray(new_vals)[bootstrap['new']].mean(axis=1) -
                     np.asarray(old_vals)[bootstrap['old']].mean(axis=1))
            records.append({'contrast': 'new_minus_old_grpo_effect', 'condition': condition,
                            'group': 'new_minus_old', 'frame': frame,
                            'n_topics': len(old_vals) + len(new_vals),
                            'estimate': mean(new_vals) - mean(old_vals),
                            'ci95': [float(x) for x in np.percentile(draws, percentiles)],
                            'n_positive_topics': None})

    for condition in [base] + adapters:
        for group in ['old', 'new']:
            vals = values(shifts, condition, group)
            records.append({'contrast': 'personal_minus_general_score',
                            'condition': condition, 'group': group,
                            'frame': 'personal_minus_general', 'n_topics': len(vals),
                            'estimate': mean(vals),
                            'ci95': ci(vals, bootstrap[group], percentiles),
                            'n_positive_topics': sum(value > 0 for value in vals)})

    topic_rows = []
    for topic in complete:
        for condition in adapters:
            topic_rows.append({
                'topic_id': topic['topic_id'], 'topic_group': topic['group'],
                'topic': topic['topic'], 'condition': condition,
                'delta_personal': deltas[(condition, topic['topic_id'], 'personal')],
                'delta_general': deltas[(condition, topic['topic_id'], 'general')],
                'framing_interaction': (deltas[(condition, topic['topic_id'], 'personal')] -
                                        deltas[(condition, topic['topic_id'], 'general')]),
            })
    return records, topic_rows, [topic['topic_id'] for topic in excluded]


def feature_summaries(rows, feature_names, protocol):
    selected = protocol['analysis']['features_of_interest']
    indices = {name: feature_names.index(name) for name in selected}
    groups = defaultdict(list)
    for row in rows:
        if len(row.get('features', [])) == len(feature_names):
            groups[(row['condition'], row['topic_group'], row['frame'])].append(row)
    out = []
    for key in sorted(groups):
        record = {'condition': key[0], 'topic_group': key[1], 'frame': key[2],
                  'n_texts': len(groups[key])}
        for name, index in indices.items():
            record[name] = mean([row['features'][index] for row in groups[key]])
        out.append(record)
    return out


def geometry_summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        if 'geometry' in row:
            groups[(row['condition'], row['topic_group'], row['frame'])].append(row['geometry'])
    out = []
    for key in sorted(groups):
        record = {'condition': key[0], 'topic_group': key[1], 'frame': key[2],
                  'n_texts': len(groups[key])}
        for name in ['z_rms', 'pca_residual_rms', 'nearest_support_kernel']:
            values = [item[name] for item in groups[key]]
            record[name + '_mean'] = mean(values)
            record[name + '_median'] = median(values)
        out.append(record)
    return out


def export_blind_review(rows, protocol, output):
    wanted_topics = set(protocol['analysis']['blind_review_topic_indices'])
    wanted_seed = protocol['analysis']['blind_review_generation_seed']
    selected = [row for row in rows
                if row['topic_index'] in wanted_topics and row['seed'] == wanted_seed]
    require(len(selected) == 60, f'Expected 60 blind-review texts, got {len(selected)}')
    require(len({row['id'] for row in selected}) == 60, 'Duplicate blind-review rows')
    random.Random(protocol['analysis']['blind_review_shuffle_seed']).shuffle(selected)
    review_path = output / 'blind_review.csv'
    key_path = output / 'blind_review_key.csv'
    with review_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'review_id', 'topic_group', 'topic_id', 'topic', 'frame', 'frame_instruction',
            'completion', 'topic_fit_0_2', 'frame_fit_0_2', 'coherence_0_2',
            'negative_semantics_0_2', 'diagnosis_symptoms_treatment_yes_no',
            'quote', 'comment'])
        writer.writeheader()
        for index, row in enumerate(selected, 1):
            writer.writerow({
                'review_id': f'R{index:03d}', 'topic_group': row['topic_group'],
                'topic_id': row['topic_id'], 'topic': row['topic'], 'frame': row['frame'],
                'frame_instruction': row['frame_instruction'], 'completion': row['completion'],
                'topic_fit_0_2': '', 'frame_fit_0_2': '', 'coherence_0_2': '',
                'negative_semantics_0_2': '', 'diagnosis_symptoms_treatment_yes_no': '',
                'quote': '', 'comment': ''})
    with key_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'review_id', 'id', 'condition', 'adapter', 'seed'])
        writer.writeheader()
        for index, row in enumerate(selected, 1):
            writer.writerow({'review_id': f'R{index:03d}', 'id': row['id'],
                             'condition': row['condition'], 'adapter': row['adapter'],
                             'seed': row['seed']})
    return {'n_texts': len(selected), 'review_sha256': sha256(review_path),
            'key_sha256': sha256(key_path)}


def markdown_report(manifest, cells, contrasts, excluded, blind):
    lines = [
        '# Topic framing v1: автоматический анализ', '',
        'Это описательный отчёт по зафиксированному протоколу. '
        '`decision_function` не является вероятностью или клинической оценкой.', '',
        f"Run status: `{manifest['status']}`; сохранено {manifest['n_saved']} текстов; "
        f"scoring errors: {manifest.get('n_scoring_errors', 0)}.", '',
        '## Scores по ячейкам', '',
        '| Модель | Группа | Форма | n | Mean score | SVC > 0 | 300-500 слов |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for row in cells:
        lines.append(f"| {row['condition']} | {row['topic_group']} | {row['frame']} | "
                     f"{row['n_valid_score']}/{row['n_total']} | {fmt(row['score_mean'])} | "
                     f"{row['n_svc_positive']}/{row['n_valid_score']} | "
                     f"{row['n_300_500_words']}/{row['n_total']} |")
    lines += ['', '## Заранее заданные тематические контрасты', '',
              '| Контраст | Модель | Группа | Форма | Оценка | 95% bootstrap CI |',
              '|---|---|---|---|---:|---:|']
    for row in contrasts:
        lines.append(f"| {row['contrast']} | {row['condition']} | {row['group']} | "
                     f"{row['frame']} | {fmt(row['estimate'])} | "
                     f"[{fmt(row['ci95'][0])}, {fmt(row['ci95'][1])}] |")
    lines += ['', f"Исключённые из полных пар темы: {', '.join(excluded) if excluded else 'нет'}.", '',
              'Интервалы описательные и не скорректированы за множественные срезы. '
              'Три generation seed усреднены внутри темы.', '',
              '## Ручная оценка', '',
              f"Экспортировано {blind['n_texts']} текстов в `blind_review.csv`; соответствие "
              'моделям находится отдельно в `blind_review_key.csv`. Не открывать key до заполнения формы.', '',
              'Geometry diagnostics, выбранные исходные признаки и все тематические значения '
              'находятся в JSON. Они не являются самостоятельным OOD-тестом или доказательством '
              'валидности классификатора.', '']
    return '\n'.join(lines)


def analyze(root, output, local_protocol):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output != root and not output.exists(), 'Use a new analysis output directory')
    protocol = json.loads(Path(local_protocol).read_text())
    source_protocol = HERE.parent.parent / protocol['source_protocol']
    require(sha256(source_protocol) == protocol['source_protocol_sha256'],
            'Frozen source protocol changed')
    manifest, rows, feature_names = validate_run(root, protocol)
    output.mkdir(parents=True)
    cells = cell_summaries(rows)
    contrasts, topic_rows, excluded = primary_analysis(rows, protocol)
    blind = export_blind_review(rows, protocol, output)
    result = {
        'experiment_id': protocol['experiment_id'], 'source_run': str(root),
        'run_manifest_sha256': sha256(root / 'run_manifest.json'),
        'analysis_script_sha256': sha256(__file__),
        'analysis_config': protocol['analysis'], 'excluded_incomplete_topics': excluded,
        'cell_summaries': cells, 'primary_contrasts': contrasts,
        'topic_contrasts': topic_rows,
        'selected_feature_summaries': feature_summaries(rows, feature_names, protocol),
        'geometry_summaries': geometry_summaries(rows), 'blind_review': blind,
    }
    write_json(output / 'analysis.json', result)
    (output / 'ANALYSIS.md').write_text(
        markdown_report(manifest, cells, contrasts, excluded, blind), encoding='utf-8')
    write_json(output / 'analysis_manifest.json', {
        'source_run_manifest_sha256': result['run_manifest_sha256'],
        'source_protocol_sha256': sha256(local_protocol),
        'analysis_script_sha256': result['analysis_script_sha256'],
        'artifact_sha256': {path.name: sha256(path) for path in output.iterdir()
                            if path.is_file() and path.name != 'analysis_manifest.json'},
    })
    print(f'Validated {len(rows)} rows and wrote analysis to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol.json')
    args = parser.parse_args()
    analyze(args.input, args.output, args.protocol)
