"""Validate a completed personal-prompt run and produce the frozen analysis."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from prepare_kaggle import HERE, SOURCE, load_protocols, validate
from run_eval import digest, grid


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def validate_run(root, protocol):
    manifest = json.loads((root / 'run_manifest.json').read_text())
    require(manifest['mode'] == 'full' and manifest['status'] == 'complete',
            'Analysis requires completed full run')
    require(manifest['n_saved'] == 300 and manifest.get('n_scoring_errors') == 0,
            'Incomplete or erroneous run')
    require(manifest['design_protocol_sha256'] == digest(HERE / 'protocol.json'),
            'Design hash mismatch')
    require(manifest['source_protocol_sha256'] == digest(SOURCE), 'Source hash mismatch')
    require(manifest['personal_runner_sha256'] == digest(HERE / 'run_eval.py'),
            'Runner hash mismatch')
    require(json.loads((root / 'protocol.json').read_text()) == protocol,
            'Resolved protocol mismatch')
    for name, expected in manifest['artifact_sha256'].items():
        require(Path(name).name == name and (root / name).is_file(), 'Unsafe artifact')
        require(digest(root / name) == expected, f'Artifact hash mismatch: {name}')
    expected = grid(protocol, 'full')
    planned = json.loads((root / 'planned_requests.json').read_text())
    generations, rows = read_jsonl(root / 'generations.jsonl'), read_jsonl(root / 'scored.jsonl')
    ids = [job['id'] for job in expected]
    require([row['id'] for row in planned] == ids, 'Planned grid mismatch')
    require([row['id'] for row in generations] == ids, 'Generation grid mismatch')
    require([row['id'] for row in rows] == ids, 'Scored grid mismatch')
    features = json.loads((root / 'feature_names.json').read_text())
    require(len(features) == 73 and len(set(features)) == 73, 'Feature schema mismatch')
    for expected_row, row in zip(expected, rows):
        for key in expected_row:
            require(row.get(key) == expected_row[key], f'Frozen input mismatch: {row["id"]}')
        require(len(row.get('features', [])) == 73 and
                all(math.isfinite(x) for x in row['features']), 'Invalid features')
        require(isinstance(row.get('feature_score'), (int, float)) and
                math.isfinite(row['feature_score']), 'Invalid feature score')
        require(abs(row['feature_score'] - row['metrics']['raw_score']) < 1e-9,
                'feature_score/raw_score mismatch')
        require(set(row.get('geometry', {})) ==
                {'z_rms', 'pca_residual_rms', 'nearest_support_kernel'}, 'Invalid geometry')
    return manifest, rows, features


def mean(values):
    return float(np.mean(values)) if len(values) else None


def cells(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['condition'], row['topic_group'])].append(row)
    result = []
    for (condition, group), rr in sorted(groups.items()):
        metrics = [row['metrics'] for row in rr]
        result.append({
            'condition': condition, 'group': group, 'n': len(rr),
            'score_mean': mean([r['feature_score'] for r in rr]),
            'score_median': float(np.median([r['feature_score'] for r in rr])),
            'n_svc_positive': sum(r['feature_score'] > 0 for r in rr),
            'n_antisem_nonzero': sum(m['antisem_rate'] > 0 for m in metrics),
            'antisem_rate_mean': mean([m['antisem_rate'] for m in metrics]),
            'n_antisem_penalty': sum(m['antisem_penalty'] > 0 for m in metrics),
            'antisem_penalty_mean': mean([m['antisem_penalty'] for m in metrics]),
            'n_sentiment_guard': sum(m['sentiment_guard'] > 0 for m in metrics),
            'n_floor': sum(bool(m['floored']) for m in metrics),
            'word_count_mean': mean([m['word_count'] for m in metrics]),
            'n_300_500_words': sum(300 <= m['word_count'] <= 500 for m in metrics),
            'rep_ratio_mean': mean([m['rep_ratio'] for m in metrics]),
            'finish_reasons': dict(sorted(defaultdict(int, {
                reason: sum(r.get('finish_reason') == reason for r in rr)
                for reason in {r.get('finish_reason') for r in rr}}).items(),
                                      key=lambda item: str(item[0]))),
        })
    return result


def bootstrap_indices(protocol):
    rng = np.random.default_rng(protocol['analysis']['bootstrap_seed'])
    n = protocol['analysis']['bootstrap_repetitions']
    return rng.integers(0, 10, size=(n, 10)), rng.integers(0, 10, size=(n, 10))


def contrast_analysis(rows, protocol):
    values = defaultdict(list)
    for row in rows:
        values[(row['condition'], row['topic_index'])].append(row['feature_score'])
    topic_means = {key: mean(scores) for key, scores in values.items()}
    topics = {topic['topic_index']: topic for topic in protocol['topics']}
    old_idx, new_idx = ([t['topic_index'] for t in protocol['topics'] if t['group'] == group]
                        for group in ('old', 'new'))
    old_draw, new_draw = bootstrap_indices(protocol)
    comparisons = [('primary', *pair) for pair in protocol['analysis']['primary_contrasts']]
    comparisons += [('secondary', *pair) for pair in protocol['analysis']['secondary_contrasts']]
    summaries, topic_rows = [], []
    lo, hi = protocol['analysis']['bootstrap_ci_percentiles']
    for kind, left, right in comparisons:
        by_group = {}
        for group, indices in [('old', old_idx), ('new', new_idx)]:
            vals = np.asarray([topic_means[left, i] - topic_means[right, i] for i in indices])
            by_group[group] = vals
            for index, value in zip(indices, vals):
                topic_rows.append({'kind': kind, 'left': left, 'right': right,
                                   **topics[index], 'delta': float(value)})
        draws_old = by_group['old'][old_draw].mean(axis=1)
        draws_new = by_group['new'][new_draw].mean(axis=1)
        for group, vals, draws in [('old', by_group['old'], draws_old),
                                   ('new', by_group['new'], draws_new),
                                   ('all_equal_group_weight',
                                    np.concatenate([by_group['old'], by_group['new']]),
                                    (draws_old + draws_new) / 2)]:
            summaries.append({
                'kind': kind, 'left': left, 'right': right, 'group': group,
                'n_topics': len(vals), 'estimate': mean(vals),
                'ci95': [float(x) for x in np.percentile(draws, [lo, hi])],
                'n_positive_topics': int(np.sum(vals > 0)),
            })
    return summaries, topic_rows


def paired_transitions(rows, protocol):
    by_id = {(r['condition'], r['topic_index'], r['seed']): r for r in rows}
    result = []
    for kind, left, right in ([('primary', *p) for p in protocol['analysis']['primary_contrasts']] +
                              [('secondary', *p) for p in protocol['analysis']['secondary_contrasts']]):
        for group in ('old', 'new', 'all'):
            topics = protocol['topics'] if group == 'all' else [t for t in protocol['topics']
                                                                if t['group'] == group]
            pairs = [(by_id[left, t['topic_index'], seed], by_id[right, t['topic_index'], seed])
                     for t in topics for seed in protocol['seeds']]
            result.append({'kind': kind, 'left': left, 'right': right, 'group': group,
                           'n_pairs': len(pairs),
                           'negative_to_positive': sum(a['feature_score'] > 0 >= b['feature_score']
                                                       for a, b in pairs),
                           'positive_to_negative': sum(a['feature_score'] <= 0 < b['feature_score']
                                                       for a, b in pairs)})
    return result


def diagnostics(rows, feature_names, protocol):
    findex = {name: feature_names.index(name) for name in protocol['analysis']['features_of_interest']}
    result = []
    for condition in [c['id'] for c in protocol['conditions']]:
        for group in ('old', 'new'):
            rr = [r for r in rows if r['condition'] == condition and r['topic_group'] == group]
            result.append({
                'condition': condition, 'group': group, 'n': len(rr),
                'features': {name: mean([r['features'][index] for r in rr])
                             for name, index in findex.items()},
                'geometry': {name: mean([r['geometry'][name] for r in rr]) for name in
                             ('z_rms', 'pca_residual_rms', 'nearest_support_kernel')},
            })
    return result


def export_review(rows, protocol, output):
    chosen = [row for row in rows
              if row['topic_index'] in protocol['analysis']['optional_review_topic_indices']
              and row['seed'] == protocol['analysis']['optional_review_generation_seed']]
    require(len(chosen) == 50, 'Unexpected review sample')
    rng = random.Random(protocol['analysis']['optional_review_shuffle_seed'])
    rng.shuffle(chosen)
    review_fields = ['review_id', 'topic', 'text', 'topic_fit_0_2',
                     'personal_form_0_2', 'coherence_grammar_0_2',
                     'negative_semantics_0_2', 'diagnosis_symptoms_treatment_yes_no',
                     'quote_if_yes', 'comment']
    with (output / 'blind_review.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields); writer.writeheader()
        for i, row in enumerate(chosen, 1):
            writer.writerow({'review_id': f'R{i:03d}', 'topic': row['topic'],
                             'text': row['completion']})
    with (output / 'blind_review_key.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['review_id', 'id', 'condition', 'topic_id', 'seed', 'feature_score']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for i, row in enumerate(chosen, 1):
            writer.writerow({'review_id': f'R{i:03d}', **{key: row[key] for key in fields[1:]}})
    return {'n_texts': len(chosen), 'shuffle_seed': protocol['analysis']['optional_review_shuffle_seed']}


def markdown(manifest, cell_rows, contrasts):
    lines = ['# Personal prompt v1: автоматический анализ', '',
             'Decision_function — не вероятность и не клиническая оценка.', '',
             f"Run `{manifest['status']}`: {manifest['n_saved']}/300 текстов, "
             f"scoring errors: {manifest.get('n_scoring_errors', 0)}.", '',
             '## Ячейки', '',
             '| Условие | Группа | Mean | SVC > 0 | Antisem > 0 | Penalty > 0 | 300-500 слов |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in cell_rows:
        lines.append(f"| {r['condition']} | {r['group']} | {r['score_mean']:+.3f} | "
                     f"{r['n_svc_positive']}/{r['n']} | {r['n_antisem_nonzero']}/{r['n']} | "
                     f"{r['n_antisem_penalty']}/{r['n']} | {r['n_300_500_words']}/{r['n']} |")
    lines += ['', '## Тематические контрасты', '',
              '| Тип | Сравнение | Группа | Delta | 95% bootstrap CI | Темы > 0 |',
              '|---|---|---|---:|---:|---:|']
    for r in contrasts:
        lines.append(f"| {r['kind']} | {r['left']} - {r['right']} | {r['group']} | "
                     f"{r['estimate']:+.3f} | [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] | "
                     f"{r['n_positive_topics']}/{r['n_topics']} |")
    lines += ['', 'Bootstrap описательный; два training seed не объединены. Все тексты входят ',
              'в первичный анализ независимо от длины, лексики и reward floor.', '',
              'Опциональная читательская форма: `blind_review.csv`; ключ открывать после оценки.', '']
    return '\n'.join(lines)


def analyze(root, output):
    design, protocol = load_protocols(); validate(design, protocol)
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists() and output != root, 'Use a new analysis directory')
    manifest, rows, feature_names = validate_run(root, protocol)
    output.mkdir(parents=True)
    cell_rows = cells(rows)
    contrasts, topic_rows = contrast_analysis(rows, protocol)
    result = {
        'experiment_id': protocol['experiment_id'],
        'source_run_manifest_sha256': digest(root / 'run_manifest.json'),
        'analysis_script_sha256': digest(__file__),
        'cell_summaries': cell_rows, 'contrasts': contrasts,
        'topic_contrasts': topic_rows,
        'paired_class_transitions': paired_transitions(rows, protocol),
        'diagnostics': diagnostics(rows, feature_names, protocol),
        'review': export_review(rows, protocol, output),
    }
    write_json(output / 'analysis.json', result)
    (output / 'ANALYSIS.md').write_text(markdown(manifest, cell_rows, contrasts), encoding='utf-8')
    write_json(output / 'analysis_manifest.json', {
        'source_run_manifest_sha256': result['source_run_manifest_sha256'],
        'design_protocol_sha256': digest(HERE / 'protocol.json'),
        'analysis_script_sha256': result['analysis_script_sha256'],
        'artifact_sha256': {path.name: digest(path) for path in output.iterdir()
                            if path.name != 'analysis_manifest.json'},
    })
    print(f'Validated {len(rows)} rows and wrote analysis to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); analyze(args.input, args.output)
