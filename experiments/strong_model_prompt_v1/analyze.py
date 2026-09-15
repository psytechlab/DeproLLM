"""Frozen topic-level analysis for GLM-5.3-Flash and Qwen references."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random

import numpy as np

from prepare_zcode import PROTOCOL, digest, grid, load_protocols, require, write_json


HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def mean(values):
    return float(np.mean(values)) if values else None


def validate_inputs(scoring_root, qwen_scored, qwen_manifest, design, personal, runner):
    scoring_root = Path(scoring_root)
    manifest = json.loads((scoring_root / 'scoring_manifest.json').read_text(encoding='utf-8'))
    require(manifest['status'] == 'scoring_complete' and manifest['n_scored'] == 180,
            'Analysis requires 180 scored GLM responses')
    require(manifest['protocol_sha256'] == digest(PROTOCOL), 'Scoring protocol mismatch')
    for name, expected_hash in manifest['artifact_sha256'].items():
        require(digest(scoring_root / name) == expected_hash, f'Scoring artifact changed: {name}')
    require(digest(qwen_scored) == design['scoring_reference']['qwen_scored_sha256'],
            'Qwen scored reference hash mismatch')
    require(digest(qwen_manifest) == design['scoring_reference']['qwen_run_manifest_sha256'],
            'Qwen run manifest hash mismatch')
    glm = read_jsonl(scoring_root / 'scored.jsonl')
    expected = grid('full', design, personal, runner)
    require([row['id'] for row in glm] == [row['id'] for row in expected],
            'GLM scored grid mismatch')
    qwen = read_jsonl(qwen_scored)
    needed = {'base_neutral', 'base_role', 'base_constrained',
              'grpo_3407_neutral', 'grpo_4407_neutral'}
    qwen = [row for row in qwen if row['condition'] in needed]
    require(len(qwen) == 300, 'Qwen reference grid must contain 300 rows')
    for row in glm + qwen:
        require(isinstance(row.get('feature_score'), (int, float)) and
                math.isfinite(row['feature_score']), 'Invalid feature score')
        require(len(row.get('features', [])) == 73 and
                all(math.isfinite(x) for x in row['features']), 'Invalid feature vector')
    return glm, qwen


def cell_summaries(rows, source):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(source, row['condition'], row['topic_group'])].append(row)
    result = []
    for (row_source, condition, group), rr in sorted(grouped.items()):
        metrics = [row['metrics'] for row in rr]
        result.append({
            'source': row_source, 'condition': condition, 'group': group, 'n': len(rr),
            'score_mean': mean([row['feature_score'] for row in rr]),
            'score_median': float(np.median([row['feature_score'] for row in rr])),
            'n_svc_positive': sum(row['feature_score'] > 0 for row in rr),
            'n_antisem_nonzero': sum(item['antisem_rate'] > 0 for item in metrics),
            'n_antisem_penalty': sum(item['antisem_penalty'] > 0 for item in metrics),
            'antisem_rate_mean': mean([item['antisem_rate'] for item in metrics]),
            'word_count_mean': mean([item['word_count'] for item in metrics]),
            'n_300_500_words': sum(300 <= item['word_count'] <= 500 for item in metrics),
            'n_floor': sum(bool(item['floored']) for item in metrics),
        })
    return result


def topic_values(glm, qwen):
    values = defaultdict(list)
    for row in glm + qwen:
        values[(row['condition'], row['topic_index'])].append(row['feature_score'])
    return {key: mean(scores) for key, scores in values.items()}


def contrasts(glm, qwen, protocol):
    values = topic_values(glm, qwen)
    topics = {topic['topic_index']: topic for topic in protocol['topics']}
    old = [index for index, topic in topics.items() if topic['group'] == 'old']
    new = [index for index, topic in topics.items() if topic['group'] == 'new']
    rng = np.random.default_rng(protocol['analysis']['bootstrap_seed'])
    reps = protocol['analysis']['bootstrap_repetitions']
    old_draw = rng.integers(0, len(old), size=(reps, len(old)))
    new_draw = rng.integers(0, len(new), size=(reps, len(new)))
    comparisons = [('primary_within_model', *pair)
                   for pair in protocol['analysis']['primary_within_model']]
    comparisons += [('secondary_within_model', *pair)
                    for pair in protocol['analysis']['secondary_within_model']]
    comparisons += [('external_descriptive', *pair)
                    for pair in protocol['analysis']['external_descriptive']]
    summaries, topic_rows = [], []
    lo, hi = protocol['analysis']['bootstrap_ci_percentiles']
    for kind, left, right in comparisons:
        by_group = {}
        for group, indices in [('old', old), ('new', new)]:
            data = np.asarray([values[left, index] - values[right, index]
                               for index in indices])
            by_group[group] = data
            for index, value in zip(indices, data):
                topic_rows.append({'kind': kind, 'left': left, 'right': right,
                                   **topics[index], 'delta': float(value)})
        draws_old = by_group['old'][old_draw].mean(axis=1)
        draws_new = by_group['new'][new_draw].mean(axis=1)
        groups = [('old', by_group['old'], draws_old),
                  ('new', by_group['new'], draws_new),
                  ('all_equal_group_weight',
                   np.concatenate([by_group['old'], by_group['new']]),
                   (draws_old + draws_new) / 2)]
        for group, data, draws in groups:
            summaries.append({
                'kind': kind, 'left': left, 'right': right, 'group': group,
                'n_topics': len(data), 'estimate': mean(data.tolist()),
                'ci95': [float(value) for value in np.percentile(draws, [lo, hi])],
                'n_positive_topics': int(np.sum(data > 0)),
            })
    return summaries, topic_rows


def diagnostics(glm, feature_names, protocol):
    indices = {name: feature_names.index(name)
               for name in protocol['analysis']['features_of_interest']}
    result = []
    for condition in [item['id'] for item in protocol['conditions']]:
        for group in ('old', 'new'):
            rows = [row for row in glm
                    if row['condition'] == condition and row['topic_group'] == group]
            result.append({
                'condition': condition, 'group': group, 'n': len(rows),
                'features': {name: mean([row['features'][index] for row in rows])
                             for name, index in indices.items()},
                'geometry': {name: mean([row['geometry'][name] for row in rows])
                             for name in ('z_rms', 'pca_residual_rms',
                                          'nearest_support_kernel')},
            })
    return result


def export_review(glm, qwen, protocol, output):
    chosen_topics = set(protocol['analysis']['optional_review_topic_indices'])
    selected = [row for row in glm if row['condition'] == 'glm53_flash_constrained'
                and row['replicate'] == protocol['analysis']['optional_review_glm_replicate']
                and row['topic_index'] in chosen_topics]
    selected += [row for row in qwen
                 if row['condition'] in {'base_constrained', 'grpo_3407_neutral',
                                         'grpo_4407_neutral'}
                 and row['seed'] == protocol['analysis']['optional_review_qwen_seed']
                 and row['topic_index'] in chosen_topics]
    require(len(selected) == 40, 'Unexpected blind review sample')
    rng = random.Random(protocol['analysis']['optional_review_shuffle_seed'])
    rng.shuffle(selected)
    public_fields = ['review_id', 'topic', 'text', 'topic_fit_0_2',
                     'personal_form_0_2', 'coherence_grammar_0_2',
                     'negative_semantics_0_2',
                     'diagnosis_symptoms_treatment_yes_no', 'quote_if_yes', 'comment']
    with (output / 'blind_review.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=public_fields); writer.writeheader()
        for index, row in enumerate(selected, 1):
            writer.writerow({'review_id': f'S{index:03d}', 'topic': row['topic'],
                             'text': row['completion']})
    with (output / 'blind_review_key.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['review_id', 'condition', 'topic_id', 'feature_score']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for index, row in enumerate(selected, 1):
            writer.writerow({'review_id': f'S{index:03d}',
                             **{key: row[key] for key in fields[1:]}})
    return {'n_texts': 40, 'shuffle_seed': protocol['analysis']['optional_review_shuffle_seed']}


def markdown(cell_rows, contrast_rows):
    lines = ['# Strong-model prompt v1: автоматический анализ', '',
             '`decision_function` — не вероятность и не клиническая оценка.', '',
             '## Условия', '',
             '| Источник | Условие | Группа | Mean | SVC > 0 | Penalty > 0 | 300-500 слов |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in cell_rows:
        lines.append(f"| {row['source']} | {row['condition']} | {row['group']} | "
                     f"{row['score_mean']:+.3f} | {row['n_svc_positive']}/{row['n']} | "
                     f"{row['n_antisem_penalty']}/{row['n']} | "
                     f"{row['n_300_500_words']}/{row['n']} |")
    lines += ['', '## Тематические контрасты', '',
              '| Тип | Сравнение | Группа | Delta | 95% bootstrap CI | Темы > 0 |',
              '|---|---|---|---:|---:|---:|']
    for row in contrast_rows:
        lines.append(f"| {row['kind']} | {row['left']} - {row['right']} | "
                     f"{row['group']} | {row['estimate']:+.3f} | "
                     f"[{row['ci95'][0]:+.3f}; {row['ci95'][1]:+.3f}] | "
                     f"{row['n_positive_topics']}/{row['n_topics']} |")
    lines += ['', 'Межмодельные контрасты описательные: они смешивают семейство модели, ',
              'decoding и ZCode wrapper. Повторы GLM не являются контролируемыми seeds.', '',
              'Опциональная читательская форма: `blind_review.csv`; ключ открывать после оценки.', '']
    return '\n'.join(lines)


def analyze(scoring_root, qwen_scored, qwen_manifest, output):
    output = Path(output).resolve()
    require(not output.exists(), f'Output already exists: {output}')
    design, personal, runner = load_protocols()
    glm, qwen = validate_inputs(scoring_root, qwen_scored, qwen_manifest,
                                design, personal, runner)
    feature_names = json.loads((Path(scoring_root) / 'feature_names.json').read_text())
    require(len(feature_names) == 73 and len(set(feature_names)) == 73,
            'Invalid feature names')
    cells = cell_summaries(glm, 'glm53_flash_zcode') + cell_summaries(qwen, 'qwen')
    contrast_rows, topic_rows = contrasts(glm, qwen, {**personal, **design})
    output.mkdir(parents=True)
    result = {
        'experiment_id': design['experiment_id'],
        'scoring_manifest_sha256': digest(Path(scoring_root) / 'scoring_manifest.json'),
        'qwen_scored_sha256': digest(qwen_scored),
        'analysis_script_sha256': digest(__file__),
        'cell_summaries': cells, 'contrasts': contrast_rows,
        'topic_contrasts': topic_rows,
        'glm_diagnostics': diagnostics(glm, feature_names, design),
        'review': export_review(glm, qwen, design, output),
    }
    write_json(output / 'analysis.json', result)
    (output / 'ANALYSIS.md').write_text(markdown(cells, contrast_rows), encoding='utf-8')
    write_json(output / 'analysis_manifest.json', {
        'protocol_sha256': digest(PROTOCOL), 'analysis_script_sha256': digest(__file__),
        'source_scoring_manifest_sha256': result['scoring_manifest_sha256'],
        'source_qwen_scored_sha256': result['qwen_scored_sha256'],
        'artifact_sha256': {path.name: digest(path) for path in output.iterdir()
                            if path.name != 'analysis_manifest.json'},
    })
    print(f'Validated and analyzed 180 GLM plus 300 Qwen rows into {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--qwen-scored', required=True, type=Path)
    parser.add_argument('--qwen-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.input, args.qwen_scored, args.qwen_manifest, args.output)
