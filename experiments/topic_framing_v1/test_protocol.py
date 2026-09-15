import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from analyze import analyze, export_blind_review, primary_analysis
from prepare_kaggle import REPO, build, validate
from run_eval import grid, safe_members, sha256, summarize, write_json


HERE = Path(__file__).resolve().parent


class TopicFramingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((HERE / 'protocol.json').read_text())

    def test_frozen_source_and_grid(self):
        validate(self.protocol)
        self.assertEqual(sha256(REPO / self.protocol['source_protocol']),
                         self.protocol['source_protocol_sha256'])
        full, pilot = grid(self.protocol, 'full'), grid(self.protocol, 'pilot')
        self.assertEqual(len(full), 360)
        self.assertEqual(len(pilot), 12)
        self.assertEqual(len({row['id'] for row in full}), 360)
        self.assertEqual(len({row['id'] for row in pilot}), 12)
        self.assertFalse({row['topic_id'] for row in full} &
                         {row['topic_id'] for row in pilot})

    def test_pairing_prompt_and_seed(self):
        rows = grid(self.protocol, 'full')
        blocks = {}
        for row in rows:
            key = row['topic_id'], row['frame'], row['seed']
            blocks.setdefault(key, []).append(row)
        self.assertEqual(len(blocks), 120)
        for members in blocks.values():
            self.assertEqual(len(members), 3)
            self.assertEqual(len({row['prompt'] for row in members}), 1)
            self.assertEqual(len({row['request_seed'] for row in members}), 1)
        for condition in self.protocol['conditions']:
            selected = [row for row in rows if row['condition'] == condition['id']]
            expected = [(topic['topic_id'], frame['id'], seed)
                        for topic in self.protocol['topics']
                        for frame in self.protocol['frames']
                        for seed in self.protocol['seeds']]
            self.assertEqual([(row['topic_id'], row['frame'], row['seed'])
                              for row in selected], expected)

    def test_zip_rejects_traversal_duplicate_and_symlink(self):
        cases = [('../oops', False), ('/absolute', False), ('link', True)]
        for name, symlink in cases:
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                info = zipfile.ZipInfo(name)
                if symlink:
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, b'target')
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive, self.assertRaises(ValueError):
                safe_members(archive)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('same', b'a')
            archive.writestr('same', b'b')
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive, self.assertRaises(ValueError):
            safe_members(archive)

    def test_summary_uses_feature_score_and_preserves_missing(self):
        rows = [
            {'condition': 'base', 'topic_group': 'old', 'frame': 'personal',
             'finish_reason': 'stop', 'feature_score': -0.2},
            {'condition': 'base', 'topic_group': 'old', 'frame': 'personal',
             'finish_reason': 'length', 'scoring_error': 'failed'},
            {'condition': 'base', 'topic_group': 'old', 'frame': 'personal',
             'finish_reason': 'stop', 'feature_score': 0.4},
        ]
        item = summarize(rows)[0]
        self.assertEqual(item['n_saved'], 3)
        self.assertEqual(item['n_valid_scores'], 2)
        self.assertEqual(item['n_missing_scores'], 1)
        self.assertEqual(item['n_positive_class'], 1)
        self.assertAlmostEqual(item['mean_decision_function'], 0.1)

    def synthetic_rows(self):
        values = {'base_neutral': (0.0, 0.0, 0.0, 0.0),
                  'grpo_3407_neutral': (2.0, 0.5, 1.0, 0.25),
                  'grpo_4407_neutral': (3.0, 1.0, 2.0, 0.5)}
        rows = []
        for job in grid(self.protocol, 'full'):
            old = job['topic_group'] == 'old'
            personal = job['frame'] == 'personal'
            index = (0 if old and personal else 1 if old else 2 if personal else 3)
            score = values[job['condition']][index] + (job['seed'] - 3408) * 0.01
            rows.append({**job, 'feature_score': score, 'completion': 'Текст'})
        return rows

    def test_primary_analysis_uses_topics_and_expected_interaction(self):
        contrasts, topic_rows, excluded = primary_analysis(
            self.synthetic_rows(), self.protocol)
        self.assertFalse(excluded)
        self.assertEqual(len(topic_rows), 40)
        target = next(row for row in contrasts
                      if row['contrast'] == 'framing_interaction'
                      and row['condition'] == 'grpo_3407_neutral'
                      and row['group'] == 'new')
        self.assertAlmostEqual(target['estimate'], 0.75)
        self.assertEqual(target['n_topics'], 10)

    def test_blind_export_is_fixed_and_hides_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_blind_review(self.synthetic_rows(), self.protocol, Path(tmp))
            self.assertEqual(result['n_texts'], 60)
            review = (Path(tmp) / 'blind_review.csv').read_text()
            key = (Path(tmp) / 'blind_review_key.csv').read_text()
            self.assertNotIn('grpo_3407_neutral', review)
            self.assertIn('grpo_3407_neutral', key)
            self.assertIn('topic_fit_0_2', review.splitlines()[0])

    def test_pilot_notebook_is_private_unexecuted_and_single_cell(self):
        downloads = REPO / 'downloads'
        downloads.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=downloads) as tmp:
            destination = Path(tmp) / 'build'
            build('pilot', destination)
            metadata = json.loads((destination / 'kernel-metadata.json').read_text())
            notebook = json.loads((destination / 'topic_framing.ipynb').read_text())
            self.assertTrue(metadata['is_private'])
            self.assertEqual(metadata['dataset_sources'], [self.protocol['runtime']['dataset']])
            code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
            self.assertEqual(len(code), 1)
            self.assertIsNone(code[0]['execution_count'])
            self.assertEqual(code[0]['outputs'], [])

    def test_end_to_end_analysis_on_complete_synthetic_run(self):
        jobs = grid(self.protocol, 'full')
        rows = []
        for row in self.synthetic_rows():
            metrics = {'raw_score': row['feature_score'], 'floored': False,
                       'word_count': 350, 'antisem_rate': 0.0,
                       'antisem_penalty': 0.0, 'sentiment_guard': 0.0,
                       'rep_ratio': 0.0}
            rows.append({**row, 'metrics': metrics, 'features': [0.0] * 73,
                         'geometry': {'z_rms': 1.0, 'pca_residual_rms': 0.2,
                                      'nearest_support_kernel': 0.5},
                         'finish_reason': 'stop'})
        with tempfile.TemporaryDirectory(dir=REPO / 'downloads') as tmp:
            root, output = Path(tmp) / 'run', Path(tmp) / 'analysis'
            root.mkdir()
            write_json(root / 'protocol.json', self.protocol)
            write_json(root / 'planned_requests.json', jobs)
            (root / 'generations.jsonl').write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            (root / 'scored.jsonl').write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            (root / 'geometry.jsonl').write_text(
                ''.join(json.dumps({'id': row['id'], **row['geometry']}) + '\n'
                        for row in rows))
            feature_names = [f'feature_{index}' for index in range(73)]
            for index, name in enumerate(self.protocol['analysis']['features_of_interest']):
                feature_names[index] = name
            write_json(root / 'feature_names.json', feature_names)
            write_json(root / 'reward_config.json', {})
            write_json(root / 'summary.json', [])
            (root / 'environment.txt').write_text('synthetic\n')
            artifacts = {path.name: sha256(path) for path in root.iterdir()
                         if path.name != 'run_manifest.json'}
            write_json(root / 'run_manifest.json', {
                'experiment_id': 'topic-framing-v1', 'mode': 'full',
                'status': 'complete', 'n_saved': 360, 'n_scoring_errors': 0,
                'protocol_sha256': sha256(root / 'protocol.json'),
                'artifact_sha256': artifacts})
            analyze(root, output, HERE / 'protocol.json')
            self.assertTrue((output / 'ANALYSIS.md').is_file())
            self.assertTrue((output / 'analysis_manifest.json').is_file())
            self.assertEqual(len((output / 'blind_review.csv').read_text().splitlines()), 61)


if __name__ == '__main__':
    unittest.main()
