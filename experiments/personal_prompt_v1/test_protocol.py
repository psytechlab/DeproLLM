import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from analyze import contrast_analysis, export_review
from prepare_kaggle import REPO, SOURCE, build, load_protocols, validate
from run_eval import digest, grid


HERE = Path(__file__).resolve().parent


class PersonalPromptTests(unittest.TestCase):
    def setUp(self):
        self.design, self.protocol = load_protocols()

    def test_frozen_source_and_grids(self):
        validate(self.design, self.protocol)
        self.assertEqual(digest(SOURCE), self.design['source_protocol_sha256'])
        full, pilot = grid(self.protocol, 'full'), grid(self.protocol, 'pilot')
        self.assertEqual(len(full), 300)
        self.assertEqual(len(pilot), 10)
        self.assertEqual(len({r['id'] for r in full}), 300)
        self.assertFalse({r['topic_id'] for r in full} & {r['topic_id'] for r in pilot})

    def test_pairing_order_prompts_and_seeds(self):
        rows = grid(self.protocol, 'full')
        blocks = {}
        for row in rows:
            blocks.setdefault((row['topic_id'], row['seed']), []).append(row)
        self.assertEqual(len(blocks), 60)
        for members in blocks.values():
            self.assertEqual(len(members), 5)
            self.assertEqual(len({r['request_seed'] for r in members}), 1)
        source = json.loads(SOURCE.read_text())
        personal = next(f for f in source['frames'] if f['id'] == 'personal')
        neutral = [r for r in rows if r['condition'] == 'base_neutral']
        for row in neutral:
            expected = source['prompt_template'].format(
                topic=row['topic'], frame_instruction=personal['instruction'])
            self.assertEqual(row['prompt'], expected)
        for condition in self.protocol['conditions']:
            selected = [r for r in rows if r['condition'] == condition['id']]
            expected = [(t['topic_id'], seed) for t in self.protocol['topics']
                        for seed in self.protocol['seeds']]
            self.assertEqual([(r['topic_id'], r['seed']) for r in selected], expected)

    def synthetic_rows(self):
        base = {'base_neutral': 0.0, 'base_role': 0.3, 'base_constrained': 0.5,
                'grpo_3407_neutral': 1.0, 'grpo_4407_neutral': 1.2}
        rows = []
        for job in grid(self.protocol, 'full'):
            score = base[job['condition']] + job['topic_index'] * 0.01 + (job['seed'] - 3408) * .001
            rows.append({**job, 'feature_score': score, 'completion': 'Текст'})
        return rows

    def test_primary_contrasts_use_topics(self):
        contrasts, topic_rows = contrast_analysis(self.synthetic_rows(), self.protocol)
        self.assertEqual(len(topic_rows), 20 * 8)
        primary = [r for r in contrasts if r['kind'] == 'primary'
                   and r['group'] == 'all_equal_group_weight']
        self.assertEqual(len(primary), 2)
        self.assertAlmostEqual(primary[0]['estimate'], 0.5)
        self.assertAlmostEqual(primary[1]['estimate'], 0.7)
        self.assertEqual(primary[0]['n_topics'], 20)

    def test_review_export_hides_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_review(self.synthetic_rows(), self.protocol, Path(tmp))
            self.assertEqual(result['n_texts'], 50)
            review = (Path(tmp) / 'blind_review.csv').read_text(encoding='utf-8-sig')
            key = (Path(tmp) / 'blind_review_key.csv').read_text()
            self.assertNotIn('grpo_3407_neutral', review)
            self.assertIn('grpo_3407_neutral', key)

    def test_pilot_build_is_private_and_unexecuted(self):
        (REPO / 'downloads').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO / 'downloads') as tmp:
            destination = Path(tmp) / 'build'
            build('pilot', destination)
            metadata = json.loads((destination / 'kernel-metadata.json').read_text())
            notebook = json.loads((destination / 'personal_prompt.ipynb').read_text())
            self.assertTrue(metadata['is_private'])
            self.assertEqual(metadata['dataset_sources'], [self.protocol['runtime']['dataset']])
            code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
            self.assertEqual(len(code), 1)
            self.assertIsNone(code[0]['execution_count'])
            self.assertEqual(code[0]['outputs'], [])


if __name__ == '__main__':
    unittest.main()
