import io
import json
from pathlib import Path
import stat
import unittest
import zipfile

from prepare_kaggle import validate
from run_eval import grid, safe_members, summarize

HERE = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((HERE / 'protocol.json').read_text())

    def test_grid_and_historical_prompts(self):
        validate(self.protocol)
        self.assertEqual(self.protocol['environment']['vllm'], '0.9.2')
        self.assertEqual(self.protocol['environment']['transformers'], '4.53.2')
        self.assertEqual(self.protocol['environment']['triton'], '3.2.0')
        full = grid(self.protocol, 'full')
        self.assertEqual(len(full), 300)
        keys = {(r['topic'], r['seed']) for r in full}
        self.assertEqual(len(keys), 60)
        for key in keys:
            rows = [r for r in full if (r['topic'], r['seed']) == key]
            self.assertEqual(len(rows), 5)
            self.assertEqual(len({r['request_seed'] for r in rows}), 1)

    def test_pilot_disjoint_and_full_length(self):
        full = grid(self.protocol, 'full')
        pilot = grid(self.protocol, 'pilot')
        self.assertEqual(len(pilot), 10)
        self.assertFalse({r['topic'] for r in full} & {r['topic'] for r in pilot})
        self.assertTrue(all('300–500' in r['prompt'] for r in pilot))

    def test_lora_prompt_is_neutral(self):
        rows = grid(self.protocol, 'full')
        neutral = {(r['topic'], r['seed']): r['prompt'] for r in rows
                   if r['condition'] == 'base_neutral'}
        for row in rows:
            if row['adapter']:
                self.assertEqual(row['prompt'], neutral[row['topic'], row['seed']])

    def test_zip_rejects_traversal_and_symlink(self):
        for name, symlink in [('../oops', False), ('/absolute', False), ('link', True)]:
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as z:
                info = zipfile.ZipInfo(name)
                if symlink:
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, b'target')
            stream.seek(0)
            with zipfile.ZipFile(stream) as z, self.assertRaises(ValueError):
                safe_members(z)

    def test_missing_score_not_negative_or_zero(self):
        rows = [dict(condition='base', topic_group='new', finish_reason='stop',
                     metrics={'raw_score': score}) for score in [None, -0.1, 0.2]]
        summary = summarize(rows)[0]
        self.assertEqual(summary['n_saved'], 3)
        self.assertEqual(summary['n_valid_scores'], 2)
        self.assertEqual(summary['n_positive_class'], 1)
        self.assertAlmostEqual(summary['mean_decision_function'], 0.05)


if __name__ == '__main__':
    unittest.main()
