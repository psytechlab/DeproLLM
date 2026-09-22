import json
from pathlib import Path
import tempfile
import unittest

from analyze import contrasts
from batch_validator import validate as validate_batch
from collect_responses import collect
from prepare_zcode import (PERSONAL, PROTOCOL, TOPIC, batches, build, digest, grid,
                           load_protocols)


class StrongModelPromptTests(unittest.TestCase):
    def setUp(self):
        self.design, self.personal, self.runner = load_protocols()

    def test_frozen_sources_and_grids(self):
        self.assertEqual(digest(PERSONAL), self.design['source_personal_protocol_sha256'])
        self.assertEqual(digest(TOPIC), self.design['source_topic_protocol_sha256'])
        full = grid('full', self.design, self.personal, self.runner)
        pilot = grid('pilot', self.design, self.personal, self.runner)
        self.assertEqual((len(full), len(pilot)), (180, 6))
        self.assertEqual(len({row['id'] for row in full}), 180)
        self.assertFalse({row['topic_id'] for row in full} &
                         {row['topic_id'] for row in pilot})
        self.assertEqual({row['replicate'] for row in full}, {0, 1, 2})

    def test_prompts_equal_personal_prompt_conditions(self):
        full = grid('full', self.design, self.personal, self.runner)
        source = self.runner.grid(self.personal, 'full')
        mapping = {'glm53_flash_neutral': 'base_neutral',
                   'glm53_flash_role': 'base_role',
                   'glm53_flash_constrained': 'base_constrained'}
        source_prompts = {(row['condition'], row['topic_id']): row['prompt'] for row in source}
        for row in full:
            self.assertEqual(row['prompt'], source_prompts[mapping[row['condition']],
                                                          row['topic_id']])

    def test_batching_is_one_condition_and_replicate_per_main_task(self):
        full = grid('full', self.design, self.personal, self.runner)
        pilot = grid('pilot', self.design, self.personal, self.runner)
        parts = batches(self.design, full, pilot)
        self.assertEqual(len(parts), 10)
        self.assertEqual(len(parts[0][2]), 6)
        for mode, _, rows in parts[1:]:
            self.assertEqual(mode, 'full')
            self.assertEqual(len(rows), 20)
            self.assertEqual(len({row['condition'] for row in rows}), 1)
            self.assertEqual(len({row['replicate'] for row in rows}), 1)

    def fill_batch(self, root):
        requests = json.loads((root / 'requests.json').read_text(encoding='utf-8'))
        template = json.loads((root / 'RUN_METADATA.template.json').read_text())
        template.update({'zcode_version': 'test',
                         'task_started_utc': '2026-09-12T00:00:00Z',
                         'task_finished_utc': '2026-09-12T00:01:00Z'})
        (root / 'RUN_METADATA.json').write_text(
            json.dumps(template, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for request in requests['requests']:
            path = root / request['output_file']
            path.write_text('Это синтетический технический ответ.\n', encoding='utf-8')
        validate_batch(root)

    def test_sanitized_kit_and_pilot_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            kit = Path(tmp) / 'kit'
            build(kit)
            files = [path for path in kit.rglob('*') if path.is_file()]
            joined = b'\n'.join(path.read_bytes() for path in files)
            for forbidden in (b'weights.npz', b'psydicts.json', b'depression_reward',
                              b'personal_prompt_full', b'scored.jsonl'):
                self.assertNotIn(forbidden, joined)
            pilot = kit / 'pilot/batch_00_pilot'
            self.fill_batch(pilot)
            output = Path(tmp) / 'collected'
            collect(kit, 'pilot', output)
            rows = [json.loads(line) for line in
                    (output / 'generations.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual({row['topic_group'] for row in rows}, {'pilot'})

    def test_topic_level_contrasts(self):
        glm = []
        base = {'glm53_flash_neutral': 0.0, 'glm53_flash_role': 0.3,
                'glm53_flash_constrained': 0.7}
        for row in grid('full', self.design, self.personal, self.runner):
            glm.append({**row, 'feature_score': base[row['condition']]})
        qwen = []
        qbase = {'base_neutral': -0.2, 'base_role': 0.1,
                 'base_constrained': 0.4, 'grpo_3407_neutral': 0.5,
                 'grpo_4407_neutral': 0.6}
        for condition, score in qbase.items():
            for topic in self.personal['topics']:
                for seed in self.personal['seeds']:
                    qwen.append({'condition': condition, 'topic_index': topic['topic_index'],
                                 'topic_group': topic['group'], 'seed': seed,
                                 'feature_score': score})
        summaries, topics = contrasts(glm, qwen, {**self.personal, **self.design})
        primary = next(row for row in summaries
                       if row['kind'] == 'primary_within_model'
                       and row['group'] == 'all_equal_group_weight')
        external = next(row for row in summaries
                        if row['left'] == 'glm53_flash_constrained'
                        and row['right'] == 'base_constrained'
                        and row['group'] == 'all_equal_group_weight')
        self.assertAlmostEqual(primary['estimate'], 0.7)
        self.assertAlmostEqual(external['estimate'], 0.3)
        self.assertEqual(len(topics), 20 * 8)


if __name__ == '__main__':
    unittest.main()
