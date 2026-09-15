"""Score a complete collected GLM batch locally with the private runtime."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments/topic_framing_v1'))

from depression_reward import DepressionReward, RewardConfig
from depression_reward.feature_names import FEATURE_NAMES
from run_eval import geometry
from prepare_zcode import PROTOCOL, digest, grid, load_protocols, require, write_json


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def score(input_root, reward_config, output):
    input_root, reward_config, output = (Path(input_root).resolve(),
                                         Path(reward_config).resolve(),
                                         Path(output).resolve())
    require(not output.exists(), f'Output already exists: {output}')
    design, personal, runner = load_protocols()
    require(digest(reward_config) == design['scoring_reference']['reward_config_sha256'],
            'Reward config hash mismatch')
    manifest_path = input_root / 'collection_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['mode'] == 'full' and manifest['status'] == 'raw_responses_collected',
            'Scoring requires a complete full collection')
    require(manifest['protocol_sha256'] == digest(PROTOCOL), 'Collection protocol mismatch')
    rows = read_jsonl(input_root / 'generations.jsonl')
    expected = grid('full', design, personal, runner)
    require([row['id'] for row in rows] == [row['id'] for row in expected],
            'Generation grid mismatch')

    reward = DepressionReward(RewardConfig.from_json(reward_config))
    scored = []
    for expected_row, row in zip(expected, rows):
        for key, value in expected_row.items():
            require(row.get(key) == value, f'Frozen input mismatch: {row["id"]}:{key}')
        completion = row['completion']
        extracted = reward.pipeline.extract(completion)
        vector = extracted.vector
        raw_score = float(reward.scorer.raw_scores(vector)[0])
        require(math.isfinite(raw_score) and len(vector) == 73, 'Invalid classifier output')
        metrics = reward.breakdown([completion])[0]
        require(abs(float(metrics['raw_score']) - raw_score) < 1e-9,
                'Breakdown/raw score mismatch')
        antisem_matches = [word for sentence in extracted.lemmas for word in sentence
                           if word.lower().replace('ё', 'е') in reward.lexicon]
        scored.append({**row, 'feature_score': raw_score, 'metrics': metrics,
                       'features': vector.tolist(),
                       'geometry': geometry(reward.scorer.model, vector),
                       'antisem_matches': antisem_matches})

    output.mkdir(parents=True)
    with (output / 'scored.jsonl').open('w', encoding='utf-8') as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(output / 'feature_names.json', FEATURE_NAMES)
    write_json(output / 'reward_config.json', asdict(reward.cfg))
    artifacts = ['scored.jsonl', 'feature_names.json', 'reward_config.json']
    write_json(output / 'scoring_manifest.json', {
        'experiment_id': design['experiment_id'], 'status': 'scoring_complete',
        'scored_utc': datetime.now(timezone.utc).isoformat(), 'n_scored': len(scored),
        'collection_manifest_sha256': digest(manifest_path),
        'protocol_sha256': digest(PROTOCOL), 'scorer_sha256': digest(__file__),
        'source_reward_config_sha256': digest(reward_config),
        'artifact_sha256': {name: digest(output / name) for name in artifacts},
        'decision_function_note': 'Not a probability or clinical assessment.',
    })
    print(f'Scored {len(scored)} responses into {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--reward-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    score(args.input, args.reward_config, args.output)
