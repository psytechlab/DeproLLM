"""Run the frozen topic-framing inference grid. Importing does no GPU work."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(canonical_json(value), encoding='utf-8')
    tmp.replace(path)


def append_row(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def grid(protocol, mode):
    require(mode in ('pilot', 'full'), 'Unknown mode')
    topics = protocol['topics'] if mode == 'full' else protocol['pilot_topics']
    seeds = protocol['seeds'] if mode == 'full' else protocol['pilot_seeds']
    rows = []
    # Frozen order: conditions, topics, frames, seeds.
    for condition in protocol['conditions']:
        for topic in topics:
            for frame in protocol['frames']:
                prompt = protocol['prompt_template'].format(
                    topic=topic['topic'], frame_instruction=frame['instruction'])
                for seed in seeds:
                    rows.append({
                        'id': f"{condition['id']}:{topic['topic_id']}:{frame['id']}:{seed}",
                        'condition': condition['id'], 'adapter': condition['adapter'],
                        'topic_index': topic['topic_index'], 'topic_id': topic['topic_id'],
                        'source_index': topic.get('source_index'), 'topic_group': topic['group'],
                        'topic': topic['topic'], 'frame': frame['id'],
                        'frame_instruction': frame['instruction'], 'seed': seed,
                        'request_seed': seed * 100 + topic['topic_index'], 'prompt': prompt,
                    })
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate evaluation keys')
    expected = protocol[f'expected_{mode}_texts']
    require(len(rows) == expected, f'Expected {expected} {mode} requests, got {len(rows)}')
    return rows


def safe_members(archive):
    names = set()
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        require(not path.is_absolute() and '..' not in path.parts and '\\' not in item.filename,
                'Unsafe ZIP path')
        require(not stat.S_ISLNK(item.external_attr >> 16), 'ZIP symlink')
        require(item.filename not in names, 'Duplicate ZIP member')
        names.add(item.filename)
    return names


def find_one(root, name):
    matches = list(Path(root).rglob(name))
    require(len(matches) == 1, f'Expected exactly one {name}, found {len(matches)}')
    return matches[0]


def unpack_inputs(protocol, input_root, scratch):
    runtime = protocol['runtime']
    matches = []
    for path in Path(input_root).rglob('manifest.json'):
        try:
            manifest = json.loads(path.read_text())
        except (ValueError, OSError, UnicodeError):
            continue
        if (manifest.get('artifact_name'), manifest.get('artifact_version')) == (
                runtime['artifact_name'], runtime['artifact_version']):
            matches.append((path, manifest))
    require(len(matches) == 1, 'Missing or ambiguous private runtime')
    path, manifest = matches[0]
    require(manifest['runtime_archive'] == 'depression_reward.payload',
            'Unexpected payload name')
    payload = path.parent / 'depression_reward.payload'
    require(manifest['runtime_archive_sha256'] == runtime['sha256'] == sha256(payload),
            'Runtime SHA256 mismatch')
    runtime_root = scratch / 'runtime'
    with zipfile.ZipFile(payload) as archive:
        safe_members(archive)
        archive.extractall(runtime_root)

    adapters, calibrations = {}, []
    for name, source in protocol['adapters'].items():
        path = find_one(input_root, f"grpo_artifacts_{source['run_id']}.zip")
        require(sha256(path) == source['archive_sha256'], f'{name}: archive SHA256 mismatch')
        with zipfile.ZipFile(path) as archive:
            safe_members(archive)
            prefix = f"grpo_run_{source['run_id']}/"
            run_manifest = json.loads(archive.read(prefix + 'run_manifest.json'))
            require(run_manifest['model']['resolved_revision'] == protocol['revision'],
                    'Adapter base revision mismatch')
            adapter_dir = scratch / name
            adapter_dir.mkdir()
            for filename, key in [('adapter_config.json', 'config_sha256'),
                                  ('adapter_model.safetensors', 'weights_sha256')]:
                rel = 'grpo_depression_lora/' + filename
                data = archive.read(prefix + rel)
                require(hashlib.sha256(data).hexdigest() == source[key] ==
                        run_manifest['artifact_sha256'][rel], 'Adapter SHA256 mismatch')
                (adapter_dir / filename).write_bytes(data)
            data = archive.read(prefix + 'calibration.json')
            require(hashlib.sha256(data).hexdigest() ==
                    run_manifest['artifact_sha256']['calibration.json'],
                    'Calibration hash mismatch')
            calibrations.append(json.loads(data))
            adapters[name] = adapter_dir
    require(len(calibrations) == 2 and calibrations[0] == calibrations[1],
            'Source calibrations differ')
    return runtime_root, adapters, calibrations[0]


@contextmanager
def heartbeat(label):
    start = time.monotonic()
    stop = threading.Event()

    def report():
        while not stop.wait(30):
            print(f'{label}: waiting, {time.monotonic() - start:.0f}s elapsed', flush=True)

    thread = threading.Thread(target=report, daemon=True)
    print(f'{label}: start', flush=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        print(f'{label}: ended after {time.monotonic() - start:.1f}s', flush=True)


def geometry(model, vector):
    import numpy as np

    vector = np.asarray(vector, dtype=np.float64).reshape(1, -1)
    scaler, pca, svc = model.steps[0][1], model.steps[1][1], model.steps[2][1]
    z = scaler.transform(vector)
    projected = pca.transform(z)
    reconstructed = pca.inverse_transform(projected)
    squared = np.maximum(np.sum((svc.support_vectors_ - projected[0]) ** 2, axis=1), 0.0)
    return {
        'z_rms': float(np.sqrt(np.mean(z ** 2))),
        'pca_residual_rms': float(np.sqrt(np.mean((z - reconstructed) ** 2))),
        'nearest_support_kernel': float(np.exp(-svc._gamma * np.min(squared))),
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['condition'], row['topic_group'], row['frame'])].append(row)
    result = []
    for (condition, group, frame), members in groups.items():
        valid = [r for r in members if isinstance(r.get('feature_score'), (int, float))]
        scores = [r['feature_score'] for r in valid]
        result.append({
            'condition': condition, 'topic_group': group, 'frame': frame,
            'n_saved': len(members), 'n_valid_scores': len(valid),
            'n_missing_scores': len(members) - len(valid),
            'n_positive_class': sum(score > 0 for score in scores),
            'mean_decision_function': sum(scores) / len(scores) if scores else None,
            'n_length_stops': sum(r.get('finish_reason') == 'length' for r in members),
        })
    return result


def run(protocol, mode, input_root, output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)
    jobs = grid(protocol, mode)
    write_json(output / 'planned_requests.json', jobs)
    manifest = {
        'experiment_id': protocol['experiment_id'], 'mode': mode,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'initializing',
        'n_expected': len(jobs), 'n_saved': 0, 'python': platform.python_version(),
        'runner_sha256': sha256(__file__), 'protocol_sha256': sha256(output / 'protocol.json'),
        'source_protocol_sha256': protocol['source_protocol_sha256'],
    }
    write_json(output / 'run_manifest.json', manifest)
    scratch = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix='deprollm_topic_framing_', dir='/tmp'))
        with heartbeat('Verify private inputs'):
            runtime_root, adapters, calibration = unpack_inputs(protocol, input_root, scratch)
        os.environ.update(PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python', VLLM_USE_V1='0',
                          TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true')
        os.environ['PYTHONPATH'] = str(runtime_root) + os.pathsep + os.environ.get('PYTHONPATH', '')
        sys.path.insert(0, str(runtime_root))
        with heartbeat('Install inference environment'):
            versions = protocol['environment']
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                                   f"vllm=={versions['vllm']}",
                                   f"transformers=={versions['transformers']}",
                                   f"numpy=={versions['numpy']}",
                                   f"peft=={versions['peft']}", '-r',
                                   str(runtime_root / 'depression_reward/requirements-colab.txt')])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                                   '--force-reinstall', '--no-deps',
                                   f"triton=={versions['triton']}"])
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'])
            subprocess.check_call([
                sys.executable, '-c',
                'from vllm import LLM, SamplingParams; import transformers, triton; '
                f'assert transformers.__version__ == {versions["transformers"]!r}; '
                f'assert triton.__version__ == {versions["triton"]!r}'
            ])
            subprocess.check_call([sys.executable, '-m', 'depression_reward.selfcheck'])
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (output / 'environment.txt').write_text(freeze, encoding='utf-8')

        import torch
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from depression_reward import DepressionReward, RewardConfig
        from depression_reward.feature_names import FEATURE_NAMES

        require(torch.cuda.is_available(), 'CUDA unavailable')
        require(torch.cuda.get_device_capability(0) >= (7, 0), 'T4 or newer GPU required')
        require(len(FEATURE_NAMES) == 73, 'Unexpected feature schema')
        manifest['gpu'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        manifest['torch'] = torch.__version__
        manifest['cuda'] = torch.version.cuda
        manifest['calibration'] = calibration
        engine_args = dict(protocol['engine'])
        manifest['engine_args'] = engine_args
        manifest['sampling'] = protocol['sampling']
        manifest['status'] = 'loading_model'
        write_json(output / 'run_manifest.json', manifest)
        with heartbeat('Load Qwen and initialize vLLM'):
            llm = LLM(model=protocol['model'], revision=protocol['revision'],
                      tokenizer_revision=protocol['revision'], **engine_args)
        tokenizer = llm.get_tokenizer()
        write_json(output / 'feature_names.json', FEATURE_NAMES)
        reward_config = RewardConfig(style_center=calibration['style_center'],
                                     style_temp=calibration['style_temp'])
        write_json(output / 'reward_config.json', asdict(reward_config))
        scorer = DepressionReward(reward_config)
        loras = {name: LoRARequest(name, index + 1, str(path))
                 for index, (name, path) in enumerate(adapters.items())}
        all_rows = []
        manifest['status'] = 'generating'
        write_json(output / 'run_manifest.json', manifest)
        for condition in protocol['conditions']:
            selected = [row for row in jobs if row['condition'] == condition['id']]
            for start in range(0, len(selected), protocol['batch_size']):
                batch = selected[start:start + protocol['batch_size']]
                require(len({row['adapter'] for row in batch}) == 1,
                        'Batch mixes adapters')
                prompts = [tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': row['prompt']}], tokenize=False,
                    add_generation_prompt=True) for row in batch]
                for rendered in prompts:
                    require(len(tokenizer.encode(rendered, add_special_tokens=False)) +
                            protocol['sampling']['max_tokens'] <= protocol['engine']['max_model_len'],
                            'Prompt exceeds context')
                params = [SamplingParams(**protocol['sampling'], seed=row['request_seed'])
                          for row in batch]
                begin = time.monotonic()
                with heartbeat(f"Generate {condition['id']} {start + 1}-{start + len(batch)}"):
                    generated = llm.generate(
                        prompts, params, lora_request=loras.get(condition['adapter']), use_tqdm=False)
                require(len(generated) == len(batch), 'Missing engine outputs')
                raw_rows = []
                for job, rendered, item in zip(batch, prompts, generated):
                    require(len(item.outputs) == 1, 'Expected one completion per request')
                    completion = item.outputs[0]
                    row = {
                        **job, 'rendered_prompt': rendered, 'completion': completion.text,
                        'token_ids': list(completion.token_ids),
                        'prompt_token_ids': list(item.prompt_token_ids),
                        'finish_reason': completion.finish_reason,
                        'stop_reason': completion.stop_reason,
                        'batch_wall_seconds': time.monotonic() - begin,
                        'batch_size': len(batch), 'batch_start': start,
                    }
                    append_row(output / 'generations.jsonl', row)
                    raw_rows.append(row)
                for row in raw_rows:
                    try:
                        row['metrics'] = scorer.breakdown([row['completion']])[0]
                        extracted = scorer.pipeline.extract(row['completion'])
                        row['features'] = extracted.vector.tolist()
                        row['feature_score'] = float(scorer.scorer.raw_scores(extracted.vector)[0])
                        row['geometry'] = geometry(scorer.scorer.model, extracted.vector)
                    except Exception as exc:
                        row['scoring_error'] = f'{type(exc).__name__}: {exc}'
                    append_row(output / 'scored.jsonl', row)
                    if 'geometry' in row:
                        append_row(output / 'geometry.jsonl', {'id': row['id'], **row['geometry']})
                    all_rows.append(row)
                manifest['n_saved'] = len(all_rows)
                write_json(output / 'run_manifest.json', manifest)
                print(f"Saved {len(all_rows)}/{len(jobs)} texts", flush=True)
        require({row['id'] for row in all_rows} == {row['id'] for row in jobs}, 'Incomplete grid')
        write_json(output / 'summary.json', summarize(all_rows))
        manifest['status'] = 'complete'
        manifest['n_scoring_errors'] = sum('scoring_error' in row for row in all_rows)
    except BaseException as exc:
        manifest['status'] = 'error'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['artifact_sha256'] = {
            path.name: sha256(path) for path in output.iterdir()
            if path.is_file() and path.name != 'run_manifest.json'
        }
        write_json(output / 'run_manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.mode, args.input, args.output)


if __name__ == '__main__':
    main()
