"""Isolated inference evaluation. Importing this module never installs or runs models."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
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


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def append_row(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def grid(protocol, mode):
    require(mode in ('pilot', 'full'), 'Unknown mode')
    topics = [('old', t) for t in protocol['old_topics']]
    topics += [('new', t) for t in protocol['new_topics']]
    seeds = protocol['seeds']
    if mode == 'pilot':
        topics = [('pilot', t) for t in protocol['pilot_topics']]
        seeds = seeds[:1]
    rows = []
    for condition in protocol['conditions']:
        for seed in seeds:
            for index, (group, topic) in enumerate(topics):
                rows.append({
                    'id': f"{condition['id']}:{group}:{index}:{seed}",
                    'condition': condition['id'], 'adapter': condition['adapter'],
                    'topic_index': index, 'topic_group': group, 'topic': topic,
                    'seed': seed, 'request_seed': seed * 100 + index,
                    'prompt': protocol['instructions'][condition['instruction']].format(topic=topic),
                })
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate evaluation keys')
    return rows


def safe_members(archive):
    names = set()
    for item in archive.infolist():
        p = PurePosixPath(item.filename)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in item.filename,
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
            m = json.loads(path.read_text())
        except (ValueError, OSError, UnicodeError):
            continue
        if (m.get('artifact_name'), m.get('artifact_version')) == (
                runtime['artifact_name'], runtime['artifact_version']):
            matches.append((path, m))
    require(len(matches) == 1, 'Missing or ambiguous private runtime')
    path, manifest = matches[0]
    require(manifest['runtime_archive'] == 'depression_reward.payload', 'Unexpected payload name')
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
            manifest_source = json.loads(archive.read(prefix + 'run_manifest.json'))
            require(manifest_source['model']['resolved_revision'] == protocol['revision'],
                    'Adapter base revision mismatch')
            adapter_dir = scratch / name
            adapter_dir.mkdir()
            for filename, key in [('adapter_config.json', 'config_sha256'),
                                  ('adapter_model.safetensors', 'weights_sha256')]:
                rel = 'grpo_depression_lora/' + filename
                data = archive.read(prefix + rel)
                require(hashlib.sha256(data).hexdigest() == source[key] ==
                        manifest_source['artifact_sha256'][rel], 'Adapter SHA256 mismatch')
                (adapter_dir / filename).write_bytes(data)
            data = archive.read(prefix + 'calibration.json')
            require(hashlib.sha256(data).hexdigest() ==
                    manifest_source['artifact_sha256']['calibration.json'], 'Calibration hash mismatch')
            calibrations.append(json.loads(data))
            adapters[name] = adapter_dir
    require(calibrations[0] == calibrations[1], 'Source calibrations differ')
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


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['condition'], row['topic_group'])].append(row)
    result = []
    for (condition, group), members in groups.items():
        valid = [r for r in members if r.get('metrics', {}).get('raw_score') is not None]
        scores = [r['metrics']['raw_score'] for r in valid]
        result.append({
            'condition': condition, 'topic_group': group, 'n_saved': len(members),
            'n_valid_scores': len(valid), 'n_missing_scores': len(members) - len(valid),
            'n_positive_class': sum(s > 0 for s in scores),
            'mean_decision_function': sum(scores) / len(scores) if scores else None,
            'n_length_stops': sum(r['finish_reason'] == 'length' for r in members),
        })
    return result


def run(protocol, mode, input_root, output):
    # No checkpoint, adapter or runtime file is ever written under output.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)
    jobs = grid(protocol, mode)
    write_json(output / 'planned_requests.json', jobs)
    manifest = {'experiment_id': protocol['experiment_id'], 'mode': mode,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'initializing',
                'n_expected': len(jobs), 'python': platform.python_version(),
                'runner_sha256': sha256(__file__), 'protocol_sha256': sha256(output / 'protocol.json')}
    write_json(output / 'run_manifest.json', manifest)
    try:
        # Keep scratch alive while the inference engine may still read adapter files.
        scratch = Path(tempfile.mkdtemp(prefix='deprollm_prompt_eval_', dir='/tmp'))
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
            # vLLM 0.9.2 LoRA kernels fail to compile on T4 with Kaggle's Triton 3.3.0.
            # This is the same override used by the successful GRPO notebook.
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                                   '--force-reinstall', '--no-deps',
                                   f"triton=={versions['triton']}"])
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'])
            # Import in a fresh process catches binary/import compatibility before model loading.
            subprocess.check_call([
                sys.executable, '-c',
                'from vllm import LLM, SamplingParams; import transformers, triton; '
                f'assert transformers.__version__ == {versions["transformers"]!r}; '
                f'assert triton.__version__ == {versions["triton"]!r}'
            ])
            subprocess.check_call([sys.executable, '-m', 'depression_reward.selfcheck'])
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (output / 'environment.txt').write_text(freeze)
        import torch
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from depression_reward import DepressionReward, RewardConfig
        from depression_reward.feature_names import FEATURE_NAMES

        require(torch.cuda.is_available(), 'CUDA unavailable')
        require(torch.cuda.get_device_capability(0) >= (7, 0), 'T4 or newer GPU required')
        manifest['gpu'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        manifest['torch'] = torch.__version__
        manifest['cuda'] = torch.version.cuda
        manifest['calibration'] = calibration
        engine_args = dict(model=protocol['model'], revision=protocol['revision'],
                           tokenizer_revision=protocol['revision'], dtype='half',
                           tensor_parallel_size=1, max_model_len=2048,
                           gpu_memory_utilization=0.88, enable_lora=True,
                           max_lora_rank=32, max_loras=1, max_cpu_loras=2,
                           max_num_seqs=protocol['batch_size'], enforce_eager=True,
                           seed=3407, generation_config='vllm', enable_prefix_caching=False)
        manifest['engine_args'] = engine_args
        manifest['status'] = 'loading_model'
        write_json(output / 'run_manifest.json', manifest)
        with heartbeat('Load Qwen and initialize vLLM'):
            llm = LLM(**engine_args)
        tokenizer = llm.get_tokenizer()
        require(len(FEATURE_NAMES) == 73, 'Unexpected feature schema')
        write_json(output / 'feature_names.json', FEATURE_NAMES)
        scorer = DepressionReward(RewardConfig(style_center=calibration['style_center'],
                                              style_temp=calibration['style_temp']))
        loras = {name: LoRARequest(name, i + 1, str(path))
                 for i, (name, path) in enumerate(adapters.items())}
        all_rows = []
        manifest['status'] = 'generating'
        write_json(output / 'run_manifest.json', manifest)
        for condition in protocol['conditions']:
            selected = [r for r in jobs if r['condition'] == condition['id']]
            for start in range(0, len(selected), protocol['batch_size']):
                batch = selected[start:start + protocol['batch_size']]
                prompts = [tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': r['prompt']}], tokenize=False,
                    add_generation_prompt=True) for r in batch]
                for p in prompts:
                    require(len(tokenizer.encode(p, add_special_tokens=False)) +
                            protocol['sampling']['max_tokens'] <= 2048, 'Prompt exceeds context')
                params = [SamplingParams(**protocol['sampling'], seed=r['request_seed']) for r in batch]
                begin = time.monotonic()
                with heartbeat(f"Generate {condition['id']} {start + 1}-{start + len(batch)}"):
                    generated = llm.generate(prompts, params,
                                             lora_request=loras.get(condition['adapter']), use_tqdm=False)
                require(len(generated) == len(batch), 'Missing engine outputs')
                # Persist every raw output BEFORE scoring any text in this batch.
                raw_rows = []
                for job, rendered, item in zip(batch, prompts, generated):
                    require(len(item.outputs) == 1, 'Expected one completion per request')
                    completion = item.outputs[0]
                    row = {**job, 'rendered_prompt': rendered, 'completion': completion.text,
                           'token_ids': list(completion.token_ids),
                           'prompt_token_ids': list(item.prompt_token_ids),
                           'finish_reason': completion.finish_reason, 'stop_reason': completion.stop_reason,
                           'batch_wall_seconds': time.monotonic() - begin,
                           'batch_size': len(batch)}
                    append_row(output / 'generations.jsonl', row)
                    raw_rows.append(row)
                for row in raw_rows:
                    try:
                        row['metrics'] = scorer.breakdown([row['completion']])[0]
                        # Preserve raw features even when the reward floors a short/repetitive text.
                        res = scorer.pipeline.extract(row['completion'])
                        row['features'] = res.vector.tolist()
                        row['feature_score'] = float(scorer.scorer.raw_scores(res.vector)[0])
                    except Exception as exc:
                        row['scoring_error'] = f'{type(exc).__name__}: {exc}'
                    append_row(output / 'scored.jsonl', row)
                    all_rows.append(row)
                manifest['n_saved'] = len(all_rows)
                write_json(output / 'run_manifest.json', manifest)
                print(f"Saved {len(all_rows)}/{len(jobs)} texts", flush=True)
        require({r['id'] for r in all_rows} == {r['id'] for r in jobs}, 'Incomplete grid')
        write_json(output / 'summary.json', summarize(all_rows))
        manifest['status'] = 'complete'
        manifest['n_scoring_errors'] = sum('scoring_error' in r for r in all_rows)
    except BaseException as exc:
        manifest['status'] = 'error'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['artifact_sha256'] = {p.name: sha256(p) for p in output.iterdir()
                                     if p.is_file() and p.name != 'run_manifest.json'}
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
