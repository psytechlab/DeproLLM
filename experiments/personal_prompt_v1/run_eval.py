"""Run the frozen personal-prompt inference grid. Importing does no GPU work."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def load_base_runner(path=None):
    path = Path(path or HERE.parent / 'topic_framing_v1/run_eval.py')
    spec = importlib.util.spec_from_file_location('personal_prompt_base_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve_protocol(design, source):
    if digest(source) != design['source_protocol_sha256']:
        raise ValueError('Frozen topic-framing protocol changed')
    inherited = json.loads(Path(source).read_text())
    resolved = {key: inherited[key] for key in design['inherit_exact_fields']}
    resolved.update({key: value for key, value in design.items()
                     if key != 'inherit_exact_fields'})
    resolved['design_protocol_sha256'] = digest(HERE / 'protocol.json') \
        if (HERE / 'protocol.json').is_file() else None
    return resolved


def prompt_for(protocol, topic, instruction):
    parts = [protocol['prompt_prefix'].format(topic=topic), instruction,
             protocol['prompt_ending']]
    return ' '.join(part for part in parts if part)


def grid(protocol, mode):
    if mode not in ('pilot', 'full'):
        raise ValueError('Unknown mode')
    topics = protocol['topics'] if mode == 'full' else protocol['pilot_topics']
    seeds = protocol['seeds'] if mode == 'full' else protocol['pilot_seeds']
    rows = []
    for condition in protocol['conditions']:
        instruction = protocol['instructions'][condition['instruction']]
        for topic in topics:
            prompt = prompt_for(protocol, topic['topic'], instruction)
            for seed in seeds:
                rows.append({
                    'id': f"{condition['id']}:{topic['topic_id']}:{seed}",
                    'condition': condition['id'], 'adapter': condition['adapter'],
                    'instruction': condition['instruction'],
                    'instruction_text': instruction,
                    'topic_index': topic['topic_index'], 'topic_id': topic['topic_id'],
                    'source_index': topic.get('source_index'),
                    'topic_group': topic['group'], 'topic': topic['topic'],
                    'frame': 'personal', 'seed': seed,
                    'request_seed': seed * 100 + topic['topic_index'],
                    'prompt': prompt,
                })
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate evaluation keys')
    expected = protocol[f'expected_{mode}_texts']
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} {mode} requests, got {len(rows)}')
    return rows


def run(design_path, source_path, base_runner_path, mode, input_root, output):
    design = json.loads(Path(design_path).read_text())
    source = Path(source_path)
    protocol = resolve_protocol(design, source)
    base = load_base_runner(base_runner_path)
    base.grid = grid
    error = None
    try:
        base.run(protocol, mode, Path(input_root), Path(output))
    except BaseException as exc:
        error = exc
    finally:
        output = Path(output)
        if output.is_dir():
            (output / 'design_protocol.json').write_text(
                Path(design_path).read_text(), encoding='utf-8')
            (output / 'source_topic_framing_protocol.json').write_text(
                source.read_text(), encoding='utf-8')
            manifest_path = output / 'run_manifest.json'
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                manifest['personal_runner_sha256'] = digest(__file__)
                manifest['base_runner_sha256'] = digest(base_runner_path)
                manifest['design_protocol_sha256'] = digest(design_path)
                manifest['source_protocol_sha256'] = digest(source)
                manifest['artifact_sha256'] = {
                    path.name: digest(path) for path in output.iterdir()
                    if path.is_file() and path.name != 'run_manifest.json'
                }
                base.write_json(manifest_path, manifest)
    if error is not None:
        raise error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--source-protocol', required=True, type=Path)
    parser.add_argument('--base-runner', required=True, type=Path)
    parser.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.source_protocol, args.base_runner, args.mode,
        args.input, args.output)


if __name__ == '__main__':
    main()
