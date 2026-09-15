"""Validate and build a private, unexecuted personal-prompt Kaggle notebook."""
import argparse
import json
from pathlib import Path
import subprocess

from run_eval import digest, grid, prompt_for, resolve_protocol


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SOURCE = HERE.parent / 'topic_framing_v1/protocol.json'
BASE_RUNNER = HERE.parent / 'topic_framing_v1/run_eval.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_protocols():
    design = json.loads((HERE / 'protocol.json').read_text())
    resolved = resolve_protocol(design, SOURCE)
    resolved['design_protocol_sha256'] = digest(HERE / 'protocol.json')
    return design, resolved


def validate(design, protocol):
    require(digest(SOURCE) == design['source_protocol_sha256'], 'Source hash mismatch')
    require(len(protocol['topics']) == 20 and len(protocol['pilot_topics']) == 2,
            'Unexpected topics')
    full, pilot = grid(protocol, 'full'), grid(protocol, 'pilot')
    require(len(full) == 300 and len(pilot) == 10, 'Unexpected grid size')
    require(not ({r['topic_id'] for r in full} & {r['topic_id'] for r in pilot}),
            'Pilot/full topics overlap')
    source = json.loads(SOURCE.read_text())
    personal = next(frame for frame in source['frames'] if frame['id'] == 'personal')
    for topic in protocol['topics'] + protocol['pilot_topics']:
        expected = source['prompt_template'].format(
            topic=topic['topic'], frame_instruction=personal['instruction'])
        require(prompt_for(protocol, topic['topic'], '') == expected,
                'Neutral personal prompt changed')
    old = json.loads((HERE.parent / 'prompt_baseline_v1/protocol.json').read_text())
    neutral = old['instructions']['neutral']
    for name in ('role', 'constrained'):
        require(old['instructions'][name] == neutral + ' ' + protocol['instructions'][name],
                f'{name} instruction changed')
    for condition in protocol['conditions']:
        require(condition['instruction'] in protocol['instructions'], 'Unknown instruction')


def check_pilot(protocol, manifest_path):
    require(manifest_path is not None, 'Full build needs --pilot-evidence')
    root = Path(manifest_path).resolve().parent
    manifest = json.loads(Path(manifest_path).read_text())
    require(manifest['mode'] == 'pilot' and manifest['status'] == 'complete',
            'Pilot did not complete')
    require(manifest['n_saved'] == 10 and manifest.get('n_scoring_errors') == 0,
            'Pilot evidence failed')
    require(manifest['design_protocol_sha256'] == digest(HERE / 'protocol.json'),
            'Design changed since pilot')
    require(manifest['source_protocol_sha256'] == digest(SOURCE),
            'Source protocol changed since pilot')
    require(manifest['personal_runner_sha256'] == digest(HERE / 'run_eval.py'),
            'Personal runner changed since pilot')
    require(manifest['base_runner_sha256'] == digest(BASE_RUNNER),
            'Base runner changed since pilot')
    required = {'planned_requests.json', 'generations.jsonl', 'scored.jsonl',
                'geometry.jsonl', 'summary.json', 'feature_names.json',
                'reward_config.json', 'environment.txt', 'protocol.json',
                'design_protocol.json', 'source_topic_framing_protocol.json'}
    require(required <= set(manifest['artifact_sha256']), 'Pilot artifacts missing')
    for name, expected in manifest['artifact_sha256'].items():
        require(Path(name).name == name and (root / name).is_file(), 'Unsafe evidence path')
        require(digest(root / name) == expected, f'Pilot artifact changed: {name}')
    rows = [json.loads(line) for line in (root / 'scored.jsonl').read_text().splitlines()
            if line.strip()]
    require([row['id'] for row in rows] == [row['id'] for row in grid(protocol, 'pilot')],
            'Pilot grid mismatch')
    for row in rows:
        require(len(row.get('features', [])) == 73, 'Pilot feature schema mismatch')
        require(isinstance(row.get('feature_score'), (int, float)), 'Pilot score missing')
        require(abs(row['feature_score'] - row['metrics']['raw_score']) < 1e-9,
                'Pilot score mismatch')


def build(mode, destination, pilot_evidence=None):
    design, protocol = load_protocols()
    validate(design, protocol)
    if mode == 'full':
        check_pilot(protocol, pilot_evidence)
    destination = Path(destination).resolve()
    require(destination.is_relative_to(REPO / 'downloads'),
            'Keep generated submissions in downloads/')
    destination.mkdir(parents=True, exist_ok=False)
    files = {
        'protocol.json': (HERE / 'protocol.json').read_text(),
        'source_protocol.json': SOURCE.read_text(),
        'run_eval.py': (HERE / 'run_eval.py').read_text(),
        'base_runner.py': BASE_RUNNER.read_text(),
    }
    writes = ''.join(
        f'(work / {name!r}).write_text({value!r}, encoding="utf-8")\n'
        for name, value in files.items())
    code = (
        'import pathlib, subprocess, sys, tempfile\n'
        'work = pathlib.Path(tempfile.mkdtemp(prefix="deprollm_personal_prompt_", dir="/tmp"))\n'
        + writes +
        'subprocess.check_call([sys.executable, "-u", str(work / "run_eval.py"),\n'
        '    "--protocol", str(work / "protocol.json"),\n'
        '    "--source-protocol", str(work / "source_protocol.json"),\n'
        '    "--base-runner", str(work / "base_runner.py"),\n'
        f'    "--mode", {mode!r}, "--output", "/kaggle/working/personal_prompt_{mode}"])\n'
    )
    notebook = {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                    'name': 'python3'}},
        'cells': [
            {'id': 'protocol', 'cell_type': 'markdown', 'metadata': {}, 'source': [
                f'# DeproLLM personal prompt v1: {mode}\n',
                'Private inference-only comparison; no training or paid API.\n',
                'Frozen design, source protocol and runners are embedded below.\n']},
            {'id': 'run', 'cell_type': 'code', 'metadata': {},
             'source': code.splitlines(keepends=True), 'execution_count': None, 'outputs': []},
        ],
    }
    notebook_path = destination / 'personal_prompt.ipynb'
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    metadata = {
        'id': f'alexdtat/deprollm-personal-prompt-v1-{mode}',
        'title': f'DeproLLM personal prompt v1 {mode}',
        'code_file': notebook_path.name, 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': [protocol['runtime']['dataset']],
        'competition_sources': [],
        'kernel_sources': [item['kernel'] for item in protocol['adapters'].values()],
    }
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    build_manifest = {
        'mode': mode,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              cwd=REPO, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'],
                                              cwd=REPO, text=True),
        'source_sha256': {name: digest(path) for name, path in {
            'protocol.json': HERE / 'protocol.json', 'run_eval.py': HERE / 'run_eval.py',
            'prepare_kaggle.py': Path(__file__), 'source_protocol.json': SOURCE,
            'base_runner.py': BASE_RUNNER}.items()},
        'notebook_sha256': digest(notebook_path),
        'pilot_evidence_manifest_sha256': digest(pilot_evidence) if pilot_evidence else None,
    }
    (destination / 'build_manifest.json').write_text(json.dumps(build_manifest, indent=2) + '\n')
    print(f"Prepared {len(grid(protocol, mode))} requests in {destination}; NOT submitted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pilot-evidence', type=Path)
    args = parser.parse_args()
    build(args.mode, args.output, args.pilot_evidence)
