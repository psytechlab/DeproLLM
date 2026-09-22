"""Build an unexecuted private Kaggle notebook; never submit it automatically."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from run_eval import grid, require, sha256


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def validate(protocol):
    source = REPO / protocol['source_protocol']
    require(sha256(source) == protocol['source_protocol_sha256'],
            'Frozen source protocol changed')
    require(len(protocol['topics']) == 20 and len(protocol['pilot_topics']) == 2,
            'Expected 20 full and 2 pilot topics')
    require(len({item['topic_id'] for item in protocol['topics'] + protocol['pilot_topics']}) == 22,
            'Duplicate topic IDs')
    require(len({item['topic_index'] for item in protocol['topics'] + protocol['pilot_topics']}) == 22,
            'Duplicate topic indices')
    require({item['group'] for item in protocol['topics']} == {'old', 'new'},
            'Unexpected full topic groups')
    require({item['id'] for item in protocol['frames']} == {'personal', 'general'},
            'Unexpected framing conditions')
    require(len(grid(protocol, 'full')) == protocol['expected_full_texts'] == 360,
            'Expected 360 full requests')
    require(len(grid(protocol, 'pilot')) == protocol['expected_pilot_texts'] == 12,
            'Expected 12 pilot requests')
    require(not {item['topic'] for item in protocol['topics']} &
            {item['topic'] for item in protocol['pilot_topics']}, 'Pilot/full topic overlap')
    source_protocol = json.loads(source.read_text())
    for key in ['model', 'revision', 'seeds', 'sampling', 'batch_size',
                'environment', 'runtime', 'adapters']:
        require(protocol[key] == source_protocol[key], f'Frozen source field changed: {key}')
    require(protocol['engine']['max_num_seqs'] == protocol['batch_size'],
            'Engine batch setting drift')
    require(protocol['analysis']['primary_score'] == 'feature_score' and
            protocol['analysis']['unit'] == 'topic', 'Primary analysis changed')


def check_pilot_evidence(protocol_text, pilot_evidence):
    require(pilot_evidence is not None, 'Full build needs --pilot-evidence run_manifest.json')
    path = Path(pilot_evidence).resolve()
    pilot = json.loads(path.read_text())
    require(pilot['experiment_id'] == 'topic-framing-v1' and pilot['mode'] == 'pilot',
            'Wrong pilot evidence')
    require(pilot['status'] == 'complete' and pilot['n_saved'] == 12 and
            pilot['n_scoring_errors'] == 0, 'Pilot did not pass')
    require(pilot['protocol_sha256'] == hashlib.sha256(protocol_text.encode()).hexdigest(),
            'Protocol changed since pilot')
    require(pilot['runner_sha256'] == sha256(HERE / 'run_eval.py'),
            'Runner changed since pilot')
    evidence_root = path.parent
    required = {'protocol.json', 'planned_requests.json', 'generations.jsonl',
                'scored.jsonl', 'geometry.jsonl', 'summary.json',
                'feature_names.json', 'reward_config.json', 'environment.txt'}
    require(required <= set(pilot['artifact_sha256']), 'Pilot evidence files missing')
    for name, digest in pilot['artifact_sha256'].items():
        require(Path(name).name == name, 'Unsafe evidence path')
        require(sha256(evidence_root / name) == digest, f'Pilot evidence mismatch: {name}')


def build(mode, destination, pilot_evidence=None):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    validate(protocol)
    protocol_text = json.dumps(protocol, ensure_ascii=False, indent=2) + '\n'
    runner_text = (HERE / 'run_eval.py').read_text()
    if mode == 'full':
        check_pilot_evidence(protocol_text, pilot_evidence)
    destination = Path(destination).resolve()
    require(destination.is_relative_to(REPO / 'downloads'),
            'Keep generated submissions in downloads/')
    destination.mkdir(parents=True, exist_ok=False)

    code = (
        'import pathlib, subprocess, sys, tempfile\n'
        'work = pathlib.Path(tempfile.mkdtemp(prefix="deprollm_framing_code_", dir="/tmp"))\n'
        f'(work / "protocol.json").write_text({protocol_text!r}, encoding="utf-8")\n'
        f'(work / "run_eval.py").write_text({runner_text!r}, encoding="utf-8")\n'
        'subprocess.check_call([sys.executable, "-u", str(work / "run_eval.py"),\n'
        f'    "--protocol", str(work / "protocol.json"), "--mode", {mode!r},\n'
        f'    "--output", "/kaggle/working/topic_framing_{mode}"])\n'
    )
    notebook = {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                    'name': 'python3'}},
        'cells': [
            {'id': 'protocol', 'cell_type': 'markdown', 'metadata': {}, 'source': [
                f'# DeproLLM topic framing v1: {mode}\n',
                'Private inference-only experiment; no training or paid API.\n',
                'The frozen protocol and runner are embedded below as source code.\n',
                'The private runtime and both private adapter archives are inputs only.\n']},
            {'id': 'run', 'cell_type': 'code', 'metadata': {},
             'source': code.splitlines(keepends=True), 'execution_count': None, 'outputs': []},
        ],
    }
    notebook_path = destination / 'topic_framing.ipynb'
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    metadata = {
        'id': f'alexdtat/deprollm-topic-framing-v1-{mode}',
        'title': f'DeproLLM topic framing v1 {mode}',
        'code_file': notebook_path.name, 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': [protocol['runtime']['dataset']],
        'competition_sources': [],
        'kernel_sources': [item['kernel'] for item in protocol['adapters'].values()],
    }
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    provenance = {
        'mode': mode,
        'git_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'git_status': subprocess.check_output(
            ['git', 'status', '--short'], cwd=REPO, text=True),
        'source_sha256': {path.name: sha256(path) for path in
                          [HERE / 'run_eval.py', HERE / 'protocol.json', Path(__file__)]},
        'source_protocol_sha256': protocol['source_protocol_sha256'],
        'notebook_sha256': sha256(notebook_path),
        'pilot_evidence_manifest_sha256': sha256(pilot_evidence) if pilot_evidence else None,
    }
    (destination / 'build_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Prepared {len(grid(protocol, mode))} requests in {destination}; NOT submitted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pilot-evidence', type=Path)
    args = parser.parse_args()
    build(args.mode, args.output, args.pilot_evidence)
