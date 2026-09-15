"""Build an unexecuted, private Kaggle notebook; never submit it automatically."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from run_eval import grid, require, sha256

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def validate(protocol):
    topics = protocol['old_topics'] + protocol['new_topics'] + protocol['pilot_topics']
    require(len(topics) == len(set(topics)), 'Overlapping topic groups')
    require(len(protocol['old_topics']) == len(protocol['new_topics']) == 10, 'Expected 10+10 topics')
    require(len(grid(protocol, 'full')) == 300, 'Expected 300 full requests')
    require(len(grid(protocol, 'pilot')) == 10, 'Expected 10 pilot requests')
    old = [json.loads(line)['prompt'] for line in
           (REPO / 'depression_reward/prompts/prompts.jsonl').read_text().splitlines() if line.strip()]
    require([protocol['instructions']['neutral'].format(topic=t)
             for t in protocol['old_topics']] == old[-10:], 'Historical holdout changed')
    require(not set(protocol['new_topics'] + protocol['pilot_topics']) &
            {p.removeprefix('Напишите эссе объёмом 300–500 слов на тему: «').removesuffix('».')
             for p in old}, 'New topics duplicate old topics')


def build(mode, destination, pilot_evidence=None):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    validate(protocol)
    protocol_text = json.dumps(protocol, ensure_ascii=False, indent=2) + '\n'
    runner_text = (HERE / 'run_eval.py').read_text()
    if mode == 'full':
        require(pilot_evidence is not None, 'Full build needs --pilot-evidence run_manifest.json')
        pilot = json.loads(Path(pilot_evidence).read_text())
        require(pilot['mode'] == 'pilot' and pilot['status'] == 'complete' and
                pilot['n_saved'] == 10 and pilot['n_scoring_errors'] == 0, 'Pilot did not pass')
        require(pilot['protocol_sha256'] == hashlib.sha256(protocol_text.encode()).hexdigest(),
                'Protocol changed since pilot')
        require(pilot['runner_sha256'] == sha256(HERE / 'run_eval.py'), 'Runner changed since pilot')
        evidence_root = Path(pilot_evidence).parent
        for name, digest in pilot['artifact_sha256'].items():
            require(Path(name).name == name, 'Unsafe evidence path')
            require(sha256(evidence_root / name) == digest, f'Pilot evidence mismatch: {name}')
    destination = Path(destination).resolve()
    require(destination.is_relative_to(REPO / 'downloads'), 'Keep generated submissions in downloads/')
    destination.mkdir(parents=True, exist_ok=False)
    # Single code cell: pip, self-check and inference run in one child process.
    # Kaggle may continue after a failed cell, so no later GPU cell is present.
    code = (
        'import pathlib, subprocess, sys, tempfile\n'
        'work = pathlib.Path(tempfile.mkdtemp(prefix="deprollm_eval_code_", dir="/tmp"))\n'
        f'(work / "protocol.json").write_text({protocol_text!r}, encoding="utf-8")\n'
        f'(work / "run_eval.py").write_text({runner_text!r}, encoding="utf-8")\n'
        'subprocess.check_call([sys.executable, "-u", str(work / "run_eval.py"),\n'
        f'    "--protocol", str(work / "protocol.json"), "--mode", {mode!r},\n'
        f'    "--output", "/kaggle/working/prompt_eval_{mode}"])\n'
    )
    notebook = {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [
            {'id': 'protocol', 'cell_type': 'markdown', 'metadata': {}, 'source': [
                f'# DeproLLM prompt baseline: {mode}\n',
                'Private inference-only experiment. No training or paid API.\n',
                'Source protocol and runner are included as plain text below.\n',
                'Runtime and two adapter archives must remain private Kaggle inputs.\n']},
            {'id': 'run', 'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(keepends=True),
             'execution_count': None, 'outputs': []},
        ],
    }
    (destination / 'prompt_eval.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    metadata = {
        'id': f'alexdtat/deprollm-prompt-baseline-v1-{mode}',
        'title': f'DeproLLM prompt baseline v1 {mode}', 'code_file': 'prompt_eval.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': [protocol['runtime']['dataset']], 'competition_sources': [],
        'kernel_sources': [a['kernel'] for a in protocol['adapters'].values()],
    }
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    provenance = {'mode': mode, 'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True),
        'source_sha256': {p.name: sha256(p) for p in
                          [HERE / 'run_eval.py', HERE / 'protocol.json', Path(__file__)]},
        'notebook_sha256': sha256(destination / 'prompt_eval.ipynb')}
    (destination / 'build_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Prepared {len(grid(protocol, mode))} requests in {destination}; NOT submitted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pilot-evidence', type=Path)
    args = parser.parse_args()
    build(args.mode, args.output, args.pilot_evidence)
