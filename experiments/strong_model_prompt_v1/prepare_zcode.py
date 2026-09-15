"""Build a sanitized, hash-pinned generation kit for ZCode Agent."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PROTOCOL = HERE / 'protocol.json'
PERSONAL = REPO / 'experiments/personal_prompt_v1/protocol.json'
TOPIC = REPO / 'experiments/topic_framing_v1/protocol.json'
PERSONAL_RUNNER = REPO / 'experiments/personal_prompt_v1/run_eval.py'
VALIDATOR = HERE / 'batch_validator.py'


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_personal_runner():
    spec = importlib.util.spec_from_file_location('strong_model_personal_runner',
                                                  PERSONAL_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_protocols():
    design = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    require(digest(PERSONAL) == design['source_personal_protocol_sha256'],
            'Frozen personal protocol changed')
    require(digest(TOPIC) == design['source_topic_protocol_sha256'],
            'Frozen topic protocol changed')
    personal = json.loads(PERSONAL.read_text(encoding='utf-8'))
    runner = load_personal_runner()
    resolved = runner.resolve_protocol(personal, TOPIC)
    return design, resolved, runner


def grid(mode, design=None, personal=None, runner=None):
    if design is None:
        design, personal, runner = load_protocols()
    require(mode in ('pilot', 'full'), f'Unknown mode: {mode}')
    topics = personal['pilot_topics'] if mode == 'pilot' else personal['topics']
    replicates = design['pilot_replicates'] if mode == 'pilot' else design['replicates']
    rows = []
    for condition in design['conditions']:
        instruction = personal['instructions'][condition['instruction']]
        for replicate in replicates:
            for topic in topics:
                prompt = runner.prompt_for(personal, topic['topic'], instruction)
                request_id = f"{condition['id']}:{topic['topic_id']}:r{replicate}"
                safe_name = request_id.replace(':', '__') + '.txt'
                rows.append({
                    'id': request_id,
                    'condition': condition['id'],
                    'instruction': condition['instruction'],
                    'topic_index': topic['topic_index'],
                    'topic_id': topic['topic_id'],
                    'topic_group': topic['group'],
                    'topic': topic['topic'],
                    'replicate': replicate,
                    'prompt': prompt,
                    'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                    'output_file': f'responses/{safe_name}',
                })
    expected = design[f'expected_{mode}_texts']
    require(len(rows) == expected and len({r['id'] for r in rows}) == expected,
            f'Invalid {mode} grid')
    return rows


def batches(design, full_rows, pilot_rows):
    result = [('pilot', 'batch_00_pilot', pilot_rows)]
    index = 1
    for condition in design['conditions']:
        for replicate in design['replicates']:
            selected = [row for row in full_rows
                        if row['condition'] == condition['id']
                        and row['replicate'] == replicate]
            require(len(selected) == 20, 'Full batch must contain twenty topics')
            result.append(('full', f"batch_{index:02d}_{condition['id']}_r{replicate}",
                           selected))
            index += 1
    require(len(result) == 10, 'Expected one pilot and nine full batches')
    return result


def task_text(batch_id, mode, n_requests):
    return f"""# ZCode generation task: {batch_id}

This is a frozen text-generation batch. Before starting, verify in the UI:

- provider: Z.ai
- connection mode: Start Plan
- model: GLM-5.3-Flash
- reasoning effort: High
- this batch directory is the workspace root

Do not read parent or external directories. Do not use network tools, plugins,
subagents, prior conversations, or files outside this directory. Do not change
any prompt and do not inspect or infer an expected classifier result.

Read `requests.json`. For each of its {n_requests} requests, follow the exact
`prompt` independently and write the resulting Russian essay as plain UTF-8
text to its `output_file`. The response file must contain only the essay: no
ID, heading, Markdown fence, commentary, or evaluation. Do not copy wording
between essays merely because they share a batch.

Do not manually improve, select, or regenerate an answer after seeing it. If
you refuse a request, save the refusal verbatim in the designated response
file. If the task fails technically, stop and report the failure instead of
silently replacing an attempt.

Before generation, create `RUN_METADATA.json` from
`RUN_METADATA.template.json`. Preserve the fixed provider/application/mode/
model/effort fields, record the actual ZCode version (or `not_recorded`) and
UTC start time. After all files are written, record the UTC finish time. Then
run exactly:

```bash
python3 validate_batch.py .
```

Do not rewrite responses based on the displayed word counts. Report whether
the validator passed and stop. This is a `{mode}` batch; no scoring or analysis
is part of this task.
"""


def write_batch(root, mode, batch_id, rows, design):
    root.mkdir(parents=True)
    (root / 'prompts').mkdir()
    (root / 'responses').mkdir()
    request_rows = []
    for row in rows:
        prompt_file = 'prompts/' + Path(row['output_file']).name
        (root / prompt_file).write_text(row['prompt'] + '\n', encoding='utf-8')
        request_rows.append({**row, 'prompt_file': prompt_file})
    write_json(root / 'requests.json', {'batch_id': batch_id, 'mode': mode,
                                        'requests': request_rows})
    shutil.copyfile(VALIDATOR, root / 'validate_batch.py')
    (root / 'TASK.md').write_text(task_text(batch_id, mode, len(rows)), encoding='utf-8')
    write_json(root / 'RUN_METADATA.template.json', {
        **{key: design['generation_surface'][key] for key in
           ('provider', 'application', 'connection_mode', 'model', 'reasoning_effort')},
        'zcode_version': '', 'task_started_utc': '', 'task_finished_utc': '',
        'notes': '',
    })
    inputs = ['requests.json', 'TASK.md', 'RUN_METADATA.template.json',
              'validate_batch.py'] + [row['prompt_file'] for row in request_rows]
    manifest = {
        'experiment_id': design['experiment_id'], 'mode': mode,
        'batch_id': batch_id, 'expected_responses': len(rows),
        'protocol_sha256': digest(PROTOCOL),
        'generation_surface': design['generation_surface'],
        'request_ids': [row['id'] for row in rows],
        'input_sha256': {name: digest(root / name) for name in inputs},
    }
    write_json(root / 'batch_manifest.json', manifest)


def build(output):
    design, personal, runner = load_protocols()
    output = Path(output).resolve()
    require(not output.exists(), f'Output already exists: {output}')
    output.mkdir(parents=True)
    full_rows = grid('full', design, personal, runner)
    pilot_rows = grid('pilot', design, personal, runner)
    write_json(output / 'generation_protocol.json', {
        'experiment_id': design['experiment_id'],
        'protocol_sha256': digest(PROTOCOL),
        'source_personal_protocol_sha256': digest(PERSONAL),
        'source_topic_protocol_sha256': digest(TOPIC),
        'generation_surface': design['generation_surface'],
        'conditions': design['conditions'],
        'replicates': design['replicates'],
        'expected_pilot_texts': design['expected_pilot_texts'],
        'expected_full_texts': design['expected_full_texts'],
    })
    for mode, batch_id, rows in batches(design, full_rows, pilot_rows):
        write_batch(output / mode / batch_id, mode, batch_id, rows, design)
    (output / 'README_RUN.md').write_text(
        '# GLM-5.3-Flash generation kit\n\nOpen only one batch directory as the ZCode '
        'workspace root. Run `pilot/batch_00_pilot` first. Do not open the '
        'DeproLLM repository in ZCode. Follow each batch `TASK.md`.\n',
        encoding='utf-8')
    files = sorted(path for path in output.rglob('*') if path.is_file())
    write_json(output / 'kit_manifest.json', {
        'experiment_id': design['experiment_id'],
        'protocol_sha256': digest(PROTOCOL),
        'builder_sha256': digest(__file__),
        'files_sha256': {str(path.relative_to(output)): digest(path) for path in files},
    })
    print(f'Built sanitized kit: {output}')
    print(f'Pilot: {len(pilot_rows)} requests; full: {len(full_rows)} requests in 9 batches')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.output)
