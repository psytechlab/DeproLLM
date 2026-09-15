"""Standalone technical validator copied into every isolated ZCode batch."""
import argparse
import hashlib
import json
from pathlib import Path
import re


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)*")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_file(root, value):
    value = Path(value)
    require(not value.is_absolute() and '..' not in value.parts, f'Unsafe path: {value}')
    path = root / value
    require(path.is_file(), f'Missing file: {value}')
    return path


def validate(root):
    root = Path(root).resolve()
    manifest_path = root / 'batch_manifest.json'
    require(manifest_path.is_file(), 'Missing batch_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, expected in manifest['input_sha256'].items():
        path = relative_file(root, name)
        require(digest(path) == expected, f'Input hash mismatch: {name}')

    requests = json.loads((root / 'requests.json').read_text(encoding='utf-8'))
    require(requests['batch_id'] == manifest['batch_id'], 'Batch ID mismatch')
    require(len(requests['requests']) == manifest['expected_responses'],
            'Unexpected request count')

    expected_paths = []
    rows = []
    for request in requests['requests']:
        path = relative_file(root, request['output_file'])
        expected_paths.append(path.resolve())
        data = path.read_bytes()
        require(data and b'\x00' not in data, f'Empty or binary response: {path.name}')
        text = data.decode('utf-8').strip()
        require(text, f'Whitespace-only response: {path.name}')
        rows.append({
            'request_id': request['id'],
            'output_file': request['output_file'],
            'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data),
            'word_count': len(WORD_RE.findall(text)),
        })

    response_root = root / 'responses'
    actual = sorted(path.resolve() for path in response_root.glob('*.txt')) \
        if response_root.is_dir() else []
    require(sorted(expected_paths) == actual, 'Missing or unexpected response files')

    metadata_path = root / 'RUN_METADATA.json'
    require(metadata_path.is_file(), 'Create RUN_METADATA.json from the template')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    expected = manifest['generation_surface']
    for key in ('provider', 'application', 'connection_mode', 'model',
                'reasoning_effort'):
        require(metadata.get(key) == expected[key], f'RUN_METADATA mismatch: {key}')
    for key in ('task_started_utc', 'task_finished_utc', 'zcode_version'):
        require(isinstance(metadata.get(key), str) and metadata[key].strip(),
                f'RUN_METADATA missing: {key}')

    report = {
        'status': 'technical_validation_passed',
        'batch_id': manifest['batch_id'],
        'expected_responses': manifest['expected_responses'],
        'responses': rows,
        'run_metadata_sha256': digest(metadata_path),
        'note': 'Word counts are audit data, not a selection or retry rule.',
    }
    (root / 'validation_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    print(f"PASS {manifest['batch_id']}: {len(rows)} responses")
    for row in rows:
        print(f"{row['request_id']}: {row['word_count']} words")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default='.', type=Path)
    args = parser.parse_args()
    validate(args.root)
