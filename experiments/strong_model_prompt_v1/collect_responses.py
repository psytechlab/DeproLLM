"""Verify completed ZCode batches and collect raw texts without scoring them."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_zcode import PROTOCOL, batches, digest, grid, load_protocols, require, write_json


def safe_file(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            f'Unsafe path: {relative}')
    path = root / relative
    require(path.is_file(), f'Missing file: {relative}')
    return path


def verify_kit(kit):
    manifest_path = kit / 'kit_manifest.json'
    require(manifest_path.is_file(), 'Missing kit_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['protocol_sha256'] == digest(PROTOCOL), 'Kit protocol hash mismatch')
    for name, expected in manifest['files_sha256'].items():
        require(digest(safe_file(kit, name)) == expected, f'Kit input changed: {name}')
    return manifest


def collect(kit, mode, output):
    kit, output = Path(kit).resolve(), Path(output).resolve()
    require(mode in ('pilot', 'full'), f'Unknown mode: {mode}')
    require(not output.exists(), f'Output already exists: {output}')
    kit_manifest = verify_kit(kit)
    design, personal, runner = load_protocols()
    expected = grid(mode, design, personal, runner)
    expected_by_id = {row['id']: row for row in expected}
    all_batches = [item for item in batches(
        design, grid('full', design, personal, runner),
        grid('pilot', design, personal, runner)) if item[0] == mode]
    rows, response_hashes, metadata = [], {}, []
    for _, batch_id, batch_rows in all_batches:
        root = kit / mode / batch_id
        batch_manifest = json.loads((root / 'batch_manifest.json').read_text(encoding='utf-8'))
        request_data = json.loads((root / 'requests.json').read_text(encoding='utf-8'))
        report_path = root / 'validation_report.json'
        require(report_path.is_file(), f'Batch not validated: {batch_id}')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        require(report.get('status') == 'technical_validation_passed',
                f'Batch validation failed: {batch_id}')
        require(report.get('batch_id') == batch_id, 'Validation batch mismatch')
        run_meta_path = root / 'RUN_METADATA.json'
        require(run_meta_path.is_file(), f'Missing RUN_METADATA.json: {batch_id}')
        run_meta = json.loads(run_meta_path.read_text(encoding='utf-8'))
        require(digest(run_meta_path) == report['run_metadata_sha256'],
                f'RUN_METADATA changed after validation: {batch_id}')
        metadata.append({'batch_id': batch_id, **run_meta})
        report_rows = {row['request_id']: row for row in report['responses']}
        require(len(report_rows) == len(batch_rows), f'Report count mismatch: {batch_id}')
        for request in request_data['requests']:
            request_id = request['id']
            require(request_id in expected_by_id, f'Unexpected request: {request_id}')
            expected_row = expected_by_id[request_id]
            for key, value in expected_row.items():
                require(request.get(key) == value, f'Frozen request changed: {request_id}:{key}')
            response_path = safe_file(root, request['output_file'])
            response_sha = digest(response_path)
            require(report_rows[request_id]['sha256'] == response_sha,
                    f'Response changed after validation: {request_id}')
            text = response_path.read_text(encoding='utf-8').strip()
            rows.append({**expected_row, 'completion': text,
                         'response_sha256': response_sha, 'batch_id': batch_id})
            response_hashes[f'{batch_id}/{request["output_file"]}'] = response_sha
        require(batch_manifest['request_ids'] == [row['id'] for row in batch_rows],
                f'Batch manifest grid mismatch: {batch_id}')

    require(len(rows) == len(expected), 'Collected row count mismatch')
    require({row['id'] for row in rows} == set(expected_by_id), 'Collected ID mismatch')
    rows.sort(key=lambda row: [item['id'] for item in expected].index(row['id']))
    output.mkdir(parents=True)
    write_json(output / 'planned_requests.json', expected)
    with (output / 'generations.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(output / 'run_metadata.json', metadata)
    write_json(output / 'design_protocol.json', design)
    manifest = {
        'experiment_id': design['experiment_id'], 'mode': mode,
        'status': 'raw_responses_collected', 'n_saved': len(rows),
        'protocol_sha256': digest(PROTOCOL),
        'kit_manifest_sha256': digest(kit / 'kit_manifest.json'),
        'collector_sha256': digest(__file__),
        'response_sha256': response_hashes,
    }
    write_json(output / 'collection_manifest.json', manifest)
    print(f'Collected {len(rows)} {mode} responses into {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kit', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('pilot', 'full'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    collect(args.kit, args.mode, args.output)
