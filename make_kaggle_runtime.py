"""Build, verify, and pin the private Kaggle reward runtime.

The generated upload ZIP contains a manifest, checksums, and one ZIP-formatted
``depression_reward.payload``. Kaggle expands files ending in ``.zip`` inside a
Dataset, so the neutral payload suffix intentionally preserves the byte stream
whose SHA-256 is pinned by the notebook.

Examples::

    .venv/bin/python make_kaggle_runtime.py check \
        downloads/kaggle/deprollm-depression-reward-runtime-v1
    .venv/bin/python make_kaggle_runtime.py pin \
        downloads/kaggle/deprollm-depression-reward-runtime-v1
    .venv/bin/python make_kaggle_runtime.py build v3

``build`` refuses to reuse a version or package uncommitted reward changes.
Create a new version instead of silently replacing an uploaded Dataset.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
REWARD_ROOT = ROOT / "depression_reward"
NOTEBOOK = ROOT / "grpo_depression.ipynb"
DEFAULT_OUTPUT_ROOT = ROOT / "downloads" / "kaggle"

ARTIFACT_NAME = "deprollm-depression-reward-runtime"
SOURCE_REPOSITORY = "psytechlab/DeproLLM"
RUNTIME_ARCHIVE = "depression_reward.payload"
LEGACY_MARKER = "# [embed_zip_in_notebook.py]"
RUNTIME_CELL_ID = "load-depression-reward-dataset"
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)

EXCLUDED_NAMES = {"model_repacked.joblib", "rebuild_model.py"}
EXCLUDED_RELATIVE_PATHS = {"README.md"}
COMPONENT_PATHS = (
    "depression_reward/model/weights.npz",
    "depression_reward/model/params.json",
    "depression_reward/model/test_features.json",
    "depression_reward/feature_names.py",
    "depression_reward/vendor/data/psydicts.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_reward_files() -> list[Path]:
    files = []
    for path in sorted(REWARD_ROOT.rglob("*")):
        relative = path.relative_to(REWARD_ROOT)
        if (
            path.is_file()
            and "__pycache__" not in relative.parts
            and path.suffix not in {".pyc", ".pyo"}
            and path.name not in EXCLUDED_NAMES
            and relative.as_posix() not in EXCLUDED_RELATIVE_PATHS
        ):
            files.append(path)
    return files


def write_deterministic_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, data in sorted(members):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data)


def parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            raise ValueError(f"invalid checksum line: {line!r}")
        name = parts[1].lstrip("*")
        if name in entries:
            raise ValueError(f"duplicate checksum entry: {name}")
        entries[name] = parts[0].lower()
    return entries


def safe_zip_names(archive: zipfile.ZipFile) -> list[str]:
    names = []
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"unsafe archive member: {info.filename!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"symlink is not allowed: {info.filename!r}")
        names.append(info.filename)
    return names


def validate_artifact(artifact_dir: Path) -> dict:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "manifest.json"
    checksums_path = artifact_dir / "SHA256SUMS"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ValueError(f"manifest.json or SHA256SUMS missing in {artifact_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_name") != ARTIFACT_NAME:
        raise ValueError("unexpected artifact_name")
    version = manifest.get("artifact_version")
    if not isinstance(version, str) or not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("unexpected artifact_version")
    runtime_name = manifest.get("runtime_archive")
    if runtime_name != RUNTIME_ARCHIVE:
        raise ValueError("unexpected runtime_archive")

    runtime_path = artifact_dir / runtime_name
    if not runtime_path.is_file():
        raise ValueError(f"runtime archive missing: {runtime_path}")

    checksums = parse_checksums(checksums_path)
    if set(checksums) != {RUNTIME_ARCHIVE, "manifest.json"}:
        raise ValueError("SHA256SUMS must contain only runtime and manifest hashes")

    actual_runtime_sha = sha256_file(runtime_path)
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_runtime_sha != manifest.get("runtime_archive_sha256"):
        raise ValueError("runtime hash differs from manifest")
    if actual_runtime_sha != checksums[RUNTIME_ARCHIVE]:
        raise ValueError("runtime hash differs from SHA256SUMS")
    if actual_manifest_sha != checksums["manifest.json"]:
        raise ValueError("manifest hash differs from SHA256SUMS")

    with zipfile.ZipFile(runtime_path) as archive:
        names = safe_zip_names(archive)
        if len(names) != len(set(names)):
            raise ValueError("runtime archive contains duplicate members")
        name_set = set(names)
        for required in COMPONENT_PATHS:
            if required not in name_set:
                raise ValueError(f"required runtime component missing: {required}")
        forbidden = [
            name
            for name in names
            if "__pycache__" in PurePosixPath(name).parts
            or name.endswith((".pyc", ".pyo"))
            or PurePosixPath(name).name in EXCLUDED_NAMES
        ]
        if forbidden:
            raise ValueError(f"forbidden runtime members: {forbidden}")
        component_sha256 = manifest.get("component_sha256")
        if not isinstance(component_sha256, dict) or set(component_sha256) != set(
            COMPONENT_PATHS
        ):
            raise ValueError("manifest has an unexpected component_sha256 set")
        for component, expected_sha256 in component_sha256.items():
            actual_sha256 = hashlib.sha256(archive.read(component)).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(f"component hash differs from manifest: {component}")

    manifest["_runtime_sha256"] = actual_runtime_sha
    return manifest


def git_source_commit() -> str:
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "depression_reward",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty_paths = []
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1]
        if path != "depression_reward/README.md":
            dirty_paths.append(path)
    if dirty_paths:
        raise ValueError(
            "runtime files have uncommitted changes; commit them before packaging: "
            + ", ".join(dirty_paths)
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build(version: str, output_root: Path) -> Path:
    if not version.startswith("v") or not version[1:].isdigit():
        raise ValueError("version must look like v1, v2, ...")

    source_commit = git_source_commit()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_root / f"{ARTIFACT_NAME}-{version}"
    upload_zip = output_root / f"{ARTIFACT_NAME}-{version}.zip"
    if artifact_dir.exists() or upload_zip.exists():
        raise FileExistsError(
            f"{version} already exists; choose a new version instead of overwriting it"
        )
    artifact_dir.mkdir()

    runtime_path = artifact_dir / RUNTIME_ARCHIVE
    reward_members = [
        (path.relative_to(ROOT).as_posix(), path.read_bytes())
        for path in included_reward_files()
    ]
    write_deterministic_zip(runtime_path, reward_members)

    component_sha256 = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in COMPONENT_PATHS
    }
    manifest = {
        "artifact_name": ARTIFACT_NAME,
        "artifact_version": version,
        "visibility": "private",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": source_commit,
        "created_date": date.today().isoformat(),
        "runtime_archive": RUNTIME_ARCHIVE,
        "runtime_archive_sha256": sha256_file(runtime_path),
        "classifier": {
            "pipeline": "StandardScaler -> PCA(46) -> SVC(rbf, C=1000)",
            "feature_count": 73,
            "score_kind": "decision_function",
            "reference_scores": {
                "depr": -0.0557977018,
                "norm": -0.1023075880,
            },
        },
        "component_sha256": component_sha256,
        "privacy_notice": (
            "Contains private TITANIS-derived code, dictionaries, and classifier "
            "parameters. Do not publish or make the Kaggle Dataset public."
        ),
        "validation_command": (
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "
            "python -m depression_reward.selfcheck"
        ),
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksums_path = artifact_dir / "SHA256SUMS"
    checksums_path.write_text(
        f"{sha256_file(runtime_path)}  {RUNTIME_ARCHIVE}\n"
        f"{sha256_file(manifest_path)}  manifest.json\n",
        encoding="utf-8",
    )
    validate_artifact(artifact_dir)

    upload_members = [
        (path.name, path.read_bytes())
        for path in (runtime_path, manifest_path, checksums_path)
    ]
    write_deterministic_zip(upload_zip, upload_members)
    return upload_zip


def runtime_cell(version: str, runtime_sha256: str) -> dict:
    source = f'''# Приватный reward runtime из подключённого Kaggle Dataset.
import hashlib, json, shutil, stat, zipfile
from pathlib import Path, PurePosixPath

EXPECTED_ARTIFACT = {ARTIFACT_NAME!r}
EXPECTED_VERSION = {version!r}
EXPECTED_RUNTIME_SHA256 = {runtime_sha256!r}
KAGGLE_INPUT = Path('/kaggle/input')
REWARD_RUNTIME_ROOT = Path('/tmp/deprollm_reward_runtime')

assert KAGGLE_INPUT.is_dir(), 'Подключите приватный reward Dataset к notebook'
_matches = []
for _manifest_path in KAGGLE_INPUT.rglob('manifest.json'):
    try:
        _manifest = json.loads(_manifest_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    if (_manifest.get('artifact_name') == EXPECTED_ARTIFACT and
            _manifest.get('artifact_version') == EXPECTED_VERSION and
            _manifest.get('runtime_archive_sha256') == EXPECTED_RUNTIME_SHA256):
        _matches.append((_manifest_path, _manifest))

assert len(_matches) == 1, (
    f'Ожидался ровно один {{EXPECTED_ARTIFACT}} {{EXPECTED_VERSION}}, найдено {{len(_matches)}}. '
    'Проверьте подключённую версию приватного Dataset.'
)
_manifest_path, _manifest = _matches[0]
_runtime_name = _manifest.get('runtime_archive')
assert _runtime_name == 'depression_reward.payload', 'Неожиданное имя runtime payload'
_runtime_path = _manifest_path.parent / _runtime_name
assert _runtime_path.is_file(), f'Не найден {{_runtime_path}}'

def _sha256(path):
    _digest = hashlib.sha256()
    with path.open('rb') as _stream:
        for _chunk in iter(lambda: _stream.read(1024 * 1024), b''):
            _digest.update(_chunk)
    return _digest.hexdigest()

_checksums_path = _manifest_path.parent / 'SHA256SUMS'
assert _checksums_path.is_file(), 'В Dataset отсутствует SHA256SUMS'
_checksums = {{}}
for _line in _checksums_path.read_text(encoding='utf-8').splitlines():
    if _line.strip():
        _parts = _line.split(maxsplit=1)
        assert len(_parts) == 2 and len(_parts[0]) == 64, f'Некорректная строка SHA256SUMS: {{_line!r}}'
        _digest, _name = _parts
        _name = _name.lstrip('*')
        assert _name not in _checksums, f'Повтор в SHA256SUMS: {{_name}}'
        _checksums[_name] = _digest.lower()
assert set(_checksums) == {{'depression_reward.payload', 'manifest.json'}}, 'Некорректный SHA256SUMS'
assert _sha256(_manifest_path) == _checksums['manifest.json'], 'Повреждён manifest.json'
_actual_sha256 = _sha256(_runtime_path)
assert _actual_sha256 == EXPECTED_RUNTIME_SHA256, 'Runtime не совпадает с закреплённым SHA-256'
assert _actual_sha256 == _checksums[_runtime_name], 'Runtime не совпадает с SHA256SUMS'

with zipfile.ZipFile(_runtime_path) as _archive:
    _names = set()
    for _info in _archive.infolist():
        _member = PurePosixPath(_info.filename)
        assert not _member.is_absolute() and '..' not in _member.parts, f'Небезопасный путь: {{_info.filename}}'
        assert not stat.S_ISLNK(_info.external_attr >> 16), f'Симлинк запрещён: {{_info.filename}}'
        _names.add(_info.filename)
    for _required in ('depression_reward/model/weights.npz',
                      'depression_reward/model/params.json',
                      'depression_reward/feature_names.py'):
        assert _required in _names, f'В runtime отсутствует {{_required}}'
    _target = REWARD_RUNTIME_ROOT
    if _target.is_symlink() or _target.is_file():
        _target.unlink()
    elif _target.is_dir():
        shutil.rmtree(_target)
    _target.mkdir(parents=True)
    _archive.extractall(_target)
REWARD_PACKAGE_DIR = REWARD_RUNTIME_ROOT / 'depression_reward'
assert REWARD_PACKAGE_DIR.is_dir(), f'После распаковки не найден {{REWARD_PACKAGE_DIR}}'
print('reward runtime:', EXPECTED_VERSION, _actual_sha256[:12] + '…', 'из', _manifest_path.parent)
'''
    return {
        "cell_type": "code",
        "id": RUNTIME_CELL_ID,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def pin_notebook(artifact_dir: Path) -> None:
    manifest = validate_artifact(artifact_dir)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    candidates = [
        index
        for index, cell in enumerate(cells)
        if cell.get("id") == RUNTIME_CELL_ID
        or LEGACY_MARKER in "".join(cell.get("source", []))[:300]
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one runtime cell, found {len(candidates)}")
    cells[candidates[0]] = runtime_cell(
        manifest["artifact_version"], manifest["_runtime_sha256"]
    )

    introduction = "".join(cells[0].get("source", [])).splitlines(keepends=True)
    dataset_instruction = (
        "2. В разделе **Input** подключите приватный Kaggle Dataset "
        f"`{ARTIFACT_NAME}` версии `{manifest['artifact_version']}`; notebook "
        "проверит manifest и SHA-256.\n"
    )
    step_two = [index for index, line in enumerate(introduction) if line.startswith("2. ")]
    if len(step_two) != 1:
        raise ValueError("expected one step 2 in notebook introduction")
    introduction[step_two[0]] = dataset_instruction
    cells[0]["source"] = introduction
    cells[0].setdefault("id", "notebook-introduction")

    for cell in cells:
        source = "".join(cell.get("source", []))
        if "Восстанавливаем встроенный `depression_reward.zip`" in source:
            source = source.replace(
                "Восстанавливаем встроенный `depression_reward.zip`, ставим "
                "зависимости и запускаем self-check",
                "Проверяем и распаковываем `depression_reward.zip` из приватного "
                "Kaggle Dataset, ставим зависимости и запускаем self-check",
            )
            cell["source"] = source.splitlines(keepends=True)
        if "requirements-colab.txt" in source:
            cell["source"] = [
                "import os, sys\n",
                "assert REWARD_PACKAGE_DIR.is_dir(), \\\n",
                "    'Reward runtime не распакован — проверьте Dataset-ячейку выше'\n",
                "_runtime_parent = str(REWARD_PACKAGE_DIR.parent)\n",
                "if _runtime_parent not in sys.path:\n",
                "    sys.path.insert(0, _runtime_parent)\n",
                "os.environ['PYTHONPATH'] = _runtime_parent + os.pathsep + os.environ.get('PYTHONPATH', '')\n",
                "requirements_path = REWARD_PACKAGE_DIR / 'requirements-colab.txt'\n",
                "!uv pip install -qqq -r {requirements_path}\n",
            ]
        if "open('depression_reward/prompts/prompts.jsonl')" in source:
            source = source.replace(
                "with open('depression_reward/prompts/prompts.jsonl') as f:",
                "with (REWARD_PACKAGE_DIR / 'prompts/prompts.jsonl').open(encoding='utf-8') as f:",
            )
            cell["source"] = source.splitlines(keepends=True)
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def check_notebook(manifest: dict) -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = json.dumps(notebook, ensure_ascii=False)
    if LEGACY_MARKER in text or "base64.b64decode" in text:
        raise ValueError("legacy embedded runtime remains in notebook")
    runtime_cells = [
        cell for cell in notebook["cells"] if cell.get("id") == RUNTIME_CELL_ID
    ]
    if len(runtime_cells) != 1:
        raise ValueError(f"expected one runtime cell, found {len(runtime_cells)}")
    runtime_source = "".join(runtime_cells[0]["source"])
    for expected in (manifest["artifact_version"], manifest["_runtime_sha256"]):
        if expected not in runtime_source:
            raise ValueError(f"notebook runtime pin missing: {expected}")
    introduction = "".join(notebook["cells"][0].get("source", []))
    expected_instruction = (
        f"`{ARTIFACT_NAME}` версии `{manifest['artifact_version']}`"
    )
    if expected_instruction not in introduction:
        raise ValueError("notebook introduction differs from the runtime pin")
    if "Path('/tmp/deprollm_reward_runtime')" not in runtime_source:
        raise ValueError("runtime must be extracted outside /kaggle/working")
    if "extractall('.')" in runtime_source:
        raise ValueError("runtime is still extracted into Kaggle output")
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for required_source in (
        "os.environ['PYTHONPATH']",
        "REWARD_PACKAGE_DIR / 'requirements-colab.txt'",
        "REWARD_PACKAGE_DIR / 'prompts/prompts.jsonl'",
    ):
        if required_source not in notebook_source:
            raise ValueError(f"notebook runtime wiring missing: {required_source}")
    dirty_cells = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("execution_count") is not None or cell.get("outputs")
    ]
    if dirty_cells:
        raise ValueError(f"notebook contains execution state in cells: {dirty_cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build a new Dataset version")
    build_parser.add_argument("version", help="new version, for example v2")
    build_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )

    for command in ("check", "pin"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("artifact_dir", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "build":
            upload_zip = build(args.version, args.output_root)
            print(f"created: {upload_zip}")
            print(f"sha256: {sha256_file(upload_zip)}")
        elif args.command == "pin":
            pin_notebook(args.artifact_dir)
            print(f"notebook pinned to: {args.artifact_dir}")
        else:
            manifest = validate_artifact(args.artifact_dir)
            check_notebook(manifest)
            print(
                "PASS:",
                manifest["artifact_version"],
                manifest["_runtime_sha256"],
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
