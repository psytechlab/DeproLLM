"""Собирает и встраивает depression_reward.zip в grpo_depression.ipynb.

Архив является игнорируемым транспортным артефактом. Он всегда собирается
заново из исходного пакета без кэшей и альтернативной joblib-копии модели.
Ячейка вставляется перед распаковкой и восстанавливает архив в Kaggle/Colab.
Запускать после каждого изменения depression_reward/:

    .venv/bin/python embed_zip_in_notebook.py
"""
import base64
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).parent
ZIP = ROOT / "depression_reward.zip"
NOTEBOOK = ROOT / "grpo_depression.ipynb"
MARKER = "# [embed_zip_in_notebook.py]"
LINE_LEN = 200

EXCLUDED_NAMES = {"model_repacked.joblib", "rebuild_model.py"}


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT / "depression_reward")
    return (
        path.is_file()
        and "__pycache__" not in relative.parts
        and path.suffix not in {".pyc", ".pyo"}
        and path.name not in EXCLUDED_NAMES
    )


with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((ROOT / "depression_reward").rglob("*")):
        if include(path):
            archive.write(path, path.relative_to(ROOT))

data = ZIP.read_bytes()
sha = hashlib.sha256(data).hexdigest()
b64 = base64.b64encode(data).decode()
chunks = [b64[i:i + LINE_LEN] for i in range(0, len(b64), LINE_LEN)]

source_text = (
    f"{MARKER} depression_reward.zip → base64, не редактировать вручную\n"
    "import base64, hashlib, os, shutil\n"
    f"_SHA = '{sha}'\n"
    "_have = os.path.exists('depression_reward.zip') and hashlib.sha256(\n"
    "    open('depression_reward.zip', 'rb').read()).hexdigest() == _SHA\n"
    "if _have:\n"
    "    print('актуальный depression_reward.zip уже в рантайме')\n"
    "else:\n"
    "    _B64 = (\n"
    + "\n".join(f'\"{c}\"' for c in chunks)
    + "\n    )\n"
    "    _data = base64.b64decode(_B64)\n"
    "    assert hashlib.sha256(_data).hexdigest() == _SHA, 'повреждён встроенный zip'\n"
    "    with open('depression_reward.zip', 'wb') as _f:\n"
    "        _f.write(_data)\n"
    "    if os.path.isdir('depression_reward'):\n"
    "        shutil.rmtree('depression_reward')  # устаревшая распаковка, пересоздастся ниже\n"
    "    print('depression_reward.zip обновлён из ноутбука:', len(_data), 'байт')\n"
)
lines = source_text.splitlines(keepends=True)

cell = {
    "cell_type": "code",
    "id": "embed-depression-reward-zip",
    "metadata": {"jupyter": {"source_hidden": True}},
    "execution_count": None,
    "outputs": [],
    "source": lines,
}

nb = json.loads(NOTEBOOK.read_text())
cells = nb["cells"]
old = [i for i, c in enumerate(cells) if MARKER in "".join(c["source"])[:200]]
if old:
    cells[old[0]] = cell
    action = f"обновлена ячейка {old[0]}"
else:
    unpack = next(
        i for i, c in enumerate(cells)
        if c["cell_type"] == "code" and "zipfile" in "".join(c["source"])
    )
    cells.insert(unpack, cell)
    action = f"вставлена ячейка {unpack} (перед распаковкой)"

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
print(f"{action}; zip {len(data)} байт, sha256 {sha[:12]}…, всего ячеек {len(cells)}")
