"""Static checks for the committed GRPO notebook.

The checks are intentionally CPU-only: they catch accidental full/smoke mode
changes, execution residue, ordering regressions, and incomplete run artifacts
before the notebook is sent to Kaggle.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "grpo_depression.ipynb"


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def python_only(cell_source: str) -> str:
    """Replace IPython-only lines while preserving Python block structure."""
    lines = []
    for line in cell_source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            indent = line[: len(line) - len(stripped)]
            lines.append(indent + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    require(isinstance(cells, list) and cells, "notebook has no cells")

    ids = [cell.get("id") for cell in cells]
    require(all(isinstance(cell_id, str) and cell_id for cell_id in ids),
            "every cell must have an id")
    require(len(ids) == len(set(ids)), "cell ids must be unique")

    dirty = [
        index for index, cell in enumerate(cells)
        if cell.get("execution_count") is not None or cell.get("outputs")
    ]
    require(not dirty, f"execution state remains in cells: {dirty}")

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    for index, cell in enumerate(code_cells):
        try:
            ast.parse(python_only(source(cell)))
        except SyntaxError as error:
            raise ValueError(
                f"invalid Python in code cell {cell.get('id')} ({index}): {error}"
            ) from error

    all_source = "\n".join(source(cell) for cell in cells)
    code_source = "\n".join(source(cell) for cell in code_cells)
    require("base64.b64decode" not in all_source,
            "embedded base64 runtime returned")
    require("# [embed_zip_in_notebook.py]" not in all_source,
            "legacy embedded-runtime marker returned")
    require(code_source.count("SMOKE_RUN = False") == 1,
            "committed notebook must contain exactly one SMOKE_RUN = False")
    require("SMOKE_RUN = True" not in code_source,
            "never commit the temporary smoke setting")

    required_fragments = (
        "N_BASELINE = 4 if SMOKE_RUN else 30",
        "EVAL_PROMPT_LIMIT = 2 if SMOKE_RUN else 10",
        "EVAL_SEEDS = [3407] if SMOKE_RUN else [3407, 3408, 3409]",
        "GRPO_MAX_STEPS = 2 if SMOKE_RUN else 300",
        "GPU_CAPABILITY >= (7, 0)",
        "training_config = training_args.to_dict()",
        "'__type__': f'{value_type.__module__}.{value_type.__qualname__}'",
        "baseline_eval_essays.jsonl",
        "grpo_eval_essays.jsonl",
        "eval_pairs.jsonl",
        "paired_comparison.json",
        "trainer_state.json",
        "artifact_sha256",
    )
    for fragment in required_fragments:
        require(fragment in all_source, f"missing notebook invariant: {fragment}")

    baseline_position = all_source.find(
        "baseline_eval, eval_pairs = generate_eval(eval_run_prompts)"
    )
    training_position = all_source.find("train_result = trainer.train()")
    trained_eval_position = all_source.find("trained_eval, trained_pairs = generate_eval(")
    require(-1 not in (baseline_position, training_position, trained_eval_position),
            "baseline, training, or trained eval stage is missing")
    require(baseline_position < training_position < trained_eval_position,
            "held-out baseline must run before training and trained eval")

    require("Welch" not in all_source, "unapproved Welch inference returned")
    require(not re.search(r"[|]t[|].{0,40}значим", all_source, re.IGNORECASE),
            "notebook still declares significance from a t threshold")

    print(
        "PASS: notebook JSON, Python syntax, clean state, full-mode default, "
        "stage ordering, and artifact invariants"
    )


if __name__ == "__main__":
    main()
