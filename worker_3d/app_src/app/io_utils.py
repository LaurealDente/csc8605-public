# training_3d/src/io_utils.py
"""I/O helpers — identique à la version 2D."""

from __future__ import annotations

import json
from pathlib import Path


def write_result(output_dir: Path, result: dict) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(result_path.resolve())
