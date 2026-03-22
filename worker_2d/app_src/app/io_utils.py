from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def write_result(output_dir: str | Path, result: Dict[str, Any]) -> str:
    """
    Write result.json inside output_dir.

    Adds:
        - written_at (UTC ISO 8601)
    Returns:
        Absolute normalized path to result.json
    """

    if not isinstance(result, dict):
        raise ValueError("Result must be a dictionary.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "result.json"

    # Copy result to avoid mutating caller dict
    result_data = dict(result)
    result_data["written_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    # Normalize path (important for Windows Docker cases)
    return str(result_path.resolve()).replace("\\", "/")