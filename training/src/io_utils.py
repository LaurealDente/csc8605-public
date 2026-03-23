from __future__ import annotations

import json
from pathlib import Path


def write_result(output_dir: Path, result: dict) -> str:
    """
    Écrit le résultat d'inférence dans un fichier result.json.

    Paramètres
    ----------
    output_dir : Path
        Répertoire de sortie de la tâche.
    result : dict
        Dictionnaire contenant les résultats (score, label, etc.).

    Retourne
    --------
    str
        Chemin absolu du fichier result.json créé.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return str(result_path.resolve())
