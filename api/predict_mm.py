# api/predict_mm.py
"""
⭐ Endpoint multimodal 3D : accepte RGB + Depth (.tiff) en upload.
Publier la tâche sur la queue tasks_3d avec xyz_filepath → le worker route vers predict-mm.

Ce fichier est un router FastAPI à inclure dans main.py via :
    from predict_mm import router as mm_router
    app.include_router(mm_router)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    IMAGES_PUBLIC_BASE,
    IMAGES_STORAGE_ROOT,
    MAX_UPLOAD_SIZE_BYTES,
    Pipeline,
    get_pipeline_config,
)
from db import insert_task_mm, update_task_status
from rabbitmq import publish_task

router = APIRouter()


def _validate_extension(filename: Optional[str]) -> str:
    suffix = Path(filename).suffix.lower() if filename else ".png"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extension non supportée: {suffix}.",
        )
    return suffix


async def _save_file(file: UploadFile, pipeline: Pipeline, subdir: str) -> Dict[str, str]:
    suffix = _validate_extension(file.filename)
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Fichier vide.")
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

    fname = f"{uuid.uuid4().hex}{suffix}"
    date_prefix = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    rel_path = f"{pipeline.value}/{subdir}/{date_prefix}/{fname}"

    abs_path = Path(IMAGES_STORAGE_ROOT) / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(content)

    image_url = f"{IMAGES_PUBLIC_BASE.rstrip('/')}/{rel_path}"

    return {
        "path": str(abs_path),
        "url": image_url,
        "relative_path": rel_path,
        "filename": fname,
    }


@router.post("/predict/3d-mm", tags=["Prediction"])
async def predict_3d_mm(
    rgb_file: UploadFile = File(..., description="Image RGB (.png, .jpg)"),
    depth_file: UploadFile = File(..., description="Depth map (.tiff)"),
    category: Optional[str] = Form(None),
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    ⭐ Prédiction 3D multimodale (RGB + Depth).

    Upload deux fichiers :
    - **rgb_file** : image RGB (.png, .jpg)
    - **depth_file** : depth map (.tiff)

    Le worker utilise le modèle Multimodal PatchCore pour la détection
    d'anomalies avec fusion tardive RGB + Depth et génération de heatmaps.
    """
    pipeline = Pipeline.three_d
    cfg = get_pipeline_config(pipeline)

    category = category or cfg["category_default"]
    model_name = model_name or cfg.get("model_name_mm_default", "mm_patchcore_3d")
    model_version = model_version or cfg.get("model_version_mm_default", "v1")

    # Sauvegarder les deux fichiers
    saved_rgb = await _save_file(rgb_file, pipeline, "uploads")
    saved_depth = await _save_file(depth_file, pipeline, "uploads_depth")

    # Insérer en DB
    try:
        task_id = insert_task_mm(
            pipeline=pipeline,
            image_path=saved_rgb["path"],
            image_url=saved_rgb["url"],
            depth_path=saved_depth["path"],
            depth_url=saved_depth["url"],
            category=category,
            model_name=model_name,
            model_version=model_version,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Payload RabbitMQ — xyz_filepath déclenche le routing vers predict-mm
    task_payload: Dict[str, Any] = {
        "task_id": task_id,
        "task_type": cfg.get("task_type_mm", "3d_anomaly_mm"),
        "model_type": "mm_patchcore",
        "pipeline": pipeline.value,
        "image_url": saved_rgb["url"],
        "image_path": saved_rgb["path"],
        "xyz_filepath": saved_depth["path"],
        "depth_filepath": saved_depth["path"],
        "category": category,
        "model_name": model_name,
        "model_version": model_version,
        "output_prefix": f"{cfg['output_prefix']}/{task_id}",
        "submitted_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    try:
        publish_task(pipeline, task_payload)
    except Exception as exc:
        try:
            update_task_status(
                pipeline=pipeline,
                task_id=task_id,
                status="failed",
                error_message=f"Publication RabbitMQ échouée: {exc}",
            )
        except Exception:
            pass
        raise HTTPException(status_code=503, detail=f"Erreur RabbitMQ : {exc}") from exc

    return {
        "status": "queued",
        "message": "Tâche 3D multimodale créée avec succès.",
        "data": {
            "task_id": task_id,
            "pipeline": "3d",
            "model_type": "mm_patchcore",
            "image_url": saved_rgb["url"],
            "depth_url": saved_depth["url"],
            "category": category,
            "model_name": model_name,
            "model_version": model_version,
        },
    }
