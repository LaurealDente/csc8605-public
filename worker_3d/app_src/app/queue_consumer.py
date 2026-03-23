# worker_3d/app_src/app/queue_consumer.py
"""
Consumer RabbitMQ pour le worker 3D.
Consomme la queue tasks_3d.

Routing automatique :
  - Si le payload contient xyz_filepath/depth_filepath ET un model_type="mm_patchcore"
    → cmd_predict_mm (Multimodal PatchCore)
  - Sinon
    → cmd_predict (V1 global k-NN)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from types import SimpleNamespace

import pika

from .config import Settings
from .main import cmd_predict, cmd_predict_mm


def _rabbitmq_url(settings: Settings) -> str:
    return (
        f"amqp://{settings.rabbitmq_user}:{settings.rabbitmq_password}"
        f"@{settings.rabbitmq_host}:{settings.rabbitmq_port}/%2F"
    )


def _should_use_mm(task: dict) -> bool:
    """
    Détermine si la tâche doit utiliser le pipeline multimodal.
    True si :
      - model_type == "mm_patchcore" (explicite)
      - OU une référence depth/xyz est présente dans le payload
    """
    if task.get("model_type", "").lower() == "mm_patchcore":
        return True

    depth_keys = [
        "xyz_filepath", "depth_filepath", "xyz_path", "depth_path",
    ]
    for key in depth_keys:
        val = task.get(key, "")
        if val and str(val).strip():
            return True

    return False


def main():
    try:
        from .server import tasks_processed_total, task_duration_seconds
        _has_metrics = True
    except Exception:
        _has_metrics = False

    settings = Settings.from_yaml("conf/config.yaml")
    params = pika.URLParameters(_rabbitmq_url(settings))
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=settings.rabbitmq_queue, durable=True)
    print(f"✅ [Worker 3D] Listening on queue={settings.rabbitmq_queue}")

    def on_message(channel, method, properties, body: bytes):
        tmp_path = None
        start = time.time()
        try:
            task = json.loads(body.decode("utf-8"))

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                json.dump(task, f, ensure_ascii=False, indent=2)
                tmp_path = f.name

            use_mm = _should_use_mm(task)

            if use_mm:
                # Multimodal PatchCore (V2)
                args = SimpleNamespace(
                    task_json=tmp_path,
                    config="/app/conf/config.yaml",
                    task_table=os.getenv("TASK_TABLE", "tasks_3d"),
                    model_dir=task.get("model_dir", None),
                    overlay_alpha=float(task.get("overlay_alpha", 0.45)),
                )
                print(f"[Worker 3D] → predict-mm (task_id={task.get('task_id')})")
                cmd_predict_mm(args)
            else:
                # V1 legacy
                args = SimpleNamespace(
                    task_json=tmp_path,
                    config="/app/conf/config.yaml",
                    task_table=os.getenv("TASK_TABLE", "tasks_3d"),
                )
                print(f"[Worker 3D] → predict V1 (task_id={task.get('task_id')})")
                cmd_predict(args)

            duration = time.time() - start
            if _has_metrics:
                tasks_processed_total.labels(status="done").inc()
                task_duration_seconds.observe(duration)
            channel.basic_ack(delivery_tag=method.delivery_tag)

        except Exception:
            print("❌ [Worker 3D] Task failed:\n", traceback.format_exc())
            if _has_metrics:
                tasks_processed_total.labels(status="failed").inc()
                task_duration_seconds.observe(time.time() - start)
            channel.basic_nack(
                delivery_tag=method.delivery_tag, requeue=False
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    ch.basic_qos(prefetch_count=1)
    ch.basic_consume(
        queue=settings.rabbitmq_queue, on_message_callback=on_message
    )
    ch.start_consuming()


if __name__ == "__main__":
    main()
