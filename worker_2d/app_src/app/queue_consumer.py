from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from types import SimpleNamespace

import pika

from .config import Settings
from .main import cmd_predict


def _rabbitmq_url(settings: Settings) -> str:
    return f"amqp://{settings.rabbitmq_user}:{settings.rabbitmq_password}@{settings.rabbitmq_host}:{settings.rabbitmq_port}/%2F"


def main():
    try:
        from .server import tasks_processed_total, task_duration_seconds, anomaly_score_histogram
        _has_metrics = True
    except Exception:
        _has_metrics = False

    settings = Settings.from_yaml("conf/config.yaml")
    params = pika.URLParameters(_rabbitmq_url(settings))
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=settings.rabbitmq_queue, durable=True)
    print(f"✅ Listening on queue={settings.rabbitmq_queue}")

    def on_message(channel, method, properties, body: bytes):
        tmp_path = None
        start = time.time()
        try:
            task = json.loads(body.decode("utf-8"))
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
                json.dump(task, f, ensure_ascii=False, indent=2)
                tmp_path = f.name
            args = SimpleNamespace(
                task_json=tmp_path,
                config="/app/conf/config.yaml",
                task_table=os.getenv("TASK_TABLE", "tasks_2d"),
            )
            cmd_predict(args)
            duration = time.time() - start
            if _has_metrics:
                tasks_processed_total.labels(status="done").inc()
                task_duration_seconds.observe(duration)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            print("❌ Task failed:\n", traceback.format_exc())
            if _has_metrics:
                tasks_processed_total.labels(status="failed").inc()
                task_duration_seconds.observe(time.time() - start)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    ch.basic_qos(prefetch_count=1)
    ch.basic_consume(queue=settings.rabbitmq_queue, on_message_callback=on_message)
    ch.start_consuming()


if __name__ == "__main__":
    main()
