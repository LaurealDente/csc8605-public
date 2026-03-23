from __future__ import annotations

import json
from typing import Any, Dict

import pika

from config import (
    RABBIT_HOST,
    RABBIT_PASS,
    RABBIT_PORT,
    RABBIT_USER,
    Pipeline,
    get_pipeline_config,
)


def publish_task(pipeline: Pipeline, task_payload: Dict[str, Any]) -> None:
    cfg = get_pipeline_config(pipeline)
    rabbit_queue = cfg["rabbit_queue"]

    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    params = pika.ConnectionParameters(
        host=RABBIT_HOST,
        port=RABBIT_PORT,
        credentials=creds,
        heartbeat=600,
        blocked_connection_timeout=300,
    )

    conn = pika.BlockingConnection(params)
    try:
        ch = conn.channel()
        ch.queue_declare(queue=rabbit_queue, durable=True)
        ch.basic_publish(
            exchange="",
            routing_key=rabbit_queue,
            body=json.dumps(task_payload, ensure_ascii=False).encode("utf-8"),
            properties=pika.BasicProperties(delivery_mode=2),
            mandatory=False,
        )
    finally:
        conn.close()


def check_rabbit_connection(pipeline: Pipeline) -> str:
    cfg = get_pipeline_config(pipeline)
    rabbit_queue = cfg["rabbit_queue"]

    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    params = pika.ConnectionParameters(
        host=RABBIT_HOST,
        port=RABBIT_PORT,
        credentials=creds,
        heartbeat=600,
        blocked_connection_timeout=300,
    )

    try:
        conn = pika.BlockingConnection(params)
        try:
            ch = conn.channel()
            ch.queue_declare(queue=rabbit_queue, durable=True)
        finally:
            conn.close()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"