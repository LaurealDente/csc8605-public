import json
import pika

RABBIT_URL = "amqp://guest:guest@localhost:5672/"
QUEUE = "tasks_2d"

task = {
  "task_id": 2,
  "task_type": "2d_anomaly",
  "image_url": "http://images.exemple.com/mvtec_ad_2/rice/train/good/000_regular.png",
  "category": "rice",
  "model_name": "resnet_knn",
  "model_version": "v1",
  "output_prefix": "results/tasks/2"
}

conn = pika.BlockingConnection(pika.URLParameters(RABBIT_URL))
ch = conn.channel()
ch.queue_declare(queue=QUEUE, durable=True)

ch.basic_publish(
    exchange="",
    routing_key=QUEUE,
    body=json.dumps(task).encode("utf-8"),
    properties=pika.BasicProperties(delivery_mode=2),
)
print("✅ sent:", task["task_id"])
conn.close()