# Worker 2D — Inférence PatchCore

Worker asynchrone consommant les tâches de prédiction **2D** depuis la file RabbitMQ `tasks_2d`. Chaque message déclenche l'inférence PatchCore (ResNet18 + k-NN), met à jour le statut de la tâche en base PostgreSQL, et expose des métriques Prometheus.

---

## Architecture

```
RabbitMQ (tasks_2d)                              PostgreSQL
       │                                              ▲
       │ consume                              update  │
       ▼                                              │
┌──────────────────────────────────────────────────────┤
│                   Worker 2D                          │
│                                                      │
│  ┌────────────────┐    ┌──────────────────────────┐  │
│  │ Queue Consumer  │──▶│  cmd_predict (main.py)   │  │
│  │ (pika)         │    │  ├─ load model (MLflow)  │  │
│  └────────────────┘    │  ├─ predict (inference)  │  │
│                        │  └─ write result (JSON)  │  │
│  ┌────────────────┐    └──────────────────────────┘  │
│  │ HTTP Server     │  FastAPI :8080                   │
│  │ /health         │  ← health check                 │
│  │ /reload-model   │  ← invalidation cache           │
│  │ /metrics        │  ← Prometheus                   │
│  └────────────────┘                                  │
└──────────────────────────────────────────────────────┘
```

Le worker fonctionne en **deux threads** :
1. **Thread principal** : consumer RabbitMQ, traitement séquentiel des tâches (prefetch=1)
2. **Thread HTTP** : serveur FastAPI sur le port 8080 pour l'administration et le monitoring

---

## Structure

```
worker_2d/
├── app_src/
│   ├── app/
│   │   ├── main.py              # Logique métier (cmd_predict, cmd_fit)
│   │   ├── queue_consumer.py    # Consumer RabbitMQ
│   │   ├── server.py            # Serveur HTTP admin (health, metrics)
│   │   ├── inference.py         # PatchCore : chargement bank, prédiction
│   │   ├── patch_inference.py   # Inférence par patchs avec heatmap
│   │   ├── data.py              # Chargement des images MVTec
│   │   ├── config.py            # Configuration (Settings depuis YAML)
│   │   ├── db.py                # Accès PostgreSQL
│   │   ├── io_utils.py          # Écriture résultats
│   │   ├── eval_test.py         # Évaluation sur le split test
│   │   └── train_finetune.py    # Fine-tuning du backbone (optionnel)
│   ├── conf/
│   │   └── config.yaml          # Configuration embarquée
│   ├── Dockerfile
│   └── requirements.txt
└── README.md
```

---

## Endpoints d'administration

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Statut du worker, version du modèle en cache, connexion MLflow |
| `POST` | `/reload-model` | Vide le cache modèle local (force re-téléchargement depuis MLflow) |
| `GET` | `/metrics` | Métriques Prometheus |

---

## Métriques Prometheus

| Métrique | Type | Description |
|----------|------|-------------|
| `worker_tasks_processed_total` | Counter | Nombre de tâches traitées (par statut : done/failed) |
| `worker_task_duration_seconds` | Histogram | Durée de traitement par tâche |
| `worker_anomaly_score` | Histogram | Distribution des scores d'anomalie |
| `worker_model_cache_version` | Gauge | Version du modèle actuellement en cache |

---

## Build et déploiement

```bash
cd worker_2d/app_src/
docker build -t laurealdente/worker-2d:latest .
docker push laurealdente/worker-2d:latest

kubectl set image deployment/worker-2d worker-2d=laurealdente/worker-2d:latest -n pfe
```

---

## Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `MLFLOW_TRACKING_URI` | URL du serveur MLflow | `http://mlflow-service:5000` |
| `MLFLOW_MODEL_NAME` | Nom du modèle dans le registry | `resnet_knn_2d` |
| `MODEL_CACHE_DIR` | Répertoire de cache des modèles | `/tmp/mlflow_model_cache` |
| `USE_MLFLOW_REGISTRY` | Activer le chargement depuis MLflow | `true` |
| `TASK_TABLE` | Nom de la table des tâches en base | `tasks_2d` |


