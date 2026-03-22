# API — Service de détection d'anomalies

API FastAPI asynchrone servant de point d’entrée HTTP pour le système de détection d’anomalies.

L'API :

1. reçoit les images envoyées par les clients

2. enregistre la tâche dans PostgreSQL

3. publie la tâche dans RabbitMQ

4. retourne immédiatement un `task_id`

Le worker 2D ou 3D consomme ensuite la tâche, exécute l’inférence et met à jour le résultat.
---

# Architecture

Client
  │
  │ HTTP
  ▼
API (FastAPI)
  │
  ├── PostgreSQL → stockage des tâches
  │
  └── RabbitMQ → file de traitement
          │
          ▼
      Worker 2D/3D → inférence 2D/3D
          │
          ▼
       MLflow → gestion et versionnement des modèles

---

# Endpoints

| Méthode | Route | Description |
|--------|------|-------------|
| `GET` | `/` | Interface HTML publique |
| `GET` | `/admin` | Interface d'administration |
| `POST` | `/predict/{pipeline}` | Soumet une image au pipeline `2d` ou `3d` |
| `GET` | `/tasks/{task_id}?pipeline=2d` | Consulte une tâche 2D |
| `GET` | `/tasks/{task_id}?pipeline=3d` | Consulte une tâche 3D |
| `POST` | `/admin/reload-model/{pipeline}` | Recharge le modèle du pipeline |
| `GET` | `/health` | Health global de la plateforme |
| `GET` | `/health/{pipeline}` | Health d'un pipeline spécifique |
| `GET` | `/ui/health` | Health global en HTML |
| `GET` | `/ui/health/{pipeline}` | Health pipeline en HTML |
| `GET` | `/ui/tasks/{task_id}?pipeline=2d` | Vue HTML d'une tâche 2D |
| `GET` | `/ui/tasks/{task_id}?pipeline=3d` | Vue HTML d'une tâche 3D |
| `GET` | `/metrics` | Métriques Prometheus |

---

# POST `/predict/{pipeline}`

Soumet une image à analyser.

### Pipelines disponibles

- `2d` : analyse d’images industrielles
- `3d` : analyse de données volumétriques

## Exemple 2D

```bash
curl -X POST "https://api.alexandremariolauret.org/predict/2d" \
  -F "file=@image.png"
```

## Exemple 3D

```bash
curl -X POST "https://api.alexandremariolauret.org/predict/3d" \
  -F "file=@image.png"
```

## Réponse :
```json
{
  "status": "queued",
  "message": "Task created successfully",
  "data": {
    "task_id": 127,
    "pipeline": "2d",
    "image_url": "http://images.alexandremariolauret.org/2d/uploads/20260316/abc123.png",
    "image_path": "/images_storage/2d/uploads/20260316/abc123.png",
    "category": "pill",
    "model_name": "resnet_knn_2d",
    "model_version": "v1"
  }
}
```
---

# GET /tasks/{task_id}

Permet de consulter le statut et le résultat d'une tâche.

Le pipeline doit être précisé.

## Exemple 2D

```bash
curl "https://api.alexandremariolauret.org/tasks/127?pipeline=2d"
```

## Exemple 3D

```bash
curl "https://api.alexandremariolauret.org/tasks/127?pipeline=3d"
```

## Réponse (tâche terminée) :
```json
{
  "status": "success",
  "data": {
    "id": 127,
    "pipeline": "2d",
    "status": "done",
    "task_type": "2d_anomaly",
    "image_url": "...",
    "category": "pill",
    "model_name": "resnet_knn_2d",
    "model_version": "v1",
    "anomaly_score": 0.42,
    "pred_label": "anomaly"
  }
}
```

Statuts possibles : `pending` → `running` → `done` / `failed`

---

# Admin — Rechargement des modèles

Permet de recharger le modèle MLflow dans un worker.

## Exemple

```bash
curl -X POST "https://api.alexandremariolauret.org/admin/reload-model/2d"
```

ou

```bash
curl -X POST "https://api.alexandremariolauret.org/admin/reload-model/3d"
```

 # Health

 
## Health global

```bash 
curl https://api.alexandremariolauret.org/health
```

## Health pipeline

```bash
curl https://api.alexandremariolauret.org/health/2d
curl https://api.alexandremariolauret.org/health/3d
```

Le health vérifie :

- PostgreSQL

- RabbitMQ

- Workers

- MLflow

- version du modèle en production

# Configuration (variables d'environnement)

Injectées via ConfigMap k8s `api-2d-config` :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `DB_HOST` | `postgres` | Hôte PostgreSQL |
| `DB_PORT` | `5432` | Port PostgreSQL |
| `DB_NAME` | `anomaly_detection` | Nom de la base |
| `DB_USER` | `admin` | Utilisateur DB |
| `DB_PASS` | `password` | Mot de passe DB |
| `RABBITMQ_HOST` | `rabbitmq` | Hôte RabbitMQ |
| `RABBITMQ_QUEUE` | `tasks_2d` | Nom de la queue |
| `IMAGES_STORAGE_ROOT` | `/images_storage` | Chemin PVC images |
| `IMAGES_PUBLIC_BASE` | `http://images.alexandremariolauret.org` | URL publique images |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | URI MLflow |
| `MLFLOW_MODEL_NAME` | `resnet_knn_2d` | Nom modèle MLflow Registry |
| `WORKER_ADMIN_URL` | `http://worker-2d:8080` | URL admin worker |

---

## Métriques Prometheus exposées

| Métrique | Type | Description |
|----------|------|-------------|
| `api_http_requests_total` | Counter | Requêtes par méthode/endpoint/status |
| `api_http_request_duration_seconds` | Histogram | Latence P50/P95/P99 |
| `api_tasks_created_total` | Counter | Tâches créées par catégorie |
| `api_tasks_by_status` | Gauge | Tâches en base par statut |
| `api_anomaly_score` | Histogram | Distribution des scores d'anomalie |
| `api_model_production_version` | Gauge | Version du modèle en production |

---

# Interfaces HTML

L'API expose des interfaces web:

| Route | Description |
|------|-------------|
| / | Interface utilisateur |
| /admin | Centre d'administration |
| /ui/tasks/{task_id}?pipeline=2d | Vue HTML tâche 2D |
| /ui/tasks/{task_id}?pipeline=3d | Vue HTML tâche 3D |
| /ui/health | Health global |
| /ui/health/2d | Health pipeline 2D |
| /ui/health/3d | Health pipeline 3D |

## Build et déploiement

```bash
cd api

# Rebuilder l'image
docker build -t tniauronis/api:vX .
docker push tniauronis/api:vX

# Déployer
kubectl set image deployment/api api=tniauronis/api:vX -n pfe
kubectl rollout status deployment/api -n pfe
```

---

## Dépendances

```
fastapi, uvicorn, pika, psycopg2-binary, sqlalchemy,
httpx, prometheus-client, python-multipart
```
