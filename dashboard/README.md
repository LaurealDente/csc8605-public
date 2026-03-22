# Dashboard — Monitoring Grafana

Dashboards Grafana préconfigurés pour le monitoring de la plateforme. Les fichiers JSON peuvent être importés directement dans Grafana via l'interface d'administration ou l'API.

---

## Dashboards disponibles

### Dashboard Infrastructure

Supervision de l'état du cluster Kubernetes et des services déployés :
- État des pods (Running, CrashLoopBackOff, Pending)
- Utilisation mémoire et CPU par pod
- Nombre de restarts par conteneur
- Latence des requêtes HTTP de l'API

### Dashboard Pipeline 2D

Monitoring du pipeline de détection d'anomalies 2D :
- Nombre de tâches traitées (done/failed)
- Durée moyenne de traitement par tâche
- Distribution des scores d'anomalie
- Version du modèle en cache dans le worker

### Dashboard Pipeline 3D

Monitoring du pipeline 3D Multimodal PatchCore :
- Tâches 3D traitées par statut
- Durée de traitement (plus long que le 2D en raison de la double extraction RGB + Depth)
- Distribution des scores d'anomalie 3D
- Version du modèle MM-PatchCore en cache

---

## Source de données

Les dashboards utilisent **Prometheus** comme source de données, qui collecte les métriques depuis :
- L'API FastAPI (`api-service:8000/metrics`)
- Le Worker 2D (`worker-2d-service:8080/metrics`)
- Le Worker 3D (`worker-3d-service:8080/metrics`)
- kube-state-metrics (état du cluster Kubernetes)
- postgres-exporter (métriques PostgreSQL)

---

## Importation

### Via l'interface Grafana

1. Ouvrir `https://grafana.alexandremariolauret.org`
2. Aller dans **Dashboards → Import**
3. Coller le contenu du fichier JSON ou uploader le fichier
4. Sélectionner la source de données Prometheus
5. Cliquer sur **Import**

### Via l'API Grafana

```bash
curl -X POST https://grafana.alexandremariolauret.org/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API_KEY>" \
  -d @dashboard/dash_3d.json
```

---

## Structure

```
dashboard/
├── dash.json          # Dashboard combiné (infrastructure + pipelines)
└── README.md
```

---

<!-- 📸 CAPTURE D'ÉCRAN : Dashboard Infrastructure Grafana
     → Ouvrir https://grafana.alexandremariolauret.org
     → Sélectionner le dashboard Infrastructure
     → Capturer le dashboard complet avec les panneaux de pods et métriques -->

<!-- 📸 CAPTURE D'ÉCRAN : Dashboard Pipeline 3D Grafana
     → Sélectionner le dashboard Pipeline 3D
     → Capturer les panneaux montrant les tâches traitées et les scores -->
