# Training 2D — PatchCore ResNet18

Pipeline d'entraînement pour la détection d'anomalies **2D** basé sur l'algorithme **PatchCore** avec un backbone **ResNet18** pré-entraîné sur ImageNet. Ce module construit une banque d'embeddings à partir des images normales du dataset, puis effectue l'inférence par recherche k-NN dans l'espace des features.

---

## Algorithme

**PatchCore** extrait des features à partir des couches intermédiaires d'un réseau pré-entraîné (ResNet18), puis construit une **memory bank** (banque de référence) à partir des patchs des images normales. Lors de l'inférence, le score d'anomalie est calculé comme la distance au plus proche voisin dans cette banque.

```
Image RGB ──▶ ResNet18 (couches 2-3) ──▶ Feature patches
                                              │
                          ┌───────────────────▼──────────────────┐
                          │  Entraînement : stockage dans la     │
                          │  memory bank (coreset sampling)      │
                          └──────────────────────────────────────┘
                          ┌───────────────────▼──────────────────┐
                          │  Inférence : distance k-NN           │
                          │  → anomaly_score, pred_label         │
                          └──────────────────────────────────────┘
```

---

## Structure

```
training/
├── src/
│   ├── main.py              # Point d'entrée CLI (fit, predict)
│   ├── inference.py          # PatchCore : extraction features, fit, predict
│   ├── patch_inference.py    # Inférence par patchs avec heatmap
│   ├── data.py               # Chargement des données MVTec
│   ├── mlflow_loader.py      # Intégration MLflow (logging, registry)
│   ├── config.py             # Configuration (Settings depuis YAML)
│   ├── db.py                 # Accès PostgreSQL (mise à jour des tâches)
│   ├── io_utils.py           # Écriture des résultats JSON
│   ├── __init__.py
│   └── __main__.py           # python -m training
├── requirements.txt
└── README.md
```

---

## Commandes CLI

### Entraînement (`fit`)

Construit la memory bank à partir des images normales :

```bash
python -m training.src fit \
  --config conf/config.yaml \
  --table-name mvtec_anomaly_detection \
  --output-model-dir models/resnet_knn_2d_v1 \
  --backbone resnet18 \
  --batch-size 64 \
  --num-workers 4
```

Les artefacts générés (embeddings, seuils, métriques) sont sauvegardés dans `--output-model-dir` et enregistrés dans **MLflow**.

### Inférence (`predict`)

Exécute la prédiction sur une tâche décrite par un fichier JSON :

```bash
python -m training.src predict \
  --task-json /tmp/task_42.json \
  --config conf/config.yaml
```

Format du `task.json` :
```json
{
  "task_id": 42,
  "image_url": "http://images.exemple.com/mvtec/bottle/test/broken/000.png",
  "category": "bottle",
  "model_name": "resnet_knn_2d",
  "model_version": "v1"
}
```

---

## Intégration MLflow

Le pipeline enregistre automatiquement dans MLflow :

- **Paramètres** : backbone, batch_size, image_size, table_name
- **Métriques** : durée d'entraînement, taille de la banque, métriques d'évaluation (AUROC, F1, precision, recall)
- **Artefacts** : `embeddings.npy`, `threshold.json`, `selection.json`
- **Model Registry** : le modèle est enregistré et peut être promu en `Production`

---

## Dépendances principales

PyTorch, torchvision, scikit-learn, NumPy, Pillow, SQLAlchemy, psycopg2, MLflow, pika, PyYAML.
