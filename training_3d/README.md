# Training 3D — Multimodal PatchCore (RGB + Depth)

Pipeline d'entraînement pour la détection d'anomalies **3D multimodal** basé sur l'algorithme **Multimodal PatchCore**. Ce module combine les informations **RGB** et de **profondeur (depth maps)** extraites du dataset MVTec 3D-AD pour détecter des anomalies sur des objets industriels en 3D.

Le pipeline prend en charge deux modes :
- **ResNet + k-NN** (`fit` / `predict`) — approche mono-modale sur les images RGB uniquement
- **Multimodal PatchCore** (`fit-mm` / `eval-mm` / `predict-mm`) — fusion RGB + Depth avec late fusion

---

## Algorithme — Multimodal PatchCore

Le Multimodal PatchCore étend PatchCore en traitant **deux modalités en parallèle** :

```
Image RGB ──▶ ResNet18 (feature extractor) ──▶ RGB patches ──▶ Coreset ──┐
                                                                          │
                                                              Late Fusion │──▶ Score anomalie
                                                              (α=0.5)    │
Depth map ──▶ ResNet18 (feature extractor) ──▶ Depth patches ──▶ Coreset ─┘
```

1. **Extraction de features** : deux backbones ResNet18 (un par modalité) extraient des feature maps multi-échelles (couches 2 et 3).
2. **Coreset subsampling** : réduction de la memory bank par sélection aléatoire (60 000 patchs par modalité).
3. **Late fusion** : les scores d'anomalie de chaque modalité sont combinés par moyenne pondérée (`α_rgb = 0.5`, `α_depth = 0.5`).
4. **Détection** : le score combiné est comparé à un seuil calibré sur les données de validation.

### Conversion des cartes de profondeur

Les fichiers `.tiff` du MVTec 3D-AD contiennent des coordonnées XYZ (3 canaux). Le pipeline extrait le **canal Z** (hauteur) et le convertit en image en niveaux de gris normalisée sur [0, 255], puis la duplique en 3 canaux pour l'adapter au backbone ResNet18.

---

## Structure

```
training_3d/
├── src/
│   ├── main.py                  # CLI : fit, predict, fit-mm, eval-mm, predict-mm
│   ├── multimodal_patchcore.py  # Classe MultimodalPatchCore (fit, predict, score)
│   ├── eval_mm_patchcore.py     # Évaluation complète (image-level, pixel-level, per-category)
│   ├── inference.py             # PatchCore mono-modal (ResNet + k-NN)
│   ├── mlflow_loader.py         # Intégration MLflow (fit, eval, artefacts, registry)
│   ├── data.py                  # Chargement MVTec 3D-AD (RGB + depth maps HTTP/cache)
│   ├── config.py                # Configuration (Settings)
│   ├── db.py                    # Accès PostgreSQL
│   ├── io_utils.py              # Écriture des résultats
│   ├── __init__.py
│   └── __main__.py
├── requirements.txt
└── README.md
```

---

## Commandes CLI

### Entraînement Multimodal PatchCore (`fit-mm`)

```bash
python -m training_3d.src fit-mm \
  --config conf/config.yaml \
  --table-name mvtec_3d_anomaly_detection \
  --output-model-dir models/mm_patchcore_v1 \
  --image-size 224 \
  --rgb-bank-size 60000 \
  --depth-bank-size 60000 \
  --n-neighbors 1
```

Ce processus :
1. Charge toutes les images normales (train split) du MVTec 3D-AD
2. Extrait les features RGB et Depth avec deux ResNet18
3. Effectue un coreset subsampling sur chaque banque
4. Calibre le seuil de décision sur les images de validation
5. Sauvegarde le modèle et enregistre le run dans MLflow

### Évaluation (`eval-mm`)

```bash
python -m training_3d.src eval-mm \
  --config conf/config.yaml \
  --model-dir models/mm_patchcore_v1 \
  --table-name mvtec_3d_anomaly_detection \
  --split test
```

Produit un rapport complet : AUROC, AP, F1, precision, recall (image-level et pixel-level), ainsi que les métriques par catégorie d'objet.

### Inférence (`predict-mm`)

```bash
python -m training_3d.src predict-mm \
  --task-json /tmp/task_42.json \
  --config conf/config.yaml
```

---

## Résultats obtenus

Évaluation sur le split test de MVTec 3D-AD (1 197 images, 10 catégories) :

| Métrique | Valeur |
|----------|--------|
| Image AUROC | **0.701** |
| Image AP | **0.903** |
| Best F1 (seuil optimal) | **0.894** |
| Pixel AUROC | **0.960** |

Résultats détaillés par catégorie :

| Catégorie | AUROC | AP | N |
|-----------|-------|----|---|
| Rope | 0.951 | 0.979 | 101 |
| Bagel | 0.932 | 0.984 | 110 |
| Cookie | 0.896 | 0.973 | 131 |
| Dowel | 0.847 | 0.960 | 130 |
| Carrot | 0.776 | 0.947 | 159 |
| Peach | 0.753 | 0.909 | 132 |
| Cable Gland | 0.696 | 0.906 | 108 |
| Foam | 0.666 | 0.897 | 100 |
| Potato | 0.625 | 0.857 | 114 |
| Tire | 0.625 | 0.846 | 112 |

> Ces résultats reproduisent fidèlement ceux de l'implémentation de référence (écart < 0.002 sur toutes les métriques), validant la bonne intégration du MM-PatchCore dans le pipeline.

---

## Intégration MLflow

Le module enregistre dans MLflow :

- **Paramètres** : image_size, alpha_rgb, alpha_depth, n_neighbors, rgb_bank_size, depth_bank_size, use_late_fusion, use_multiscale
- **Métriques** : durée d'entraînement, toutes les métriques d'évaluation (image-level, pixel-level, per-category)
- **Artefacts** : `rgb_bank.pt`, `depth_bank.pt`, `threshold.json`, `calibration_stats.json`
- **Model Registry** : enregistrement automatique du modèle `mm_patchcore_3d`

![alt text](../img/mlflow_3D_versionning.png)

---

## Dépendances principales

PyTorch, torchvision, scikit-learn, NumPy, Pillow, tifffile, MLflow, SQLAlchemy, psycopg2, PyYAML.
