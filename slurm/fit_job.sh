#!/bin/bash
#SBATCH --job-name=fit_2d
#SBATCH --output=logs/fit_2d_%j.out
#SBATCH --error=logs/fit_2d_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu

# ============================================================
# Entraînement PatchCore 2D (ResNet18 + k-NN)
#
# Usage :
#   sbatch fit_job.sh [CATEGORY]
#
# Exemples :
#   sbatch fit_job.sh                  # Toutes les catégories
#   sbatch fit_job.sh bottle           # Catégorie bottle seule
#   sbatch fit_job.sh metal_nut        # Catégorie metal_nut
# ============================================================

set -euo pipefail

CATEGORY="${1:-}"

# Charger l'environnement (conda/venv)
source slurm/setup_env.sh

# Créer le dossier de logs
mkdir -p logs

# Config
CONFIG="conf/config.yaml"
TABLE_NAME="mvtec_anomaly_detection"

if [ -n "$CATEGORY" ]; then
    MODEL_DIR="models/resnet_knn_2d_${CATEGORY}_v1"
    CATEGORY_FLAG="--category $CATEGORY"
    echo "=== Entraînement PatchCore 2D — catégorie: $CATEGORY ==="
else
    MODEL_DIR="models/resnet_knn_2d_v1"
    CATEGORY_FLAG=""
    echo "=== Entraînement PatchCore 2D — toutes catégories ==="
fi

echo "  Config      : $CONFIG"
echo "  Table       : $TABLE_NAME"
echo "  Model dir   : $MODEL_DIR"
echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU')"
echo ""

# Lancer le fit
python -m training.src fit \
    --config "$CONFIG" \
    --table-name "$TABLE_NAME" \
    --output-model-dir "$MODEL_DIR" \
    --backbone resnet18 \
    --batch-size 64 \
    --num-workers 8 \
    $CATEGORY_FLAG

echo ""
echo "=== Entraînement terminé ==="
echo "  Modèle sauvé dans : $MODEL_DIR"
echo "  Fichiers :"
ls -la "$MODEL_DIR"/ 2>/dev/null || echo "  (répertoire non trouvé)"
echo ""
echo "  → Aller sur l'UI MLflow pour promouvoir en Production."