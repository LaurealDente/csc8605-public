#!/bin/bash
#SBATCH --job-name=fit_mm_3d
#SBATCH --output=logs/fit_mm_3d_%j.out
#SBATCH --error=logs/fit_mm_3d_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu

# ============================================================
# Entraînement Multimodal PatchCore 3D (RGB + Depth)
#
# Usage :
#   sbatch fit_mm_job.sh [CATEGORY]
#
# Exemples :
#   sbatch fit_mm_job.sh                  # Toutes les catégories
#   sbatch fit_mm_job.sh bagel            # Catégorie bagel seule
#   sbatch fit_mm_job.sh cable_gland      # Catégorie cable_gland
# ============================================================

set -euo pipefail

CATEGORY="${1:-}"

# Charger l'environnement (conda/venv)
source slurm/setup_env.sh

# Créer le dossier de logs
mkdir -p logs

# Config
CONFIG="conf/config.yaml"
TABLE_NAME="mvtec_3d_anomaly_detection"

if [ -n "$CATEGORY" ]; then
    MODEL_DIR="models/mm_patchcore_3d_${CATEGORY}_v1"
    CATEGORY_FLAG="--category $CATEGORY"
    echo "=== Entraînement MM-PatchCore 3D — catégorie: $CATEGORY ==="
else
    MODEL_DIR="models/mm_patchcore_3d_all_v1"
    CATEGORY_FLAG=""
    echo "=== Entraînement MM-PatchCore 3D — toutes catégories ==="
fi

echo "  Config      : $CONFIG"
echo "  Table       : $TABLE_NAME"
echo "  Model dir   : $MODEL_DIR"
echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU')"
echo ""

# Lancer le fit
python -m training_3d.src fit-mm \
    --config "$CONFIG" \
    --table-name "$TABLE_NAME" \
    --model-dir "$MODEL_DIR" \
    --fit-split train \
    --val-split validation \
    --normal-only \
    --alpha-rgb 0.5 \
    --alpha-depth 0.5 \
    --k 1 \
    --max-patches 200000 \
    --coreset-pre-sample-size 60000 \
    --coreset-proj-dim 128 \
    --image-size 224 \
    $CATEGORY_FLAG

echo ""
echo "=== Entraînement terminé ==="
echo "  Modèle sauvé dans : $MODEL_DIR"
echo "  Fichiers :"
ls -la "$MODEL_DIR"/ 2>/dev/null || echo "  (répertoire non trouvé)"
echo ""
echo "  → Aller sur l'UI MLflow pour promouvoir en Production."