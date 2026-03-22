#!/bin/bash
# =============================================================================
# setup_env.sh — Configuration de l'environnement sur le cluster Slurm
# Usage : source slurm/setup_env.sh
# =============================================================================

set -e

PROJECT_DIR="/mnt/hdd/homes/alauret/csc8605"
ENV_NAME="pfe"

echo "=== Setup environnement PFE ==="

# --- Conda ---
if ! command -v conda &> /dev/null; then
    echo "❌ conda introuvable. Assurez-vous que Miniconda/Anaconda est installé."
    return 1
fi

# Créer l'environnement s'il n'existe pas
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "📦 Création de l'environnement conda '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

echo "🔄 Activation de l'environnement '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# --- Dépendances Python ---
echo "📦 Installation des dépendances..."
pip install --quiet --upgrade pip

pip install --quiet \
    torch \
    torchvision \
    numpy \
    pillow \
    requests \
    scikit-learn \
    sqlalchemy \
    psycopg2-binary \
    pyyaml \
    pydantic \
    pandas \
    mlflow \
    tifffile

# --- Variables d'environnement ---
ENV_FILE="${PROJECT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    echo "🔑 Chargement des variables depuis ${ENV_FILE}..."
    set -a
    source "${ENV_FILE}"
    set +a
else
    echo "⚠️  Fichier .env introuvable : ${ENV_FILE}"
fi

echo "✅ Environnement prêt."
echo "   Python : $(python --version)"
echo "   PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "   MLflow : $(python -c 'import mlflow; print(mlflow.__version__)')"
echo "   Projet : ${PROJECT_DIR}"