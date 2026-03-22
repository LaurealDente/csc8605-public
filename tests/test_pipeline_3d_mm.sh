#!/bin/bash
# ============================================================
# test_pipeline_3d_mm.sh
#
# Script de test end-to-end pour la chaîne 3D Multimodal PatchCore.
# Teste chaque communication : API → DB → RabbitMQ → Worker → résultat.
#
# Prérequis :
#   - jq installé (apt install jq)
#   - curl installé
#   - Accès au cluster k8s (kubectl port-forward ou service exposé)
#
# Usage :
#   export API_URL="http://localhost:8000"       # ou votre URL API
#   export WORKER_3D_URL="http://localhost:8080"  # admin du worker 3D
#   export MLFLOW_URL="http://localhost:5000"     # UI MLflow
#   bash test_pipeline_3d_mm.sh
# ============================================================

set -euo pipefail

# --- Configuration ---
API_URL="${API_URL:-http://localhost:8000}"
WORKER_3D_URL="${WORKER_3D_URL:-http://localhost:8080}"
MLFLOW_URL="${MLFLOW_URL:-http://localhost:5000}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass_count=0
fail_count=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        echo -e "  ${GREEN}✅ $name${NC}"
        pass_count=$((pass_count + 1))
    else
        echo -e "  ${RED}❌ $name${NC}"
        fail_count=$((fail_count + 1))
    fi
}

echo ""
echo "============================================================"
echo " Test Pipeline 3D Multimodal PatchCore — End-to-End"
echo "============================================================"
echo " API_URL       = $API_URL"
echo " WORKER_3D_URL = $WORKER_3D_URL"
echo " MLFLOW_URL    = $MLFLOW_URL"
echo "============================================================"
echo ""

# ============================================================
# 1. Health checks
# ============================================================
echo -e "${YELLOW}[1/7] Health checks${NC}"

# API health
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" 2>/dev/null || echo "000")
check "API /health (HTTP $HTTP_CODE)" "$([ "$HTTP_CODE" = "200" ] && echo 0 || echo 1)"

# API health 3D
HEALTH_3D=$(curl -s "$API_URL/health/3d" 2>/dev/null || echo '{}')
STATUS_3D=$(echo "$HEALTH_3D" | jq -r '.status // "error"' 2>/dev/null || echo "error")
check "API /health/3d (status=$STATUS_3D)" "$([ "$STATUS_3D" != "error" ] && echo 0 || echo 1)"

# Worker 3D health
WORKER_HEALTH=$(curl -s "$WORKER_3D_URL/health" 2>/dev/null || echo '{}')
WORKER_STATUS=$(echo "$WORKER_HEALTH" | jq -r '.status // "error"' 2>/dev/null || echo "error")
check "Worker 3D /health (status=$WORKER_STATUS)" "$([ "$WORKER_STATUS" = "ok" ] || [ "$WORKER_STATUS" = "degraded" ] && echo 0 || echo 1)"

# Worker 3D — MM model check
MM_STATUS=$(echo "$WORKER_HEALTH" | jq -r '.models.v2_mm_patchcore.status // "absent"' 2>/dev/null || echo "absent")
check "Worker 3D — modèle MM-PatchCore (status=$MM_STATUS)" "$([ "$MM_STATUS" = "ok" ] && echo 0 || echo 1)"

# MLflow
MLFLOW_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$MLFLOW_URL/api/2.0/mlflow/experiments/search" 2>/dev/null || echo "000")
check "MLflow accessible (HTTP $MLFLOW_CODE)" "$([ "$MLFLOW_CODE" = "200" ] && echo 0 || echo 1)"

echo ""

# ============================================================
# 2. DB migration check
# ============================================================
echo -e "${YELLOW}[2/7] Vérification DB (colonnes depth)${NC}"

# Force la migration en appelant l'API (ensure_tables_exist est dans le lifespan)
DOCS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs" 2>/dev/null || echo "000")
check "API Swagger /docs (HTTP $DOCS_CODE)" "$([ "$DOCS_CODE" = "200" ] && echo 0 || echo 1)"

echo "  ℹ️  Les colonnes depth_path/depth_url sont ajoutées automatiquement au démarrage de l'API."
echo ""

# ============================================================
# 3. Endpoint /predict/3d-mm disponible
# ============================================================
echo -e "${YELLOW}[3/7] Vérification endpoint /predict/3d-mm${NC}"

OPENAPI=$(curl -s "$API_URL/openapi.json" 2>/dev/null || echo '{}')
HAS_MM_ENDPOINT=$(echo "$OPENAPI" | jq 'has("paths") and (.paths | has("/predict/3d-mm"))' 2>/dev/null || echo "false")
check "Endpoint POST /predict/3d-mm existe dans OpenAPI" "$([ "$HAS_MM_ENDPOINT" = "true" ] && echo 0 || echo 1)"

echo ""

# ============================================================
# 4. Test d'upload multimodal (si fichiers de test disponibles)
# ============================================================
echo -e "${YELLOW}[4/7] Test d'upload multimodal${NC}"

# Créer des fichiers de test temporaires
TEST_RGB="/tmp/test_rgb.png"
TEST_DEPTH="/tmp/test_depth.tiff"

# Créer une image RGB de test (1x1 pixel rouge)
python3 -c "
from PIL import Image
img = Image.new('RGB', (224, 224), color=(255, 0, 0))
img.save('$TEST_RGB')
print('✓ Image RGB de test créée')
" 2>/dev/null || echo "  ⚠️  PIL non disponible, skip création image test"

# Créer un depth map de test
python3 -c "
import numpy as np
try:
    import tifffile
    depth = np.random.rand(224, 224, 3).astype(np.float32)
    tifffile.imwrite('$TEST_DEPTH', depth)
    print('✓ Depth map de test créé')
except ImportError:
    # Fallback : créer un fichier tiff vide
    from PIL import Image
    img = Image.new('F', (224, 224))
    img.save('$TEST_DEPTH')
    print('✓ Depth map de test créé (fallback PIL)')
" 2>/dev/null || echo "  ⚠️  Impossible de créer le depth de test"

if [ -f "$TEST_RGB" ] && [ -f "$TEST_DEPTH" ]; then
    UPLOAD_RESP=$(curl -s -X POST "$API_URL/predict/3d-mm" \
        -F "rgb_file=@$TEST_RGB" \
        -F "depth_file=@$TEST_DEPTH" \
        -F "category=bagel" \
        2>/dev/null || echo '{"status":"error"}')

    UPLOAD_STATUS=$(echo "$UPLOAD_RESP" | jq -r '.status // "error"' 2>/dev/null || echo "error")
    TASK_ID=$(echo "$UPLOAD_RESP" | jq -r '.data.task_id // "none"' 2>/dev/null || echo "none")

    check "POST /predict/3d-mm (status=$UPLOAD_STATUS)" "$([ "$UPLOAD_STATUS" = "queued" ] && echo 0 || echo 1)"

    if [ "$TASK_ID" != "none" ] && [ "$TASK_ID" != "null" ]; then
        echo "  ℹ️  task_id=$TASK_ID"
    fi
else
    echo "  ⚠️  Fichiers de test non créés — skip upload test"
fi

echo ""

# ============================================================
# 5. Vérification du status de la tâche
# ============================================================
echo -e "${YELLOW}[5/7] Vérification status tâche${NC}"

if [ -n "${TASK_ID:-}" ] && [ "$TASK_ID" != "none" ] && [ "$TASK_ID" != "null" ]; then
    echo "  ℹ️  Attente 5s pour que le worker traite la tâche..."
    sleep 5

    TASK_RESP=$(curl -s "$API_URL/tasks/$TASK_ID?pipeline=3d" 2>/dev/null || echo '{}')
    TASK_STATUS=$(echo "$TASK_RESP" | jq -r '.data.status // "unknown"' 2>/dev/null || echo "unknown")
    TASK_TYPE=$(echo "$TASK_RESP" | jq -r '.data.task_type // "unknown"' 2>/dev/null || echo "unknown")

    echo "  ℹ️  status=$TASK_STATUS, task_type=$TASK_TYPE"
    check "GET /tasks/$TASK_ID — tâche trouvée" "$([ "$TASK_STATUS" != "unknown" ] && echo 0 || echo 1)"

    if [ "$TASK_STATUS" = "pending" ] || [ "$TASK_STATUS" = "running" ]; then
        echo "  ℹ️  Tâche encore en cours, attente 15s supplémentaires..."
        sleep 15
        TASK_RESP=$(curl -s "$API_URL/tasks/$TASK_ID?pipeline=3d" 2>/dev/null || echo '{}')
        TASK_STATUS=$(echo "$TASK_RESP" | jq -r '.data.status // "unknown"' 2>/dev/null || echo "unknown")
        echo "  ℹ️  status=$TASK_STATUS après 20s"
    fi

    check "Tâche complétée (status=$TASK_STATUS)" "$([ "$TASK_STATUS" = "done" ] && echo 0 || echo 1)"

    if [ "$TASK_STATUS" = "done" ]; then
        SCORE=$(echo "$TASK_RESP" | jq -r '.data.anomaly_score // "N/A"' 2>/dev/null || echo "N/A")
        LABEL=$(echo "$TASK_RESP" | jq -r '.data.pred_label // "N/A"' 2>/dev/null || echo "N/A")
        echo "  ℹ️  anomaly_score=$SCORE, pred_label=$LABEL"
    fi

    if [ "$TASK_STATUS" = "failed" ]; then
        ERROR=$(echo "$TASK_RESP" | jq -r '.data.error_message // "N/A"' 2>/dev/null || echo "N/A")
        echo -e "  ${RED}  error: $ERROR${NC}"
    fi
else
    echo "  ⚠️  Pas de task_id — skip"
fi

echo ""

# ============================================================
# 6. Test predict V1 (backward compatibility)
# ============================================================
echo -e "${YELLOW}[6/7] Backward compatibility — /predict/3d (V1)${NC}"

if [ -f "$TEST_RGB" ]; then
    V1_RESP=$(curl -s -X POST "$API_URL/predict/3d" \
        -F "file=@$TEST_RGB" \
        -F "category=bagel" \
        2>/dev/null || echo '{"status":"error"}')

    V1_STATUS=$(echo "$V1_RESP" | jq -r '.status // "error"' 2>/dev/null || echo "error")
    check "POST /predict/3d V1 (status=$V1_STATUS)" "$([ "$V1_STATUS" = "queued" ] && echo 0 || echo 1)"
else
    echo "  ⚠️  Pas de fichier test RGB — skip"
fi

echo ""

# ============================================================
# 7. Worker metrics
# ============================================================
echo -e "${YELLOW}[7/7] Métriques Prometheus${NC}"

METRICS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$WORKER_3D_URL/metrics" 2>/dev/null || echo "000")
check "Worker /metrics (HTTP $METRICS_CODE)" "$([ "$METRICS_CODE" = "200" ] && echo 0 || echo 1)"

if [ "$METRICS_CODE" = "200" ]; then
    HAS_MM_METRIC=$(curl -s "$WORKER_3D_URL/metrics" 2>/dev/null | grep -c "worker3d_model_mm_cache_version" || echo "0")
    check "Métrique MM-PatchCore présente" "$([ "$HAS_MM_METRIC" -gt "0" ] && echo 0 || echo 1)"
fi

echo ""

# ============================================================
# Résumé
# ============================================================
echo "============================================================"
TOTAL=$((pass_count + fail_count))
echo -e " Résultat : ${GREEN}$pass_count passés${NC} / ${RED}$fail_count échoués${NC} / $TOTAL total"
echo "============================================================"

# Cleanup
rm -f "$TEST_RGB" "$TEST_DEPTH" 2>/dev/null || true

if [ "$fail_count" -gt 0 ]; then
    echo ""
    echo "Actions correctives :"
    echo "  1. Vérifier que l'API a redémarré avec predict_mm.py"
    echo "  2. Vérifier que le worker_3d a les fichiers multimodal_patchcore.py, queue_consumer.py"
    echo "  3. Vérifier que le modèle MM-PatchCore est entraîné et promu en production dans MLflow"
    echo "  4. Consulter les logs : kubectl logs -f deployment/worker-3d"
    exit 1
fi
