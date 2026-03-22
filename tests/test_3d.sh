#!/bin/bash
# test_3d.sh — Test du pipeline 3D Multimodal
# Usage : ./test_3d.sh

API="https://api.alexandremariolauret.org"
DIR="demo/3D"
CATEGORY="carrot"

echo "=========================================="
echo "  TEST PIPELINE 3D MM — $CATEGORY"
echo "=========================================="

# Trouver les paires RGB/Depth (même nom, .png + .tiff)
for RGB in "$DIR"/*.png; do
  BASENAME=$(basename "$RGB" .png)
  DEPTH="$DIR/${BASENAME}.tiff"

  if [ ! -f "$DEPTH" ]; then
    echo ""
    echo "--- $BASENAME : ⚠️  Pas de depth map ($DEPTH), skip ---"
    continue
  fi

  echo ""
  echo "--- $BASENAME (RGB + Depth) ---"

  # Soumettre
  RESULT=$(curl -s -X POST "$API/predict/3d-mm" \
    -F "rgb_file=@$RGB" \
    -F "depth_file=@$DEPTH" \
    -F "category=$CATEGORY")

  TASK_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['task_id'])" 2>/dev/null)

  if [ -z "$TASK_ID" ]; then
    echo "  ❌ Erreur soumission : $RESULT"
    continue
  fi

  echo "  Task ID : $TASK_ID"
  echo "  Polling..."

  # Attendre le résultat
  while true; do
    TASK=$(curl -s "$API/tasks/${TASK_ID}?pipeline=3d")
    STATUS=$(echo "$TASK" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['status'])" 2>/dev/null)

    if [ "$STATUS" = "done" ]; then
      echo "$TASK" | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
score = d.get('anomaly_score', 'N/A')
label = d.get('pred_label', 'N/A')
emoji = '🔴' if label == 'anomaly' else '🟢'
print(f'  {emoji} Score: {score}')
print(f'  {emoji} Label: {label}')
print(f'  🔗 Détail: $API/ui/tasks/{d[\"id\"]}?pipeline=3d')
"
      break
    elif [ "$STATUS" = "failed" ]; then
      echo "$TASK" | python3 -c "import sys,json; print('  ❌ Erreur:', json.load(sys.stdin)['data'].get('error_message','?'))"
      break
    fi
    sleep 2
  done
done

echo ""
echo "=========================================="
echo "  TEST 3D TERMINÉ"
echo "=========================================="
