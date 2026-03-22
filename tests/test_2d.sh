#!/bin/bash
# test_2d.sh — Test du pipeline 2D
# Usage : ./test_2d.sh

API="https://api.alexandremariolauret.org"
DIR="demo/2D"
CATEGORY="pill"

echo "=========================================="
echo "  TEST PIPELINE 2D — $CATEGORY"
echo "=========================================="

for IMG in "$DIR"/*.png; do
  FILENAME=$(basename "$IMG")
  echo ""
  echo "--- $FILENAME ---"

  # Soumettre
  RESULT=$(curl -s -X POST "$API/predict/2d" \
    -F "file=@$IMG" \
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
    TASK=$(curl -s "$API/tasks/${TASK_ID}?pipeline=2d")
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
print(f'  🔗 Détail: $API/ui/tasks/{d[\"id\"]}?pipeline=2d')
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
echo "  TEST 2D TERMINÉ"
echo "=========================================="
