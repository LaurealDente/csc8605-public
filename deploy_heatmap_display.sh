#!/bin/bash
set -e

# ============================================================
# deploy_heatmap_display.sh
#
# 1. Monte les outputs workers dans l'image-server (k8s)
# 2. Patche l'API pour afficher heatmaps dans le détail tâche
# 3. Rebuild + deploy
#
# Usage : sudo ./deploy_heatmap_display.sh
# ============================================================

REPO_DIR="/home/csc8605"

echo "=========================================="
echo "  AJOUT AFFICHAGE HEATMAPS"
echo "=========================================="

# ============================================================
# 1. Image-server : monter les outputs
# ============================================================
echo "[1/3] Patch image-server..."

kubectl apply -f - -n pfe << 'YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: image-server
  namespace: pfe
spec:
  replicas: 1
  selector:
    matchLabels:
      app: image-server
  template:
    metadata:
      labels:
        app: image-server
    spec:
      containers:
      - name: image-server
        image: halverneus/static-file-server:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          protocol: TCP
        env:
        - name: FOLDER
          value: /web
        volumeMounts:
        - name: images
          mountPath: /web
        - name: outputs-2d
          mountPath: /web/outputs_2d
        - name: outputs-3d
          mountPath: /web/outputs_3d
        resources: {}
      volumes:
      - name: images
        hostPath:
          path: /home/mario/pfe-fast-data/database-pfe/images_storage
      - name: outputs-2d
        hostPath:
          path: /home/mario/pfe-fast-data/database-pfe/worker_outputs_2d
          type: DirectoryOrCreate
      - name: outputs-3d
        hostPath:
          path: /home/mario/pfe-fast-data/database-pfe/worker_outputs_3d
          type: DirectoryOrCreate
YAML

kubectl rollout status deployment/image-server -n pfe --timeout=120s
echo "   ✅ Image-server patché"


# ============================================================
# 2. Patch API main.py
# ============================================================
echo "[2/3] Patch API main.py..."

python3 << 'PYEOF'
import sys

API = "/home/csc8605/api/main.py"
IMAGES_BASE = "http://images.exemple.com"

with open(API, "r") as f:
    lines = f.readlines()

# --- Find the task_status_ui function ---
func_start = None
for i, line in enumerate(lines):
    if "def task_status_ui" in line:
        func_start = i
        break

if func_start is None:
    print("FATAL: Cannot find task_status_ui function")
    sys.exit(1)

print(f"Found task_status_ui at line {func_start + 1}")

# --- Check if already patched ---
for line in lines[func_start:func_start+80]:
    if "heatmap_block" in line:
        print("Already patched, skipping.")
        sys.exit(0)

# --- Find "result_block" line (where we insert heatmap code BEFORE) ---
insert_before = None
for i in range(func_start, min(func_start + 60, len(lines))):
    if "result_block" in lines[i] and '= ""' in lines[i]:
        insert_before = i
        break

if insert_before is None:
    print("FATAL: Cannot find result_block line")
    sys.exit(1)

print(f"Inserting heatmap code before line {insert_before + 1}")

# --- Build heatmap code block ---
heatmap_code = '''
    # --- Heatmap & Overlay ---
    outputs_subdir = "outputs_2d" if pipeline.value == "2d" else "outputs_3d"
    heatmap_name = "heatmap.png" if pipeline.value == "2d" else "heatmap_fused.png"
    overlay_name = "overlay.png" if pipeline.value == "2d" else "overlay_fused.png"
    heatmap_url = f"''' + IMAGES_BASE + '''/{outputs_subdir}/{task_id}/{heatmap_name}"
    overlay_url = f"''' + IMAGES_BASE + '''/{outputs_subdir}/{task_id}/{overlay_name}"

    heatmap_block = ""
    if status == "done":
        heatmap_block = f"""
        <div class="card">
          <h3>Localisation de l\\'anomalie</h3>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <div>
              <p style="color:var(--muted);font-size:0.85rem;margin:0 0 8px;">Overlay</p>
              <img src="{overlay_url}" alt="Overlay" class="preview-image"
                   onerror="this.parentElement.style.display='none'" />
            </div>
            <div>
              <p style="color:var(--muted);font-size:0.85rem;margin:0 0 8px;">Heatmap</p>
              <img src="{heatmap_url}" alt="Heatmap" class="preview-image"
                   onerror="this.parentElement.style.display='none'" />
            </div>
          </div>
        </div>
        """

'''

# --- Insert heatmap code ---
lines.insert(insert_before, heatmap_code)
print("OK: heatmap code inserted")

# --- Find {image_block} in the HTML template and add {heatmap_block} ---
content = "".join(lines)
old_pattern = "{image_block}\n      </div>\n    </div>"
new_pattern = "{image_block}\n        {heatmap_block}\n      </div>\n    </div>"

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern, 1)
    print("OK: {heatmap_block} added to HTML template")
else:
    # Try without exact whitespace
    import re
    match = re.search(r'\{image_block\}\s*\n\s*</div>\s*\n\s*</div>', content)
    if match:
        old = match.group(0)
        new = old.replace("{image_block}", "{image_block}\n        {heatmap_block}")
        content = content.replace(old, new, 1)
        print("OK: {heatmap_block} added (regex match)")
    else:
        print("WARNING: Could not find {image_block} pattern in template")
        print("You may need to manually add {heatmap_block} after {image_block}")

with open(API, "w") as f:
    f.write(content)

print("DONE: API patched successfully")
PYEOF

echo "   ✅ API patchée"


# ============================================================
# 3. Rebuild + deploy API
# ============================================================
echo "[3/3] Rebuild + deploy API..."

cd "$REPO_DIR/api"
docker build --no-cache -t laurealdente/api:v11 .
docker push laurealdente/api:v11
kubectl set image deployment/api api=laurealdente/api:v11 -n pfe
kubectl rollout status deployment/api -n pfe --timeout=120s
echo "   ✅ API déployée"

# Nettoyage
docker system prune -f > /dev/null 2>&1 || true

echo ""
echo "=========================================="
echo "  TERMINÉ"
echo "=========================================="
echo ""
echo "Lance une prédiction 2D ou 3D puis clique 'Voir le détail'."
echo "L'overlay et la heatmap s'affichent sous l'image soumise."
echo ""
echo "Si les images n'apparaissent pas, vérifie que les workers"
echo "ont bien créé les fichiers :"
echo "  ls /home/mario/pfe-fast-data/database-pfe/worker_outputs_2d/<task_id>/"
echo "  ls /home/mario/pfe-fast-data/database-pfe/worker_outputs_3d/<task_id>/"
echo ""
