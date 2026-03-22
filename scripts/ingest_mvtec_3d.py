#!/usr/bin/env python3
"""
Ingestion du vrai dataset MVTec 3D-AD dans PostgreSQL.

Ce script :
  1. DROP la table mvtec_3d_anomaly_detection existante (qui contenait des données 2D par erreur)
  2. Crée une nouvelle table avec les colonnes adaptées au 3D (rgb + xyz + gt)
  3. Scanne le filesystem pour trouver toutes les images RGB
  4. Associe automatiquement les depth maps (.tiff) et masques GT correspondants
  5. Insère tout dans PostgreSQL

Structure attendue du dataset MVTec 3D-AD sur le filesystem :
    <root>/
        <category>/          (bagel, cable_gland, carrot, cookie, dowel, foam, peach, potato, rope, tire)
            train/
                good/
                    rgb/000.png, 001.png, ...
                    xyz/000.tiff, 001.tiff, ...
            test/
                good/
                    rgb/...  xyz/...
                <defect_type>/
                    rgb/...  xyz/...  gt/...
            validation/
                good/
                    rgb/...  xyz/...

Usage :
    python ingest_mvtec_3d.py

    # Avec DB externe (hors cluster k8s) :
    DB_HOST=db.example.com python ingest_mvtec_3d.py

    # Via kubectl exec :
    kubectl exec deployment/postgres-pfe -n pfe -- psql -U admin -d anomaly_detection -f /tmp/ingest.sql
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Racine du dataset MVTec 3D-AD sur le serveur
DATASET_ROOT = os.getenv(
    "MVTEC_3D_ROOT",
    "/home/mario/pfe-fast-data/database-pfe/images_storage/mvtec_3d_anomaly_detection",
)

# Préfixe à stocker en DB (relatif à images_storage, cohérent avec le 2D)
# Les filepath en DB seront : images_storage/mvtec_3d_anomaly_detection/peach/train/good/rgb/000.png
DB_FILEPATH_PREFIX = os.getenv(
    "DB_FILEPATH_PREFIX",
    "images_storage/mvtec_3d_anomaly_detection",
)

# Les 10 catégories officielles du MVTec 3D-AD
EXPECTED_CATEGORIES = {
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
}

# Table cible
TABLE_NAME = os.getenv("TABLE_NAME", "mvtec_3d_anomaly_detection")

# Connexion PostgreSQL
DB_HOST = os.getenv("DB_HOST", "db.example.com")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "anomaly_detection")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "password")


# ---------------------------------------------------------------------------
# Scan du filesystem
# ---------------------------------------------------------------------------

def scan_dataset(root: str) -> list[dict]:
    """
    Scanne le dataset MVTec 3D-AD et retourne une liste de dicts :
    {category, split, label, rgb_filepath, xyz_filepath, gt_filepath}
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    entries = []

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        if category not in EXPECTED_CATEGORIES:
            print(f"  [skip] '{category}' n'est pas une catégorie MVTec 3D-AD")
            continue

        for split_dir in sorted(category_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name

            if split not in ("train", "test", "validation"):
                continue

            for label_dir in sorted(split_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                label = label_dir.name

                rgb_dir = label_dir / "rgb"
                xyz_dir = label_dir / "xyz"
                gt_dir = label_dir / "gt"

                if not rgb_dir.exists():
                    print(f"  [warn] Pas de dossier rgb/ dans {label_dir}")
                    continue

                for rgb_file in sorted(rgb_dir.glob("*.png")):
                    sample_id = rgb_file.stem  # "000", "001", etc.

                    # Chemin relatif pour la DB (cohérent avec le 2D)
                    rgb_rel = f"{DB_FILEPATH_PREFIX}/{category}/{split}/{label}/rgb/{rgb_file.name}"

                    # Depth map correspondante
                    xyz_file = xyz_dir / f"{sample_id}.tiff"
                    xyz_rel = None
                    if xyz_file.exists():
                        xyz_rel = f"{DB_FILEPATH_PREFIX}/{category}/{split}/{label}/xyz/{xyz_file.name}"

                    # Masque GT (seulement pour les défauts dans test/)
                    gt_file = gt_dir / f"{sample_id}.png"
                    gt_rel = None
                    if gt_file.exists():
                        gt_rel = f"{DB_FILEPATH_PREFIX}/{category}/{split}/{label}/gt/{gt_file.name}"

                    entries.append({
                        "category": category,
                        "split": split,
                        "label": label,
                        "filepath": rgb_rel,
                        "xyz_filepath": xyz_rel,
                        "gt_filepath": gt_rel,
                    })

    return entries


# ---------------------------------------------------------------------------
# Insertion en DB
# ---------------------------------------------------------------------------

def ingest_to_postgres(entries: list[dict]) -> None:
    """DROP + CREATE + INSERT dans PostgreSQL."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 non installé. pip install psycopg2-binary")
        sys.exit(1)

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # 1. DROP ancienne table (celle qui contenait les données 2D par erreur)
        print(f"\n1. DROP TABLE IF EXISTS {TABLE_NAME}...")
        cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE;")

        # 2. CREATE avec les colonnes adaptées au 3D
        print(f"2. CREATE TABLE {TABLE_NAME}...")
        cur.execute(f"""
            CREATE TABLE {TABLE_NAME} (
                id            SERIAL PRIMARY KEY,
                category      VARCHAR(64)  NOT NULL,
                split         VARCHAR(16)  NOT NULL,
                label         VARCHAR(64)  NOT NULL,
                filepath      TEXT         NOT NULL,
                xyz_filepath  TEXT,
                gt_filepath   TEXT
            );
        """)

        # 3. INSERT par batch
        print(f"3. INSERT {len(entries)} rows...")
        insert_sql = f"""
            INSERT INTO {TABLE_NAME} (category, split, label, filepath, xyz_filepath, gt_filepath)
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        batch = [
            (
                e["category"],
                e["split"],
                e["label"],
                e["filepath"],
                e["xyz_filepath"],
                e["gt_filepath"],
            )
            for e in entries
        ]

        cur.executemany(insert_sql, batch)

        # 4. Index pour les requêtes fréquentes
        print("4. Création des index...")
        cur.execute(f"CREATE INDEX idx_{TABLE_NAME}_split ON {TABLE_NAME}(split);")
        cur.execute(f"CREATE INDEX idx_{TABLE_NAME}_category ON {TABLE_NAME}(category);")
        cur.execute(f"CREATE INDEX idx_{TABLE_NAME}_label ON {TABLE_NAME}(label);")

        conn.commit()
        print(f"\n✅ Table {TABLE_NAME} créée avec {len(entries)} entrées.")

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ---------------------------------------------------------------------------
# Génération SQL alternative (si pas de psycopg2 disponible)
# ---------------------------------------------------------------------------

def generate_sql_file(entries: list[dict], output_path: str = "/tmp/ingest_mvtec_3d.sql") -> None:
    """Génère un fichier .sql pour injection manuelle via psql."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"-- Ingestion MVTec 3D-AD ({len(entries)} entrées)\n")
        f.write(f"-- Généré automatiquement\n\n")

        f.write(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE;\n\n")

        f.write(f"""CREATE TABLE {TABLE_NAME} (
    id            SERIAL PRIMARY KEY,
    category      VARCHAR(64)  NOT NULL,
    split         VARCHAR(16)  NOT NULL,
    label         VARCHAR(64)  NOT NULL,
    filepath      TEXT         NOT NULL,
    xyz_filepath  TEXT,
    gt_filepath   TEXT
);\n\n""")

        f.write(f"INSERT INTO {TABLE_NAME} (category, split, label, filepath, xyz_filepath, gt_filepath) VALUES\n")

        lines = []
        for e in entries:
            xyz = f"'{e['xyz_filepath']}'" if e["xyz_filepath"] else "NULL"
            gt = f"'{e['gt_filepath']}'" if e["gt_filepath"] else "NULL"
            lines.append(
                f"  ('{e['category']}', '{e['split']}', '{e['label']}', "
                f"'{e['filepath']}', {xyz}, {gt})"
            )

        f.write(",\n".join(lines))
        f.write(";\n\n")

        f.write(f"CREATE INDEX idx_{TABLE_NAME}_split ON {TABLE_NAME}(split);\n")
        f.write(f"CREATE INDEX idx_{TABLE_NAME}_category ON {TABLE_NAME}(category);\n")
        f.write(f"CREATE INDEX idx_{TABLE_NAME}_label ON {TABLE_NAME}(label);\n\n")

        f.write(f"-- Vérification\n")
        f.write(f"SELECT split, label, count(*) FROM {TABLE_NAME} GROUP BY split, label ORDER BY split, label;\n")
        f.write(f"SELECT category, split, count(*) FROM {TABLE_NAME} GROUP BY category, split ORDER BY category, split;\n")

    print(f"\n📄 Fichier SQL généré : {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Ingestion MVTec 3D-AD dans PostgreSQL")
    print("=" * 60)
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Table cible  : {TABLE_NAME}")
    print(f"DB           : {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    # Scan
    print(f"\nScan du dataset...")
    entries = scan_dataset(DATASET_ROOT)

    if not entries:
        print("❌ Aucune entrée trouvée. Vérifier DATASET_ROOT.")
        sys.exit(1)

    # Stats
    splits = {}
    categories = set()
    xyz_count = sum(1 for e in entries if e["xyz_filepath"])
    gt_count = sum(1 for e in entries if e["gt_filepath"])

    for e in entries:
        key = (e["split"], e["label"])
        splits[key] = splits.get(key, 0) + 1
        categories.add(e["category"])

    print(f"\n--- Résumé ---")
    print(f"Catégories : {sorted(categories)}")
    print(f"Total      : {len(entries)} échantillons")
    print(f"Avec XYZ   : {xyz_count}")
    print(f"Avec GT    : {gt_count}")
    print(f"\nDistribution :")
    for (split, label), cnt in sorted(splits.items()):
        print(f"  {split:12s} | {label:20s} | {cnt}")

    # Insertion
    mode = os.getenv("MODE", "auto")  # "auto", "sql", "db"

    if mode == "sql":
        generate_sql_file(entries)
    elif mode == "db":
        ingest_to_postgres(entries)
    else:
        # Auto : essayer psycopg2, sinon générer SQL
        try:
            import psycopg2  # noqa: F401
            print("\n→ psycopg2 disponible, insertion directe en DB...")
            ingest_to_postgres(entries)
        except ImportError:
            print("\n→ psycopg2 non disponible, génération du fichier SQL...")
            generate_sql_file(entries)

    # Vérification
    print("\n--- Vérification post-ingestion ---")
    print("Lancer cette commande pour vérifier :")
    print(f"""
kubectl exec deployment/postgres-pfe -n pfe -- psql -U admin -d anomaly_detection -c "
SELECT split, label, count(*) FROM {TABLE_NAME} GROUP BY split, label ORDER BY split, label;
SELECT category, split, count(*) FROM {TABLE_NAME} GROUP BY category, split ORDER BY category, split;
"
""")


if __name__ == "__main__":
    main()
