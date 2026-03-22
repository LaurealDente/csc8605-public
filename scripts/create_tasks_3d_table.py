#!/usr/bin/env python3
"""
Création de la table tasks_3d dans PostgreSQL.
Miroir de tasks_2d, pour le pipeline d'inférence 3D.

Usage :
    python create_tasks_3d_table.py
    
    # Ou via psql directement :
    kubectl exec deployment/postgres-pfe -n pfe -- psql -U admin -d anomaly_detection -c "
    CREATE TABLE IF NOT EXISTS tasks_3d (
        id            BIGSERIAL PRIMARY KEY,
        status        TEXT DEFAULT 'pending',
        task_type     TEXT DEFAULT '3d_anomaly',
        image_path    TEXT,
        image_url     TEXT,
        category      TEXT,
        model_name    TEXT,
        model_version TEXT,
        output_dir    TEXT,
        anomaly_score DOUBLE PRECISION,
        pred_label    TEXT,
        result_json   TEXT,
        error_message TEXT,
        created_at    TIMESTAMPTZ DEFAULT NOW(),
        updated_at    TIMESTAMPTZ DEFAULT NOW()
    );
    "
"""

import os
import sys

DB_HOST = os.getenv("DB_HOST", "db.example.com")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "anomaly_detection")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "password")

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS tasks_3d (
    id            BIGSERIAL PRIMARY KEY,
    status        TEXT DEFAULT 'pending',
    task_type     TEXT DEFAULT '3d_anomaly',
    image_path    TEXT,
    image_url     TEXT,
    category      TEXT,
    model_name    TEXT,
    model_version TEXT,
    output_dir    TEXT,
    anomaly_score DOUBLE PRECISION,
    pred_label    TEXT,
    result_json   TEXT,
    error_message TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);
"""

def main():
    try:
        import psycopg2
    except ImportError:
        print("psycopg2 non disponible. Utiliser la commande kubectl ci-dessus.")
        sys.exit(1)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(CREATE_SQL)
    cur.close()
    conn.close()
    print("✅ Table tasks_3d créée.")


if __name__ == "__main__":
    main()
