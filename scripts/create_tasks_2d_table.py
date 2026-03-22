from sqlalchemy import create_engine, text
import os

DB_HOST = os.getenv("DB_HOST", "100.65.87.91")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "anomaly_detection")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "password")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)

SQL = """
CREATE TABLE IF NOT EXISTS tasks_2d (
  id            BIGSERIAL PRIMARY KEY,
  created_at    TIMESTAMP DEFAULT NOW(),
  updated_at    TIMESTAMP DEFAULT NOW(),

  status        TEXT NOT NULL DEFAULT 'pending',
  task_type     TEXT NOT NULL DEFAULT '2d_anomaly',

  image_path    TEXT NOT NULL,
  image_url     TEXT,

  category      TEXT,
  model_name    TEXT DEFAULT 'resnet_knn',
  model_version TEXT DEFAULT 'v1',

  anomaly_score DOUBLE PRECISION,
  pred_label    TEXT,

  output_dir    TEXT,
  result_json   TEXT,

  error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_2d_status
  ON tasks_2d(status);

CREATE INDEX IF NOT EXISTS idx_tasks_2d_created_at
  ON tasks_2d(created_at);
"""

def main():
    with engine.begin() as conn:
        conn.execute(text(SQL))
    print("✅ Table tasks_2d créée (ou déjà existante)")

if __name__ == "__main__":
    main()