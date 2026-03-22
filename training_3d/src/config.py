# training_3d/src/config.py
"""
Configuration pour le pipeline 3D.
Identique à la version 2D, seuls les defaults changent.
"""

from __future__ import annotations

from pathlib import Path
import os
import yaml
from pydantic import BaseModel, Field


class Settings(BaseModel):
    config_path: Path = Field(default=Path("conf/config.yaml"))

    # Database
    db_host: str
    db_port: int = 5432
    db_name: str
    db_user: str
    db_password: str

    # Images
    images_root: str
    images_url: str

    # Worker parameters
    threshold: float = 0.5
    knn_k: int = 5
    outputs_dir: str = "outputs"

    # RabbitMQ
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_queue: str = "tasks_3d"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        db = raw.get("database", {})
        paths = raw.get("paths", {})
        network = raw.get("network", {})

        settings = cls(
            config_path=path,
            db_host=os.getenv("DB_HOST", db.get("host")),
            db_port=int(os.getenv("DB_PORT", db.get("port", 5432))),
            db_name=os.getenv("DB_NAME", db.get("name")),
            db_user=os.getenv("DB_USER", db.get("user")),
            db_password=os.getenv("DB_PASSWORD", db.get("pass")),
            images_root=os.getenv("IMAGES_ROOT", paths.get("images_root")),
            images_url=os.getenv("IMAGES_URL", network.get("images_url")),
            threshold=float(os.getenv("THRESHOLD", 0.5)),
            knn_k=int(os.getenv("KNN_K", 5)),
            outputs_dir=os.getenv("OUTPUTS_DIR", "outputs"),
            rabbitmq_host=os.getenv("RABBITMQ_HOST", "localhost"),
            rabbitmq_port=int(os.getenv("RABBITMQ_PORT", 5672)),
            rabbitmq_user=os.getenv("RABBITMQ_USER", "guest"),
            rabbitmq_password=os.getenv("RABBITMQ_PASSWORD", "guest"),
            rabbitmq_queue=os.getenv("RABBITMQ_QUEUE", "tasks_3d"),
        )
        settings._validate_required_fields()
        return settings

    def _validate_required_fields(self) -> None:
        missing = []
        required = {
            "db_host": self.db_host,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password": self.db_password,
            "images_root": self.images_root,
            "images_url": self.images_url,
        }
        for field, value in required.items():
            if value is None:
                missing.append(field)
        if missing:
            raise ValueError(
                f"Missing required config fields: {missing}. "
                f"Check YAML file or environment variables."
            )

    def build_db_url(self) -> str:
        return (
            f"postgresql+psycopg2://"
            f"{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
