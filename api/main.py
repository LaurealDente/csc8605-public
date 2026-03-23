from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from predict_mm import router as mm_router
from db import (
    get_distinct_categories_from_table,
    check_db_connection,
    ensure_tables_exist,
    get_task,
    get_task_counts_by_status,
    insert_task,
    update_task_status,
)

from mlflow_utils import get_pipeline_mlflow_health, get_pipeline_production_version
from rabbitmq import check_rabbit_connection, publish_task

from config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    APP_TITLE,
    APP_VERSION,
    IMAGES_PUBLIC_BASE,
    IMAGES_STORAGE_ROOT,
    MAX_UPLOAD_SIZE_BYTES,
    Pipeline,
    get_pipeline_config,
    PUBLIC_API_URL,
    ADMINER_PUBLIC_URL,
    GRAFANA_PUBLIC_URL,
    IMAGES_PUBLIC_UI_URL,
    MLFLOW_PUBLIC_URL,
    PREFECT_PUBLIC_URL,
    PROMETHEUS_PUBLIC_URL,
)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

http_requests_total = Counter(
    "api_http_requests_total",
    "Nombre total de requêtes HTTP",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "api_http_request_duration_seconds",
    "Latence HTTP",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

tasks_created_total = Counter(
    "api_tasks_created_total",
    "Nombre de tâches créées",
    ["pipeline", "category"],
)

tasks_by_status = Gauge(
    "api_tasks_by_status",
    "Nombre de tâches par statut",
    ["pipeline", "status"],
)

model_production_version = Gauge(
    "api_model_production_version",
    "Version du modèle en Production",
    ["pipeline", "model_name"],
)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_tables_exist()
    yield


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="""
API unifiée de détection d'anomalies industrielles 2D et 3D.

Projet de fin d'études réalisé par :
- Tatiana Niauronis
- Selim Jerbi
- Alexandre Lauret

Cette plateforme expose une architecture MLOps complète permettant :

• la soumission d’images d’inspection  
• la création de tâches asynchrones via RabbitMQ  
• l’exécution d’inférences par des workers spécialisés  
• le suivi d’état des traitements  
• l’administration des modèles  
• le monitoring de la plateforme  

### Architecture technique

- **FastAPI** — API REST
- **PostgreSQL** — stockage des tâches
- **RabbitMQ** — orchestration asynchrone
- **MLflow** — gestion des modèles
- **Prometheus** — métriques
- **Kubernetes** — déploiement

Cette API est conçue comme démonstrateur d’industrialisation d’un système d’IA.
""",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "UI", "description": "Interface visuelle d'accueil"},
        {"name": "Prediction", "description": "Soumission d'images 2D/3D"},
        {"name": "Tasks", "description": "Consultation des tâches"},
        {"name": "Admin", "description": "Administration et rechargement des modèles"},
        {"name": "Health", "description": "Santé de l'API, DB, RabbitMQ, workers et MLflow"},
        {"name": "Monitoring", "description": "Métriques Prometheus"},
    ],
    terms_of_service="https://api.exemple.com",
    contact={
        "name": "Tatiana Niauronis, Selim Jerbi, Alexandre Lauret",
        "email": "team@exemple.com",
    },
    license_info={
        "name": "MIT",
    },
)
app.include_router(mm_router)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=str(response.status_code),
    ).inc()

    http_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    response.headers["X-Process-Time"] = f"{duration:.6f}"
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.status_code,
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):  # noqa: BLE001
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "code": 500,
            "detail": f"Erreur interne inattendue: {exc}",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def validate_file_extension(filename: Optional[str]) -> str:
    suffix = Path(filename).suffix.lower() if filename else ".png"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Extension non supportée: {suffix}. "
                f"Extensions autorisées: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )
    return suffix


def validate_content_type(content_type: Optional[str]) -> None:
    if not content_type:
        return
    if content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Content-Type non supporté: {content_type}. "
                f"Types autorisés: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )


async def save_uploaded_file(file: UploadFile, pipeline: Pipeline) -> Dict[str, str]:
    validate_content_type(file.content_type)
    suffix = validate_file_extension(file.filename)

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (> {MAX_UPLOAD_SIZE_BYTES} octets).",
        )

    fname = f"{uuid.uuid4().hex}{suffix}"
    date_prefix = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    rel_path = f"{pipeline.value}/uploads/{date_prefix}/{fname}"

    abs_path = Path(IMAGES_STORAGE_ROOT) / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(content)

    image_url = f"{IMAGES_PUBLIC_BASE.rstrip('/')}/{rel_path}"

    return {
        "image_path": str(abs_path),
        "image_url": image_url,
        "relative_path": rel_path,
        "filename": fname,
    }


async def ping_worker(pipeline: Pipeline) -> Dict[str, Any]:
    cfg = get_pipeline_config(pipeline)
    url = cfg["worker_health_url"]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()
            return {
                "status": "ok",
                "url": url,
                "response": resp.json(),
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "url": url,
            "error": str(exc),
        }


async def call_worker_reload(pipeline: Pipeline, force: bool) -> Dict[str, Any]:
    cfg = get_pipeline_config(pipeline)
    worker_reload_url = f"{cfg['worker_admin_url'].rstrip('/')}/reload-model"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                worker_reload_url,
                params={"force": str(force).lower()},
                timeout=60.0,
            )
            resp.raise_for_status()

            return {
                "status": "reload_triggered",
                "pipeline": pipeline.value,
                "worker_url": worker_reload_url,
                "worker_response": resp.json(),
            }

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Le worker {pipeline.value} a retourné une erreur lors du reload : "
                f"{exc.response.status_code} — {exc.response.text}"
            ),
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de joindre le worker {pipeline.value} à {worker_reload_url} : {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["UI"],
    summary="Page d'accueil de la plateforme",
    description="""
Interface visuelle publique de la plateforme.

Cette page présente :

• la plateforme d’inspection intelligente
• les pipelines disponibles
• les endpoints principaux
• l’architecture technique

Elle permet également de soumettre directement des images
pour démonstration des pipelines 2D et 3D.
""",
    response_description="Interface web HTML de la plateforme"
)
def home() -> str:
    return """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <title>PFE API — Plateforme de détection d'anomalies 2D / 3D</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {
      --bg: #07111f;
      --bg-soft: #0d1b2e;
      --card: rgba(15, 27, 46, 0.88);
      --card-2: rgba(20, 37, 63, 0.94);
      --text: #eef4ff;
      --muted: #b7c5df;
      --primary: #60a5fa;
      --primary-2: #38bdf8;
      --accent: #34d399;
      --danger: #f87171;
      --warning: #fbbf24;
      --border: rgba(255,255,255,0.08);
      --shadow: 0 20px 55px rgba(0,0,0,0.35);
      --radius-xl: 28px;
      --radius-lg: 20px;
      --radius-md: 14px;
      --max-width: 1240px;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 10% 10%, rgba(96,165,250,0.18), transparent 26%),
        radial-gradient(circle at 90% 0%, rgba(52,211,153,0.13), transparent 22%),
        radial-gradient(circle at 50% 100%, rgba(56,189,248,0.10), transparent 28%),
        linear-gradient(180deg, #06101d 0%, #081220 100%);
      min-height: 100vh;
    }

    a {
      color: inherit;
      text-decoration: none;
    }

    .container {
      max-width: var(--max-width);
      margin: 0 auto;
      padding: 24px;
    }

    .navbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      margin-top: 10px;
      border: 1px solid var(--border);
      background: rgba(10, 18, 32, 0.72);
      backdrop-filter: blur(18px);
      border-radius: 999px;
      position: sticky;
      top: 16px;
      z-index: 30;
      box-shadow: var(--shadow);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 800;
      font-size: 1rem;
    }

    .brand-logo {
      width: 38px;
      height: 38px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      display: grid;
      place-items: center;
      color: #06101d;
      font-weight: 900;
      box-shadow: 0 10px 24px rgba(52, 211, 153, 0.22);
    }

    .nav-links {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
    }

    .nav-links a {
      padding: 10px 14px;
      border-radius: 999px;
      color: var(--muted);
      transition: 0.2s ease;
    }

    .nav-links a:hover {
      background: rgba(255,255,255,0.06);
      color: var(--text);
    }

    .hero {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 28px;
      align-items: stretch;
      padding: 46px 0 24px;
    }

    .hero-card,
    .card {
      background: linear-gradient(180deg, rgba(15,27,46,0.88), rgba(18,31,53,0.92));
      border: 1px solid var(--border);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .hero-main {
      padding: 34px;
      position: relative;
      overflow: hidden;
    }

    .hero-main::after {
      content: "";
      position: absolute;
      top: -120px;
      right: -90px;
      width: 260px;
      height: 260px;
      background: radial-gradient(circle, rgba(96,165,250,0.30), transparent 65%);
      pointer-events: none;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(96,165,250,0.12);
      border: 1px solid rgba(96,165,250,0.22);
      color: #beddff;
      font-size: 0.92rem;
      margin-bottom: 20px;
    }

    h1 {
      margin: 0 0 18px;
      font-size: clamp(2.1rem, 5vw, 4rem);
      line-height: 1.04;
      letter-spacing: -0.03em;
    }

    .hero-main p {
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.75;
      max-width: 760px;
    }

    .hero-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-bottom: 26px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 14px 18px;
      border-radius: 14px;
      border: 1px solid transparent;
      font-weight: 700;
      transition: 0.2s ease;
      cursor: pointer;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: #07111f;
      box-shadow: 0 16px 36px rgba(52, 211, 153, 0.18);
    }

    .btn-primary:hover {
      transform: translateY(-1px);
    }

    .btn-secondary {
      background: rgba(255,255,255,0.04);
      border-color: var(--border);
      color: var(--text);
    }

    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .badge {
      padding: 9px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      font-size: 0.92rem;
    }

    .hero-side {
      padding: 26px;
      display: grid;
      gap: 16px;
    }

    .stat-card {
      padding: 18px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
      border: 1px solid var(--border);
    }

    .stat-label {
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 10px;
    }

    .stat-value {
      font-size: 1.9rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .stat-sub {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.95rem;
      line-height: 1.5;
    }

    .section {
      margin-top: 28px;
    }

    .section-header {
      margin-bottom: 18px;
    }

    .section-header h2 {
      margin: 0 0 8px;
      font-size: 1.7rem;
      letter-spacing: -0.02em;
    }

    .section-header p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 22px;
    }

    .grid-3 {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 22px;
    }

    .card {
      padding: 24px;
    }

    .pipeline-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }

    .pipeline-title h3 {
      margin: 0;
      font-size: 1.25rem;
    }

    .mini-badge {
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(52,211,153,0.12);
      border: 1px solid rgba(52,211,153,0.22);
      color: #b6f7dd;
      font-size: 0.84rem;
    }

    .card p,
    .card li {
      color: var(--muted);
      line-height: 1.7;
    }

    .list {
      padding-left: 18px;
      margin: 14px 0 0;
    }

    .endpoint {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      margin-bottom: 10px;
      overflow-x: auto;
    }

    .method {
      min-width: 58px;
      text-align: center;
      font-size: 0.82rem;
      font-weight: 800;
      padding: 7px 10px;
      border-radius: 10px;
      color: #07111f;
    }

    .post { background: #34d399; }
    .get  { background: #60a5fa; }
    .admin { background: #fbbf24; }

    .endpoint code {
      white-space: nowrap;
      background: transparent;
      padding: 0;
      color: var(--text);
      font-size: 0.95rem;
    }

    form {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }

    select {
      width: 100%;
      padding: 14px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.05);
      color: var(--text);
      outline: none;
    }

    .progress-wrapper {
      margin-top: 16px;
    }

    .progress-label {
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.95rem;
    }

    .progress-bar {
      width: 100%;
      height: 14px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255,255,255,0.08);
      border: 1px solid var(--border);
    }

    .progress-fill {
      width: 0%;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      transition: width 0.4s ease;
    }

    input[type="file"] {
      width: 100%;
      padding: 14px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.05);
      color: var(--text);
      outline: none;
    }

    button {
      border: none;
      border-radius: 16px;
      padding: 16px 18px;
      font-weight: 800;
      font-size: 1rem;
      cursor: pointer;
      background: linear-gradient(135deg, var(--primary), #7bdcb5);
      color: #07111f;
      transition: 0.2s ease;
    }

    button:hover {
      transform: translateY(-1px);
    }

    button:disabled {
      opacity: 0.7;
      cursor: not-allowed;
      transform: none;
    }

    .form-hint {
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.5;
    }

    .result-box {
      display: none;
      margin-top: 16px;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      line-height: 1.7;
    }

    .result-box.success {
      display: block;
      border-color: rgba(52,211,153,0.35);
      background: rgba(52,211,153,0.08);
    }

    .result-box.error {
      display: block;
      border-color: rgba(248,113,113,0.35);
      background: rgba(248,113,113,0.08);
    }

    .result-box.loading {
      display: block;
      border-color: rgba(96,165,250,0.35);
      background: rgba(96,165,250,0.08);
    }

    .result-title {
      font-weight: 800;
      margin-bottom: 8px;
      font-size: 1rem;
    }

    .result-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }

    .result-link {
      display: inline-block;
      padding: 10px 14px;
      border-radius: 12px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      color: var(--text);
      text-decoration: none;
      font-weight: 700;
    }

    .architecture {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 14px;
      margin-top: 18px;
    }

    .arch-box {
      padding: 18px 16px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      text-align: center;
    }

    .arch-box strong {
      display: block;
      margin-bottom: 8px;
      font-size: 1rem;
    }

    .arch-box span {
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.5;
    }

    .footer {
      margin: 34px 0 16px;
      padding: 22px 0 8px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      text-align: center;
      line-height: 1.7;
    }

    @media (max-width: 1100px) {
      .hero,
      .grid-3,
      .architecture,
      .grid-2 {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 760px) {
      .navbar {
        border-radius: 24px;
        align-items: flex-start;
        flex-direction: column;
      }

      .hero-main,
      .hero-side,
      .card {
        padding: 20px;
      }

      .container {
        padding: 16px;
      }

      h1 {
        font-size: 2.2rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <nav class="navbar">
      <div class="brand">
        <div class="brand-logo">AI</div>
        <div>PFE Anomaly Detection Platform</div>
      </div>
      <div class="nav-links">
        <a href="#pipelines">Pipelines</a>
        <a href="#architecture">Architecture</a>
        <a href="#api">API</a>
        <a href="/admin">Admin</a>
        <a href="/docs">Swagger</a>
        <a href="/ui/health">Health</a>
      </div>
    </nav>

    <section class="hero">
      <div class="hero-card hero-main">
        <div class="eyebrow">Plateforme MLOps industrielle • Projet de fin d’études</div>
        <h1>Détection d’anomalies 2D / 3D pour l’inspection intelligente</h1>
        <p>
          Cette plateforme unifie la soumission d’images, l’orchestration asynchrone des tâches,
          le suivi d’état, l’administration des modèles et le monitoring opérationnel.
          Elle s’appuie sur une architecture distribuée intégrant FastAPI, PostgreSQL,
          RabbitMQ, MLflow, Prometheus et des workers spécialisés 2D / 3D.
        </p>

        <div class="hero-actions">
          <a class="btn btn-primary" href="/docs">Explorer l’API</a>
          <a class="btn btn-secondary" href="/redoc">Documentation technique</a>
          <a class="btn btn-secondary" href="/admin">Centre d’administration</a>
        </div>

        <div class="badges">
          <div class="badge">FastAPI</div>
          <div class="badge">PostgreSQL</div>
          <div class="badge">RabbitMQ</div>
          <div class="badge">MLflow</div>
          <div class="badge">Prometheus</div>
          <div class="badge">Kubernetes</div>
        </div>
      </div>

      <div class="hero-card hero-side">
        <div class="stat-card">
          <div class="stat-label">Pipelines supportés</div>
          <div class="stat-value">2</div>
          <div class="stat-sub">Inspection 2D et 3D avec workers dédiés et orchestration indépendante.</div>
        </div>

        <div class="stat-card">
          <div class="stat-label">Architecture</div>
          <div class="stat-value">Asynchrone</div>
          <div class="stat-sub">Soumission rapide des tâches, consommation RabbitMQ, traitement par workers, stockage des résultats en base.</div>
        </div>

        <div class="stat-card">
          <div class="stat-label">Objectif</div>
          <div class="stat-value">Industrialisation</div>
          <div class="stat-sub">Exposer une API exploitable, supervisable et démontrable dans un contexte de production MLOps.</div>
        </div>
      </div>
    </section>

    <section id="pipelines" class="section">
      <div class="section-header">
        <h2>Pipelines d’inférence</h2>
        <p>
          Deux pipelines spécialisés sont disponibles afin de traiter différents types de données
          d’inspection industrielle tout en conservant une API unifiée.
        </p>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="pipeline-title">
            <h3>Pipeline 2D</h3>
            <span class="mini-badge">Vision industrielle</span>
          </div>
          <p>
            Pipeline orienté analyse d’images 2D pour la détection d’anomalies visuelles
            sur objets industriels, avec configuration backend par défaut.
          </p>
          <ul class="list">
            <li>Soumission via <code>POST /predict/2d</code></li>
            <li>Worker dédié au traitement 2D</li>
            <li>Suivi d’état via la table <code>tasks_2d</code></li>
            <li>Chargement du modèle piloté par MLflow</li>
          </ul>

          <form id="predict-form-2d" enctype="multipart/form-data">
            <select name="category" id="category-2d" required>
              <option value="">Chargement des catégories 2D...</option>
            </select>
            <input type="file" name="file" accept="image/*" required />
            <button type="submit">Lancer la prédiction 2D</button>
          </form>
          <div class="form-hint">
            Sélectionnez une catégorie d’objet puis une image.
          </div>
          <div class="progress-wrapper" id="progress-wrapper-2d" style="display:none;">
            <div class="progress-label" id="progress-label-2d">Préparation...</div>
            <div class="progress-bar">
              <div class="progress-fill" id="progress-fill-2d"></div>
            </div>
          </div>

          <div id="predict-result-2d" class="result-box"></div>
        </div>

        <div class="card">
          <div class="pipeline-title">
            <h3>Pipeline 3D</h3>
            <span class="mini-badge">Inspection avancée</span>
          </div>
          <p>
            Pipeline dédié aux données 3D / MVTec 3D-AD, avec séparation claire des tâches,
            de la queue RabbitMQ et du worker, tout en gardant une interface unique côté API.
          </p>
          <ul class="list">
            <li>Soumission via <code>POST /predict/3d</code></li>
            <li>Worker dédié au traitement 3D</li>
            <li>Suivi d’état via la table <code>tasks_3d</code></li>
            <li>Rechargement dynamique depuis MLflow</li>
          </ul>

          <form id="predict-form-3d" enctype="multipart/form-data">
            <select name="category" id="category-3d" required>
              <option value="">Chargement des catégories 3D...</option>
            </select>
            <label for="rgb-file-3d"><strong>Image RGB</strong></label>
            <input id="rgb-file-3d" type="file" name="rgb_file" accept="image/*" required />

            <label for="depth-file-3d"><strong>Carte de profondeur</strong></label>
            <input id="depth-file-3d" type="file" name="depth_file" accept="image/*" required />
            <button type="submit">Lancer la prédiction 3D</button>
          </form>
          <div class="form-hint">
            Sélectionnez une catégorie 3D, puis :
            1) l’image RGB,
            2) la carte de profondeur.
          </div>
          <div class="progress-wrapper" id="progress-wrapper-3d" style="display:none;">
            <div class="progress-label" id="progress-label-3d">Préparation...</div>
            <div class="progress-bar">
              <div class="progress-fill" id="progress-fill-3d"></div>
            </div>
          </div>
          <div id="predict-result-3d" class="result-box"></div>
        </div>
      </div>
    </section>

    <section id="architecture" class="section">
      <div class="section-header">
        <h2>Architecture technique</h2>
        <p>
          La plateforme repose sur une séparation claire des responsabilités pour garantir
          extensibilité, observabilité et maintien en condition opérationnelle.
        </p>
      </div>

      <div class="card">
        <div class="architecture">
          <div class="arch-box">
            <strong>Client</strong>
            <span>Soumet les images et consulte le cycle de vie des tâches.</span>
          </div>
          <div class="arch-box">
            <strong>FastAPI</strong>
            <span>Valide, persiste, publie et expose les APIs publiques.</span>
          </div>
          <div class="arch-box">
            <strong>RabbitMQ</strong>
            <span>Découple la réception des requêtes du traitement d’inférence.</span>
          </div>
          <div class="arch-box">
            <strong>Workers</strong>
            <span>Exécutent les traitements spécialisés 2D / 3D et stockent les résultats.</span>
          </div>
          <div class="arch-box">
            <strong>MLflow / DB</strong>
            <span>Versionnement modèle, traçabilité, supervision et exploitation.</span>
          </div>
        </div>
      </div>
    </section>

    <section id="api" class="section">
      <div class="section-header">
        <h2>Endpoints principaux</h2>
        <p>
          Les routes ci-dessous couvrent les besoins de démonstration, d’exploitation
          et d’administration de la plateforme.
        </p>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="endpoint">
            <span class="method post">POST</span>
            <code>/predict/2d</code>
          </div>
          <div class="endpoint">
            <span class="method post">POST</span>
            <code>/predict/3d</code>
          </div>
          <div class="endpoint">
            <span class="method get">GET</span>
            <code>/tasks/{task_id}?pipeline=2d</code>
          </div>
          <div class="endpoint">
            <span class="method get">GET</span>
            <code>/tasks/{task_id}?pipeline=3d</code>
          </div>
        </div>

        <div class="card">
          <div class="endpoint">
            <span class="method admin">POST</span>
            <code>/admin/reload-model/2d</code>
          </div>
          <div class="endpoint">
            <span class="method admin">POST</span>
            <code>/admin/reload-model/3d</code>
          </div>
          <div class="endpoint">
            <span class="method get">GET</span>
            <code>/ui/health</code>
          </div>
          <div class="endpoint">
            <span class="method get">GET</span>
            <code>/metrics</code>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="grid-3">
        <div class="card">
          <h3>Observabilité</h3>
          <p>
            L’API expose des métriques Prometheus et des endpoints de santé détaillés
            pour suivre l’état de la base, de RabbitMQ, des workers et des modèles.
          </p>
        </div>
        <div class="card">
          <h3>Maintenabilité</h3>
          <p>
            Le découpage modulaire facilite les évolutions, la revue de code et la
            démonstration d’une architecture logicielle propre.
          </p>
        </div>
        <div class="card">
          <h3>Dimension PFE</h3>
          <p>
            Cette interface sert à la fois de vitrine de projet, d’outil de démonstration
            et de façade technique exploitable dans Kubernetes.
          </p>
        </div>
      </div>
    </section>

    <footer class="footer">
      PFE — Plateforme de détection d’anomalies 2D / 3D • FastAPI • MLOps • Kubernetes
    </footer>
  </div>

  <script>
    async function loadCategories() {
      try {
        const response = await fetch("/categories");
        const payload = await response.json();

        console.log("Categories payload:", payload);

        if (!response.ok) {
          throw new Error(payload.detail || "Impossible de charger les catégories.");
        }

        const categories2d = payload?.data?.["2d"] || [];
        const categories3d = payload?.data?.["3d"] || [];

        fillCategorySelect("category-2d", categories2d, "Sélectionner une catégorie 2D");
        fillCategorySelect("category-3d", categories3d, "Sélectionner une catégorie 3D");
      } catch (error) {
        console.error("Erreur catégories:", error);
        fillCategorySelect("category-2d", [], "Erreur chargement catégories 2D");
        fillCategorySelect("category-3d", [], "Erreur chargement catégories 3D");
      }
    }

    function fillCategorySelect(selectId, categories, placeholder) {
      const select = document.getElementById(selectId);
      if (!select) {
        console.error("Select introuvable:", selectId);
        return;
      }

      if (!categories || !categories.length) {
        select.innerHTML = `<option value="">Aucune catégorie disponible</option>`;
        return;
      }

      select.innerHTML =
        `<option value="">${placeholder}</option>` +
        categories.map(cat => `<option value="${cat}">${cat}</option>`).join("");
    }

    function updateProgress(progressFillId, progressLabelId, status) {
      const fill = document.getElementById(progressFillId);
      const label = document.getElementById(progressLabelId);

      let width = 10;
      let text = "Préparation...";

      if (status === "queued" || status === "pending") {
        width = 30;
        text = "Tâche en attente...";
      } else if (status === "running") {
        width = 70;
        text = "Traitement en cours...";
      } else if (status === "done" || status === "success") {
        width = 100;
        text = "Traitement terminé.";
      } else if (status === "failed" || status === "error") {
        width = 100;
        text = "Le traitement a échoué.";
      }

      if (fill) fill.style.width = `${width}%`;
      if (label) label.textContent = text;
    }

    async function pollTaskStatus(taskId, pipeline, resultBox, progressFillId, progressLabelId) {
      const maxAttempts = 120;

      for (let i = 0; i < maxAttempts; i++) {
        try {
          const response = await fetch(`/tasks/${taskId}?pipeline=${pipeline}`);
          const data = await response.json();

          if (!response.ok) {
            throw new Error(data.detail || "Impossible de suivre la tâche.");
          }

          const task = data?.data || {};
          const taskStatus = task.status || "unknown";
          updateProgress(progressFillId, progressLabelId, taskStatus);

          if (taskStatus === "done") {
            resultBox.className = "result-box success";
            resultBox.innerHTML = `
              <div class="result-title">Prédiction terminée</div>
              <div><strong>Score anomalie :</strong> ${task.anomaly_score ?? "N/A"}</div>
              <div><strong>Label prédit :</strong> ${task.pred_label ?? "N/A"}</div>
              <div class="result-actions">
                <a class="result-link" href="/ui/tasks/${taskId}?pipeline=${pipeline}" target="_blank">
                  Voir le détail
                </a>
              </div>
            `;
            return;
          }

          if (taskStatus === "failed") {
            resultBox.className = "result-box error";
            resultBox.innerHTML = `
              <div class="result-title">Prédiction échouée</div>
              <div>${task.error_message || "Une erreur est survenue."}</div>
              <div class="result-actions">
                <a class="result-link" href="/ui/tasks/${taskId}?pipeline=${pipeline}" target="_blank">
                  Voir le détail
                </a>
              </div>
            `;
            return;
          }

          await new Promise(resolve => setTimeout(resolve, 2000));
        } catch (error) {
          resultBox.className = "result-box error";
          resultBox.innerHTML = `
            <div class="result-title">Erreur de suivi</div>
            <div>${error}</div>
          `;
          return;
        }
      }

      resultBox.className = "result-box error";
      resultBox.innerHTML = `
        <div class="result-title">Timeout</div>
        <div>Le suivi de la tâche a dépassé le temps maximum d'attente.</div>
      `;
    }

    function registerPredictionForm(formId, pipeline, resultId, progressWrapperId, progressFillId, progressLabelId) {
      const form = document.getElementById(formId);
      const resultBox = document.getElementById(resultId);
      const progressWrapper = document.getElementById(progressWrapperId);

      if (!form) {
        console.error("Form introuvable:", formId);
        return;
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(form);
        const submitButton = form.querySelector('button[type="submit"]');
        const category = formData.get("category");

        if (pipeline === "2d") {
          const fileInput = form.querySelector('input[name="file"]');

          if (!fileInput || !fileInput.files.length) {
            resultBox.className = "result-box error";
            resultBox.innerHTML = `
              <div class="result-title">Fichier manquant</div>
              <div>Merci de sélectionner une image avant de lancer la prédiction.</div>
            `;
            return;
          }
        }

        if (pipeline === "3d") {
          const rgbInput = form.querySelector('input[name="rgb_file"]');
          const depthInput = form.querySelector('input[name="depth_file"]');

          if (!rgbInput || !rgbInput.files.length) {
            resultBox.className = "result-box error";
            resultBox.innerHTML = `
              <div class="result-title">Fichier RGB manquant</div>
              <div>Merci de sélectionner une image RGB.</div>
            `;
            return;
          }

          if (!depthInput || !depthInput.files.length) {
            resultBox.className = "result-box error";
            resultBox.innerHTML = `
              <div class="result-title">Carte de profondeur manquante</div>
              <div>Merci de sélectionner une carte de profondeur.</div>
            `;
            return;
          }
        }

        if (!category) {
          resultBox.className = "result-box error";
          resultBox.innerHTML = `
            <div class="result-title">Catégorie manquante</div>
            <div>Merci de sélectionner une catégorie.</div>
          `;
          return;
        }

        submitButton.disabled = true;
        submitButton.textContent = "Envoi en cours...";

        if (progressWrapper) progressWrapper.style.display = "block";
        updateProgress(progressFillId, progressLabelId, "queued");

        resultBox.className = "result-box loading";
        resultBox.innerHTML = `
          <div class="result-title">Traitement en cours</div>
          <div>Envoi de l’image au pipeline ${pipeline.toUpperCase()}...</div>
        `;

        try {
          const response = await fetch(pipeline === '3d' ? '/predict/3d-mm' : `/predict/${pipeline}`, {
            method: "POST",
            body: formData
          });

          const data = await response.json();

          if (!response.ok) {
            resultBox.className = "result-box error";
            resultBox.innerHTML = `
              <div class="result-title">Erreur</div>
              <div>${data.detail || data.message || "Une erreur est survenue."}</div>
            `;
            updateProgress(progressFillId, progressLabelId, "failed");
            return;
          }

          const taskId = data?.data?.task_id;

          resultBox.className = "result-box loading";
          resultBox.innerHTML = `
            <div class="result-title">Tâche créée</div>
            <div><strong>Task ID :</strong> ${taskId}</div>
            <div><strong>Pipeline :</strong> ${pipeline.toUpperCase()}</div>
            <div><strong>Catégorie :</strong> ${category}</div>
          `;

          await pollTaskStatus(taskId, pipeline, resultBox, progressFillId, progressLabelId);
        } catch (error) {
          resultBox.className = "result-box error";
          resultBox.innerHTML = `
            <div class="result-title">Erreur réseau</div>
            <div>${error}</div>
          `;
          updateProgress(progressFillId, progressLabelId, "failed");
        } finally {
          submitButton.disabled = false;
          submitButton.textContent = pipeline === "2d"
            ? "Lancer la prédiction 2D"
            : "Lancer la prédiction 3D";
        }
      });
    }

    document.addEventListener("DOMContentLoaded", () => {
      loadCategories();

      registerPredictionForm(
        "predict-form-2d",
        "2d",
        "predict-result-2d",
        "progress-wrapper-2d",
        "progress-fill-2d",
        "progress-label-2d"
      );

      registerPredictionForm(
        "predict-form-3d",
        "3d",
        "predict-result-3d",
        "progress-wrapper-3d",
        "progress-fill-3d",
        "progress-label-3d"
      );
    });
  </script>
</body>
</html>
    """
@app.get(
    "/categories",
    tags=["Prediction"],
    summary="Liste des catégories disponibles pour les pipelines 2D et 3D",
    description="Retourne les catégories distinctes disponibles dans les tables dataset 2D et 3D.",
)
def categories() -> Dict[str, Any]:
    return {
        "status": "success",
        "data": {
            "2d": get_distinct_categories_from_table("mvtec_anomaly_detection"),
            "3d": get_distinct_categories_from_table("mvtec_3d_anomaly_detection"),
        },
    }

@app.get(
    "/admin", 
    response_class=HTMLResponse, 
    tags=["UI"],
    summary="Centre d'administration",
    description="""
Interface d'administration de la plateforme.

Elle permet d'accéder rapidement à :

• la documentation API  
• les endpoints de monitoring  
• les outils MLOps (MLflow, Grafana, Prometheus, Prefect)  
• l'administration des modèles  

Cette page est conçue comme un portail de supervision
de l'architecture technique.
""",
    response_description="Interface web d'administration"
)

def admin_page() -> str:
    return f"""
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <title>Admin — PFE Anomaly Detection Platform</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {{
      --bg: #07111f;
      --card: #121d35;
      --card2: #1a2748;
      --text: #eef4ff;
      --muted: #b7c5df;
      --primary: #60a5fa;
      --accent: #34d399;
      --border: rgba(255,255,255,0.08);
      --shadow: 0 18px 48px rgba(0,0,0,0.35);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(96,165,250,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(52,211,153,0.12), transparent 24%),
        var(--bg);
      padding: 28px;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}

    .header {{
      margin-bottom: 24px;
    }}

    .header h1 {{
      margin: 0 0 10px;
      font-size: 2.4rem;
    }}

    .header p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }}

    .top-actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}

    .btn {{
      display: inline-block;
      padding: 12px 16px;
      border-radius: 14px;
      text-decoration: none;
      font-weight: 700;
      border: 1px solid var(--border);
    }}

    .btn-primary {{
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: #07111f;
      border: none;
    }}

    .btn-secondary {{
      background: rgba(255,255,255,0.04);
      color: var(--text);
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
      margin-top: 28px;
    }}

    .card {{
      background: linear-gradient(180deg, var(--card), var(--card2));
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}

    .card h2 {{
      margin-top: 0;
      font-size: 1.2rem;
    }}

    .card p {{
      color: var(--muted);
      line-height: 1.7;
      margin-bottom: 18px;
    }}

    .tool-link {{
      display: block;
      margin-top: 10px;
      color: var(--text);
      text-decoration: none;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
    }}

    .reload-actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .reload-actions form {{
      margin: 0;
    }}

    .reload-actions button {{
      border: none;
      padding: 12px 16px;
      border-radius: 12px;
      font-weight: 700;
      cursor: pointer;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: #07111f;
    }}

    .note {{
      margin-top: 18px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.95rem;
    }}

    @media (max-width: 980px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Centre d’administration</h1>
      <p>
        Portail d’administration, d’observabilité et de supervision de la plateforme
        de détection d’anomalies 2D / 3D.
      </p>
      <div class="top-actions">
        <a class="btn btn-primary" href="/">Retour à l’accueil</a>
        <a class="btn btn-secondary" href="{PUBLIC_API_URL}/docs" target="_blank">Swagger</a>
        <a class="btn btn-secondary" href="{PUBLIC_API_URL}/redoc" target="_blank">ReDoc</a>
        <a class="btn btn-secondary" href="{PUBLIC_API_URL}/ui/health" target="_blank">Health global</a>
        <a class="btn btn-secondary" href="{PUBLIC_API_URL}/metrics" target="_blank">Metrics</a>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Documentation & supervision</h2>
        <p>Accès rapide à la documentation API et aux endpoints de monitoring.</p>
        <a class="tool-link" href="{PUBLIC_API_URL}/docs" target="_blank">Swagger UI</a>
        <a class="tool-link" href="{PUBLIC_API_URL}/redoc" target="_blank">ReDoc</a>
        <a class="tool-link" href="{PUBLIC_API_URL}/ui/health" target="_blank">Health global</a>
        <a class="tool-link" href="{PUBLIC_API_URL}/ui/health/2d" target="_blank">Health pipeline 2D</a>
        <a class="tool-link" href="{PUBLIC_API_URL}/ui/health/3d" target="_blank">Health pipeline 3D</a>
        <a class="tool-link" href="{PUBLIC_API_URL}/metrics" target="_blank">Metrics Prometheus</a>
      </div>

      <div class="card">
        <h2>Outils plateforme</h2>
        <p>Liens vers les composants techniques et outils MLOps exposés publiquement.</p>
        <a class="tool-link" href="{MLFLOW_PUBLIC_URL}" target="_blank">MLflow</a>
        <a class="tool-link" href="{PREFECT_PUBLIC_URL}" target="_blank">Prefect</a>
        <a class="tool-link" href="{GRAFANA_PUBLIC_URL}" target="_blank">Grafana</a>
        <a class="tool-link" href="{PROMETHEUS_PUBLIC_URL}" target="_blank">Prometheus</a>
        <a class="tool-link" href="{ADMINER_PUBLIC_URL}" target="_blank">Adminer</a>
        <a class="tool-link" href="{IMAGES_PUBLIC_UI_URL}" target="_blank">Image Server</a>
      </div>

      <div class="card">
        <h2>Administration des modèles</h2>
        <p>Déclenchement manuel du rechargement des modèles servis par les workers.</p>
        <div class="reload-actions">
          <form action="/admin/reload-model/2d" method="post">
            <button type="submit">Reload modèle 2D</button>
          </form>
          <form action="/admin/reload-model/3d" method="post">
            <button type="submit">Reload modèle 3D</button>
          </form>
        </div>
        <div class="note">
          Tous les liens ci-dessus utilisent les URLs publiques exposées via Ingress.
        </div>
      </div>
    </div>
  </div>
</body>
</html>
    """


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@app.post(
    "/predict/{pipeline}", 
    tags=["Prediction"],
    summary="Soumettre une image pour détection d'anomalie",
    description="""
Soumet une image au pipeline d'inférence.

Le pipeline est sélectionné via le paramètre `pipeline` :

• `2d` — analyse d’images industrielles  
• `3d` — analyse de données MVTec 3D-AD  

Processus :

1️⃣ validation du fichier  
2️⃣ sauvegarde de l’image  
3️⃣ création d’une tâche en base  
4️⃣ publication dans RabbitMQ  
5️⃣ traitement par un worker  

Le résultat est ensuite stocké en base et accessible via `/tasks/{task_id}`.
""",
    response_description="Tâche créée et placée en file d'attente"
)

async def predict(
    pipeline: Pipeline,
    file: Optional[UploadFile] = File(None),
    rgb_file: Optional[UploadFile] = File(None),
    depth_file: Optional[UploadFile] = File(None),
    category: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    model_version: Optional[str] = Form(None),
) -> Dict[str, Any]:

    cfg = get_pipeline_config(pipeline)

    category = category or cfg["category_default"]
    model_name = model_name or cfg["model_name_default"]
    model_version = model_version or cfg["model_version_default"]

    saved = None
    saved_rgb = None
    saved_depth = None

    if pipeline.value == "2d":
        if file is None:
            raise HTTPException(status_code=400, detail="Le fichier image est obligatoire pour le pipeline 2D.")
        saved = await save_uploaded_file(file, pipeline)

    elif pipeline.value == "3d":
        if rgb_file is None:
            raise HTTPException(status_code=400, detail="Le fichier RGB est obligatoire pour le pipeline 3D.")
        if depth_file is None:
            raise HTTPException(status_code=400, detail="Le fichier depth est obligatoire pour le pipeline 3D.")

        saved_rgb = await save_uploaded_file(rgb_file, pipeline)
        saved_depth = await save_uploaded_file(depth_file, pipeline)

    main_saved = saved if pipeline.value == "2d" else saved_rgb

    try:
        task_id = insert_task(
            pipeline=pipeline,
            image_path=main_saved["image_path"],
            image_url=main_saved["image_url"],
            category=category,
            model_name=model_name,
            model_version=model_version,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    task_payload: Dict[str, Any] = {
        "task_id": task_id,
        "task_type": cfg["task_type"],
        "pipeline": pipeline.value,
        "image_url": main_saved["image_url"],
        "image_path": main_saved["image_path"],
        "category": category,
        "model_name": model_name,
        "model_version": model_version,
        "output_prefix": f"{cfg['output_prefix']}/{task_id}",
        "submitted_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    if pipeline.value == "3d":
      task_payload["model_type"] = "mm_patchcore"
      task_payload["depth_path"] = saved_depth["image_path"]
      task_payload["depth_url"] = saved_depth["image_url"]

    try:
        publish_task(pipeline, task_payload)
    except Exception as exc:  # noqa: BLE001
        try:
            update_task_status(
                pipeline=pipeline,
                task_id=task_id,
                status="failed",
                error_message=f"Publication RabbitMQ échouée: {exc}",
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=503,
            detail=f"Erreur RabbitMQ : {exc}",
        ) from exc

    tasks_created_total.labels(
        pipeline=pipeline.value,
        category=category,
    ).inc()

    return {
        "status": "queued",
        "message": "Tâche créée avec succès.",
        "data": {
            "task_id": task_id,
            "pipeline": pipeline.value,
            "image_url": main_saved["image_url"],
            "image_path": main_saved["image_path"],
            "category": category,
            "model_name": model_name,
            "model_version": model_version,
            "depth_url": saved_depth["image_url"] if pipeline.value == "3d" else None,
            "depth_path": saved_depth["image_path"] if pipeline.value == "3d" else None,
        },
    }


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@app.get(
    "/tasks/{task_id}", 
    tags=["Tasks"],
    summary="Consulter l'état d'une tâche",
    description="""
Retourne l'état d'une tâche d'inférence.

Une tâche peut être dans les états suivants :

• `pending` — en attente dans RabbitMQ  
• `running` — en cours de traitement par un worker  
• `done` — traitement terminé  
• `failed` — erreur lors du traitement  

Cette route permet de récupérer :

• l'image d'origine  
• le modèle utilisé  
• l'état du traitement  
• les résultats d'inférence
""",
    response_description="Informations complètes sur la tâche"
)

def task_status(
    task_id: int,
    pipeline: Pipeline = Query(..., description="2d ou 3d"),
) -> Dict[str, Any]:
    try:
        task = get_task(task_id, pipeline)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Tâche {pipeline.value} {task_id} introuvable.",
        )

    return {
        "status": "success",
        "data": task,
    }


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


@app.post(
    "/admin/reload-model/{pipeline}",
    tags=["Admin"],
    summary="Recharger un modèle",
    description="""
Demande au worker de recharger le modèle depuis MLflow.

Cette opération permet :

• de déployer une nouvelle version du modèle  
• de synchroniser le worker avec MLflow  
• d'actualiser l'inférence sans redéployer le service  

Le rechargement est déclenché via l'endpoint admin du worker.
""",
    response_description="Rechargement du modèle déclenché"
)

async def reload_model(
    pipeline: Pipeline,
    force: bool = False,
) -> Dict[str, Any]:
    result = await call_worker_reload(pipeline, force)

    return {
        "status": "success",
        "message": f"Reload modèle demandé pour le pipeline {pipeline.value}.",
        "data": result,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get(
    "/health/{pipeline}",
    tags=["Health"],
    summary="État d'un pipeline",
    description="""
Retourne l'état de santé détaillé d'un pipeline.

Les vérifications incluent :

• connexion PostgreSQL
• disponibilité RabbitMQ
• état du worker associé
• état MLflow
""",
    response_description="État de santé détaillé d'un pipeline"
)
async def health_pipeline_json(pipeline: Pipeline) -> Dict[str, Any]:
    cfg = get_pipeline_config(pipeline)

    result: Dict[str, Any] = {
        "status": "ok",
        "pipeline": pipeline.value,
        "display_name": cfg["display_name"],
        "checks": {
            "database": None,
            "rabbitmq": None,
            "worker": None,
            "mlflow": None,
        },
    }

    db_status = check_db_connection()
    result["checks"]["database"] = {"status": db_status}
    if db_status != "ok":
        result["status"] = "degraded"

    rabbit_status = check_rabbit_connection(pipeline)
    result["checks"]["rabbitmq"] = {"status": rabbit_status}
    if rabbit_status != "ok":
        result["status"] = "degraded"

    worker_status = await ping_worker(pipeline)
    result["checks"]["worker"] = worker_status
    if worker_status["status"] != "ok":
        result["status"] = "degraded"

    mlflow_health = get_pipeline_mlflow_health(pipeline)
    result["checks"]["mlflow"] = mlflow_health
    if mlflow_health["status"] != "ok":
        result["status"] = "degraded"

    prod_version = get_pipeline_production_version(pipeline)
    if prod_version is not None:
        model_production_version.labels(
            pipeline=pipeline.value,
            model_name=cfg["mlflow_model_name"],
        ).set(float(prod_version))

    return result


@app.get(
    "/health",
    tags=["Health"],
    summary="État global de la plateforme",
    description="""
Retourne l'état global de la plateforme.

Les vérifications incluent :

• connexion PostgreSQL
• état global des pipelines
• disponibilité RabbitMQ
• état MLflow
""",
    response_description="État de santé global de la plateforme"
)
async def health_json() -> Dict[str, Any]:
    global_status = "ok"
    pipelines: Dict[str, Any] = {}

    db_status = check_db_connection()

    for pipeline in Pipeline:
        pipe_health = await health_pipeline_json(pipeline)
        pipelines[pipeline.value] = pipe_health

        if pipe_health["status"] != "ok":
            global_status = "degraded"

    if db_status != "ok":
        global_status = "degraded"

    return {
        "status": global_status,
        "application": APP_TITLE,
        "version": APP_VERSION,
        "database": db_status,
        "pipelines": pipelines,
    }


@app.get(
    "/ui/health",
    response_class=HTMLResponse,
    tags=["UI"],
    summary="Vue HTML du health global",
    description="Affiche l'état global de la plateforme dans une interface web."
)
async def health_ui() -> str:
    data = await health_json()

    def render_pipeline_card(pipe_key: str, pipe_data: Dict[str, Any]) -> str:
        checks = pipe_data.get("checks", {})

        def check_status(name: str, value: Dict[str, Any]) -> str:
            st = value.get("status", "unknown") if isinstance(value, dict) else "unknown"
            cls = "ok" if st == "ok" else "degraded"
            return f'<div class="kv"><span>{name}</span><strong class="{cls}">{st}</strong></div>'

        pipeline_label = pipe_data.get("pipeline", pipe_key)

        return f"""
        <div class="card">
          <h3>{pipe_data.get("display_name", pipe_key)}</h3>
          <div class="kv"><span>Statut pipeline</span><strong class="{'ok' if pipe_data.get('status') == 'ok' else 'degraded'}">{pipe_data.get('status')}</strong></div>
          {check_status("Database", checks.get("database", {}))}
          {check_status("RabbitMQ", checks.get("rabbitmq", {}))}
          {check_status("Worker", checks.get("worker", {}))}
          {check_status("MLflow", checks.get("mlflow", {}))}
          <div class="actions">
            <a class="btn btn-secondary" href="/ui/health/{pipeline_label}">Détail pipeline</a>
          </div>
        </div>
        """

    pipeline_cards = "".join(
        render_pipeline_card(pipe_key, pipe_data)
        for pipe_key, pipe_data in data.get("pipelines", {}).items()
    )

    global_class = "ok" if data.get("status") == "ok" else "degraded"

    return f"""
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <title>Health global — PFE API</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {{
      --bg: #07111f;
      --card: #121d35;
      --card2: #1a2748;
      --text: #eef4ff;
      --muted: #b7c5df;
      --primary: #60a5fa;
      --accent: #34d399;
      --border: rgba(255,255,255,0.08);
      --shadow: 0 18px 48px rgba(0,0,0,0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(96,165,250,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(52,211,153,0.12), transparent 24%),
        var(--bg);
      padding: 28px;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .header {{ display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
    .header h1 {{ margin:0 0 8px; font-size:2.2rem; }}
    .header p {{ margin:0; color:var(--muted); }}
    .badge {{
      display:inline-block; padding:10px 14px; border-radius:999px; font-weight:800;
      background: rgba(255,255,255,0.06); border: 1px solid var(--border);
    }}
    .badge.ok {{ color:#b6f7dd; background:rgba(52,211,153,0.15); }}
    .badge.degraded {{ color:#fecaca; background:rgba(248,113,113,0.15); }}
    .grid {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:22px; }}
    .card {{
      background: linear-gradient(180deg, var(--card), var(--card2));
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .card h3 {{ margin-top:0; }}
    .kv {{
      display:flex; justify-content:space-between; gap:16px;
      padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.06);
    }}
    .kv span {{ color:var(--muted); }}
    .ok {{ color:#86efac; }}
    .degraded {{ color:#fca5a5; }}
    .actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }}
    .btn {{
      display:inline-block; padding:12px 16px; border-radius:12px;
      text-decoration:none; font-weight:700;
    }}
    .btn-primary {{ background: linear-gradient(135deg, var(--primary), var(--accent)); color:#07111f; }}
    .btn-secondary {{ background: rgba(255,255,255,0.05); color:var(--text); border:1px solid var(--border); }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>Health global</h1>
        <p>État global de la plateforme d’anomaly detection</p>
      </div>
      <div class="badge {global_class}">{data.get("status", "unknown").upper()}</div>
    </div>

    <div class="actions" style="margin-bottom:22px;">
      <a class="btn btn-primary" href="/">Retour à l’accueil</a>
      <a class="btn btn-secondary" href="/admin">Centre d’administration</a>
    </div>

    <div class="card" style="margin-bottom:22px;">
      <h3>Vue d’ensemble</h3>
      <div class="kv"><span>Application</span><strong>{data.get("application", "N/A")}</strong></div>
      <div class="kv"><span>Version</span><strong>{data.get("version", "N/A")}</strong></div>
      <div class="kv"><span>Base de données</span><strong class="{'ok' if data.get('database') == 'ok' else 'degraded'}">{data.get("database", "unknown")}</strong></div>
    </div>

    <div class="grid">
      {pipeline_cards}
    </div>
  </div>
</body>
</html>
    """


@app.get(
    "/ui/health/{pipeline}",
    response_class=HTMLResponse,
    tags=["UI"],
    summary="Vue HTML du health pipeline",
    description="Affiche l'état détaillé d'un pipeline dans une interface web."
)
async def health_pipeline_ui(pipeline: Pipeline) -> str:
    data = await health_pipeline_json(pipeline)
    checks = data.get("checks", {})

    def line(label: str, block: Dict[str, Any]) -> str:
        st = block.get("status", "unknown") if isinstance(block, dict) else "unknown"
        cls = "ok" if st == "ok" else "degraded"
        details = ""
        if isinstance(block, dict):
            for key, value in block.items():
                if key != "status":
                    details += f"<div class='subline'><span>{key}</span><strong>{value}</strong></div>"
        return f"""
        <div class="card">
          <h3>{label}</h3>
          <div class="kv"><span>Statut</span><strong class="{cls}">{st}</strong></div>
          {details}
        </div>
        """

    return f"""
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <title>Health {pipeline.value.upper()} — PFE API</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {{
      --bg: #07111f;
      --card: #121d35;
      --card2: #1a2748;
      --text: #eef4ff;
      --muted: #b7c5df;
      --primary: #60a5fa;
      --accent: #34d399;
      --border: rgba(255,255,255,0.08);
      --shadow: 0 18px 48px rgba(0,0,0,0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin:0; font-family:Inter, Arial, sans-serif; color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(96,165,250,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(52,211,153,0.12), transparent 24%),
        var(--bg);
      padding:28px;
    }}
    .container {{ max-width:1100px; margin:0 auto; }}
    .header {{ display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
    .header h1 {{ margin:0 0 8px; font-size:2.2rem; }}
    .header p {{ margin:0; color:var(--muted); }}
    .badge {{
      display:inline-block; padding:10px 14px; border-radius:999px; font-weight:800;
      background: rgba(255,255,255,0.06); border:1px solid var(--border);
    }}
    .badge.ok {{ color:#86efac; background:rgba(52,211,153,0.15); }}
    .badge.degraded {{ color:#fca5a5; background:rgba(248,113,113,0.15); }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
    .card {{
      background: linear-gradient(180deg, var(--card), var(--card2));
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .card h3 {{ margin-top:0; }}
    .kv, .subline {{
      display:flex; justify-content:space-between; gap:16px;
      padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.06);
    }}
    .kv span, .subline span {{ color:var(--muted); }}
    .ok {{ color:#86efac; }}
    .degraded {{ color:#fca5a5; }}
    .actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:22px; }}
    .btn {{
      display:inline-block; padding:12px 16px; border-radius:12px;
      text-decoration:none; font-weight:700;
    }}
    .btn-primary {{ background: linear-gradient(135deg, var(--primary), var(--accent)); color:#07111f; }}
    .btn-secondary {{ background: rgba(255,255,255,0.05); color:var(--text); border:1px solid var(--border); }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>Health pipeline {pipeline.value.upper()}</h1>
        <p>{data.get("display_name", pipeline.value)}</p>
      </div>
      <div class="badge {'ok' if data.get('status') == 'ok' else 'degraded'}">{data.get("status", "unknown").upper()}</div>
    </div>

    <div class="actions">
      <a class="btn btn-primary" href="/ui/health">Retour health global</a>
      <a class="btn btn-secondary" href="/admin">Centre d’administration</a>
    </div>

    <div class="grid">
      {line("Database", checks.get("database", {}))}
      {line("RabbitMQ", checks.get("rabbitmq", {}))}
      {line("Worker", checks.get("worker", {}))}
      {line("MLflow", checks.get("mlflow", {}))}
    </div>
  </div>
</body>
</html>
    """

@app.get(
    "/ui/tasks/{task_id}",
    response_class=HTMLResponse,
    tags=["UI"],
    summary="Vue HTML d'une tâche",
    description="Affiche une tâche dans une interface web lisible."
)
def task_status_ui(
    task_id: int,
    pipeline: Pipeline = Query(..., description="2d ou 3d"),
) -> str:
    task_response = task_status(task_id, pipeline)
    task = task_response["data"]

    status = task.get("status", "unknown")
    status_class = {
        "pending": "warning",
        "running": "info",
        "done": "success",
        "failed": "error",
    }.get(status, "info")

    image_url = task.get("image_url")
    error_message = task.get("error_message")
    anomaly_score = task.get("anomaly_score")
    pred_label = task.get("pred_label")

    image_block = ""
    if image_url:
        image_block = f"""
        <div class="card">
          <h3>Image soumise</h3>
          <img src="{image_url}" alt="Image soumise" class="preview-image" />
          <div class="actions">
            <a class="btn btn-secondary" href="{image_url}" target="_blank">Ouvrir l’image</a>
          </div>
        </div>
        """


    # --- Heatmap & Overlay ---
    outputs_subdir = "outputs_2d" if pipeline.value == "2d" else "outputs_3d"
    heatmap_name = "heatmap.png" if pipeline.value == "2d" else "heatmap_fused.png"
    overlay_name = "overlay.png" if pipeline.value == "2d" else "overlay_fused.png"
    heatmap_url = f"http://images.exemple.com/{outputs_subdir}/{task_id}/{heatmap_name}"
    overlay_url = f"http://images.exemple.com/{outputs_subdir}/{task_id}/{overlay_name}"

    heatmap_block = ""
    if status == "done" and pred_label == "anomaly":
        heatmap_block = f"""
        <div class="card">
          <h3>Localisation de l\'anomalie</h3>
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

    result_block = ""
    if status == "failed" and error_message:
        result_block = f"""
        <div class="card border-error">
          <h3>Erreur</h3>
          <p>{error_message}</p>
        </div>
        """
    else:
        result_block = f"""
        <div class="card">
          <h3>Résultat</h3>
          <div class="kv"><span>Score anomalie</span><strong>{anomaly_score if anomaly_score is not None else "N/A"}</strong></div>
          <div class="kv"><span>Label prédit</span><strong>{pred_label if pred_label is not None else "N/A"}</strong></div>
        </div>
        """

    return f"""
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <title>Tâche #{task_id} — Pipeline {pipeline.value.upper()}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {{
      --bg: #07111f;
      --card: #121d35;
      --card2: #1a2748;
      --text: #eef4ff;
      --muted: #b7c5df;
      --primary: #60a5fa;
      --accent: #34d399;
      --warning: #fbbf24;
      --danger: #f87171;
      --border: rgba(255,255,255,0.08);
      --shadow: 0 18px 48px rgba(0,0,0,0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(96,165,250,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(52,211,153,0.12), transparent 24%),
        var(--bg);
      padding: 28px;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}
    .title h1 {{ margin: 0 0 8px; font-size: 2.2rem; }}
    .title p {{ margin: 0; color: var(--muted); }}
    .badge {{
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      font-weight: 800;
      font-size: 0.9rem;
      border: 1px solid var(--border);
    }}
    .badge.success {{ background: rgba(52,211,153,0.15); color: #b6f7dd; }}
    .badge.warning {{ background: rgba(251,191,36,0.15); color: #fde68a; }}
    .badge.error {{ background: rgba(248,113,113,0.15); color: #fecaca; }}
    .badge.info {{ background: rgba(96,165,250,0.15); color: #bfdbfe; }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 22px;
    }}
    .card {{
      background: linear-gradient(180deg, var(--card), var(--card2));
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 22px;
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }}
    .border-error {{ border-color: rgba(248,113,113,0.35); }}
    .card h3 {{ margin-top: 0; }}
    .kv {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .kv span {{ color: var(--muted); }}
    .preview-image {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--border);
      display: block;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    .btn {{
      display: inline-block;
      padding: 12px 16px;
      border-radius: 12px;
      text-decoration: none;
      font-weight: 700;
    }}
    .btn-primary {{
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: #07111f;
    }}
    .btn-secondary {{
      background: rgba(255,255,255,0.05);
      color: var(--text);
      border: 1px solid var(--border);
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title">
        <h1>Tâche #{task_id}</h1>
        <p>Pipeline {pipeline.value.upper()} • Suivi détaillé d’une prédiction</p>
      </div>
      <div class="badge {status_class}">{status.upper()}</div>
    </div>

    <div class="actions" style="margin-bottom:22px;">
        <a class="btn btn-primary" href="/">Retour à l’accueil</a>
        <a class="btn btn-secondary" href="/ui/health/{pipeline.value}">Voir l’état du pipeline</a>
    </div>

    <div class="grid">
      <div>
        <div class="card">
          <h3>Informations principales</h3>
          <div class="kv"><span>Task ID</span><strong>{task.get("id", task_id)}</strong></div>
          <div class="kv"><span>Pipeline</span><strong>{task.get("pipeline", pipeline.value)}</strong></div>
          <div class="kv"><span>Statut</span><strong>{status}</strong></div>
          <div class="kv"><span>Type</span><strong>{task.get("task_type", "N/A")}</strong></div>
          <div class="kv"><span>Catégorie</span><strong>{task.get("category", "N/A")}</strong></div>
          <div class="kv"><span>Modèle</span><strong>{task.get("model_name", "N/A")}</strong></div>
          <div class="kv"><span>Version modèle</span><strong>{task.get("model_version", "N/A")}</strong></div>
          <div class="kv"><span>Créée le</span><strong>{task.get("created_at", "N/A")}</strong></div>
          <div class="kv"><span>Mise à jour</span><strong>{task.get("updated_at", "N/A")}</strong></div>
        </div>
        {result_block}
      </div>

      <div>
        {image_block}
        {heatmap_block}
      </div>
    </div>
  </div>
</body>
</html>
    """


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Métriques Prometheus",
    description="""
Expose les métriques Prometheus de l'API.

Les métriques incluent :

• nombre de requêtes HTTP  
• latence des endpoints  
• nombre de tâches par pipeline  
• nombre de tâches par statut  
• version du modèle en production  

Cette route est utilisée par Prometheus et Grafana
pour le monitoring de la plateforme.
""",
    response_description="Métriques Prometheus"
)

def metrics():
    for pipeline in Pipeline:
        counts = get_task_counts_by_status(pipeline)
        for status in ("pending", "running", "done", "failed"):
            tasks_by_status.labels(
                pipeline=pipeline.value,
                status=status,
            ).set(counts.get(status, 0))

    return Response(
        generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
