# flows/training_flow_3d_mm.py
"""
Flow Prefect d'orchestration de l'entraînement Multimodal PatchCore 3D sur Slurm.

Appelle : training_3d.src fit-mm
  - Extraction patches multiscale RGB + Depth
  - Coreset greedy reduction
  - Calibration seuils sur validation
  - Évaluation complète (image + pixel + par catégorie) sur val et test
  - Toutes les métriques loguées dans MLflow (~60+)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import paramiko
from prefect import flow, get_run_logger, task

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLURM_HOST: str = os.getenv("SLURM_HOST", "slurm-login.votre-domaine.fr")
SLURM_PORT: int = int(os.getenv("SLURM_PORT", "22"))
SLURM_USER: str = os.getenv("SLURM_USER", "pfe_user")
SLURM_SSH_KEY_PATH: str = os.getenv("SLURM_SSH_KEY_PATH", "/secrets/slurm_ssh_key")
SLURM_PROJECT_DIR: str = os.getenv("SLURM_PROJECT_DIR", "/home/pfe_user/csc8605")
SLURM_MODELS_DIR: str = os.getenv("SLURM_MODELS_DIR", "/mnt/hdd/homes/alauret/csc8605/models")
SLURM_CONDA_ENV: str = os.getenv("SLURM_CONDA_ENV", "pfe_env")

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.exemple.com")
MLFLOW_MODEL_NAME_MM: str = os.getenv("MLFLOW_MODEL_NAME_MM", "mm_patchcore_3d")

SLURM_PARTITION: str = os.getenv("SLURM_PARTITION", "normal")
SLURM_GPUS: str = os.getenv("SLURM_GPUS", "1")
SLURM_CPUS: str = os.getenv("SLURM_CPUS", "8")
SLURM_MEM: str = os.getenv("SLURM_MEM", "32G")
SLURM_TIME: str = os.getenv("SLURM_TIME", "04:00:00")
SLURM_POLL_INTERVAL: int = int(os.getenv("SLURM_POLL_INTERVAL", "30"))
SLURM_JOB_TIMEOUT: int = int(os.getenv("SLURM_JOB_TIMEOUT", "14400"))


@dataclass
class FitMMJobConfig:
    table_name: str = "mvtec_3d_anomaly_detection"
    category: str = ""
    model_version: str = "v1"
    alpha_rgb: float = 0.5
    alpha_depth: float = 0.5
    n_neighbors: int = 1
    max_patches: int = 200000
    image_size: int = 224
    coreset_pre_sample_size: int = 60000
    coreset_proj_dim: int = 128
    slurm_partition: str = SLURM_PARTITION
    slurm_gpus: str = SLURM_GPUS
    slurm_cpus: str = SLURM_CPUS
    slurm_mem: str = SLURM_MEM
    slurm_time: str = SLURM_TIME


# ---------------------------------------------------------------------------
# SSH Helper (identique au flow V1)
# ---------------------------------------------------------------------------

def _ssh_connect() -> paramiko.SSHClient:
    gateway = paramiko.SSHClient()
    gateway.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gateway.connect(
        hostname=os.getenv("SSH_JUMP_HOST", "ssh.exemple.edu"),
        username=SLURM_USER,
        key_filename=SLURM_SSH_KEY_PATH,
        timeout=30,
    )

    transport = gateway.get_transport()
    dest_addr = (os.getenv("SLURM_TARGET_IP", "10.0.0.1"), 22)
    local_addr = ("127.0.0.1", 0)
    channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.getenv("SLURM_TARGET_IP", "10.0.0.1"),
        username=SLURM_USER,
        key_filename=SLURM_SSH_KEY_PATH,
        sock=channel,
        timeout=30,
    )
    client._gateway = gateway
    return client


def _ssh_run(client: paramiko.SSHClient, command: str) -> tuple[str, str, int]:
    _, stdout, stderr = client.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    return (
        stdout.read().decode("utf-8").strip(),
        stderr.read().decode("utf-8").strip(),
        exit_code,
    )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="generate-sbatch-script-3d-mm", retries=0)
def generate_sbatch_script(config: FitMMJobConfig) -> str:
    logger = get_run_logger()

    cat_suffix = f"_{config.category}" if config.category else "_all"
    model_output_dir = f"{SLURM_MODELS_DIR}/mm_patchcore_3d{cat_suffix}_{config.model_version}"

    category_flag = ("    --category " + config.category + " \\\n") if config.category else ""

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --job-name=pfe_fit_mm_3d{cat_suffix}_{config.model_version}\n"
        f"#SBATCH --partition={config.slurm_partition}\n"
        f"#SBATCH --gres=gpu:{config.slurm_gpus}\n"
        f"#SBATCH --cpus-per-task={config.slurm_cpus}\n"
        f"#SBATCH --mem={config.slurm_mem}\n"
        f"#SBATCH --time={config.slurm_time}\n"
        f"#SBATCH --output={SLURM_PROJECT_DIR}/logs/fit_mm_3d_%j.out\n"
        f"#SBATCH --error={SLURM_PROJECT_DIR}/logs/fit_mm_3d_%j.err\n"
        "#SBATCH --mail-type=FAIL\n"
        "\n"
        "source ${CONDA_PROFILE_PATH:-/opt/conda/etc/profile.d/conda.sh}\n"
        f"conda activate {SLURM_CONDA_ENV}\n"
        "\n"
        f'export MLFLOW_TRACKING_URI="{MLFLOW_TRACKING_URI}"\n'
        f'export MLFLOW_MODEL_NAME_MM="{MLFLOW_MODEL_NAME_MM}"\n'
        'export DB_HOST="${DB_HOST:-db.exemple.com}"\n'
        'export DB_PORT="${DB_PORT:-5432}"\n'
        'export DB_NAME="${DB_NAME:-anomaly_detection}"\n'
        'export DB_USER="${DB_USER:-admin}"\n'
        'export DB_PASS="${DB_PASS:-password}"\n'
        'export PFE_IMG_CACHE="${PFE_IMG_CACHE:-/tmp/pfe_img_cache}"\n'
        "\n"
        f"mkdir -p {model_output_dir}\n"
        f"mkdir -p {SLURM_PROJECT_DIR}/logs\n"
        "\n"
        f'echo "[$(date)] Pipeline       : 3D MM-PatchCore"\n'
        f'echo "[$(date)] Category       : {config.category or "all"}"\n'
        f'echo "[$(date)] Version        : {config.model_version}"\n'
        f'echo "[$(date)] Table          : {config.table_name}"\n'
        f'echo "[$(date)] Alpha RGB      : {config.alpha_rgb}"\n'
        f'echo "[$(date)] Alpha Depth    : {config.alpha_depth}"\n'
        f'echo "[$(date)] N neighbors    : {config.n_neighbors}"\n'
        f'echo "[$(date)] Max patches    : {config.max_patches}"\n'
        f'echo "[$(date)] Image size     : {config.image_size}"\n'
        f'echo "[$(date)] MLflow         : {MLFLOW_TRACKING_URI}"\n'
        f'echo "[$(date)] Model name     : {MLFLOW_MODEL_NAME_MM}"\n'
        "\n"
        f"cd {SLURM_PROJECT_DIR}\n"
        "git pull origin main 2>/dev/null || true\n"
        "\n"
        f"python -m training_3d.src fit-mm \\\n"
        f"    --config conf/config.yaml \\\n"
        f"    --table-name {config.table_name} \\\n"
        f"    --model-dir {model_output_dir} \\\n"
        f"    --fit-split train \\\n"
        f"    --val-split validation \\\n"
        f"    --normal-only \\\n"
        f"{category_flag}"
        f"    --alpha-rgb {config.alpha_rgb} \\\n"
        f"    --alpha-depth {config.alpha_depth} \\\n"
        f"    --k {config.n_neighbors} \\\n"
        f"    --max-patches {config.max_patches} \\\n"
        f"    --image-size {config.image_size} \\\n"
        f"    --coreset-pre-sample-size {config.coreset_pre_sample_size} \\\n"
        f"    --coreset-proj-dim {config.coreset_proj_dim}\n"
        "\n"
        "EXIT_CODE=$?\n"
        'echo "[$(date)] Job terminé : $EXIT_CODE"\n'
        "exit $EXIT_CODE\n"
    )

    logger.info(
        f"Script sbatch MM-PatchCore généré : "
        f"category={config.category or 'all'}, version={config.model_version}"
    )
    return script


@task(name="submit-slurm-job-3d-mm", retries=2, retry_delay_seconds=60)
def submit_slurm_job(sbatch_script: str) -> str:
    logger = get_run_logger()
    client = _ssh_connect()

    try:
        timestamp = int(time.time())
        remote_script_path = f"/tmp/pfe_fit_mm_3d_{timestamp}.sh"

        sftp = client.open_sftp()
        try:
            with sftp.open(remote_script_path, 'w') as f:
                f.write(sbatch_script)
        finally:
            sftp.close()

        _ssh_run(client, f"chmod +x {remote_script_path}")

        logger.info(f"Soumission sbatch MM-PatchCore : {remote_script_path}")
        stdout, stderr, exit_code = _ssh_run(client, f"sbatch {remote_script_path}")
        _ssh_run(client, f"rm -f {remote_script_path}")

        if exit_code != 0:
            raise RuntimeError(f"sbatch échoué (exit {exit_code}) : {stderr}")

        if "Submitted batch job" not in stdout:
            raise RuntimeError(f"Sortie sbatch inattendue : {stdout}")

        job_id = stdout.split()[-1]
        logger.info(f"Job Slurm MM-PatchCore soumis : job_id={job_id}")
        return job_id

    finally:
        client.close()


@task(name="wait-for-slurm-job-3d-mm", retries=0)
def wait_for_slurm_job(job_id: str) -> str:
    logger = get_run_logger()
    start_time = time.time()
    last_state = "UNKNOWN"

    while True:
        elapsed = time.time() - start_time
        if elapsed > SLURM_JOB_TIMEOUT:
            raise TimeoutError(
                f"Job MM-PatchCore {job_id} timeout après {SLURM_JOB_TIMEOUT}s"
            )

        client = _ssh_connect()
        try:
            stdout, _, _ = _ssh_run(
                client,
                f"squeue -j {job_id} -h -o '%T' 2>/dev/null || "
                f"sacct -j {job_id} -n -o State --parsable2 2>/dev/null | head -1",
            )
        finally:
            client.close()

        current_state = stdout.strip().upper()

        if not current_state:
            client = _ssh_connect()
            try:
                sacct_out, _, _ = _ssh_run(
                    client,
                    f"sacct -j {job_id} -n -o State --parsable2 | head -1",
                )
            finally:
                client.close()
            current_state = sacct_out.strip().upper().split("|")[0]

        if current_state != last_state:
            logger.info(
                f"Job MM-PatchCore {job_id} : {last_state} → {current_state} "
                f"({elapsed:.0f}s écoulées)"
            )
            last_state = current_state

        if current_state == "COMPLETED":
            logger.info(
                f"Job MM-PatchCore {job_id} terminé avec succès en {elapsed:.0f}s"
            )
            return current_state

        if current_state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
            raise RuntimeError(
                f"Job MM-PatchCore {job_id} en erreur : {current_state}"
            )

        time.sleep(SLURM_POLL_INTERVAL)


@task(name="verify-mlflow-registration-3d-mm", retries=3, retry_delay_seconds=30)
def verify_mlflow_registration(expected_model_name: str) -> dict:
    logger = get_run_logger()

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{expected_model_name}'")
    except Exception as exc:
        raise RuntimeError(f"MLflow Registry injoignable : {exc}") from exc

    if not versions:
        raise RuntimeError(f"Aucune version pour '{expected_model_name}'")

    latest = max(versions, key=lambda v: int(v.version))

    logger.info(
        f"MLflow Registry OK : {expected_model_name} "
        f"version {latest.version} (stage={latest.current_stage}, "
        f"run_id={latest.run_id})"
    )

    # Récupérer les métriques clés du run
    run = client.get_run(latest.run_id)
    metrics_summary = {}
    for key in [
        "eval_test_image_f1", "eval_test_image_auroc", "eval_test_image_ap",
        "eval_test_image_accuracy", "eval_test_image_precision",
        "eval_test_image_recall", "eval_test_pixel_pixel_auroc",
        "eval_test_pixel_pixel_ap", "eval_test_image_best_f1",
        "fit_duration_seconds", "rgb_bank_size", "depth_bank_size",
    ]:
        v = run.data.metrics.get(key)
        if v is not None:
            metrics_summary[key] = v

    logger.info(f"Métriques clés du run :")
    for k, v in metrics_summary.items():
        logger.info(f"  {k}: {v:.4f}")

    return {
        "model_name": expected_model_name,
        "latest_version": str(latest.version),
        "run_id": latest.run_id,
        "stage": latest.current_stage,
        "status": "registered",
        "metrics": metrics_summary,
    }


# ---------------------------------------------------------------------------
# Flow principal MM-PatchCore
# ---------------------------------------------------------------------------

@flow(
    name="train-mm-patchcore-3d-slurm",
    description=(
        "Entraînement Multimodal PatchCore 3D (RGB+Depth) sur MVTec 3D-AD via Slurm H100. "
        "Inclut fit, calibration, évaluation complète (image + pixel + par catégorie), "
        "et logging de 60+ métriques dans MLflow."
    ),
    log_prints=True,
)
def train_mm_patchcore_3d_flow(
    table_name: str = "mvtec_3d_anomaly_detection",
    category: str = "",
    model_version: str = "v1",
    alpha_rgb: float = 0.5,
    alpha_depth: float = 0.5,
    n_neighbors: int = 1,
    max_patches: int = 200000,
    image_size: int = 224,
    slurm_partition: str = SLURM_PARTITION,
    slurm_gpus: str = SLURM_GPUS,
    slurm_time: str = SLURM_TIME,
) -> dict:
    logger = get_run_logger()

    logger.info(
        f"Démarrage flow MM-PatchCore 3D : "
        f"category={category or 'all'}, version={model_version}, "
        f"alpha_rgb={alpha_rgb}, alpha_depth={alpha_depth}, k={n_neighbors}"
    )

    config = FitMMJobConfig(
        table_name=table_name,
        category=category,
        model_version=model_version,
        alpha_rgb=alpha_rgb,
        alpha_depth=alpha_depth,
        n_neighbors=n_neighbors,
        max_patches=max_patches,
        image_size=image_size,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_time=slurm_time,
    )

    sbatch_script = generate_sbatch_script(config)
    job_id = submit_slurm_job(sbatch_script)

    logger.info(f"Job Slurm MM-PatchCore soumis : {job_id}. Attente...")

    final_state = wait_for_slurm_job(job_id)
    mlflow_info = verify_mlflow_registration(MLFLOW_MODEL_NAME_MM)

    result = {
        "pipeline": "3d_mm_patchcore",
        "slurm_job_id": job_id,
        "slurm_final_state": final_state,
        "category": category or "all",
        "model_version": model_version,
        "alpha_rgb": alpha_rgb,
        "alpha_depth": alpha_depth,
        "n_neighbors": n_neighbors,
        **mlflow_info,
    }

    logger.info(
        f"Flow MM-PatchCore 3D terminé.\n"
        f"  Job Slurm    : {job_id} ({final_state})\n"
        f"  Modèle MLflow: {mlflow_info['model_name']} v{mlflow_info['latest_version']}\n"
        f"  → Promouvoir en Production dans l'UI MLflow, "
        f"puis POST /admin/reload-model/3d"
    )

    return result


# ---------------------------------------------------------------------------
# Déploiement
# ---------------------------------------------------------------------------

def deploy_flow() -> None:
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    deployment = Deployment.build_from_flow(
        flow=train_mm_patchcore_3d_flow,
        name="weekly-training-3d-mm",
        schedule=CronSchedule(cron="0 5 * * 1", timezone="Europe/Paris"),
        parameters={
            "table_name": "mvtec_3d_anomaly_detection",
            "category": "",
            "model_version": "auto",
            "alpha_rgb": 0.5,
            "alpha_depth": 0.5,
            "n_neighbors": 1,
            "max_patches": 200000,
            "image_size": 224,
        },
        tags=["training", "slurm", "mlops", "3d", "mm-patchcore", "multimodal"],
        description=(
            "Entraînement hebdomadaire du modèle Multimodal PatchCore 3D "
            "sur MVTec 3D-AD. Inclut évaluation image-level + pixel-level "
            "et logging complet dans MLflow."
        ),
    )
    deployment.apply()
    print("✅ Flow 'train-mm-patchcore-3d-slurm' déployé sur Prefect.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        deploy_flow()
    else:
        train_mm_patchcore_3d_flow()
