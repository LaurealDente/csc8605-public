# flows/training_flow.py
"""
Flow Prefect d'orchestration de l'entraînement sur le cluster Slurm.

Ce module est le seul endroit où les décorateurs @flow et @task de Prefect
sont utilisés. Il orchestre le cycle complet d'entraînement :

  1. Génération du script sbatch personnalisé
  2. Soumission du job sur le serveur Slurm via SSH
  3. Polling de l'état du job jusqu'à completion
  4. Vérification que MLflow a bien reçu le nouveau modèle
  5. Notification du résultat (succès ou échec)

Ce flow est pensé pour être déclenché :
  - Manuellement depuis l'UI Prefect
  - Via un schedule cron (ex: re-entraînement hebdomadaire)
  - Via l'API Prefect (POST /api/deployments/{id}/create_flow_run)

Architecture :
  [Prefect Server k8s] → SSH → [Slurm login node] → sbatch → [H100 node]
                                                               ↓
  [MLflow Server k8s]  ←────────────────── mlflow log ───────┘

Prérequis :
  - pip install prefect paramiko mlflow
  - Secret Kubernetes "slurm-ssh-key" monté dans le pod Prefect
  - Accès réseau entre le serveur Slurm et le serveur k8s (port MLflow 5000)
"""

from __future__ import annotations

import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

import paramiko
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret

# ---------------------------------------------------------------------------
# Configuration — variables d'environnement pour les deux serveurs
# ---------------------------------------------------------------------------

# --- Serveur Slurm ---
SLURM_HOST: str = os.getenv("SLURM_HOST", "slurm-login.votre-domaine.fr")
SLURM_PORT: int = int(os.getenv("SLURM_PORT", "22"))
SLURM_USER: str = os.getenv("SLURM_USER", "pfe_user")

# Chemin de la clé SSH privée dans le pod Prefect
# Monter le secret k8s "slurm-ssh-key" dans ce chemin via volumeMount
SLURM_SSH_KEY_PATH: str = os.getenv(
    "SLURM_SSH_KEY_PATH",
    "/secrets/slurm_ssh_key",
)

# Chemin du projet worker_2d sur le serveur Slurm
# Le code doit être cloné ou synchronisé sur le serveur GPU
SLURM_PROJECT_DIR: str = os.getenv(
    "SLURM_PROJECT_DIR",
    "/home/pfe_user/worker_2d",
)

# Chemin où les modèles entraînés sont stockés sur le serveur Slurm
# Ce chemin doit être accessible en écriture par le job Slurm
SLURM_MODELS_DIR: str = os.getenv(
    "SLURM_MODELS_DIR",
    "/mnt/hdd/homes/alauret/csc8605/models",
)

# Environnement conda/venv à activer sur Slurm avant d'exécuter le job
SLURM_CONDA_ENV: str = os.getenv("SLURM_CONDA_ENV", "pfe_env")

# --- Serveur Kubernetes (visible depuis Slurm) ---
# MLflow doit être joignable depuis le réseau du serveur Slurm
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://mlflow.example.com",  # URL publique ou IP du service k8s
)
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "resnet_knn_2d")

# --- Paramètres Slurm par défaut pour le job d'entraînement ---
SLURM_PARTITION: str = os.getenv("SLURM_PARTITION", "normal")
SLURM_GPUS: str = os.getenv("SLURM_GPUS", "1")             # 1 H100
SLURM_CPUS: str = os.getenv("SLURM_CPUS", "8")
SLURM_MEM: str = os.getenv("SLURM_MEM", "32G")
SLURM_TIME: str = os.getenv("SLURM_TIME", "02:00:00")       # 2h max

# Intervalle de polling pour vérifier l'état du job Slurm (secondes)
SLURM_POLL_INTERVAL: int = int(os.getenv("SLURM_POLL_INTERVAL", "30"))

# Timeout total d'attente de completion du job (secondes) — 3h par défaut
SLURM_JOB_TIMEOUT: int = int(os.getenv("SLURM_JOB_TIMEOUT", "10800"))


# ---------------------------------------------------------------------------
# Dataclass de configuration du job d'entraînement
# ---------------------------------------------------------------------------


@dataclass
class FitJobConfig:
    """
    Paramètres complets d'un job d'entraînement.

    Regroupe les hyperparamètres métier et les paramètres Slurm pour
    générer le script sbatch correspondant.

    Attributs
    ---------
    backbone : str
        Architecture du backbone ResNet ("resnet18" ou "resnet50").
    table_name : str
        Nom de la table PostgreSQL contenant les images d'entraînement.
    batch_size : int
        Taille des batches pour l'extraction des embeddings.
    num_workers : int
        Nombre de workers PyTorch DataLoader.
    model_version : str
        Tag de version pour nommer le répertoire de sortie du modèle.
    slurm_partition : str
        Partition Slurm à utiliser (gpu, gpu_long…).
    slurm_gpus : str
        Nombre de GPUs à réserver.
    slurm_cpus : str
        Nombre de CPUs à réserver.
    slurm_mem : str
        Mémoire RAM à réserver (ex: "32G").
    slurm_time : str
        Durée maximale du job au format HH:MM:SS.
    """

    backbone: str = "resnet18"
    table_name: str = "mvtec_anomaly_detection"
    batch_size: int = 64
    num_workers: int = 8
    model_version: str = "v1"
    slurm_partition: str = SLURM_PARTITION
    slurm_gpus: str = SLURM_GPUS
    slurm_cpus: str = SLURM_CPUS
    slurm_mem: str = SLURM_MEM
    slurm_time: str = SLURM_TIME
    category: Optional[str] = None
    feature_layer: str = "layer3"
    normal_only: bool = True


# ---------------------------------------------------------------------------
# Helper SSH
# ---------------------------------------------------------------------------


def _ssh_connect() -> paramiko.SSHClient:
    import socket
    
    # Connexion au gateway d'abord
    gateway = paramiko.SSHClient()
    gateway.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gateway.connect(
        hostname=os.getenv("SSH_JUMP_HOST", "ssh.example.edu"),
        username=SLURM_USER,
        key_filename=SLURM_SSH_KEY_PATH,
        timeout=30,
    )
    
    # Tunnel via le gateway vers Slurm
    transport = gateway.get_transport()
    dest_addr = (os.getenv("SLURM_TARGET_IP", "10.0.0.1"), 22)
    local_addr = ("127.0.0.1", 0)
    channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
    
    # Connexion finale à Slurm via le tunnel
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.getenv("SLURM_TARGET_IP", "10.0.0.1"),
        username=SLURM_USER,
        key_filename=SLURM_SSH_KEY_PATH,
        sock=channel,
        timeout=30,
    )
    
    # Garder le gateway en vie (stocker dans le client)
    client._gateway = gateway
    return client


def _ssh_run(client: paramiko.SSHClient, command: str) -> tuple[str, str, int]:
    """
    Exécute une commande shell via SSH et retourne stdout, stderr, exit_code.

    Paramètres
    ----------
    client : paramiko.SSHClient
        Client SSH connecté.
    command : str
        Commande à exécuter sur le serveur distant.

    Retourne
    --------
    tuple[str, str, int]
        (stdout, stderr, exit_code)
    """
    _, stdout, stderr = client.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()

    return (
        stdout.read().decode("utf-8").strip(),
        stderr.read().decode("utf-8").strip(),
        exit_code,
    )


# ---------------------------------------------------------------------------
# Tasks Prefect
# ---------------------------------------------------------------------------


@task(
    name="generate-sbatch-script",
    description="Génère le script sbatch pour le job d'entraînement Slurm",
    retries=0,
)
def generate_sbatch_script(config: FitJobConfig) -> str:
    """
    Génère le contenu du script sbatch à soumettre sur Slurm.

    Le script est conçu pour :
      - Activer l'environnement conda sur le nœud GPU
      - Configurer les variables d'env MLflow pour pointer vers k8s
      - Lancer worker_2d fit avec les bons paramètres
      - Logger dans un fichier de log Slurm horodaté

    Paramètres
    ----------
    config : FitJobConfig
        Configuration complète du job.

    Retourne
    --------
    str
        Contenu complet du script sbatch (à écrire dans un fichier .sh).
    """
    logger = get_run_logger()

    # Chemin de sortie du modèle sur le serveur Slurm
    # Convention : models/<backbone>_<version>/
    model_output_dir = (
        f"{SLURM_MODELS_DIR}/{config.backbone}_{config.model_version}"
    )

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --job-name=pfe_fit_{config.backbone}_{config.model_version}\n"
        f"#SBATCH --partition={config.slurm_partition}\n"
        f"#SBATCH --gres=gpu:{config.slurm_gpus}\n"
        f"#SBATCH --cpus-per-task={config.slurm_cpus}\n"
        f"#SBATCH --mem={config.slurm_mem}\n"
        f"#SBATCH --time={config.slurm_time}\n"
        f"#SBATCH --output={SLURM_PROJECT_DIR}/logs/fit_%j.out\n"
        f"#SBATCH --error={SLURM_PROJECT_DIR}/logs/fit_%j.err\n"
        "#SBATCH --mail-type=FAIL\n"
        "\n"
        "source ${CONDA_PROFILE_PATH:-/opt/conda/etc/profile.d/conda.sh}\n"
        f"conda activate {SLURM_CONDA_ENV}\n"
        "\n"
        f"export MLFLOW_TRACKING_URI=\"{MLFLOW_TRACKING_URI}\"\n"
        f"export MLFLOW_MODEL_NAME=\"{MLFLOW_MODEL_NAME}\"\n"
        "export DB_HOST=\"${DB_HOST:-db.example.com}\"\n"
        "export DB_PORT=\"${DB_PORT:-5432}\"\n"
        "export DB_NAME=\"${DB_NAME:-anomaly_detection}\"\n"
        "export DB_USER=\"${DB_USER:-admin}\"\n"
        "export DB_PASS=\"${DB_PASS:-password}\"\n"
        "export PFE_IMG_CACHE=\"${PFE_IMG_CACHE:-/tmp/pfe_img_cache}\"\n"
        "\n"
        f"mkdir -p {model_output_dir}\n"
        f"mkdir -p {SLURM_PROJECT_DIR}/logs\n"
        "\n"
        f"echo \"[$(date)] Backbone : {config.backbone}\"\n"
        f"echo \"[$(date)] Version  : {config.model_version}\"\n"
        f"echo \"[$(date)] MLflow   : {MLFLOW_TRACKING_URI}\"\n"
        "\n"
        f"cd {SLURM_PROJECT_DIR}\n"
        f"git fetch origin main\n"
	f"git reset --hard origin/main\n"
        "\n"
        f"python -m training.src fit \\\n"
        f"    --config conf/config.yaml \\\n"
        f"    --table-name {config.table_name} \\\n"
        f"    --output-model-dir {model_output_dir} \\\n"
        f"    --backbone {config.backbone} \\\n"
        f"    --batch-size {config.batch_size} \\\n"
        f"    --num-workers {config.num_workers} \\\n"
        f"    --feature-layer {config.feature_layer} \\\n"
        f"    --normal-only"
        f"{' --category ' + config.category if config.category else ''}\n"
        "\n"
        "EXIT_CODE=$?\n"
        "echo \"[$(date)] Job terminé : $EXIT_CODE\"\n"
        "exit $EXIT_CODE\n"
    )

    logger.info(
        f"Script sbatch généré pour backbone={config.backbone} "
        f"version={config.model_version}"
    )

    return script


@task(
    name="submit-slurm-job",
    description="Soumet le script sbatch sur le cluster Slurm via SSH",
    retries=2,
    retry_delay_seconds=60,
)
def submit_slurm_job(sbatch_script: str) -> str:
    """
    Soumet le script sbatch sur le cluster Slurm via SSH.

    Flow :
      1. Connexion SSH au login node
      2. Écriture du script dans /tmp sur le serveur distant
      3. Soumission via sbatch
      4. Extraction du job_id depuis la sortie de sbatch

    Paramètres
    ----------
    sbatch_script : str
        Contenu du script sbatch à soumettre.

    Retourne
    --------
    str
        Identifiant du job Slurm (ex: "12345").

    Lève
    ----
    RuntimeError
        Si sbatch échoue ou ne retourne pas de job_id parseable.
    """
    logger = get_run_logger()

    client = _ssh_connect()

    try:
        # Écriture du script dans /tmp avec un nom unique
        timestamp = int(time.time())
        remote_script_path = f"/tmp/pfe_fit_{timestamp}.sh"

        # Écriture via SFTP — plus fiable que heredoc avec paramiko
        sftp = client.open_sftp()
        try:
            with sftp.open(remote_script_path, 'w') as f:
                f.write(sbatch_script)
        finally:
            sftp.close()
        # Rendre exécutable
        _ssh_run(client, f"chmod +x {remote_script_path}")


        # Soumission du job
        logger.info(f"Soumission du job sbatch : {remote_script_path}")
        stdout, stderr, exit_code = _ssh_run(
            client, f"sbatch {remote_script_path}"
        )

        # Nettoyage du script temporaire (best-effort)
        _ssh_run(client, f"rm -f {remote_script_path}")

        if exit_code != 0:
            raise RuntimeError(
                f"sbatch a échoué (exit {exit_code}) : {stderr}"
            )

        # Sortie attendue de sbatch : "Submitted batch job 12345"
        if "Submitted batch job" not in stdout:
            raise RuntimeError(
                f"Sortie sbatch inattendue (job_id non trouvé) : {stdout}"
            )

        job_id = stdout.split()[-1]
        logger.info(f"Job Slurm soumis avec succès : job_id={job_id}")

        return job_id

    finally:
        client.close()


@task(
    name="wait-for-slurm-job",
    description="Attend la completion du job Slurm en interrogeant squeue",
    # Pas de retry ici : on ne veut pas re-soumettre le job par erreur
    retries=0,
)
def wait_for_slurm_job(job_id: str) -> str:
    """
    Attend la fin du job Slurm en interrogeant régulièrement squeue.

    L'interrogation se fait toutes les SLURM_POLL_INTERVAL secondes.
    Une fois le job terminé, l'état final est retourné.

    États Slurm possibles :
      - PENDING   : en attente de ressources
      - RUNNING   : en cours d'exécution
      - COMPLETED : terminé avec succès
      - FAILED    : terminé en erreur
      - CANCELLED : annulé manuellement
      - TIMEOUT   : durée maximale dépassée

    Paramètres
    ----------
    job_id : str
        Identifiant du job Slurm à surveiller.

    Retourne
    --------
    str
        État final du job ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"…).

    Lève
    ----
    TimeoutError
        Si le job n'est pas terminé après SLURM_JOB_TIMEOUT secondes.
    RuntimeError
        Si le job se termine en FAILED, CANCELLED ou TIMEOUT.
    """
    logger = get_run_logger()

    start_time = time.time()
    last_state = "UNKNOWN"

    while True:
        # Vérification du timeout global
        elapsed = time.time() - start_time
        if elapsed > SLURM_JOB_TIMEOUT:
            raise TimeoutError(
                f"Job Slurm {job_id} toujours en cours après "
                f"{SLURM_JOB_TIMEOUT}s. "
                f"Vérifier sur le serveur Slurm : squeue -j {job_id}"
            )

        client = _ssh_connect()
        try:
            # squeue avec format personnalisé pour extraire uniquement l'état
            stdout, _, _ = _ssh_run(
                client,
                f"squeue -j {job_id} -h -o '%T' 2>/dev/null || "
                f"sacct -j {job_id} -n -o State --parsable2 2>/dev/null | head -1",
            )
        finally:
            client.close()

        current_state = stdout.strip().upper()

        # Le job n'apparaît plus dans squeue → il est terminé
        # On interroge sacct pour obtenir l'état final
        if not current_state or current_state == "":
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
                f"Job {job_id} : {last_state} → {current_state} "
                f"({elapsed:.0f}s écoulées)"
            )
            last_state = current_state

        # États terminaux
        if current_state == "COMPLETED":
            logger.info(f"Job {job_id} terminé avec succès en {elapsed:.0f}s")
            return current_state

        if current_state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
            raise RuntimeError(
                f"Job Slurm {job_id} terminé en erreur : état={current_state}. "
                f"Consulter les logs sur le serveur Slurm : "
                f"cat {SLURM_PROJECT_DIR}/logs/fit_<jobid>.err"
            )

        time.sleep(SLURM_POLL_INTERVAL)


@task(
    name="verify-mlflow-registration",
    description="Vérifie que le modèle a bien été enregistré dans MLflow Registry",
    retries=3,
    retry_delay_seconds=30,
)
def verify_mlflow_registration(expected_model_name: str) -> dict:
    """
    Vérifie qu'une nouvelle version du modèle est apparue dans MLflow Registry.

    Cette tâche interroge directement le serveur MLflow sur k8s pour confirmer
    que le job Slurm a bien réussi à enregistrer son modèle. C'est la validation
    de bout en bout du pipeline d'entraînement.

    Paramètres
    ----------
    expected_model_name : str
        Nom du modèle attendu dans le Registry (ex: "resnet_knn_2d").

    Retourne
    --------
    dict
        {"model_name": str, "latest_version": str, "run_id": str, "status": str}

    Lève
    ----
    RuntimeError
        Si aucune version n'est trouvée ou si le Registry est injoignable.
    """
    logger = get_run_logger()

    import mlflow  # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Récupère toutes les versions (tous stages confondus)
    try:
        versions = client.search_model_versions(f"name='{expected_model_name}'")
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de joindre MLflow Registry à {MLFLOW_TRACKING_URI} : {exc}"
        ) from exc

    if not versions:
        raise RuntimeError(
            f"Aucune version trouvée pour le modèle '{expected_model_name}' "
            f"dans MLflow Registry. "
            f"Le job Slurm a peut-être échoué avant l'enregistrement."
        )

    # Trie par numéro de version décroissant pour prendre la plus récente
    latest = max(versions, key=lambda v: int(v.version))

    logger.info(
        f"MLflow Registry OK : {expected_model_name} "
        f"version {latest.version} (stage={latest.current_stage}, "
        f"run_id={latest.run_id})"
    )
    logger.info(
        f"→ Aller dans l'UI MLflow pour promouvoir la version "
        f"{latest.version} en Production."
    )

    return {
        "model_name": expected_model_name,
        "latest_version": str(latest.version),
        "run_id": latest.run_id,
        "stage": latest.current_stage,
        "status": "registered",
    }


# ---------------------------------------------------------------------------
# Flow principal
# ---------------------------------------------------------------------------


@flow(
    name="train-resnet-knn-slurm",
    description=(
        "Soumet un job d'entraînement ResNet+kNN sur le cluster Slurm H100 "
        "et enregistre le modèle dans MLflow Registry."
    ),
    # Les logs Prefect sont envoyés au serveur Prefect sur k8s
    log_prints=True,
)
def train_resnet_knn_flow(
    backbone: str = "resnet18",
    table_name: str = "mvtec_anomaly_detection",
    batch_size: int = 64,
    num_workers: int = 8,
    model_version: str = "v1",
    slurm_partition: str = SLURM_PARTITION,
    slurm_gpus: str = SLURM_GPUS,
    slurm_time: str = SLURM_TIME,
    category: Optional[str] = None,
    feature_layer: str = "layer3",
    normal_only: bool = True,
) -> dict:
    """
    Flow principal d'entraînement — à déployer sur le serveur Prefect k8s.

    Ce flow est le point d'entrée de l'orchestration MLOps. Il est appelé
    manuellement depuis l'UI Prefect ou automatiquement par un schedule.

    Paramètres (modifiables depuis l'UI Prefect lors du déclenchement)
    ----------
    backbone : str
        Architecture backbone ("resnet18" ou "resnet50").
    table_name : str
        Table PostgreSQL source des images d'entraînement.
    batch_size : int
        Taille des batches pour l'extraction des embeddings.
    num_workers : int
        Nombre de workers DataLoader PyTorch.
    model_version : str
        Tag de version pour nommer le modèle (ex: "v2", "20240315").
    slurm_partition : str
        Partition Slurm à utiliser.
    slurm_gpus : str
        Nombre de GPUs à réserver.
    slurm_time : str
        Durée max du job Slurm (HH:MM:SS).

    Retourne
    --------
    dict
        Résultat complet incluant job_id, état final et infos MLflow.
    """
    logger = get_run_logger()

    logger.info(
        f"Démarrage du flow d'entraînement : "
        f"backbone={backbone}, version={model_version}, table={table_name}"
    )

    # Construction de la configuration du job
    config = FitJobConfig(
        backbone=backbone,
        table_name=table_name,
        batch_size=batch_size,
        num_workers=num_workers,
        model_version=model_version,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        category=category,
        feature_layer=feature_layer,
        normal_only=normal_only,
    )

    # ── Étape 1 : Génération du script sbatch ────────────────────────────
    sbatch_script = generate_sbatch_script(config)

    # ── Étape 2 : Soumission sur Slurm ───────────────────────────────────
    job_id = submit_slurm_job(sbatch_script)

    logger.info(f"Job Slurm soumis : {job_id}. Attente de la completion...")

    # ── Étape 3 : Attente de la fin du job ───────────────────────────────
    final_state = wait_for_slurm_job(job_id)

    # ── Étape 4 : Vérification de l'enregistrement dans MLflow ──────────
    mlflow_info = verify_mlflow_registration(MLFLOW_MODEL_NAME)

    result = {
        "slurm_job_id": job_id,
        "slurm_final_state": final_state,
        "backbone": backbone,
        "model_version": model_version,
        **mlflow_info,
    }

    logger.info(
        f"Flow terminé avec succès.\n"
        f"  Job Slurm    : {job_id} ({final_state})\n"
        f"  Modèle MLflow: {mlflow_info['model_name']} "
        f"version {mlflow_info['latest_version']}\n"
        f"  → Promouvoir la version {mlflow_info['latest_version']} "
        f"en Production dans l'UI MLflow, puis appeler POST /admin/reload-model"
    )

    return result


# ---------------------------------------------------------------------------
# Déploiement du flow (à exécuter une seule fois pour enregistrer le flow)
# ---------------------------------------------------------------------------


def deploy_flow() -> None:
    """
    Enregistre le flow sur le serveur Prefect avec un schedule optionnel.

    Cette fonction est à exécuter manuellement une fois pour créer le
    déploiement dans Prefect :

        python -m flows.training_flow deploy

    Après déploiement, le flow apparaît dans l'UI Prefect et peut être
    déclenché manuellement ou via le schedule configuré ici.
    """
    from prefect.deployments import Deployment  # noqa: PLC0415
    from prefect.server.schemas.schedules import CronSchedule  # noqa: PLC0415

    deployment = Deployment.build_from_flow(
        flow=train_resnet_knn_flow,
        name="weekly-training",
        # Re-entraînement automatique chaque lundi à 2h du matin
        schedule=CronSchedule(cron="0 2 * * 1", timezone="Europe/Paris"),
        # Paramètres par défaut modifiables depuis l'UI lors du déclenchement
        parameters={
            "backbone": "resnet18",
            "table_name": "mvtec_anomaly_detection",
            "batch_size": 64,
            "num_workers": 0,
            "model_version": "auto",
        },
        tags=["training", "slurm", "mlops"],
        description=(
            "Entraînement hebdomadaire automatique du modèle ResNet+kNN "
            "sur le cluster Slurm avec H100."
        ),
    )

    deployment.apply()
    print("✅ Flow 'train-resnet-knn-slurm' déployé sur Prefect.")
    print("   Aller dans l'UI Prefect pour déclencher manuellement ou attendre le schedule.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        deploy_flow()
    else:
        # Exécution directe pour test local (sans Prefect Server)
        train_resnet_knn_flow()


# ---------------------------------------------------------------------------
# --- RÉSUMÉ DU FICHIER ---
#
# AVANT : Ce fichier n'existait pas. Prefect tournait dans le cluster k8s
#   (pod prefect-777c776ffb-p58tf) mais n'orchestrait absolument rien.
#   Aucun @flow, aucun @task, aucun déploiement enregistré. Le pod était
#   un serveur vide qui attendait des instructions qui n'arrivaient jamais.
#   Les entraînements devaient être lancés manuellement en ligne de commande
#   directement sur le serveur disposant du GPU.
#
# MAINTENANT :
#   Ce fichier définit le flow Prefect complet d'orchestration de l'entraînement.
#   Il connecte Prefect à ton infrastructure en deux directions :
#
#   → VERS SLURM (soumission du job) :
#     generate_sbatch_script() construit le script d'entraînement personnalisé
#     submit_slurm_job()       le soumet via SSH sur le login node Slurm
#     wait_for_slurm_job()     poll squeue/sacct jusqu'à completion
#
#   → VERS MLFLOW (vérification) :
#     verify_mlflow_registration() confirme que le modèle est dans le Registry
#
#   Le flow est paramétrable depuis l'UI Prefect : backbone, batch_size,
#   table_name, version, ressources Slurm — tout peut être surchargé au
#   moment du déclenchement sans modifier le code.
#
# POSITION DANS L'ARCHITECTURE :
#   Ce fichier est le chef d'orchestre du pipeline MLOps. Il se place entre :
#     [UI Prefect / schedule cron]
#         → [flows/training_flow.py]  (ce fichier, tourne dans le pod Prefect k8s)
#             → SSH → [Slurm login node]
#                 → sbatch → [H100 node] → [worker_2d fit] → [MLflow k8s]
#
#   Après l'exécution du flow :
#     [Opérateur] → UI MLflow → promote version → POST /admin/reload-model
#         → [worker-2d k8s] recharge le modèle Production
#
#   Pour déployer : python -m flows.training_flow deploy
#   Pour déclencher manuellement : UI Prefect → Deployments → Run
# ---------------------------------------------------------------------------
