from __future__ import annotations
from pfe_data import PFEDataManager

MINIO_BASE = "http://images.exemple.com"
BUCKET = "mvtec_ad_2"

DISK_PREFIX = "/home/mario/database-pfe/images_storage/"
OLD_DATASET_NAME = "mvtec_anomaly_detection"


def to_minio_url(fp: str) -> str:
    if not fp:
        raise ValueError("Empty filepath")

    s = str(fp).strip().replace("\\", "/")

    # 1️⃣ enlever le prefix disque
    if s.startswith(DISK_PREFIX):
        s = s[len(DISK_PREFIX):]

    # 2️⃣ remplacer nom dataset si nécessaire
    if s.startswith(OLD_DATASET_NAME):
        s = s.replace(OLD_DATASET_NAME, BUCKET, 1)

    # 3️⃣ s'assurer qu'on ne double pas le bucket
    if s.startswith(BUCKET + "/"):
        object_key = s[len(BUCKET)+1:]
    else:
        object_key = s

    return f"{MINIO_BASE}/{BUCKET}/{object_key}"


def main():
    dm = PFEDataManager()

    df = dm.get_dataset(
        table="mvtec_ad",
        limit=None,
        load_images=False,
        verbose=True
    )

    df = df[(df["split"] == "train") & (df["label"] == "good")].copy()

    out = "normal_images_from_db.txt"

    with open(out, "w", encoding="utf-8") as f:
        for fp in df["filepath"]:
            url = to_minio_url(fp)
            f.write(url + "\n")

    print(f"✅ wrote {len(df)} urls -> {out}")
    print("🔎 exemple:", to_minio_url(df["filepath"].iloc[0]))


if __name__ == "__main__":
    main()