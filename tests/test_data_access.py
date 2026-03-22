from pfe_data import PFEDataManager

def main():
    dm = PFEDataManager()  # lit conf/config.yaml

    print("1) Test SQL: récupération dataset…")
    df = dm.get_dataset(table="mvtec_anomaly_detection", limit=20, load_images=False, verbose=True)
    if df.empty:
        raise SystemExit("❌ Dataset vide → problème DB/table ou VPN.")

    print("\nAperçu:")
    print(df.head(5))

    print("\n2) Résumé dataset:")
    dm.get_summary(df)

    print("\n3) Test chargement 1 image (local OU via http://images.example.com)")
    sample_path = df.iloc[0]["filepath"]
    img = dm.load_image(sample_path)
    print("✅ Image chargée:", img.size, img.mode)

    out_path = "test_loaded_image.jpg"
    img.save(out_path)
    print(f"✅ Image sauvegardée localement: {out_path}")

    print("\n4) Galerie (ouvre une fenêtre / inline selon ton environnement)")
    dm.show_gallery(df, n=5, seed=42)

if __name__ == "__main__":
    main()
