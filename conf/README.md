# Configuration

Fichier de configuration partagé par l'ensemble des composants du projet. Il centralise les paramètres de connexion à la base de données, les chemins vers le stockage d'images et les URLs réseau.

---

## Structure

```
conf/
├── config.yaml      # Configuration principale
└── README.md
```

---

## Format du fichier `config.yaml`

```yaml
database:
  host: "db.alexandremariolauret.org"
  port: 5432
  name: "anomaly_detection"
  user: "admin"
  pass: "password"

paths:
  images_root: "/home/mario/pfe-fast-data/database-pfe/images_storage"

network:
  images_url: "http://images.alexandremariolauret.org"
```

---

## Paramètres

### `database`

| Champ | Description |
|-------|-------------|
| `host` | Adresse du serveur PostgreSQL |
| `port` | Port PostgreSQL (défaut : 5432) |
| `name` | Nom de la base de données |
| `user` | Utilisateur de connexion |
| `pass` | Mot de passe |

### `paths`

| Champ | Description |
|-------|-------------|
| `images_root` | Chemin absolu vers le répertoire racine des images du dataset MVTec |

### `network`

| Champ | Description |
|-------|-------------|
| `images_url` | URL publique du serveur d'images (utilisé par les workers pour télécharger les images) |

---

## Utilisation

Ce fichier est lu par la classe `Settings` présente dans chaque composant (`training/src/config.py`, `worker_3d/app_src/app/config.py`, etc.) :

```python
settings = Settings.from_yaml("conf/config.yaml")
```

Les workers Docker embarquent une copie de ce fichier dans leur image (`COPY conf/ /app/conf/`).
