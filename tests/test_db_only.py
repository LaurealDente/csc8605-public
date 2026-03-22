from sqlalchemy import create_engine, text
from pfe_data import PFEDataManager

dm = PFEDataManager()
engine = create_engine(dm.get_connection_url())

with engine.connect() as conn:
    v = conn.execute(text("SELECT version();")).fetchone()
    print("✅ DB OK:", v[0])