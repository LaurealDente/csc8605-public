from sqlalchemy import create_engine, text
from pfe_data import PFEDataManager

def main():
    dm = PFEDataManager()
    engine = create_engine(dm.get_connection_url())

    with engine.connect() as conn:
        tables = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
            ORDER BY table_name;
        """)).fetchall()

        print("=== Tables (public) ===")
        for (t,) in tables:
            print("-", t)

        # Cherche tables qui ressemblent à des tasks
        task_like = [t[0] for t in tables if "task" in t[0] or "job" in t[0]]
        print("\n=== Tables task/job candidates ===")
        for t in task_like:
            print("-", t)

        # Affiche colonnes pour candidates
        for t in task_like:
            cols = conn.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name=:t
                ORDER BY ordinal_position;
            """), {"t": t}).fetchall()
            print(f"\n--- Columns of {t} ---")
            for c, dt in cols:
                print(f"{c:25s} {dt}")

if __name__ == "__main__":
    main()