"""
database/connection.py
SQLAlchemy 2.0 ryšys su DB.
Naudojama tik 2.0 sintaksė: select() + session.execute() — be session.query().
"""
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base, Customer
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "churn.db")
DB_URL  = f"sqlite:///{os.path.abspath(DB_PATH)}"

engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Sukuria visas lenteles (jei dar neegzistuoja)."""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Grąžina naują DB sesiją."""
    return SessionLocal()


def db_exists() -> bool:
    """
    Tikrina ar DB jau turi klientų duomenų.
    SQLAlchemy 2.0: select() + session.execute() — be session.query().
    """
    try:
        session = SessionLocal()
        stmt = select(func.count()).select_from(Customer)
        count = session.execute(stmt).scalar()
        session.close()
        return count > 0
    except Exception:
        return False
