from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Index, Float, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from sqlalchemy.schema import DDL
from sqlalchemy import text
from config.conf import DATABASE_USERNAME, DATABASE_PASSWORD, DATABASE_HOSTNAME, DATABASE_PORT, DATABASE_NAME


DATABASE_URL = f'postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOSTNAME}:{DATABASE_PORT}/{DATABASE_NAME}'

engine = create_engine(
    DATABASE_URL,
    pool_size=20,  # Increased pool size
    max_overflow=30,  # Increased max overflow
    pool_timeout=60,  # Increased timeout
    pool_pre_ping=True  # Enable connection health checks
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from models.model import PDFFile, Sentence