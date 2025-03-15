
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Index, Float, DateTime
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from config.postgres_db import Base

class PDFFile(Base):
    __tablename__ = "pdf_files"
    
    pdf_id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, unique=True, nullable=False)
    file_path = Column(String, nullable=False)  # Store full path for reference
    created_at = Column(DateTime, default=datetime.utcnow)
    total_sentences = Column(Integer, default=0)  # Track number of sentences
    sentences = relationship("Sentence", back_populates="pdf", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_filename', 'filename'),  # B-tree for exact filename lookups
        Index('idx_created_at_brin', 'created_at', postgresql_using='brin'),  # BRIN for time-based queries
    )

# Sentences Table
class Sentence(Base):
    __tablename__ = "sentences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_id = Column(Integer, ForeignKey("pdf_files.pdf_id", ondelete="CASCADE"), nullable=False)
    sentence = Column(Text, nullable=False)
    sentence_index = Column(Integer, nullable=False)  # Position in document
    
    pdf = relationship("PDFFile", back_populates="sentences")

    __table_args__ = (
        Index('idx_sentence_pdf', 'pdf_id'),
        Index('idx_pdf_sentence_brin', 'pdf_id', 'sentence_index', postgresql_using='brin', 
              postgresql_with={'pages_per_range': '128'})  # Each range covers ~128 pages
    )