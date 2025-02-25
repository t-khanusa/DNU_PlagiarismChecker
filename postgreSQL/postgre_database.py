from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Index, Float, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from sqlalchemy.schema import DDL
from sqlalchemy import text

# Define PostgreSQL Connection URL
DATABASE_URL = "postgresql://similarity:123456@192.168.167.251:5434/Sentence_Similarity"

engine = create_engine(
    DATABASE_URL,
    pool_size=20,  # Increased pool size
    max_overflow=30,  # Increased max overflow
    pool_timeout=60,  # Increased timeout
    pool_pre_ping=True  # Enable connection health checks
)
Base = declarative_base()

# PDF Files Table
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
        # B-tree index on pdf_id alone for fast initial PDF lookup
        Index('idx_sentence_pdf', 'pdf_id'),
        
        # BRIN index for the pdf_id + sentence_index combination
        # This works well because:
        # 1. Data is naturally ordered (400-500 sentences per PDF)
        # 2. We have 1M+ rows total
        # 3. We typically query ranges within a single PDF
        Index('idx_pdf_sentence_brin', 'pdf_id', 'sentence_index', postgresql_using='brin', 
              postgresql_with={'pages_per_range': '128'})  # Each range covers ~128 pages
    )

# Create Tables in Database
Base.metadata.create_all(engine)

# Create a Session Factory
SessionLocal = sessionmaker(bind=engine)

# Execute all DDL statements within a connection context
with engine.connect() as connection:
    # Configure table storage parameters for optimal performance
    connection.execute(text("""
        ALTER TABLE sentences SET (
            autovacuum_vacuum_scale_factor = 0.1,    -- Vacuum after 10% changes
            autovacuum_analyze_scale_factor = 0.05,  -- Analyze after 5% changes
            fillfactor = 95                          -- Pack pages more fully since updates are rare
        );
    """))
    
    # Create extended statistics
    connection.execute(text("""
        CREATE STATISTICS IF NOT EXISTS sentences_stats (dependencies) 
        ON pdf_id, sentence_index FROM sentences;
    """))
    
    # First, get the threshold value for recent PDFs
    result = connection.execute(text("SELECT COALESCE(MAX(pdf_id), 0) - 100 FROM pdf_files"))
    threshold = result.scalar() or 0
    
    # Create partial index for recent PDFs using the computed threshold
    connection.execute(text(f"""
        CREATE INDEX IF NOT EXISTS idx_recent_pdf_sentence 
        ON sentences (pdf_id, sentence_index) 
        WHERE pdf_id > {threshold};
    """))
    
    # Analyze tables
    connection.execute(text("ANALYZE VERBOSE sentences;"))
    connection.execute(text("ANALYZE VERBOSE pdf_files;"))
    
    # Cluster the sentences table
    connection.execute(text("""
        ALTER TABLE sentences CLUSTER ON idx_sentence_pdf;
        CLUSTER VERBOSE sentences;
    """))
    
    # Commit all changes
    connection.commit()
