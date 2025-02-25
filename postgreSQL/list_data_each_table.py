from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

# Define PostgreSQL Connection URL
DATABASE_URL = "postgresql://similarity:123456@localhost:5434/Sentence_Similarity"

def check_database():
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("Successfully connected to database")
            
            # Get table names
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            print(f"\nFound tables: {table_names}")
            
            # Check pdf_files table
            print("\nChecking pdf_files table:")
            result = conn.execute(text("SELECT COUNT(*) FROM pdf_files"))
            count = result.scalar()
            print(f"Total records in pdf_files: {count}")
            
            if count > 0:
                print("\nPDF Files entries:")
                result = conn.execute(text("""
                    SELECT pdf_id, filename, file_path, created_at, total_sentences 
                    FROM pdf_files 
                    ORDER BY pdf_id
                """))
                for row in result:
                    print(f"ID: {row.pdf_id}, File: {row.filename}")
                    print(f"Path: {row.file_path}")
                    print(f"Created: {row.created_at}")
                    print(f"Total Sentences: {row.total_sentences}")
                    print("-" * 50)
            
            # Check sentences table
            print("\nChecking sentences table:")
            result = conn.execute(text("SELECT COUNT(*) FROM sentences"))
            count = result.scalar()
            print(f"Total records in sentences: {count}")
            
            if count > 0:
                print("\nSample of sentences (first 5):")
                result = conn.execute(text("""
                    SELECT s.id, s.pdf_id, s.sentence_index, s.sentence, p.filename
                    FROM sentences s
                    JOIN pdf_files p ON s.pdf_id = p.pdf_id
                    ORDER BY s.id
                    LIMIT 5
                """))
                for row in result:
                    print(f"ID: {row.id}, PDF ID: {row.pdf_id}, Index: {row.sentence_index}")
                    print(f"From file: {row.filename}")
                    print(f"Sentence: {row.sentence}")
                    print("-" * 50)

    except Exception as e:
        print(f"Error accessing database: {str(e)}")
        raise

if __name__ == "__main__":
    check_database()
