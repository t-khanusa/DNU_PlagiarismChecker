from sqlalchemy import create_engine, text

# Define PostgreSQL Connection URL
DATABASE_URL = "postgresql://similarity:123456@localhost:5434/Sentence_Similarity"

# Create engine
engine = create_engine(DATABASE_URL)

# Drop all tables
with engine.connect() as connection:
    # Drop sentences table first due to foreign key constraint
    connection.execute(text("DROP TABLE IF EXISTS sentences CASCADE;"))
    # Drop pdf_files table
    connection.execute(text("DROP TABLE IF EXISTS pdf_files CASCADE;"))
    connection.commit()

print("All tables have been dropped successfully") 