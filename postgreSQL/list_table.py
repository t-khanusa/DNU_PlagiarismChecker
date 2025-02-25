from sqlalchemy import create_engine, text

# Define PostgreSQL Connection URL
DATABASE_URL = "postgresql://similarity:123456@localhost:5434/Sentence_Similarity"

# Create engine
engine = create_engine(DATABASE_URL)

# Check tables
with engine.connect() as connection:
    result = connection.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
    tables = result.fetchall()

# Print table names
print("Tables in the database:", [table[0] for table in tables])
