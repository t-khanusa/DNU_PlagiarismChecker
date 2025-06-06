from config.postgres_db import engine, Base
from models.model import PDFFile, Sentence

def create_tables():
    """Create all database tables."""
    print("🔧 Creating database tables...")
    
    try:
        # Import models to register them with Base
        from models.model import PDFFile, Sentence
        
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine)
        
        print("✅ Successfully created database tables:")
        print("   - pdf_files table")
        print("   - sentences table")
        print("   - All indexes and constraints")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

if __name__ == "__main__":
    success = create_tables()
    if success:
        print("\n🎉 Database setup complete! You can now run create_corpus.py")
    else:
        print("\n💥 Database setup failed. Please check your configuration.")