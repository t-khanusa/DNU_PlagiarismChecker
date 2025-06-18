import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.postgres_db import engine, SessionLocal
    from models.model import PDFFile, Sentence
    from sqlalchemy import text
    
    def test_postgresql_connection():
        """Test PostgreSQL database connection"""
        print("🔍 Testing PostgreSQL connection...")
        
        try:
            # Test 1: Basic connection
            print("📡 Testing basic connection...")
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                print(f"✅ Connected to PostgreSQL!")
                print(f"   Version: {version[:80]}...")
            
            # Test 2: Test database URL
            print(f"🔗 Database URL: {engine.url}")
            
            # Test 3: Check if tables exist
            print("📋 Checking database tables...")
            with SessionLocal() as session:
                try:
                    pdf_count = session.query(PDFFile).count()
                    sentence_count = session.query(Sentence).count()
                    print(f"✅ Database tables accessible:")
                    print(f"   - PDF files table: {pdf_count} records")
                    print(f"   - Sentences table: {sentence_count} records")
                except Exception as table_error:
                    print(f"⚠️  Tables not found or not accessible: {table_error}")
                    print("💡 You may need to create the database tables first")
            
            print("🎉 PostgreSQL connection test PASSED!")
            return True
            
        except Exception as e:
            print(f"❌ PostgreSQL connection test FAILED: {e}")
            print("\n🔧 Troubleshooting tips:")
            print("1. Check if PostgreSQL Docker container is running:")
            print("   sudo docker ps | grep postgres")
            print("2. Verify connection parameters in config/.env")
            print("3. Make sure PostgreSQL is accepting connections on port 5434")
            return False
    
    if __name__ == "__main__":
        test_postgresql_connection()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you have the required modules and config files")
    print("💡 You may need to create config/.env with database parameters")
    
    # Show what should be in the .env file
    print("\n📝 Your config/.env should contain:")
    print("DATABASE_USERNAME=postgres")
    print("DATABASE_PASSWORD=postgres") 
    print("DATABASE_HOSTNAME=localhost")
    print("DATABASE_PORT=5434")
    print("DATABASE_NAME=postgres") 