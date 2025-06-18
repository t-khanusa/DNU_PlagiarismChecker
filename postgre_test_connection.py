import psycopg2
import sys

def test_connection():
    try:
        # Connection parameters based on your Docker container
        conn = psycopg2.connect(
            host="localhost",
            port="5434",
            database="Sentence_Similarity", 
            user="similarity",
            password="123456"  # Default password or check your Docker setup
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ PostgreSQL connection successful!")
        print(f"Version: {version[0]}")
        
        # List tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        tables = cursor.fetchall()
        print(f"📋 Tables found: {[t[0] for t in tables]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Try checking if PostgreSQL is running and accessible")

if __name__ == "__main__":
    test_connection()