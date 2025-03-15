from pymilvus import connections, utility, Collection
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from postgreSQL.postgre_database import Base

# PostgreSQL connection settings
DATABASE_URL = "postgresql://similarity:123456@localhost:5434/Sentence_Similarity"

# Milvus connection settings
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_DB = "vector_database"
MILVUS_ALIAS = "default"  # Adding default alias for Milvus connection

def clear_postgresql():
    """Clear all data from PostgreSQL database"""
    print("Clearing PostgreSQL database...")
    
    # Create engine and connect
    engine = create_engine(DATABASE_URL)
    
    try:
        # Drop all tables
        Base.metadata.drop_all(engine)
        # Recreate all tables
        Base.metadata.create_all(engine)
        print("Successfully cleared PostgreSQL database")
    except Exception as e:
        print(f"Error clearing PostgreSQL database: {e}")

def clear_milvus():
    """Clear all data from Milvus database"""
    print("Clearing Milvus database...")
    
    try:
        # Connect to Milvus
        connections.connect(
            alias=MILVUS_ALIAS,
            host=MILVUS_HOST, 
            port=MILVUS_PORT, 
            db_name=MILVUS_DB
        )
        
        # Get list of all collections
        collections = utility.list_collections()
        
        # Drop each collection
        for collection_name in collections:
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                collection.drop()
                print(f"Dropped collection: {collection_name}")
        
        print("Successfully cleared Milvus database")
    except Exception as e:
        print(f"Error clearing Milvus database: {e}")
    finally:
        # Disconnect from Milvus with alias
        connections.disconnect(MILVUS_ALIAS)

if __name__ == "__main__":
    # Clear both databases
    clear_postgresql()
    clear_milvus()
    print("Database clearing completed!") 