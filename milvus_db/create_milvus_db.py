from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, db
import numpy as np
from config.conf import MILVUS_DB_NAME, MILVUS_HOST, MILVUS_PORT
# from create_corpus import CorpusCreator
# Connect to Milvus


# # First connect to default database to create our target database
# conn = connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# # Create database if it doesn't exist
# try:
#     # Check if database exists
#     database_list = db.list_database()
#     if MILVUS_DB_NAME not in database_list:
#         db.create_database(MILVUS_DB_NAME)
#         print(f"Created database: {MILVUS_DB_NAME}")
#     else:
#         print(f"Database {MILVUS_DB_NAME} already exists")
# except Exception as e:
#     print(f"Error checking/creating database: {e}")

# # Now connect to our specific database
# connections.disconnect("default")
conn = connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, db_name=MILVUS_DB_NAME)
print(f"Connected to database: {MILVUS_DB_NAME}")

# collection_name = "sentence_similarity"
# collection = Collection(name=collection_name)

# Define the schema for the collection
fields = [
    # Primary key from PostgreSQL - required for direct lookups and joins
    FieldSchema(name="sentence_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    
    # Document context - required for filtering and ordering
    FieldSchema(name="pdf_id", dtype=DataType.INT64, description="Foreign key to PostgreSQL pdf_files.pdf_id"),
    FieldSchema(name="sentence_index", dtype=DataType.INT64, description="Position in document"),
    
    # Vector data
    FieldSchema(name="sentence_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, "Sentence similarity collection with PostgreSQL alignment")

# Create or recreate the collection
collection_name = "sentence_similarity"
# if utility.has_collection(collection_name):
#     utility.drop_collection(collection_name)
collection = Collection(name=collection_name, schema=schema)

# Create IVF_FLAT index for faster similarity search
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index(
    field_name="sentence_vector", 
    index_params=index_params
)

# Create index on pdf_id for document filtering
collection.create_index(
    field_name="pdf_id",
    index_name="idx_pdf",
    index_params={"index_type": "STL_SORT"}
)

# Create index on sentence_index for ordering within documents
collection.create_index(
    field_name="sentence_index",
    index_name="idx_sentence_pos",
    index_params={"index_type": "STL_SORT"}
)


def insert_vectors(sentence_ids, pdf_ids, sentence_indices, vectors):
    """
    Insert vectors into Milvus with PostgreSQL alignment
    
    Args:
        sentence_ids (List[int]): PostgreSQL sentence.id values (required for joins)
        pdf_ids (List[int]): PostgreSQL pdf_files.pdf_id values
        sentence_indices (List[int]): Position of sentences in documents
        vectors (List[List[float]]): Sentence embeddings
    """
    data = [
        sentence_ids,      # For direct lookups and joins
        pdf_ids,          # For document context
        sentence_indices,  # For ordering within document
        vectors           # For similarity search
    ]
    collection.insert(data)
    collection.flush()



def search_similar_sentences(query_vector, pdf_id=None, top_k=5, context_window=0):
    """
    Search for similar sentences with optional PDF filtering and context
    
    Args:
        query_vector (List[float]): Query embedding vector
        pdf_id (int, optional): Filter results to specific PDF
        top_k (int): Number of results to return
        context_window (int): Number of surrounding sentences to include (0 for none)
    
    Returns:
        List of tuples (sentence_id, pdf_id, sentence_index, similarity_score)
    """
    collection.load()
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 16}
    }
    
    # Prepare expression for filtering if pdf_id is provided
    expr = f"pdf_id == {pdf_id}" if pdf_id is not None else None
    
    # Search in the collection
    results = collection.search(
        data=[query_vector],
        anns_field="sentence_vector",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["sentence_id", "pdf_id", "sentence_index"]
    )
    
    # Format results and include context if requested
    similar_sentences = []
    for hits in results:
        for hit in hits:
            sentence_id = hit.entity.get('sentence_id')
            pdf_id = hit.entity.get('pdf_id')
            sentence_index = hit.entity.get('sentence_index')
            
            # Add the main match
            similar_sentences.append((sentence_id, pdf_id, sentence_index, hit.score))
            
            # Add context sentences if requested
            if context_window > 0:
                context_expr = (
                    f"pdf_id == {pdf_id} && "
                    f"sentence_index >= {sentence_index - context_window} && "
                    f"sentence_index <= {sentence_index + context_window} && "
                    f"sentence_id != {sentence_id}"
                )
                context = collection.query(
                    expr=context_expr,
                    output_fields=["sentence_id", "pdf_id", "sentence_index"],
                    sort_fields=["sentence_index"]
                )
                for ctx in context:
                    similar_sentences.append((
                        ctx.get('sentence_id'),
                        ctx.get('pdf_id'),
                        ctx.get('sentence_index'),
                        1.0  # Context sentences get full score
                    ))
    
    return similar_sentences

def get_document_sentences(pdf_id, start_index=None, end_index=None):
    """
    Retrieve sentences from a specific document in order
    
    Args:
        pdf_id (int): PostgreSQL pdf_files.pdf_id
        start_index (int, optional): Start position
        end_index (int, optional): End position
    
    Returns:
        List of sentences in document order
    """
    collection.load()
    
    expr = f"pdf_id == {pdf_id}"
    if start_index is not None and end_index is not None:
        expr += f" && sentence_index >= {start_index} && sentence_index <= {end_index}"
    
    results = collection.query(
        expr=expr,
        output_fields=["sentence_id", "sentence_index", "sentence_vector"],
        sort_fields=["sentence_index"]
    )
    
    return [(r.get('sentence_id'), r.get('sentence_index'), r.get('sentence_vector')) for r in results]

