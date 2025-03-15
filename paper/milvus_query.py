from pymilvus import connections, Collection, utility, MilvusClient
import numpy as np
import time
from routers.create_corpus import CorpusCreator
from sklearn.metrics.pairwise import cosine_similarity
from create_milvus_db import collection

# Load the collection
collection.load()

# Check collection size
num_vectors = collection.num_entities
print(f"Number of vectors in collection: {num_vectors}")

checker = CorpusCreator("./embedding_models/snapshots/ca1bafe673133c99ee38d9782690a144758cb338")


# Define the search parameters
search_params = {
    "metric_type": "COSINE",  # Cosine similarity
    "params": {"nprobe": 10}
}

import time

t0 = time.time()
# Load the query vectors (example vectors)

_, query_vectors1 = checker.process_document("/hdd1/similarity/CheckSimilarity/database/IT/PhamQuangThanh__DATN.pdf")
_, query_vectors2 = checker.process_document("/hdd1/similarity/CheckSimilarity/database/IT/1451020171_Nguyễn Thị Nhung.pdf")

# print(query_vectors1.shape)
# print(query_vectors2.shape)

cos = cosine_similarity(query_vectors1, query_vectors2)
print(cos.shape)
# Count values > 0.9 in each column
# Get max value in each column
max_values = np.max(cos, axis=0)

# Create mask of same shape as cos, initialized to zeros
mask = np.zeros_like(cos)

# For each column, set True only for the max value position
for col in range(cos.shape[1]):
    max_idx = np.argmax(cos[:, col])
    mask[max_idx, col] = True

# Apply mask to keep only max values, rest become 0
filtered_cos = np.where(mask, cos, 0)

# Count values > 0.9 in each column
high_similarity_counts = np.sum(filtered_cos > 0.9, axis=0)

# print("\nNumber of high similarity matches (>0.9) per sentence in document 2:")
# for i, count in enumerate(high_similarity_counts):
#     if count > 0:
#         print(f"Sentence {i+1}: {count} matches")

total_matches = np.sum(high_similarity_counts)
print(f"\nTotal number of high similarity matches: {total_matches}")
print(f"Average matches per sentence: {total_matches/len(high_similarity_counts):.2f}")


# # query_vectors = query_vectors.tolist()
# print(query_vectors.shape)

def search_with_batch_size(query_vectors, batch_size):
    start = time.time()
    all_results = []
    
    for i in range(0, len(query_vectors), batch_size):
        batch_query_vectors = query_vectors[i:i + batch_size]
        results = collection.search(
            data=batch_query_vectors,
            anns_field="sentence_vector",
            param=search_params,
            limit=1,
            output_fields=["pdf_id", "sentence_id"]
        )
        all_results.extend(results)
        
        # Print results for current batch
        for j, result in enumerate(results):
            for res in result:
                print(f"Batch {i//batch_size + 1}, Vector {i + j + 1}: PDF ID: {res.entity.get('pdf_id')}, "
                      f"Sentence ID: {res.entity.get('sentence_id')}, Similarity: {res.distance}")
    
    total_time = time.time() - start
    print(f"\nProcessed {len(query_vectors)} vectors in {total_time:.2f} seconds")
    print(f"Average time per vector: {total_time/len(query_vectors):.4f} seconds")
    return all_results

# # Test with different batch sizes
# print("\nTesting with batch_size = len(query_vectors1):")
# results_full = search_with_batch_size(query_vectors1, len(query_vectors1))

# print("\nTesting with batch_size = 5:")
# results_batched = search_with_batch_size(query_vectors1, 5)

# print(results.shape)

# print(results-cos_value)

# t1 = time.time()
# print(f"Total search time: {t1 - t0} seconds")