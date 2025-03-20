from milvus_db.create_corpus import CorpusCreator
import zipfile
import os
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import json

# Initialize the corpus creator once for reuse
corpus = CorpusCreator()
corpus.batch_size=32

def unzip_file(zip_path, extract_to):
    """Extract files from a zip archive to a specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def process_pdf(file_path):
    file_name = os.path.basename(file_path)
    try:
        raw_sentences, embeddings = corpus.process_document(file_path)
        sentence_count = len(raw_sentences)
        
        if sentence_count > 0:
            return file_path, file_name, sentence_count, embeddings
        else:
            print(f"Warning: No sentences extracted from {file_name}")
            return file_path, file_name, 0, None
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return file_path, file_name, 0, None

def process_files_parallel(file_paths, max_workers=4):
    """Process multiple PDF files in parallel.
    
    Args:
        file_paths: List of PDF file paths
        max_workers: Number of worker threads
        
    Returns:
        Dictionary mapping file paths to (file_name, sentence_count, embeddings)
    """
    start_time = time.time()
    processed_count = 0
    total_sentences = 0
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pdf, file_path) for file_path in file_paths]
        
        for future in futures:
            try:
                file_path, file_name, sentence_count, embeddings = future.result()
                if sentence_count > 0:
                    results[file_path] = (file_name, sentence_count, embeddings)
                    processed_count += 1
                    total_sentences += sentence_count
            except Exception as e:
                print(f"Exception occurred: {e}")
    
    end_time = time.time()
    print(f"Processed {processed_count} files with {total_sentences} sentences in {end_time - start_time:.2f} seconds")
    
    return results


def compare_file_pair(file_data, file1, file2, similarity_threshold=0.8):
    if file1 not in file_data or file2 not in file_data:
        return 0.0, 0, file1, file2
    
    _, _, embeddings1 = file_data[file1]
    _, _, embeddings2 = file_data[file2]
    
    similarity_matrix = corpus.model.similarity(embeddings1, embeddings2).numpy()
    column_max_scores = np.max(similarity_matrix, axis=0)
    match_count = np.sum(column_max_scores >= similarity_threshold)
    
    if len(column_max_scores) > 0:
        similarity_percentage = match_count / len(column_max_scores) * 100
    else:
        similarity_percentage = 0.0

    return similarity_percentage, match_count, file1, file2


def compare_all_files_parallel(file_data, similarity_threshold=0.8, max_workers=4):
    """Compare all possible file pairs in parallel.
    
    Args:
        file_data: Dictionary mapping file paths to their data
        similarity_threshold: Threshold for considering sentences similar
        max_workers: Number of worker threads
        
    Returns:
        List of tuples (similarity_percentage, match_count, file1, file2) sorted by similarity
    """
    processed_files = list(file_data.keys())
    file_pairs = []
    
    # Generate all unique file pairs
    for i, file1 in enumerate(processed_files):
        for j in range(i+1, len(processed_files)):
            file_pairs.append((file1, processed_files[j]))
    
    total_comparisons = len(file_pairs)
    print(f"Comparing {len(processed_files)} files ({total_comparisons} comparisons)...")
    
    similarity_scores = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a function that takes a pair and returns the similarity results
        def compare_pair(pair):
            file1, file2 = pair
            return compare_file_pair(file_data, file1, file2, similarity_threshold)
        
        # Submit all comparison tasks and gather results
        results = list(executor.map(compare_pair, file_pairs))
        
        # Filter and add results that have matches
        for similarity_percentage, match_count, file1, file2 in results:
            if similarity_percentage > 0:
                similarity_scores.append((
                    similarity_percentage,
                    match_count,
                    file1,
                    file2
                ))
    
    # Sort by similarity percentage (descending)
    similarity_scores.sort(reverse=True)
    
    end_time = time.time()
    print(f"Completed {total_comparisons} comparisons in {end_time - start_time:.2f} seconds")
    
    return similarity_scores


def find_similar_files(file_data, similarity_threshold=0.9, min_similarity_percent=None, parallel=True, max_workers=4):
    """Find similar file pairs based on threshold percentage.
    
    Args:
        file_data: Dictionary mapping file paths to their data
        similarity_threshold: Threshold for considering sentences similar
        min_similarity_percent: Minimum similarity percentage to include in results
        parallel: Whether to use parallel comparison
        max_workers: Number of worker threads
        
    Returns:
        List of all similar file pairs above the threshold
    """
    if parallel:
        similarity_scores = compare_all_files_parallel(
            file_data, similarity_threshold, max_workers
        )
    
    # Sort by similarity percentage (descending)
    similarity_scores.sort(reverse=True)
    
    # Filter by minimum similarity percentage
    if min_similarity_percent is not None:
        filtered_scores = [
            score for score in similarity_scores 
            if score[0] >= min_similarity_percent
        ]
        print(f"Found {len(filtered_scores)} file pairs with similarity >= {min_similarity_percent}%")
        return filtered_scores
    
    # Return all scores if no threshold is specified
    print(f"Found {len(similarity_scores)} similar file pairs")
    return similarity_scores


def format_results_to_json(similar_pairs, file_data, threshold, time_taken):

    pairs = []
    
    for similarity, matches, file1, file2 in similar_pairs:
        file1_name, file1_count, _ = file_data[file1]
        file2_name, file2_count, _ = file_data[file2]
        
        pair_data = {
            "similarity": float(round(similarity, 2)),
            "matching_sentences": int(matches),
            "files": [
                {
                    "name": file1_name,
                    "total_sentences": int(file1_count)
                },
                {
                    "name": file2_name,
                    "total_sentences": int(file2_count)
                }
            ]
        }
        pairs.append(pair_data)
    
    result = {
        "data": {
            "total_pairs": int(len(similar_pairs)),
            "threshold": threshold,
            "pairs": pairs,
            "time_taken_seconds": round(time_taken, 2)
        }
    }
    
    return result

# def save_json_results(results, output_file="similarity_results.json"):
#     """Save results to a JSON file with proper error handling.
    
#     Args:
#         results: Dictionary to serialize to JSON
#         output_file: Path to the output file
#     """
#     try:
#         # Ensure all numeric values are properly converted to avoid serialization issues
#         if isinstance(results, dict) and "data" in results:
#             data = results["data"]
#             if "pairs" in data and isinstance(data["pairs"], list):
#                 for pair in data["pairs"]:
#                     # Ensure numeric values are properly rounded
#                     if "similarity" in pair:
#                         pair["similarity"] = float(round(pair["similarity"], 2))
#                     if "matching_sentences" in pair:
#                         pair["matching_sentences"] = int(pair["matching_sentences"])
                    
#                     # Ensure file data is properly formatted
#                     if "files" in pair and isinstance(pair["files"], list):
#                         for file_info in pair["files"]:
#                             if "total_sentences" in file_info:
#                                 file_info["total_sentences"] = int(file_info["total_sentences"])
        
#         # Write to a temporary file first
#         temp_file = f"{output_file}.tmp"
#         with open(temp_file, 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
        
#         # Then rename to the final file name (this is more atomic)
#         os.replace(temp_file, output_file)
#         print(f"Results successfully saved to {output_file}")
#     except Exception as e:
#         print(f"Error saving results to {output_file}: {str(e)}")
        
#         # Try saving in a simplified format as backup
#         try:
#             simplified_data = {
#                 "data": {
#                     "total_pairs": len(results["data"]["pairs"]),
#                     "threshold": results["data"]["threshold"],
#                     "error": "Full data could not be saved, simplified version provided",
#                     "pairs": []
#                 }
#             }
            
#             # Add only essential information for each pair
#             for pair in results["data"]["pairs"]:
#                 simplified_pair = {
#                     "similarity": round(float(pair["similarity"]), 2),
#                     "files": [file["name"] for file in pair["files"]]
#                 }
#                 simplified_data["data"]["pairs"].append(simplified_pair)
            
#             backup_file = f"{output_file}.backup"
#             with open(backup_file, 'w', encoding='utf-8') as f:
#                 json.dump(simplified_data, f, indent=2, ensure_ascii=False)
#             print(f"Simplified backup saved to {backup_file}")
#         except Exception as backup_error:
#             print(f"Could not save backup file: {str(backup_error)}")

# def main():
#     """Main function to run the PDF similarity comparison."""
#     # Configuration
#     zip_path = "/hdd1/similarity/CheckSimilarity/myfiles.zip"
#     extract_to = "/hdd1/similarity/CheckSimilarity/unzip_folder"
#     output_file = "zip_test_results.json"
#     similarity_threshold = 0.9
#     min_similarity_percent = 20.0  # Only show pairs with similarity ≥ 20%
    
#     # Start timing
#     start_time = time.time()
    
#     # Uncomment to extract files if needed
#     # unzip_file(zip_path, extract_to)
    
#     # Find all PDF files in the extract directory
#     pdf_files = glob.glob(os.path.join(extract_to, "**/*.pdf"), recursive=True)
    
#     if not pdf_files:
#         print(f"No PDF files found in {extract_to}")
#         return
    
#     print(f"Found {len(pdf_files)} PDF files to process")
    
#     # Process all files in parallel
#     file_data = process_files_parallel(pdf_files, max_workers=4)
    
#     # Find ALL similar file pairs above the threshold
#     similar_pairs = find_similar_files(
#         file_data,
#         similarity_threshold=similarity_threshold,
#         min_similarity_percent=min_similarity_percent,
#         parallel=True,
#         max_workers=4
#     )
    
#     # Calculate total time
#     end_time = time.time()
#     total_time = end_time - start_time
    
#     # Create a generate and print a report if needed
#     print(f"Found {len(similar_pairs)} similar file pairs with similarity >= {min_similarity_percent}%")
#     print(f"Total time taken: {total_time:.2f} seconds")
        
#     # Format results to JSON and save
#     json_results = format_results_to_json(
#         similar_pairs, 
#         file_data, 
#         min_similarity_percent,
#         total_time
#     )
    
#     # Save results using the improved error-handling function
#     save_json_results(json_results, output_file)


# if __name__ == "__main__":
#     main()

