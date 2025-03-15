from postgreSQL.postgre_database import SessionLocal, PDFFile, Sentence
from sentence_transformers import SentenceTransformer
from routers.create_milvus_db import collection
from routers.create_corpus import CorpusCreator
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm
import numpy as np
import fitz
import time
import os
import json
import re
from difflib import SequenceMatcher
import nltk
import threading

nltk.download('punkt')

results = {}
lock = threading.Lock()

# Initialize the corpus creator with the embedding model
checker = CorpusCreator()

# def similar(a, b):
#     """Calculate text similarity ratio between two strings."""
#     return SequenceMatcher(None, a, b).ratio()

# def extract_sentences_with_positions(doc):
#     """
#     Extract all sentences with their positions from a PDF document in a single pass.
    
#     Returns:
#         Dict: Mapping of page numbers to sentences with their positions
#     """
#     page_sentences = {}
    
#     for page_num, page in enumerate(doc):
#         # Extract text with positions
#         page_sentences[page_num] = []
        
#         # First try with search_for method for exact matches
#         text = page.get_text()
#         blocks = page.get_text("dict")["blocks"]
        
#         for block in blocks:
#             if "lines" not in block:
#                 continue
                
#             block_text = ""
#             block_rects = []
            
#             for line in block["lines"]:
#                 for span in line["spans"]:
#                     block_text += span["text"] + " "
#                     block_rects.append([
#                         float(span["bbox"][0]), 
#                         float(span["bbox"][1]),
#                         float(span["bbox"][2]), 
#                         float(span["bbox"][3])
#                     ])
            
#             # Store this block of text with its positions
#             if block_text.strip():
#                 page_sentences[page_num].append({
#                     "text": block_text.strip(),
#                     "rects": block_rects
#                 })
    
#     return page_sentences


# def locate_sentences_in_pdf(doc, sentences, min_similarity=0.7):
#     """
#     Find sentences in the PDF and return their locations.
    
#     Args:
#         doc: The PDF document
#         sentences: List of sentences to find
#         min_similarity: Minimum similarity threshold for fuzzy matching
        
#     Returns:
#         Dict: Mapping sentences to their locations in the PDF
#     """
#     start_time = time.time()
    
#     # Extract all text blocks with positions
#     page_sentences = extract_sentences_with_positions(doc)
    
#     # Mapping of sentences to their locations
#     sentence_locations = {}
    
#     # Clean sentences for better matching
#     clean_sentences = {}
#     for sentence in sentences:
#         clean = re.sub(r'\s+', ' ', sentence).strip().lower()
#         clean_sentences[sentence] = clean
    
#     # Track which sentences we've found
#     found_sentences = set()
    
#     # First pass: try direct search which is faster
#     for page_num, page in enumerate(doc):
#         for sentence in sentences:
#             if sentence in found_sentences:
#                 continue
                
#             # Direct search for exact matches
#             text_instances = page.search_for(sentence)
#             if text_instances:
#                 if sentence not in sentence_locations:
#                     sentence_locations[sentence] = []
                
#                 # Format the rectangles
#                 rects = [[float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)] 
#                         for rect in text_instances]
                
#                 sentence_locations[sentence].append({
#                     "page_num": page_num,
#                     "rects": rects
#                 })
                
#                 found_sentences.add(sentence)
    
#     # Second pass: try fuzzy matching for sentences we haven't found yet
#     remaining_sentences = [s for s in sentences if s not in found_sentences]
#     if remaining_sentences:
#         for page_num, blocks in page_sentences.items():
#             for sentence in remaining_sentences:
#                 if sentence in found_sentences:
#                     continue
                
#                 clean_sentence = clean_sentences[sentence]
                
#                 # Skip if too short
#                 if len(clean_sentence) < 15:
#                     continue
                
#                 # Check each block on this page
#                 for block in blocks:
#                     block_text = block["text"].lower()
                    
#                     # Try substring match first
#                     if clean_sentence in block_text:
#                         if sentence not in sentence_locations:
#                             sentence_locations[sentence] = []
                        
#                         sentence_locations[sentence].append({
#                             "page_num": page_num,
#                             "rects": block["rects"]
#                         })
                        
#                         found_sentences.add(sentence)
#                         break
                    
#                     # If not found, try fuzzy matching
#                     elif len(clean_sentence) > 30 and similar(clean_sentence, block_text) > min_similarity:
#                         if sentence not in sentence_locations:
#                             sentence_locations[sentence] = []
                        
#                         sentence_locations[sentence].append({
#                             "page_num": page_num,
#                             "rects": block["rects"]
#                         })
                        
#                         found_sentences.add(sentence)
#                         break
    
#     print(f"Found locations for {len(found_sentences)} out of {len(sentences)} sentences in {time.time() - start_time:.2f}s")
#     return sentence_locations
def check_box(doc, sentences, key):    
    # Mapping of sentences to their locations
    print("?????????????????????", key)
    sentence_locations = {}

    # Track which sentences we've found
    found_sentences = set()
    
    # First pass: try direct search which is faster
    for page_num, page in enumerate(doc):
        for sentence in sentences:
            if sentence in found_sentences:
                continue
    
            # Direct search for exact matches
            text_instances = page.search_for(sentence)
            if text_instances:
                if sentence not in sentence_locations:
                    sentence_locations[sentence] = []
                
                # Format the rectangles
                rects = [[float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)] 
                        for rect in text_instances]
                
                sentence_locations[sentence].append({
                    "page_num": page_num,
                    "rects": rects
                })
                
                found_sentences.add(sentence)
    with lock:
        results[key] = sentence_locations 
        # return 

def check_plagiarism_details(file_path: str, min_similarity: float = 0.9) -> Dict:
    """
    Check document for plagiarism against the vector database.
    
    Args:
        file_path: Path to the PDF file to check
        min_similarity: Minimum similarity threshold (default 0.9)
        
    Returns: 
        Dict with plagiarism information in the format required for visualization:
        {
            "data": {
                "total_percent": float,
                "size_page": {"width": float, "height": float},
                "similarity_documents": [
                    {
                        "name": str,
                        "similarity_value": int,
                        "similarity_box_sentences": [
                            {
                                "pageNumber": int,
                                "similarity_content": [
                                    {
                                        "content": str,
                                        "rects": [[float, float, float, float], ...]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
    """
    start_time = time.time()
    filename = os.path.basename(file_path)
    print(f"Processing document: {filename}")
    # Process document to get sentences and embeddings
    raw_sentences, embeddings = checker.process_document(file_path)
    time_process_document = time.time()
    print(f"Time taken to process document: {time_process_document - start_time:.2f} seconds")

    if not raw_sentences:
        return {"error": "No valid raw_sentences found in document"}
    
    print(f"Number of raw_sentences processed: {len(raw_sentences)}")

    # Get document page size and prepare for text extraction
    try:
        document = fitz.open(file_path)
        page_size = {"width": document[0].rect.width, "height": document[0].rect.height}
    except Exception as e:
        print(f"Error opening PDF: {e}")
        page_size = {"width": 595.0, "height": 842.0}  # Default A4 size
        document = None

    # Configure search parameters for vector search
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    
    # Initialize results containers
    document_matches = {}
    # Process in batches with progress bar
    batch_size = len(raw_sentences)
    matched_sentence_set = set()
    # with tqdm(total=len(raw_sentences), desc="Checking for plagiarism") as pbar:
    for i in range(0, len(raw_sentences), batch_size):
        batch_end = min(i + batch_size, len(raw_sentences))
        batch_vectors = embeddings[i:batch_end].tolist()
        
        start_search = time.time()
        # Search Milvus database
        search_results = collection.search(
            data=batch_vectors,
            anns_field="sentence_vector",
            param=search_params,
            limit=1,
            output_fields=["pdf_id", "sentence_id"]
        )
        end_search = time.time()
        print(f"Time taken to search Milvus: {end_search - start_search:.2f} seconds")

        with SessionLocal() as db:
            for idx, hits in enumerate(search_results):
                sentence_idx = i + idx
                current_sentence = raw_sentences[sentence_idx]
                
                for hit in hits:
                    similarity_score = float(hit.distance)
                    
                    if similarity_score >= min_similarity:
                        pdf_id = hit.entity.get('pdf_id')
                        
                        # Retrieve matching document
                        pdf_file = db.query(PDFFile).filter(PDFFile.pdf_id == pdf_id).first()

                        if pdf_file.filename not in document_matches:
                            document_matches[pdf_file.filename] = []
                        document_matches[pdf_file.filename].append(current_sentence) 
                        matched_sentence_set.add(current_sentence)
            

    # Get top 5 similar documents
    top_docs = sorted(document_matches.items(), 
                     key=lambda x: len(x[1]), 
                     reverse=True)[:5]  # Limit to top 5

    # Initialize similarity_documents list to store all document matches
    similarity_documents = []
    
    # Calculate total number of sentences for percentage calculation
    total_sentences = len(raw_sentences)
    
    result = {
        "similarity_documents": []
    }
    found_sentences = set()
    sentence_locations = {}
    # for docs in top_docs:
    #     (filename, matched_sentences) = docs
    threads = []
    num_threads = min(5, len(top_docs))
    for i in range(num_threads):
        thread = threading.Thread(target=check_box, args=(document, top_docs[i][1], str(i)))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    # print("Kết quả tổng hợp:", results)
    # for page_num, page in enumerate(document):
    #     for docs in top_docs:
    #         (filename, matched_sentences) = docs
    #         similarity_box_sentences_list = []
    #         for item, sentence in enumerate(matched_sentences):
    #             if item not in found_sentences:
    #                 text_instances = page.search_for(sentence)
                    
    #                 if len(text_instances) > 0:
    #                     print("???", text_instances)
    #                     if sentence not in sentence_locations:
    #                         sentence_locations[sentence] = []
                        
    #                     rects = [
    #                         [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)] 
    #                         for rect in text_instances
    #                     ]
    #                     found_sentences.add(item)
    #                     similarity_box_sentences_list.append({
    #                         "pageNumber": page_num,
    #                         "similarity_content": {
    #                             "content": sentence,
    #                             "rects": rects
    #                         }
    #                     })
    #         similarity_value = 0
    #         if total_sentences > 0:
    #             total_matched = len(matched_sentences)
    #             similarity_value = int((total_matched / total_sentences) * 100)
    #             similarity_value = min(similarity_value, 100)
    #         result['similarity_documents'].append({
    #             'name': filename,
    #             'similarity_value': similarity_value,
    #             'similarity_box_sentences': similarity_box_sentences_list
    #         })
    
    # Get page size from the document
    page_size = {"width": 595.0, "height": 842.0}  # Default A4 size
    if document and hasattr(document, 'mediabox'):
        page_size = {
            "width": document.mediabox[2],
            "height": document.mediabox[3]
        }
    
    # Format final output
    result = {
        "data": {
            "total_percent": len(matched_sentence_set) / total_sentences * 100,  # Always 100% for the document being checked
            "size_page": page_size,
            "similarity_documents": similarity_documents
        }
    }
    
    # # Print summary statistics
    # print(f"\nPlagiarism Check Summary:")
    # print(f"Total sentences: {total_sentences}")
    # print(f"Documents with matches above {min_similarity}: {len(document_matches)}")
    
    # print(f"\nTop similar documents:")
    # for doc in similarity_documents:
    #     sent_count = sum(len(page["similarity_content"]) for page in doc["similarity_box_sentences"])
    #     print(f"- {doc['name']}: {doc['similarity_value']}% ({sent_count} sentences located)")
    
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print(len(matched_sentence_set) / total_sentences * 100)
    print(len(matched_sentence_set))
    return result

def main():
    """Run a plagiarism check on a sample document and save results to JSON."""
    # Example file path - adjust as needed
    file_to_check = "/hdd1/similarity/CheckSimilarity/database/IT/Báo cáo đồ án_Nguyễn Thị Linh_1451020141.pdf"
    
    start_time = time.time()
    results = check_plagiarism_details(file_to_check)
    end_time = time.time()

    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    # Save results to JSON file
    output_file = "plagiarism_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()