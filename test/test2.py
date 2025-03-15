from postgreSQL.postgre_database import SessionLocal, PDFFile, Sentence
from sentence_transformers import SentenceTransformer
from create_milvus_db import collection
from create_corpus import CorpusCreator
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm
import numpy as np
import fitz
import time
import os
import json
import re
from collections import defaultdict

# Initialize the corpus creator with the embedding model
checker = CorpusCreator()

def check_box(doc, sentences):
    sentence_locations = {}
    found_sentences = set()
    sentence_set = set(sentences)  # Dùng set để kiểm tra nhanh hơn

    for page_num, page in enumerate(doc):
        for sentence in sentence_set - found_sentences:  # Chỉ duyệt những câu chưa tìm thấy
            text_instances = page.search_for(sentence)
            if text_instances:
                rects = [[float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                         for rect in text_instances]

                sentence_locations.setdefault(sentence, []).append({
                    "page_num": page_num,
                    "rects": rects
                })
                found_sentences.add(sentence)  # Đánh dấu đã tìm thấy để bỏ qua lần sau

    return sentence_locations


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
    
    for filename, matched_sentences in top_docs:
        # Lấy vị trí các câu khớp trong document
        sentence_locations = check_box(document, matched_sentences)

        # Dùng defaultdict để tránh kiểm tra key tồn tại
        page_sentences = defaultdict(list)

        for sentence, locations in sentence_locations.items():
            if not locations:
                continue

            for location in locations:
                page_num = location["page_num"]
                rects = location["rects"]

                # Dùng set để kiểm tra nhanh hơn
                existing_sentences = {s['content'] for s in page_sentences[page_num]}
                if sentence not in existing_sentences:
                    page_sentences[page_num].append({'content': sentence, 'rects': rects})

        # Format kết quả đầu ra
        similarity_box_sentences = [
            {'pageNumber': page_num, 'similarity_content': sentences}
            for page_num, sentences in sorted(page_sentences.items())
        ]

        # Tính toán % độ giống
        similarity_value = min(int((len(matched_sentences) / total_sentences) * 100), 100) if total_sentences else 0

        # Thêm vào danh sách kết quả
        similarity_documents.append({
            'name': filename,
            'similarity_value': similarity_value,
            'similarity_box_sentences': similarity_box_sentences
        })
    
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