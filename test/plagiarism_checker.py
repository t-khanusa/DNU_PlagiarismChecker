from postgreSQL.postgre_database import SessionLocal, PDFFile, Sentence
from minio_file.get_pdffile_aws import read_presigned_url
from fastapi.responses import JSONResponse
from routers.create_milvus_db import collection
from routers.create_corpus import CorpusCreator
from typing import List, Dict, Tuple
import numpy as np
import fitz
import time
import requests
from tqdm import tqdm
import os
from fastapi import FastAPI
import uvicorn

baseURL = os.getenv('baseURL')
app = FastAPI()
checker = CorpusCreator()


def check_box(doc, sentences):    
    # Mapping of sentences to their locations
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
    return sentence_locations

@app.post("/plagiarism_checker/")
async def check_plagiarism(subject_id: str, file_name: str, ) -> Dict:
    min_similarity = 0.9
    """
    Check document for plagiarism against the vector database.
    
    Args:
        subject_id: Subject ID
        file_name: File name

    Returns: 
        "total_percent": float,
    """

    try:
        file_path = read_presigned_url(subject_id, file_name)
        
    except Exception as e:
        return {"error": "No valid file path found"}

    # Process document to get sentences and embeddings
    raw_sentences, embeddings = checker.process_document(file_path)

    if not raw_sentences:
        return {"error": "No valid raw_sentences found in document"}    
    print(f"Number of raw_sentences processed: {len(raw_sentences)}")

    # Configure search parameters for vector search
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    
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

        for idx, hits in enumerate(search_results):
            sentence_idx = i + idx
            current_sentence = raw_sentences[sentence_idx]
            
            for hit in hits:
                similarity_score = float(hit.distance)
                
                if similarity_score >= min_similarity:
                    matched_sentence_set.add(current_sentence)
    return JSONResponse(content={"total_percent": len(matched_sentence_set) / len(raw_sentences) * 100}, status_code=200)


@app.post("/plagiarism_checker_details/")    
async def check_plagiarism_details(subject_id: str, file_name: str, file_check_id: str) -> Dict:
    min_similarity = 0.9
    url = f"{baseURL}/api/checker/files/{file_check_id}/result"
    
    headers = {
    'Content-Type': 'application/json'
    }


    """
    Check document for plagiarism against the vector database.
    
    Args:
        subject_id: Subject ID
        file_name: File name
        
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
    try:
        file_path = read_presigned_url(subject_id, file_name)
        result = requests.request("POST", url, headers=headers)
    except Exception as e:
        return {"error": "No valid file path found"}

    filename = os.path.basename(file_path)
    # Process document to get sentences and embeddings
    raw_sentences, embeddings = checker.process_document(file_path)

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
    
    for docs in top_docs:
        (filename, matched_sentences) = docs

        # Process each sentence that matches with this document
        sentence_locations = check_box(document, matched_sentences)

        # Initialize page_sentences dictionary to store sentences by page number
        page_sentences = {}
        
        # Process each matched sentence and its locations
        for sentence, locations in sentence_locations.items():
            # Skip if no locations found for this sentence
            if not locations:
                continue
                
            # Process each location where this sentence was found
            for location in locations:
                page_num = location["page_num"]
                
                if page_num not in page_sentences:
                    page_sentences[page_num] = []
                
                # Check if this sentence is already added to this page
                found = False
                for existing in page_sentences[page_num]:
                    if existing['content'] == sentence:
                        found = True
                        break
                
                if not found:
                    page_sentences[page_num].append({
                        'content': sentence,
                        'rects': location["rects"]
                    })
        
        # Format page sentences for output
        similarity_box_sentences = []
        for page_num, sentences in sorted(page_sentences.items()):
            similarity_box_sentences.append({
                'pageNumber': page_num,
                'similarity_content': sentences
            })
        
        # Calculate similarity percentage based on number of matched sentences
        similarity_value = 0
        if total_sentences > 0:
            # Count total sentences found in this document
            total_matched = len(matched_sentences)
            similarity_value = int((total_matched / total_sentences) * 100)
            # Cap at 100%
            similarity_value = min(similarity_value, 100)
        
        # Add document to similarity documents
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


    response = requests.request("POST", url, headers=headers, data=result)
    # # Print summary statistics
    # print(f"\nPlagiarism Check Summary:")
    # print(f"Total sentences: {total_sentences}")
    # print(f"Documents with matches above {min_similarity}: {len(document_matches)}")
    
    # print(f"\nTop similar documents:")
    # for doc in similarity_documents:
    #     sent_count = sum(len(page["similarity_content"]) for page in doc["similarity_box_sentences"])
    #     print(f"- {doc['name']}: {doc['similarity_value']}% ({sent_count} sentences located)")
    
    # print(f"Processing time: {time.time() - start_time:.2f} seconds")
    # print(len(matched_sentence_set) / total_sentences * 100)
    # print(len(matched_sentence_set))
    return JSONResponse(content={"result":"success"}, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8688)

# def main():
#     """Main function to run plagiarism checks"""
#     # try:
#     # Example usage
#     file_to_check = "/hdd1/similarity/CheckSimilarity/database/IT/PhamQuangThanh__DATN.pdf"
#     results = check_plagiarism(file_to_check)
    
#     # Print summary
#     print("\nPlagiarism Check Summary:")
#     print(f"Total Similarity Percentage: {results['total_similarity_percent']:.2f}%")
    
#     print("\nTop Similar Documents:")
#     for doc, score in zip(results['top_similarity_documents'], results['top_similarity_values']):
#         print(f"- {doc}: {score:.2f}")
    
#     # print("\nDetailed Sentence Matches:")
#     # for sentence, matches in results['similarity_sentences'].items():
#     #     print(f"\nOriginal: {sentence}")
#     #     for match in matches:
#     #         print(f"- Found in: {match['document']}")
#     #         print(f"  Similarity: {match['similarity_score']:.2f}")
#     #         print(f"  Text: {match['matched_text']}")
    
#     print(f"\nProcessing Time: {results['processing_time']:.2f} seconds")

#     # except:
#     #     print("Error")

