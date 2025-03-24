from config.postgres_db import SessionLocal
from config.conf import BASEURL
from models.model import PDFFile, Sentence
from aws_file.get_pdffile_aws import read_presigned_url
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from milvus_db.create_milvus_db import collection
from milvus_db.create_corpus import CorpusCreator
from typing import List, Dict, Tuple
import numpy as np
import fitz
import time
import requests
from tqdm import tqdm
import os
import json
import shutil
from pathlib import Path
import glob

router = APIRouter(
    tags=["plagiarism-checker"],
)

baseURL = BASEURL
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

@router.post("/plagiarism_checker_details/") 
async def check_file(background_tasks: BackgroundTasks, subject_id: str, file_name: str, file_check_id: str):
    url = f"{baseURL}/api/checker/files/{file_check_id}/result"
    headers = {
        'Content-Type': 'application/json'
    }
    time_start = time.time()
    try:
        file_path = read_presigned_url(subject_id, file_name)
        requests.request("POST", url, headers=headers)
        background_tasks.add_task(check_plagiarism_details, file_path, file_check_id, time_start)
    except Exception as e:
        return JSONResponse(content={"message": 'Không tìm thấy file'}, status_code=200)
    
    return JSONResponse(content={"message": 'Tìm thấy file'}, status_code=200)


# @router.post("/plagiarism_checker_details/")    
async def check_plagiarism_details(file_path, file_check_id, time_start) -> Dict:
    min_similarity = 0.9
    url = f"{baseURL}/api/checker/files/{file_check_id}/result"
    raw_sentences, embeddings = checker.process_document(file_path)

    if not raw_sentences:
        return {"error": "No valid raw_sentences found in document"}    
    print(f"Number of raw_sentences processed: {len(raw_sentences)}")
    try:
        document = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        document = None

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    document_matches = {}
    batch_size = len(raw_sentences)
    matched_sentence_set = set()
    # with tqdm(total=len(raw_sentences), desc="Checking for plagiarism") as pbar:

    for i in range(0, len(raw_sentences), batch_size):
        batch_end = min(i + batch_size, len(raw_sentences))
        batch_vectors = embeddings[i:batch_end].tolist()  
        start_search = time.time()
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
                        pdf_file = db.query(PDFFile).filter(PDFFile.pdf_id == pdf_id).first()
                        if pdf_file.filename not in document_matches:
                            document_matches[pdf_file.filename] = []
                        document_matches[pdf_file.filename].append(current_sentence) 
                        matched_sentence_set.add(current_sentence)

    top_docs = sorted(document_matches.items(), 
                     key=lambda x: len(x[1]), 
                     reverse=True)[:5]  # Limit to top 5
    similarity_documents = []
    total_sentences = len(raw_sentences)
    
    for docs in top_docs:
        (filename, matched_sentences) = docs
        sentence_locations = check_box(document, matched_sentences)
        page_sentences = {}
        for sentence, locations in sentence_locations.items():
            if not locations:
                continue
            for location in locations:
                page_num = location["page_num"]
                
                if page_num not in page_sentences:
                    page_sentences[page_num] = []
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

        similarity_box_sentences = []
        for page_num, sentences in sorted(page_sentences.items()):
            similarity_box_sentences.append({
                'pageNumber': page_num,
                'similarity_content': sentences
            })
        
        similarity_value = 0
        if total_sentences > 0:
            total_matched = len(matched_sentences)
            similarity_value = int((total_matched / total_sentences) * 100)
            similarity_value = min(similarity_value, 100)
        
        similarity_documents.append({
            "file_resource_id": "0a58b8ab-60e7-42e3-b765-6d633a1f2d44",
            'result': similarity_box_sentences
        })
    time_process = time.time() - time_start
    result = {
        "file_check_id": file_check_id,
        "result": similarity_documents,
        "create_at": "2024-06-12T15:04:05Z",
        "duration": time_process
    }
    # output_file = "abcd.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)
    headers_auth = {
        'Content-Type': 'application/json',
        'Cookie': 'jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDMwOTExNDUsImlzcyI6Ijg0MDhhZjZkLWZlOGMtNDM4NS05N2IyLTU1MzFiNjIwM2U1OSIsInVzZXJuYW1lIjoiaGluZTEyIiwiZW1haWwiOiJoaW5lMTIiLCJyb2xlIjoic3UifQ.bnNPUsjj5ADfEgRbNL5Q7ScHsrjds1Sat31o_1rjKVw'
    }
    response = requests.request("POST", url, headers=headers_auth, data=json.dumps(result))
    return {"message": "Thành công"}
