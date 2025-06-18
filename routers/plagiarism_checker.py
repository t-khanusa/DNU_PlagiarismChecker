from config.postgres_db import SessionLocal
from config.conf import BASEURL, COOKIE
from models.model import PDFFile, Sentence
from aws_file.get_pdffile_aws import read_presigned_url, check_file_in_s3
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from milvus_db.create_milvus_db import collection
from milvus_db.create_corpus import CorpusCreator
from typing import List, Dict, Tuple
from .spell_correction import spell_checker
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
from datetime import datetime, timezone
import asyncio
from pydantic import BaseModel


router = APIRouter(
    tags=["plagiarism-checker"],
)

baseURL = BASEURL
checker = CorpusCreator()

def run_spell_check_sync(file_path, file_check_id, time_start):
    """Sync wrapper for spell_checker"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(spell_checker(file_path, file_check_id, time_start))
    finally:
        loop.close()

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

# @router.get("/plagiarism_checker_details/") 
# async def check_file(background_tasks: BackgroundTasks, subject_id: str, file_name: str, file_check_id: str):    
#     print("File name: ", file_name)

#     is_file_exist, error = check_file_in_s3(subject_id, file_name)
#     if not is_file_exist:
#         return JSONResponse(content={"message": 'Không tìm thấy file'}, status_code=404)
#     else:
#         JSONResponse(content={"message": 'Tìm thấy file'}, status_code=200)
#     time_start = time.time()
#     try:
#         file_path = read_presigned_url(subject_id, file_name)
#         print(f"Tải file thành công: {file_path}")
        
#     except Exception as e:
#         print(f"Bug cmnr: {e}")
#         return JSONResponse(content={"message": 'Không tải được file'}, status_code=400)
    
#     # Get the current event loop
#     loop = asyncio.get_event_loop()
    
#     # Run both CPU-intensive tasks in separate threads
#     task1 = loop.run_in_executor(None, check_plagiarism_details, file_path, file_check_id, time_start)
#     task2 = loop.run_in_executor(None, run_spell_check_sync, file_path, file_check_id, time_start)
    
#     # Run tasks in background without waiting for completion
#     background_tasks.add_task(asyncio.gather, task1, task2)

#     return JSONResponse(content={"message": 'Tải file thành công'}, status_code=200)


# Request body model
class FileCheckRequest(BaseModel):
    subject_id: str
    file_name: str
    file_check_id: str

@router.post("/plagiarism_checker_details/")
async def check_file_post(background_tasks: BackgroundTasks, data: FileCheckRequest):
    subject_id = data.subject_id
    file_name = data.file_name
    file_check_id = data.file_check_id

    print("File name: ", file_name)
    
    is_file_exist, error = check_file_in_s3(subject_id, file_name)
    if not is_file_exist:
        return JSONResponse(content={"message": 'Không tìm thấy file'}, status_code=404)

    time_start = time.time()
    try:
        file_path = read_presigned_url(subject_id, file_name)
        print(f"Tải file thành công: {file_path}")
    except Exception as e:
        print(f"Bug cmnr: {e}")
        return JSONResponse(content={"message": 'Không tải được file'}, status_code=400)

    # Run các task nặng trong background
    loop = asyncio.get_event_loop()
    task1 = loop.run_in_executor(None, check_plagiarism_details, file_path, file_check_id, time_start)
    task2 = loop.run_in_executor(None, run_spell_check_sync, file_path, file_check_id, time_start)

    background_tasks.add_task(asyncio.gather, task1, task2)

    return JSONResponse(content={"message": 'Tải file thành công'}, status_code=200)


  
def check_plagiarism_details(file_path, file_check_id, time_start) -> Dict:
    min_similarity = 0.9
    url = f"{baseURL}/api/checker/files/{file_check_id}/result"

    try:
        document = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        document = None

    raw_sentences, embeddings = checker.process_document(file_path)

    if not raw_sentences:
        return {"error": "No valid raw_sentences found in document"}    
    print(f"Number of raw_sentences processed: {len(raw_sentences)}")

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
    total_similarity_percent = int(len(matched_sentence_set) / total_sentences * 100)
 
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
                    # Lambda function to merge rectangles on the same line
                    merge_rects = lambda rects: [
                        [min(group, key=lambda r: r[0])[0],  # min x0
                         min(group, key=lambda r: r[1])[1],  # min y0  
                         max(group, key=lambda r: r[2])[2],  # max x1
                         max(group, key=lambda r: r[3])[3]]  # max y1
                        for group in [
                            [rect for rect in rects if abs(rect[1] - y) < 5]  # Group by similar y-coordinate (within 5 pixels)
                            for y in sorted(set(rect[1] for rect in rects))
                        ] if group
                    ]
                    
                    merged_rects = merge_rects(location["rects"])
                    
                    page_sentences[page_num].append({
                        'content': sentence,
                        'rects': merged_rects
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
            "similarity_value": similarity_value,
            'page_result': similarity_box_sentences
        })
    time_process = time.time() - time_start

    result = {
        "file_check_id": file_check_id,
        "data_result": similarity_documents,
        "create_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_similarity_percent": total_similarity_percent,
        "height": document[0].rect.height,
        "width": document[0].rect.width,
        "duration": time_process
    }
    output_file = f"plagiarism_check_{file_check_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"plagiarism_check results saved to {output_file}")

    # output_file = "abcd.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)
    headers_auth = {
        'Content-Type': 'application/json',
        'Cookie': 'jwt='+COOKIE
    }
    response = requests.request("POST", url, headers=headers_auth, data=json.dumps(result))
    # print("check plagiarism details thành công")
    # response2 = requests.request("POST", url2, headers=headers_auth, data=json.dumps(spell_result))
    return {"message": "Thành công"}

