from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from milvus_db.create_corpus import CorpusCreator
from config.conf import BASEURL
import time
import os
import shutil
from pathlib import Path
import glob
import requests
import json

# Import functions from zip_test.py
from zip_test import (
    unzip_file, 
    process_files_parallel,
    find_similar_files,
    format_results_to_json
)

checker = CorpusCreator()

router = APIRouter(
    tags=["update-database"],
)

baseURL = BASEURL
# Create directories if they don't exist
UPLOAD_DIR = Path("uploads")
EXTRACT_DIR = Path("extracted_files")
UPLOAD_DIR.mkdir(exist_ok=True)
EXTRACT_DIR.mkdir(exist_ok=True)

@router.post("/upload_zip_file/{concat}")
async def upload_zip_file(
    concat: str,
    file: UploadFile = File(...),
):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    # # Create temporary directories with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    job_upload_dir = UPLOAD_DIR / timestamp
    job_extract_dir = EXTRACT_DIR / timestamp
    
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_extract_dir, exist_ok=True)
    
    try:
        # Save the uploaded file
        zip_path = job_upload_dir / file.filename
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return JSONResponse(
            content={
                "message": "Nhận file zip thành công",
                "concat": concat,
            },
            status_code=200
        )
    except Exception as e:
        try:
            shutil.rmtree(job_upload_dir)
            shutil.rmtree(job_extract_dir)
        except:
            pass
        return JSONResponse(
            content={
                "message": "Nhận file zip thất bại",
            },
            status_code=409)


@router.post("/find_pair_similar_docs/{concat}")
async def upload_zip_file(
    concat: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    similarity_threshold: float = 0.9,
    min_similarity_percent: float = 20.0
):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    # Create temporary directories with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    job_upload_dir = UPLOAD_DIR / timestamp
    job_extract_dir = EXTRACT_DIR / timestamp
    
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_extract_dir, exist_ok=True)
    
    try:
        # Save the uploaded file
        zip_path = job_upload_dir / file.filename
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Start processing in background
        background_tasks.add_task(
            create_corpus_database, 
            zip_path=str(zip_path),
            extract_dir=str(job_extract_dir),
            concat=concat,
            similarity_threshold=similarity_threshold,
            min_similarity_percent=min_similarity_percent,
            
        )
        return JSONResponse(
            content={
                "message": "Nhận file zip thành công",
                "concat": concat,
            },
            status_code=200
        )
    except Exception as e:
        try:
            shutil.rmtree(job_upload_dir)
            shutil.rmtree(job_extract_dir)
        except:
            pass
        return JSONResponse(
            content={
                "message": "Nhận file zip thất bại",
            },
            status_code=409)

async def create_corpus_database(
    zip_path: str,
    extract_dir: str,
    concat: str,
    similarity_threshold: float = 0.9,
    min_similarity_percent: float = 20.0
):
    try:
        print(f"Starting processing of {zip_path}")
        start_time = time.time()
        # Unzip the file
        unzip_file(zip_path, extract_dir)
        print(f"Extraction completed for {zip_path}")
        # Find all PDF files in the extract directory
        pdf_files = glob.glob(os.path.join(extract_dir, "**/*.pdf"), recursive=True)
        total_file = len(pdf_files)
        if not pdf_files:
            print(f"No PDF files found in {zip_path}")
            # Clean up
            cleanup_files(os.path.dirname(zip_path), extract_dir)
            return
        
        print(f"Processing {len(pdf_files)} PDF files")
        checker.create_corpus(extract_dir)


    except Exception as e:
        print(f"Error processing ZIP file: {str(e)}")
        # Clean up on error
        cleanup_files(os.path.dirname(zip_path), extract_dir)        



def cleanup_files(upload_dir, extract_dir):
    """Clean up temporary directories after processing."""
    try:
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
    except Exception as e:
        print(f"Error cleaning up directories: {str(e)}")