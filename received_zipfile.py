from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import time
import glob

# Import functions from zip_test.py
from zip_test import (
    unzip_file, 
    process_files_parallel,
    find_similar_files,
    format_results_to_json
)

# Create FastAPI app
app = FastAPI(
    title="PDF Similarity API",
    description="API for comparing similarity between PDF documents in a ZIP file",
    version="1.0.0"
)

# Create directories if they don't exist
UPLOAD_DIR = Path("uploads")
EXTRACT_DIR = Path("extracted_files")
UPLOAD_DIR.mkdir(exist_ok=True)
EXTRACT_DIR.mkdir(exist_ok=True)


@app.post("/upload/")
async def upload_zip_file(
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
            process_similarity_task, 
            zip_path=str(zip_path),
            extract_dir=str(job_extract_dir),
            similarity_threshold=similarity_threshold,
            min_similarity_percent=min_similarity_percent
        )
        
        # Return immediate response
        return JSONResponse(
            content={
                "message": "ZIP file received successfully and processing has started",
                "file_info": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size_bytes": file.size if hasattr(file, "size") else None,
                    "received_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            },
            status_code=200
        )
    except Exception as e:
        # Clean up on error
        try:
            shutil.rmtree(job_upload_dir)
            shutil.rmtree(job_extract_dir)
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Error receiving ZIP file: {str(e)}"
        )

async def process_similarity_task(
    zip_path: str,
    extract_dir: str,
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
        
        if not pdf_files:
            print(f"No PDF files found in {zip_path}")
            # Clean up
            cleanup_files(os.path.dirname(zip_path), extract_dir)
            return
        
        print(f"Processing {len(pdf_files)} PDF files")
        
        # Process all files in parallel
        file_data = process_files_parallel(pdf_files, max_workers=4)
        
        print(f"Finding similar documents")
        
        # Find similar file pairs
        similar_pairs = find_similar_files(
            file_data,
            similarity_threshold=similarity_threshold,
            min_similarity_percent=min_similarity_percent,
            parallel=True,
            max_workers=4
        )
        
        # Calculate total time
        end_time = time.time()
        total_time = end_time - start_time
        results = format_results_to_json(similar_pairs, file_data, min_similarity_percent, total_time)
        print(f"Processing completed in {total_time:.2f} seconds. Found {len(similar_pairs)} similar file pairs.")
        print(results)
        # Clean up
        cleanup_files(os.path.dirname(zip_path), extract_dir)

        return JSONResponse(
            content=results
        )
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8689) 