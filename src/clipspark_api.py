import os
import shutil
import logging
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from celery.result import AsyncResult
from src.clipspark_worker import process_video_task

# === SHARED VOLUME CONFIG ===
SHARED_DIR = Path("/app/shared")
TEMP_DIR = SHARED_DIR / "uploads"
OUTPUT_DIR = SHARED_DIR / "output"
CLIPS_DIR = OUTPUT_DIR / "clips"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ClipSpark API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clipspark_api")

@app.post("/process-video", status_code=status.HTTP_202_ACCEPTED)
async def process_video(file: UploadFile = File(...)):
    try:
        # Use UUID for collision-free filenames
        ext = Path(file.filename).suffix
        job_id = str(uuid.uuid4())
        safe_filename = f"{job_id}{ext}"
        
        file_path = TEMP_DIR / safe_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Send absolute paths to worker
        task = process_video_task.apply_async(
            args=[str(file_path.absolute()), str(OUTPUT_DIR.absolute())],
            task_id=job_id
        )
        
        return {"job_id": job_id, "status": "queued"}
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    task = AsyncResult(job_id)
    # Simple status mapping
    state_map = {
        "PENDING": "queued",
        "STARTED": "processing",
        "SUCCESS": "completed",
        "FAILURE": "failed"
    }
    return {"job_id": job_id, "status": state_map.get(task.state, "processing")}

@app.get("/job-result/{job_id}")
async def get_job_result(job_id: str):
    task = AsyncResult(job_id)
    if task.state == "SUCCESS":
        return task.result
    elif task.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(task.result))
    else:
        raise HTTPException(status_code=404, detail="Job not ready")

@app.get("/clips/{filename}")
async def get_clip(filename: str):
    file_path = CLIPS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(file_path)