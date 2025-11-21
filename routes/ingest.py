from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from scripts.ingest import ingest_files
from utils.logger import get_logger
from utils.config import INPUT_FOLDER
import os

router = APIRouter()
log = get_logger("[IngestRoute]")

class IngestRequest(BaseModel):
    filename: str

@router.post("")
async def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger ingestion process in background for a specific file located in INPUT_FOLDER.
    """
    try:
        file_path = os.path.join(INPUT_FOLDER, request.filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found in input folder: {request.filename}")
            
        log.info(f"Triggering ingestion for file: {file_path}")

        # Running in background to avoid blocking, passing ONLY the new file
        background_tasks.add_task(ingest_files, files_to_ingest=[file_path])
        
        return {"message": f"Ingestion started for {request.filename} in background"}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error triggering ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
