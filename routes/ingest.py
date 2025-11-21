from fastapi import APIRouter, HTTPException, BackgroundTasks
from scripts.ingest import ingest_files
from utils.logger import get_logger

router = APIRouter()
log = get_logger("[IngestRoute]")

@router.post("")
async def ingest_endpoint(background_tasks: BackgroundTasks):
    """
    Trigger ingestion process in background.
    """
    try:
        # Running in background to avoid blocking
        background_tasks.add_task(ingest_files)
        return {"message": "Ingestion started in background"}
    except Exception as e:
        log.error(f"Error triggering ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
