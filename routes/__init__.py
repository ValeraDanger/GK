from fastapi import APIRouter
from .ask import router as ask_router
from .ingest import router as ingest_router

api_router = APIRouter()
api_router.include_router(ask_router, prefix="/ask", tags=["ask"])
api_router.include_router(ingest_router, prefix="/ingest", tags=["ingest"])

__all__ = ["api_router"]
