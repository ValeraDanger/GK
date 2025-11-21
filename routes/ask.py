from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scripts.ask import answer_query
from utils.logger import get_logger

router = APIRouter()
log = get_logger("[AskRoute]")

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

from typing import List, Dict

class SourceItem(BaseModel):
    source: str
    content: str

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

@router.post("", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    try:
        result = answer_query(request.question, request.top_k)
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        log.error(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
