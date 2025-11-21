import logging
from fastapi import FastAPI
from routes import api_router
from starlette.middleware.cors import CORSMiddleware
from utils.logger import setup_logging, get_logger

setup_logging(level=logging.DEBUG, log_to_file=True)
log = get_logger("[API]")

app = FastAPI(title="RAG Gorkiy API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "ok"}
