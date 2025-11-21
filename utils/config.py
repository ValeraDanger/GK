import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_USER = os.getenv('NEO4J_USER')

QDRANT_PATH = os.getenv('QDRANT_PATH')
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION')
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
CLOUD_API_KEY = os.getenv('CLOUD_API_KEY')
CLOUD_RU_URL = os.getenv('CLOUD_RU_URL')

INPUT_FOLDER = os.getenv('INPUT_FOLDER', str(BASE_DIR / "input_files"))
TEXT_OUTPUT_FOLDER = os.getenv('TEXT_OUTPUT_FOLDER', str(BASE_DIR / "ocr_texts"))

VECTOR_SIZE = int(os.getenv('VECTOR_SIZE', 1024)) # Default value if not set
SEMANTIC_BREAKPOINT_TYPE = os.getenv('SEMANTIC_BREAKPOINT_TYPE', 'percentile')
SEMANTIC_BREAKPOINT_THRESHOLD = float(os.getenv('SEMANTIC_BREAKPOINT_THRESHOLD', 0.9))

MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', 2))
HYBRID_ALPHA = float(os.getenv('HYBRID_ALPHA', 0.5))
