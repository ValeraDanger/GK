from services.rag_system import HybridRAGSystem
from services.embeddings import CloudRuEmbeddings
from utils.config import (
    QDRANT_PATH, QDRANT_COLLECTION, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    CLOUD_API_KEY, CLOUD_RU_URL
)
from utils.logger import get_logger

log = get_logger("[AskScript]")

def answer_query(
    question: str,
    top_k: int = 5,
) -> str:
    log.info(f"Processing question: {question}")

    embeddings = CloudRuEmbeddings(api_key=CLOUD_API_KEY, base_url=CLOUD_RU_URL)
    rag = HybridRAGSystem(
        embeddings=embeddings,
        qdrant_path=QDRANT_PATH,
        collection_name=QDRANT_COLLECTION,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        llm_api_key=CLOUD_API_KEY,
        llm_base_url=CLOUD_RU_URL
    )
    answer, results = rag.rag(question, top_k=top_k)

    log.info("Answer generated successfully")
    return answer
