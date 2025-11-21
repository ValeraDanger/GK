# ask.py

import sys

# Импорт классов из вашего init-файла или отдельных файлов
from init import HybridRAGSystem, CloudRuEmbeddings
from config import *

def answer_query(
    question,
    qdrant_path=QDRANT_PATH,
    qdrant_collection=QDRANT_COLLECTION,
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    cloud_api_key=CLOUD_API_KEY,
    cloud_ru_url=CLOUD_RU_URL,
    top_k=5,
    alpha=0.5,
):
    print("="*60)
    print("🧠 Ответ на вопрос по базе знаний RAG")
    print(f"Вопрос: {question}")
    print("="*60)

    embeddings = CloudRuEmbeddings(api_key=cloud_api_key, base_url=cloud_ru_url)
    rag = HybridRAGSystem(
        embeddings=embeddings,
        qdrant_path=qdrant_path,
        collection_name=qdrant_collection,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        llm_api_key=cloud_api_key,
        llm_base_url=cloud_ru_url
    )
    answer, results = rag.rag(question, top_k=top_k)

    print("\n--- Ответ LLM --------------------------------------------------")
    print(answer)
    print("---------------------------------------------------------------\n")

    print("--- Чанки, на которых строился ответ -----------------------")
    for i, r in enumerate(results, 1):
        print(f"[{i}] ({r.source}) score={r.score:.3f}: {r.content[:200]} ...")
    print("---------------------------------------------------------------")

    # возможность вернуть ответ для web/cli или API
    return answer

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Введите ваш вопрос: ")
    answer_query(user_query)
