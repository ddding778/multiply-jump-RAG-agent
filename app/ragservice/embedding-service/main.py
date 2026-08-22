import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 配置
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tech_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1024))
EMBEDDING_API_KEY = os.getenv("ALIYUN_API_KEY")  # 或 OPENAI_API_KEY
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化客户端
qdrant = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)
embedding_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

app = FastAPI(title="RAG Service")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResponse(BaseModel):
    documents: list[str]

def get_embedding(text: str):
    resp = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIM
    )
    return resp.data[0].embedding

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    try:
        # 1. 向量化用户查询
        query_vec = get_embedding(req.query)
        # 2. Qdrant 检索
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,           
            limit=req.top_k,
            with_payload=True
        )
        # 3. 提取文本片段
        points = search_result.points
        docs = [point.payload.get("text", "") for point in points if point.payload]
        return SearchResponse(documents=docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)