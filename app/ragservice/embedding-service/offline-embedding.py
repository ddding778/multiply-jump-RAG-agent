#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 离线索引脚本：使用阿里云百炼 text-embedding-v3 模型将文档向量化并存入 Qdrant。
"""

import os
import sys
import hashlib
from pathlib import Path

# 尝试加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 第三方库
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ==================== 配置 ====================
# 阿里云百炼配置
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")
if not ALIYUN_API_KEY:
    raise ValueError("请设置环境变量 ALIYUN_API_KEY")
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 1024   # 默认 1024，可根据需求改为 768, 512 等

# Qdrant 配置
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tech_docs")

# 文档目录（相对于本脚本的位置：项目根目录下的 docs 文件夹）
DOCS_DIR = Path(__file__).parent.parent.parent.parent/ "docs"

# 文本分割器配置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)

# ==================== 初始化客户端 ====================
# 阿里云百炼 OpenAI 兼容客户端
client = OpenAI(
    api_key=ALIYUN_API_KEY,
    base_url=ALIYUN_BASE_URL,
)

# Qdrant 客户端（gRPC 连接）
qdrant = QdrantClient(
    host=QDRANT_HOST,
    grpc_port=QDRANT_GRPC_PORT,
    prefer_grpc=True
)

# ==================== 辅助函数 ====================
def get_embedding(text: str):
    """调用 text-embedding-v3 模型返回向量列表"""
    try:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIM   # 指定维度（与集合创建时一致）
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"Embedding API 调用失败: {e}")
        raise

def ensure_collection():
    """检查并创建 Qdrant 集合"""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"已创建集合: {COLLECTION_NAME}")
    else:
        print(f"集合已存在: {COLLECTION_NAME}")

def index_document(file_path: Path):
    """索引单个文档文件"""
    print(f"处理文件: {file_path.name}")
    content = file_path.read_text(encoding='utf-8')
    chunks = text_splitter.split_text(content)
    if not chunks:
        print(f"  文件无有效内容，跳过")
        return

    points = []
    for i, chunk in enumerate(chunks):
        # 生成唯一 ID（基于文件绝对路径和块索引）
        unique_id = hashlib.md5(f"{file_path.absolute()}_{i}".encode()).hexdigest()
        try:
            vec = get_embedding(chunk)
        except Exception as e:
            print(f"  向量化失败 (块 {i}): {e}")
            continue
        points.append(PointStruct(
            id=unique_id,
            vector=vec,
            payload={"text": chunk, "source": file_path.name}
        ))

    if points:
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"  成功索引 {len(points)} 个文本块")
        except Exception as e:
            print(f"  写入 Qdrant 失败: {e}")
    else:
        print(f"  没有成功向量化的文本块")

# ==================== 主流程 ====================
def main():
    # 检查文档目录
    if not DOCS_DIR.exists():
        print(f"错误: 文档目录不存在: {DOCS_DIR}")
        sys.exit(1)

    # 确保 Qdrant 集合存在
    ensure_collection()

    # 遍历文档目录中的所有 .md 和 .txt 文件
    files_processed = 0
    for file_path in DOCS_DIR.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in ('.md', '.txt'):
            continue
        index_document(file_path)
        files_processed += 1

    if files_processed == 0:
        print("警告: 未找到任何 .md 或 .txt 文件，请将文档放入 docs/ 目录")
    else:
        print("所有文档索引完成")

if __name__ == "__main__":
    main()