from langchain_milvus import Milvus
import json
from langchain_ollama import OllamaEmbeddings
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Any
from langchain_core.documents import Document


with open("milvus_config.json", "r") as f:
    milvus_config = json.load(f)

ollama_model_name = "dengcao/Qwen3-Embedding-8B:Q8_0"
embedding_model = OllamaEmbeddings(
    model=ollama_model_name
)

COLLECTION_NAME = "LangChainCollection"
vector_store = Milvus(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    connection_args=milvus_config,
    auto_id=True
)

app = FastAPI(
    title="向量搜索 API (Vector Search API)",
    description="一个使用 FastAPI, LangChain, Ollama 和 Milvus 构建的向量搜索服务。",
    version="1.0.0",
)


class SearchResultMetadata(BaseModel):
    source: Optional[str] = None
    doc_id: Optional[int] = None
    pk: Optional[int] = None
    # 在这里添加你所有可能的元数据字段


class SearchResultItem(BaseModel):
    page_content: str
    metadata: SearchResultMetadata
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]

   
@app.get("/search", response_model=SearchResponse)
def search_vector(
        query: str,
        k: int = Query(default=4, ge=1, le=20, description="要返回的结果数量"),
        expr: Optional[str] = Query(default=None,
                                    description="用于元数据过滤的 Milvus 表达式 (例如, 'doc_id in [1, 5]')")
):
    """
    根据给定的查询文本执行向量相似度搜索。

    - **query**: 要搜索的文本。
    - **k**: 返回的最相似结果的数量。
    - **expr**: (可选) 一个 Milvus 布尔表达式，用于在搜索前进行元数据过滤。
    """
    print(f"收到查询: query='{query}', k={k}, expr='{expr}'")

    try:
        # 执行带有分数和过滤的相似度搜索
        search_results_with_score = vector_store.similarity_search_with_score(
            query=query,
            k=k,
            expr=expr,
        )

        # 格式化返回结果
        formatted_results = []
        for doc, score in search_results_with_score:
            # 清理元数据，只保留我们定义过的字段
            cleaned_metadata = SearchResultMetadata(**doc.metadata)
            formatted_results.append(
                SearchResultItem(
                    page_content=doc.page_content,
                    metadata=cleaned_metadata,
                    score=score
                )
            )

        return SearchResponse(query=query, results=formatted_results)

    except Exception as e:
        # 捕获可能的运行时错误，例如 Milvus 连接断开
        print(f"❌ 查询处理时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


# (可选) 添加一个根路径，用于健康检查
@app.get("/")
def read_root():
    return {"status": "ok", "message": "向量搜索服务正在运行"}