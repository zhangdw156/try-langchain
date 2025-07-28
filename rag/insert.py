from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from pymilvus import utility, connections
from langchain_core.documents import Document

# --- 1. 连接到 Milvus 服务 ---
# 如果你是本地 Docker 启动的 Milvus
milvus_config = {
    "user": "root",
    "password": "Milvus",
    "uri": "http://127.0.0.1:19530",
}

# 连接 Milvus
connections.connect(**milvus_config)
print("🔗 连接到 Milvus 成功！")

# --- 2. 准备 Embedding 模型 (使用 Ollama) ---
ollama_model_name = "dengcao/Qwen3-Embedding-8B:Q8_0"

# 实例化 OllamaEmbeddings
embedding_model = OllamaEmbeddings(
    model=ollama_model_name
)

print(f"🔗 已连接到 Ollama 服务，将使用 Embedding 模型 '{ollama_model_name}'！")

# --- 3. 定义你的 Collection 名称 ---
# Collection 在 Milvus 中类似于关系型数据库中的“表”
COLLECTION_NAME = "LangChainCollection"

vector_store = Milvus(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    connection_args=milvus_config,
    auto_id=True
)
print(f"✅ Milvus 向量存储准备就绪，将使用 Collection: '{COLLECTION_NAME}'")
print("-" * 30)

# 模拟要写入的数据流，每条数据都是一个字符串
documents_to_add = [
    "LangChain 是一个强大的语言模型应用开发框架。",
    "Milvus 是一个专为向量搜索设计的高性能数据库。",
    "Zilliz Cloud 提供了托管的 Milvus 服务，非常方便。",
    "Embedding 模型可以将文本转换为高维向量。",
    "向量相似度搜索是许多 AI 应用的核心技术。"
]

# 循环遍历数据，一条一条地写入
for i, text in enumerate(documents_to_add):
    print(f"✍️ 正在写入第 {i + 1} 条数据: '{text}'")

    doc = Document(page_content=text, metadata={"source": "manual_input", "doc_id": i + 1})

    vector_store.add_documents([doc])

print("\n🎉 所有数据已逐条写入 Milvus！")
