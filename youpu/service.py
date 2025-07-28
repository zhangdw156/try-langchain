import requests, os, json, requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Any
from pymilvus import utility, connections
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from enum import Enum
from langchain_core.documents import Document
from loguru import logger

with open("milvus_config.json", "r") as f:
    milvus_config = json.load(f)

with open("youpu_config.json", "r") as f:
    youpu_config = json.load(f)


def delete_collection(collection_name: str):
    """
    根据collection_name删除milvus中的collection
    :param collection_name:
    :return:
    """
    # --- 1. 配置你的 Milvus 连接信息 ---
    # 替换成你的 Milvus 实例的地址和端口
    MILVUS_HOST = "localhost"  # 或者你的 Milvus 服务器 IP
    MILVUS_PORT = "19530"
    COLLECTION_NAME_TO_DELETE = collection_name  # <--- 在这里填入你要删除的 Collection 名称

    # 定义一个别名，方便管理多个连接
    alias = "default"

    try:
        # --- 2. 连接到 Milvus ---
        logger.info(f"正在连接到 Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
        connections.connect(alias=alias, host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("连接成功！")

        # --- 3. (推荐) 检查 Collection 是否存在 ---
        logger.info(f"正在检查 Collection '{COLLECTION_NAME_TO_DELETE}' 是否存在...")
        if utility.has_collection(COLLECTION_NAME_TO_DELETE, using=alias):
            logger.info(f"Collection '{COLLECTION_NAME_TO_DELETE}' 存在，准备删除。")

            # --- 4. 执行删除操作 ---

            utility.drop_collection(COLLECTION_NAME_TO_DELETE, using=alias)
            logger.info(f"✅ Collection '{COLLECTION_NAME_TO_DELETE}' 已成功删除！")


        else:
            logger.info(f"🤷‍ Collection '{COLLECTION_NAME_TO_DELETE}' 不存在，无需删除。")

    except Exception as e:
        logger.info(f"发生错误: {e}")

    finally:
        # --- 5. 断开连接 ---
        # 检查是否有名为 'default' 的连接存在
        existing_connections = connections.list_connections()
        if any(conn[0] == alias for conn in existing_connections):
            connections.disconnect(alias)
            logger.info("已断开与 Milvus 的连接。")


class DB:
    """
    管理与数据库交互时的参数，包括了不同数据库的api接口，哪些数据要存储为向量，哪些数据要存储为元数据
    """

    def __init__(self, api: str, keys_to_store: set[str], keys_to_meta: set[str], collection_name: str):
        self.api = api
        self.keys_to_store = keys_to_store
        self.keys_to_meta = keys_to_meta
        self.collection_name = collection_name
        self.vector_store = Milvus(
            embedding_function=OllamaEmbeddings(
                model=youpu_config["ollama_model_name"]
            ),
            collection_name=self.collection_name,
            connection_args=milvus_config,
            auto_id=True
        )


class Job(Enum):
    """
    区分不同的数据库，不同的数据库有不同的api接口，不同的数据库有不同的keys_to_store和keys_to_meta
    """
    FORMS = DB(api="formSubmissionHistory",
               keys_to_store={'userName', 'department', 'formName', 'procurementProjectName', 'procurementMethod',
                              'projectType',
                              'packageName', },
               keys_to_meta={'id', 'userId', 'createTime', 'updateTime', 'submitTime'},
               collection_name="forms",
               )
    FEEDBACKS = DB(api="userBehaviorAnnotation",
                   keys_to_store={'contextInfo', 'fieldName', 'recommendContent', 'userFeedback', 'annotation',
                                  'userFeedbackRemark'},
                   keys_to_meta={'id', 'formId', 'formSubmitTime', 'createTime', 'updateTime'},
                   collection_name="feedbacks",
                   )
    KNOWLEDGE = DB(api="procurementDocumentLibrary",
                   keys_to_store={'projectName', 'packageNumber'},
                   keys_to_meta={'id', 'documentPublishTime', 'createTime', 'updateTime'},
                   collection_name="knowledge",
                   )


def prepare_data(job: Job = Job.FORMS):
    """
    根据job.value.api获取数据，根据job.value.keys_to_store和job.value.keys_to_meta处理数据，最后将数据存储到job.value.vector_store中
    :param job:
    :return:
    """
    base_url = youpu_config["backend_url"]
    url = base_url + job.value.api + "/queryAll"
    logger.info(url)
    data = requests.get(url).json()['result']
    logger.info(f"一共有{len(data)}条数据")
    for item in data:
        logger.info(item)
        item_to_store = {key: (value if value is not None else "") for key, value in item.items() if
                         key in job.value.keys_to_store} if job.value.keys_to_store is not None else item
        item_to_meta = {key: (value if value is not None else "") for key, value in item.items() if
                        key in job.value.keys_to_meta}
        logger.info(item_to_store)
        logger.info(item_to_meta)
        doc = Document(page_content=json.dumps(item_to_store, ensure_ascii=False), metadata=item_to_meta)
        logger.info(doc)
        job.value.vector_store.add_documents([doc])


if __name__ == '__main__':
    # prepare_data(Job.KNOWLEDGE)
    pass
