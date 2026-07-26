"""
向量存储模块：封装 ChromaDB 的读写操作。

为什么选 ChromaDB 而非 FAISS 或 Milvus：
- FAISS：C++ 实现，Windows 上编译困难，需要 Visual Studio Build Tools，
  且纯内存模式数据不持久化，IndexFlatL2 存盘需要额外序列化代码
- Milvus：需要 Docker，Windows 上 Docker Desktop 占用资源大，杀鸡用牛刀
- ChromaDB：纯 Python + SQLite，pip install 即用，自动持久化，
  自带元数据（metadata）存储，检索时直接返回文档内容+来源章节，无需额外查映射表

核心概念：
- Client：数据库连接（PersistentClient = SQLite 文件持久化）
- Collection：类比 SQL 表，存储 (id, document, embedding, metadata) 四元组
- 查询：输入 query_embedding，返回最相似的 n 条记录及距离
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import VECTOR_STORE_DIR, COLLECTION_NAME


def get_chroma_client() -> chromadb.PersistentClient:
    """获取 ChromaDB 持久化客户端。

    为什么用 PersistentClient 而非 EphemeralClient：
    EphemeralClient 数据只在内存中，进程退出即丢失。每次启动 `python main.py`
    都要重新走一遍 文档加载 → 分块 → embedding → 写入，即使知识库没变。
    对于 25 个 chunk，全流程约 30 秒（含模型加载），纯属浪费。
    PersistentClient 把向量写入 VECTOR_STORE_DIR 的 SQLite 文件，
    重启后毫秒级加载，零等待。

    anonymized_telemetry=False：关闭 ChromaDB 的匿名遥测上报，避免网络请求延迟。
    """
    return chromadb.PersistentClient(
        path=VECTOR_STORE_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection(name: str = None):
    """获取已有 collection，不存在则创建新的。

    get_or_create 的语义：
    - collection 已存在 → 返回已有，保留全部数据
    - collection 不存在 → 创建空的，count() 返回 0

    为什么用 get_or_create 而非 get_collection：
    get_collection 在 collection 不存在时抛 ValueError，调用方要写 try/except，
    代码冗长。get_or_create 一个调用覆盖两种情况。

    调用方通过返回的 count() 判断是否需要重建：
    - count == 0 → 首次使用，需走 build_index
    - count > 0 → 已有索引，跳过 build（除非用户指定 --rebuild）
    """
    client = get_chroma_client()
    coll_name = name or COLLECTION_NAME
    return client.get_or_create_collection(name=coll_name)


def rebuild_collection(name: str = None):
    """删除旧 collection 并创建新的空 collection。

    为什么需要重建而非追加（add）新 chunk：
    如果知识库内容有修改（新增、删减、修正），追加模式会导致：
    1. 旧内容的向量仍然可被检索 → 用户可能看到过时/矛盾的信息
    2. 同一内容出现多份向量（旧版 + 新版）→ 检索结果重复
    3. 被删除的内容继续存在 → 向量库和知识库不一致

    重建是唯一能保证向量库 == 知识库 的操作。
    对于 25 个 chunk 的小库，重建开销可忽略（几秒 embedding）。

    为什么先 try delete 再 create 而非直接 create：
    ChromaDB 不允许创建同名 collection，先删后建是最干净的方案。
    try/except 处理了首次运行时 collection 不存在的情况。
    """
    client = get_chroma_client()
    coll_name = name or COLLECTION_NAME

    try:
        client.delete_collection(name=coll_name)
    except Exception:
        pass  # collection 不存在 → 无需删除，继续创建

    return client.create_collection(name=coll_name)


def add_chunks_to_store(collection, chunks: list[dict], embeddings: list[list[float]]):
    """将文档块及其向量批量写入 ChromaDB。

    参数:
        collection: get_or_create_collection() 或 rebuild_collection() 的返回值
        chunks: [{"content": "文本", "metadata": {"source": "校园美食", "chunk_index": 0}}, ...]
        embeddings: List[List[float]], shape=(N, 1024)，和 chunks 一一对应

    ID 生成策略：用 "章节名_序号" 而非随机 UUID。
    好处：
    1. 可读性强：一看 "校园美食_0" 就知道是校园美食章节的第一个 chunk
    2. 幂等性：重复调用 add 时，相同 ID 会被 upsert 而非插入重复记录
       （ChromaDB 的 add 在 ID 冲突时默认行为是 update）
    3. 如果改用 UUID，每次 rebuild 都会产生新 ID，旧向量不会被覆盖，
       白白占用磁盘直到手动清理
    """
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        meta = chunk["metadata"]
        chunk_id = f"{meta['source']}_{meta['chunk_index']}"
        ids.append(chunk_id)
        documents.append(chunk["content"])
        metadatas.append(meta)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_collection(collection, query_embedding: list[float], n_results: int = 5):
    """检索与查询向量最相似的 top-n 文档块。

    参数:
        collection: ChromaDB collection
        query_embedding: 查询文本的 embedding 向量, shape=(1024,)
        n_results: 返回 top-K 个最相似结果

    返回 ChromaDB 原生结构:
        {
            "ids": [["校园美食_0", "学社简介_0", ...]],
            "documents": [["北餐有葛辉饺子...", "太原理工大学创新学社秉承...", ...]],
            "metadatas": [[{"source": "校园美食", "chunk_index": 0}, ...]],
            "distances": [[0.0234, 0.0312, ...]],
        }
    注意：所有值都是一层额外的列表包裹（ChromaDB 支持批量查询的遗留设计），
    取结果时用 results["documents"][0][i] 而非 results["documents"][i]。

    distances 是 L2 距离（欧几里得距离），范围 [0, 2]（归一化向量时）。
    距离越小越相似，0 = 完全相同的向量。
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return results
