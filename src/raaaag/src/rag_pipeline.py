"""
RAG 主流程：索引构建 + 问答检索（Retrieve-Augment-Generate）。

两个核心操作：
- build_index(): 文档 → 分块 → embedding → ChromaDB（离线，一次性）
- query(): 用户问题 → embedding → 检索 → LLM 生成回答（在线，每次调用）

为什么把建索引和问答放在同一个文件：
build_index 和 query 共享同一套依赖链（document_loader → embedder → vector_store），
拆成两个文件会导致大量重复 import。对于 300 行的项目，一个文件更清晰。
"""

from src.document_loader import chunk_document
from src.embedder import embed_texts
from src.vector_store import (
    rebuild_collection,
    add_chunks_to_store,
    get_or_create_collection,
    query_collection,
)
from src.llm import chat

from config import KNOWLEDGE_BASE_PATH, RETRIEVAL_TOP_K


def build_index(force: bool = False):
    """构建（或跳过）向量索引。

    参数:
        force: True = 强制重建（用户指定 --rebuild 时）
               False = 如果已有索引则跳过（默认启动时）

    流程:
    1. 检查是否需要重建（force=False 且 collection 非空 → 跳过）
    2. 加载知识库 → parse_sections → RecursiveCharacterTextSplitter 分块
    3. 逐块生成 embedding 向量（BGE-large-zh-v1.5, 1024 维）
    4. 重建 ChromaDB collection + 批量写入

    为什么不支持增量更新（只重建变化的章节）：
    增量更新需要比较新旧文档的 diff，找到变化的具体章节→只删该章节旧向量→重新嵌入，
    实现复杂度高（需要 diff 算法 + 章节→chunk 的映射管理），
    对于 25 个 chunk 的小库，全量重建只需 5-10 秒，投入产出比太低。
    如果知识库增长到数千 chunk，再考虑增量更新。
    """
    if not force:
        existing = get_or_create_collection()
        if existing.count() > 0:
            print(f"[索引] 向量库已存在 ({existing.count()} 条记录)，跳过构建")
            print("[索引] 如需重建请运行: python main.py --rebuild")
            return

    print("[索引] 开始加载和分块知识库...")
    chunks = chunk_document(KNOWLEDGE_BASE_PATH)
    print(f"[索引] 共 {len(chunks)} 个文本块")

    if not chunks:
        print("[索引] 警告: 知识库为空，未生成任何文本块。")
        print("[索引] 可能原因: 知识库文件格式不匹配（检查章节标题是否符合 关键词：内容 格式）")
        return

    # 提取待嵌入的文本列表
    texts = [c["content"] for c in chunks]

    print("[索引] 生成 embedding 向量...")
    embeddings = embed_texts(texts)
    print(f"[索引] 向量维度: {embeddings.shape[1]}, 数量: {len(embeddings)}")

    print("[索引] 写入向量数据库...")
    # 先删旧再写新，保证幂等性（force=True 时多次运行结果一致）
    collection = rebuild_collection()
    # embeddings 是 numpy float32 数组，ChromaDB 需要 Python list[list[float]]
    # tolist() 做转换，对于 25×1024 的数组零开销（几毫秒）
    add_chunks_to_store(collection, chunks, embeddings.tolist())

    print(f"[索引] 完成! 已将 {len(chunks)} 条记录写入向量库。")


def query(question: str, top_k: int = None) -> dict:
    """执行 RAG 问答：检索 + 增强 + 生成。

    参数:
        question: 用户输入的自然语言问题
        top_k: 检索返回的文档块数，默认从 config 读取

    返回:
        {
            "question": "原始问题",
            "answer": "LLM 生成的回答",
            "sources": [
                {"content": "块文本", "source": "校园美食", "distance": 0.0234},
                ...
            ],
        }

    为什么返回 dict 而非纯字符串：
    用户需要看到检索来源才能判断回答可信度。
    如果 LLM 回答有偏差，用户可以翻 sources 追溯原始资料自行判断，
    而不是盲信模型的输出。这是 RAG 相比纯 LLM 的核心优势——可验证性。

    如果向量库为空怎么办：
    返回友好的"请先重建索引"提示而非抛异常，避免用户看到 traceback 而困惑。
    """
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    # 1. 检查向量库状态
    collection = get_or_create_collection()
    if collection.count() == 0:
        return {
            "question": question,
            "answer": "向量库为空，请先运行 python main.py --rebuild 构建索引",
            "sources": [],
        }

    # 2. 问题向量化 + 相似度检索
    # embed_texts 接受列表，即使只有一条查询也要包装成列表传入
    print(f"[检索] 正在检索与问题相关的资料 (top-{top_k})...")
    q_embedding = embed_texts([question])
    # q_embedding[0] 取第一条（也是唯一一条）的向量
    results = query_collection(collection, q_embedding[0].tolist(), n_results=top_k)

    # 3. 整理检索结果
    # ChromaDB 的 query 返回结果有多层列表包裹（为批量查询设计），
    # results["documents"] 是 [[doc1, doc2, ...]]，取 [0] 得到内层列表
    sources = []
    context_chunks = []
    for i in range(len(results["documents"][0])):
        sources.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "未知"),
            "distance": round(results["distances"][0][i], 4),
        })
        context_chunks.append(results["documents"][0][i])

    # 4. 调用 LLM 生成回答
    print("[LLM] 正在生成回答...")
    answer = chat(question, context_chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
