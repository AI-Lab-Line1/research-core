"""
Embedding 模块：将文本转为稠密向量。
从 ModelScope Hub 下载中文 BGE 嵌入模型，用 sentence-transformers 加载推理。

为什么不用 modelscope pipeline 做 embedding：
初版代码用 modelscope.pipelines.pipeline(Tasks.sentence_embedding, ...)，
但它依赖的 transformers 版本与 BertConfig 的 is_decoder 属性不兼容，
报错 'BertConfig' object has no attribute 'is_decoder'。
modelscope pipeline 对底层 transformers 版本高度敏感，换个环境就炸。

改用方案：modelscope 只负责下载模型文件（snapshot_download），
sentence-transformers 负责加载和推理。两者分工明确，互不干扰。
sentence-transformers 直接加载模型目录中的 config.json + pytorch_model.bin，
不需要 modelscope pipeline 那层脆弱的封装。
"""

import os
import numpy as np

from config import MODELSCOPE_API_KEY, EMBEDDING_MODEL_ID, EMBEDDING_DEVICE


# 模块级模型缓存，避免重复加载。
# 加载一个 1.3GB 的模型需要 5-10 秒（含硬盘读取 + 内存分配），
# 如果每次 embed_texts() 都重新加载，交互式问答的每次查询要等 10 秒才出结果。
# 单例模式保证了只在首次调用时加载，后续调用直接复用。
_model = None


def _download_model() -> str:
    """从 ModelScope Hub 下载模型到本地缓存目录。

    返回模型文件的本地路径（如 C:\\Users\\...\\.cache\\modelscope\\...\\snapshots\\master）。

    snapshot_download 的行为：
    - 首次调用：从 ModelScope 服务器下载全部模型文件（config.json, pytorch_model.bin,
      tokenizer.json, ...），存入 ~/.cache/modelscope/，耗时取决于网速
    - 后续调用：检测本地缓存是否完整，完整则直接返回路径，零网络开销
    - 模型更新：如果远程有新版本，自动增量下载变更文件

    为什么不直接用 sentence-transformers 的 "modelscope/模型ID" 前缀格式：
    sentence-transformers 2.7+ 声称支持但实际测试不稳定（报 Path not found），
    直接下载到本地再加载是最稳妥的方式。
    """
    os.environ["MODELSCOPE_API_TOKEN"] = MODELSCOPE_API_KEY
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[Embedding] 下载模型: {EMBEDDING_MODEL_ID} ...")
    local_path = snapshot_download(EMBEDDING_MODEL_ID)
    print(f"[Embedding] 模型路径: {local_path}")
    return local_path


def get_embedding_model():
    """获取（或延迟加载）sentence-transformers 模型实例。

    首次调用：
    1. 从 ModelScope 下载模型（或验证本地缓存）
    2. 用 sentence-transformers 加载到指定设备（CPU）
    3. 缓存到模块全局变量

    后续调用：直接返回缓存的模型实例，毫秒级。
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        local_path = _download_model()
        print(f"[Embedding] 加载模型到 {EMBEDDING_DEVICE} ...")
        _model = SentenceTransformer(local_path, device=EMBEDDING_DEVICE)
        print("[Embedding] 模型加载完成")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """将文本列表转为归一化向量矩阵。

    参数:
        texts: 文本字符串列表，长度任意（包括单条查询和多条批量建库）

    返回:
        numpy 数组, dtype=float32, shape=(len(texts), 1024)
        BGE-large 的向量维度是 1024。

    为什么 normalize_embeddings=True：
    - 归一化后向量模长为 1，此时内积 = 余弦相似度
    - ChromaDB 默认用 L2 距离，L2 在归一化向量上等价于余弦距离
    - 如果不归一化，不同长度的文本会产生不同模长的向量，
      单靠 L2 距离无法准确反映语义相似度（长文本天然距离更大）

    为什么 show_progress_bar 只在超过 10 条时显示：
    - 单条查询（交互式问答）不需要进度条，反而干扰输出
    - 批量建索引（25 条）显示进度条让用户知道没有卡死
    """
    if not texts:
        # 返回空数组而非 None：上游代码可能直接对返回值做 shape 操作，
        # None 会导致 AttributeError，空数组则安全通过
        return np.array([], dtype=np.float32)

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10,
    )
    # 显式转为 float32：sentence-transformers 在 GPU 上可能返回 float16，
    # ChromaDB 接受 float32，float16 可能导致精度损失或类型错误
    return embeddings.astype(np.float32)
