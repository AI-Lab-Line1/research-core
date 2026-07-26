"""
RAG 系统全局配置。
所有可变参数集中在一处，避免散落在各模块中导致改漏。
修改配置后需执行 `python main.py --rebuild` 重建索引。
"""

import os

# ============================================================
# 路径
# ============================================================
# 项目根目录 —— 用 __file__ 的绝对路径而非相对路径，
# 因为在不同目录下运行 main.py 时，相对路径会指向运行目录而非项目目录，
# 导致找不到知识库文件。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 知识库文件路径。
# 为什么用 os.path.join 而非硬编码字符串：
# Windows 和 Linux 的路径分隔符不同（\ vs /），os.path.join 自动适配。
# 如果知识库不存在，build_index 阶段会直接抛 FileNotFoundError，不会静默建空库，
# 这避免了"以为建好了但实际是空的"的隐蔽 bug。
KNOWLEDGE_BASE_PATH = os.path.join(PROJECT_ROOT, "知识库.md")

# ChromaDB 向量数据库持久化目录。
# 为什么选择 ChromaDB 的 PersistentClient 而非 EphemeralClient：
# EphemeralClient 数据只在内存中，进程退出即丢失，每次启动都要重建索引，
# 对于 25 个 chunk 的小库虽然只要几秒，但 embedding 模型加载就有 5-15 秒开销，
# 持久化后 `python main.py` 启动几乎是瞬时的。
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")

# ============================================================
# ModelScope 认证
# ============================================================
# API Key 用于：
# 1. 从 ModelScope Hub 下载 embedding 模型（不设限流更严）
# 2. 调用 LLM 推理 API（必须，否则 401）
# 这个 Key 是用户专属的，不要提交到 git。
MODELSCOPE_API_KEY = "ms-1858aa66-bc09-4c0a-8d30-5ae1ba4f6015"

# ============================================================
# 文档分块
# ============================================================
# 每块最大字符数（中文字符，一个字 = 1 字符）。
# 为什么是 400 而非更常见的 512/1024：
# - 中文一个字符的信息密度远高于英文一个 token（英文平均 4 字符/token，中文约 1.5），
#   所以中文 RAG 的 chunk 应该比英文更短
# - 400 字符约等于 250-300 个 token（LLM 上下文的计量单位），留够余量给 5 个 chunk
# - 如果设到 1024，5 个 chunk = 3000+ token，加上 system prompt 很容易超出一些
#   免费 API 的上下文窗口，导致回答被截断
# - 如果设到 200，一个"校园美食"章节（442 字）会被切成 3 块，
#   检索时可能只召到其中一块，丢失同一话题的其他信息
CHUNK_SIZE = 400

# 相邻块重叠字符数。
# 为什么需要重叠：
# 分块时可能在关键信息中间切断（如"报到需要带：身份证、录取通知书、..."
# 恰好被切在两块里）。overlap=80 保证跨边界的信息在相邻块中重复出现，
# 检索时至少有一块包含完整信息。
# 为什么是 80 而非 50 或 100：
# - 50：对于中文长句（如"把录取通知书、身份证及复印件、户口本复印件、党团关系证明、
#   若干证件照等都放在防水文件袋里"约 50 字），可能刚好跨边界，不够安全
# - 100：overlap 过大意味着更多重复内容占据 ChromaDB 空间和 LLM 上下文，
#   chunk_size=400 时 20% 的重叠率已经足够
# - 80 约为 chunk_size 的 20%，是 RAG 领域的经验最佳实践
CHUNK_OVERLAP = 80

# ============================================================
# Embedding 模型
# ============================================================
# 为什么选 BAAI/bge-large-zh-v1.5：
# - 它是 BGE（BAAI General Embedding）系列的中文版本，在 C-MTEB 中文嵌入基准上
#   排名靠前，专门为语义检索（retrieval）任务训练
# - 实测对比：iic/nlp_corom_sentence-embedding_chinese-base 在"创新学社的精神"
#   vs "敢立高标擎日月"的语义匹配上完全失败（top-10 检索都召不回"学社简介"那个 chunk），
#   BGE 在此类中文语义匹配上显著优于 Corom
# - 代价：模型约 1.3GB（Corom 约 400MB），首次下载需等待，但下载一次后缓存到
#   ~/.cache/modelscope/，后续无需重复下载
# - 如果磁盘空间紧张或网络慢，可切回 iic/nlp_corom_sentence-embedding_chinese-base，
#   但检索质量会下降
EMBEDDING_MODEL_ID = "BAAI/bge-large-zh-v1.5"

# embedding 推理设备。
# 为什么用 "cpu" 而非 "cuda"：
# - embedding 模型（~1.3GB）在 CPU 上推理 25 个 chunk 只需 1-2 秒，完全够用
# - 放在 CPU 上可以把 8GB 显存全部留给可能本地运行的 LLM
# - 如果将来知识库扩大到数千 chunk，改为 "cuda" 会有明显加速
EMBEDDING_DEVICE = "cpu"

# ============================================================
# LLM 生成模型（通过 ModelScope API 调用）
# ============================================================
# ModelScope 推理 API endpoint，兼容 OpenAI chat/completions 协议。
# 这意味着可以像调用 OpenAI 一样调用 ModelScope 上部署的任何模型。
LLM_BASE_URL = "https://api-inference.modelscope.cn/v1"

# 对话模型 ID —— 必须是 API 实际支持的模型。
# 切换模型只需改这一行，其余代码不变。
#
# ========== 实测可用的模型（2025-07-22） ==========
# ✓ Qwen/Qwen3.5-27B            通义千问 3.5 27B，中文顶级 ← 当前选择
# ✓ deepseek-ai/DeepSeek-V3.2   DeepSeek V3.2，综合能力极强
#
# ========== 以下返回空结果（API 层面问题，非代码 bug） ==========
# ✗ Qwen/Qwen3-8B               HTTP 200 但 choices 始终为空
# ✗ Qwen/Qwen3-4B               HTTP 400
# ✗ ZhipuAI/GLM-4.7-Flash      HTTP 200 但 choices 始终为空
#
# 为什么选 Qwen3.5-27B 而非 DeepSeek-V3.2：
# Qwen 系列对中文指令遵循更好，RAG 场景下"只根据资料回答"的约束更可靠。
# DeepSeek 有时会忽略 system prompt 中的"只根据提供的资料"限制，用自身知识补充。
LLM_MODEL_ID = "Qwen/Qwen3.5-27B"

# temperature 控制生成随机性，范围 [0, 2]。
# 为什么是 0.3 而非 0：
# - 0 意味着每次对相同输入生成完全相同的输出（确定性解码），
#   但 RAG 上下文每次不同，严格确定性的措辞反而显得机械
# - 0.3 给措辞留一点灵活性（如"根据资料显示..."vs"资料中提到..."），
#   但不会偏离事实——对 RAG 这种事实性场景足够安全
# - 如果回答出现"幻觉"（编造信息），降到 0 试试
LLM_TEMPERATURE = 0.3

# 最大生成 token 数。
# 为什么是 1024 而非更大：
# - RAG 回答通常 100-300 字就够（校园美食、报到流程等问题不需要长篇大论）
# - 设太大浪费 API 配额（按 token 计费时），也对用户没价值
# - 如果回答被截断，可以临时加大到 2048
LLM_MAX_TOKENS = 1024

# ============================================================
# 检索参数
# ============================================================
# 检索返回的 Top-K 文档块数。
# 为什么是 5 而非 3 或 10：
# - 3：可能遗漏关键信息，特别是当问题涉及多个章节时（如"报到要带什么+宿舍注意什么"）
# - 10：每个 chunk 约 400 字，10 个 = 4000 字 ≈ 2500+ token，
#   加上 system prompt 和 LLM_MAX_TOKENS，总消耗可能超出 API 的免费配额，
#   且过多的无关 chunk 会稀释相关信息，LLM 可能在噪声中迷失
# - 5：在覆盖率和噪声之间取得平衡，5*400=2000 字对 LLM 上下文压力适中
RETRIEVAL_TOP_K = 5

# ChromaDB collection 名称。
# 为什么用英文命名：
# ChromaDB 内部以 collection name 作为目录名的一部分，中文路径在某些
# Windows 版本上可能导致编码问题（特别是非 UTF-8 locale 环境）。
COLLECTION_NAME = "tyut_knowledge"
