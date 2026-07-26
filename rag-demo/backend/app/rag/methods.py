from .models import MethodOption


METHODS = [
    MethodOption(
        id="structure", name="标题 / 段落结构切分", category="chunking", status="available",
        description="依据空行与标题前缀切分，保留章节元数据。",
        advantages=["语义边界较完整", "结果直观", "适合结构清晰的小文档"],
        limitations=["依赖原文结构质量", "超长段落仍需二次切分"],
    ),
    MethodOption(
        id="fixed_length", name="固定长度切分", category="chunking", status="available",
        description="按字符窗口和 overlap 切分，用作结构切分的对照组。",
        advantages=["简单稳定", "参数容易控制"],
        limitations=["可能截断完整语义", "边界不理解标题结构"],
    ),
    MethodOption(
        id="semantic", name="TF-IDF 语义凝聚度切分", category="chunking", status="available",
        description="将段落表示为 TF-IDF 向量，按相邻余弦相似度下降和最大长度共同确定边界。",
        advantages=["边界分数可解释", "可合并相邻同主题段落", "无需下载额外模型"],
        limitations=["依赖词项重合，不等同于 Embedding 深层语义", "阈值需要结合语料调试"],
    ),
    MethodOption(
        id="tfidf", name="TF-IDF 稀疏向量检索", category="retrieval", status="available",
        description="将词及词组映射为稀疏向量，再按余弦相似度召回。",
        advantages=["本地可复现", "无需模型下载", "词权重容易解释"],
        limitations=["不理解深层语义", "近义表达召回能力有限"],
    ),
    MethodOption(
        id="bm25", name="BM25 关键词检索", category="retrieval", status="available",
        description="按词频、逆文档频率和文档长度计算关键词相关度。",
        advantages=["精确词匹配强", "分数来源容易解释", "无需模型下载"],
        limitations=["不理解近义表达", "分词质量会影响效果"],
    ),
    MethodOption(
        id="dense_embedding", name="Embedding 稠密向量检索", category="retrieval", status="planned",
        description="接入中文 embedding 模型，对比真正的语义检索。",
        advantages=["可召回语义相近表达"], limitations=["需要额外模型与推理资源"],
    ),
    MethodOption(
        id="hybrid", name="TF-IDF + BM25 混合检索", category="retrieval", status="available",
        description="归一化并融合 TF-IDF 余弦分数与 BM25 关键词分数。",
        advantages=["兼顾向量空间与关键词统计", "可观察分数融合"], limitations=["仍不具备稠密语义能力", "融合权重需要调参"],
    ),
    MethodOption(
        id="none", name="不重排", category="reranking", status="available",
        description="直接采用初次检索的相似度顺序。",
        advantages=["速度快", "便于观察原始召回"], limitations=["候选片段的精排能力有限"],
    ),
    MethodOption(
        id="term_coverage", name="查询词覆盖率重排", category="reranking", status="available",
        description="先扩大候选召回，再按原始分数与查询词覆盖率重新排序。",
        advantages=["无需额外模型", "排序过程透明", "可用于理解两阶段检索"],
        limitations=["仍依赖字面词匹配", "不能替代 Cross-Encoder"],
    ),
    MethodOption(
        id="cross_encoder", name="Cross-Encoder 重排", category="reranking", status="planned",
        description="逐对判断问题与候选片段的相关性。",
        advantages=["排序准确度通常更高"], limitations=["增加模型调用和延迟"],
    ),
    MethodOption(
        id="extractive", name="结构化教学抽取回答", category="generation", status="available",
        description="识别枚举、流程和事实意图，将证据句整理为可追踪的答案要点。",
        advantages=["结果可复现", "每个要点有引用", "可观察选句依据", "无调用费用"],
        limitations=["语言组织能力有限", "不等同于真实 LLM 生成"],
    ),
    MethodOption(
        id="longcat", name="LongCat-2.0 证据生成", category="generation", status="available",
        description="通过 OpenAI 兼容接口将检索上下文交给 LongCat-2.0 综合回答。",
        advantages=["回答自然", "可综合多段资料", "支持列表与步骤组织"], limitations=["依赖外部服务", "存在延迟与调用成本", "仍需校验引用与幻觉"],
    ),
]
