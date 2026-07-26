"""
LLM 对话模块：通过 ModelScope API 调用大语言模型生成回答。
使用 OpenAI 兼容协议，ModelScope 推理 API 完全兼容 openai Python SDK。

为什么用 ModelScope API 而非本地运行大模型：
- Qwen3.5-27B 需要约 54GB 显存（FP16），远超 RTX 5060 的 8GB
- 量化到 4-bit 仍需约 15GB，放不下
- API 调用零显存占用，延迟 2-5 秒，适合交互式问答
- 代价：依赖网络，可能有速率限制
"""

from openai import OpenAI

from config import MODELSCOPE_API_KEY, LLM_BASE_URL, LLM_MODEL_ID, LLM_TEMPERATURE, LLM_MAX_TOKENS


# 模块级客户端复用。
# OpenAI 客户端内部维护 TCP 连接池，重复创建会：
# 1. 每次重新 TCP 握手（TLS 协商 ~200ms）
# 2. 浪费服务端连接资源（可能触发限流）
# 单例模式复用连接池，后续调用省掉握手开销。
_client = None


def _get_client() -> OpenAI:
    """获取（或延迟创建）OpenAI 客户端实例。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=MODELSCOPE_API_KEY,
            base_url=LLM_BASE_URL,
        )
    return _client


def build_prompt(query: str, context_chunks: list[str]) -> tuple[str, str]:
    """将用户问题和检索到的上下文组装为 LLM prompt。

    参数:
        query: 用户的原始问题，如"食堂有什么好吃的"
        context_chunks: 检索返回的 top-K 文档块内容列表

    返回: (system_prompt, user_message) 两个字符串

    为什么用 system + user 双消息结构：
    - ModelScope API 兼容 OpenAI /v1/chat/completions 协议，要求 messages 数组
    - system 消息设定角色和行为约束（"你是学社助手"），不会被用户的追问覆盖
    - 如果全塞在一条 user 消息里，多轮对话时 system 指令会逐渐被挤出上下文窗口

    为什么在 system prompt 中强调"只根据提供的资料回答"：
    这是 RAG 的核心契约——没有这个约束，LLM 会用训练数据中的知识编造答案。
    举例：如果问"学校食堂有什么"，检索失败召回了风马牛不相及的内容，
    LLM 可能凭训练数据中的通用知识回答"一般学校食堂有盖浇饭、麻辣烫..."，
    用户无法区分这是真实信息还是模型编的。
    加了约束后，检索失败 → LLM 回答"资料中未找到" → 用户知道是检索的问题，
    可以换种问法或检查知识库。

    为什么 context_text 前面不加"以下是参考资料"之类的引导词：
    直接在 user message 里放参考资料 → LLM 天然把它当成回答的依据。
    加引导词反而可能让模型困惑（"以下是参考"和"用户问题"都是 user 说的，
    模型可能不理解哪个是问题哪个是参考）。
    """
    # 组装上下文：每个 chunk 前标注序号，方便 LLM 区分不同来源
    context_text = "\n\n---\n\n".join(
        f"[来源: {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    system_prompt = (
        "你是太原理工大学创新学社的智能助手，负责回答关于学社的各项问题。\n\n"
        "回答规则：\n"
        "1. 只根据下面提供的资料内容回答，不要使用你自己的知识\n"
        '2. 如果提供的资料中没有相关信息，直接说"资料中未找到相关信息"，不要猜测\n'
        "3. 回答要准确、简洁，优先引用资料中的原文表述\n"
        "4. 如果问题涉及多个方面，逐条回答但不要编号过多"
    )

    user_message = (
        f"参考资料：\n{context_text}\n\n"
        f"用户问题：{query}\n\n"
        f"请根据以上参考资料回答问题。"
    )

    return system_prompt, user_message


def chat(query: str, context_chunks: list[str]) -> str:
    """调用 LLM 生成回答。

    参数:
        query: 用户问题，如"有哪些美食推荐"
        context_chunks: 检索到的相关文档块列表，按相似度降序排列

    返回:
        生成的回答文本，或错误描述字符串（不抛异常，上层直接展示给用户）

    为什么 temperature 设为 LLM_TEMPERATURE (0.3) 而非更常见的 0.7：
    - 0.7 是创意写作的默认值，RAG 场景下可能导致"创造性发挥"→ 偏离资料
    - 0.3 在保持措辞灵活性的同时抑制了模型"自由发挥"的冲动
    - 极端对比：0 = 贪婪解码，回答干瘪但绝不出错；1.0 = 创造力拉满但可能胡编

    为什么错误处理返回字符串而非 raise：
    RAG pipeline 调用 chat() 后直接展示给用户。如果抛异常，
    pipeline 层需要 try/except 并自行构造错误消息，重复代码。
    直接在 chat() 里处理并返回友好提示，pipeline 层一行搞定。
    """
    system_prompt, user_message = build_prompt(query, context_chunks)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        # response.choices 是 OpenAI SDK 的类型化对象，正常情况下 choices[0].message.content 存在
        # 但 ModelScope API 实测发现某些模型返回 choices=None（如 Qwen3-8B），
        # 此时 response.choices[0] 会抛 TypeError: 'NoneType' object is not subscriptable
        # 这个异常被外层 except 捕获并返回友好提示
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        # 按 HTTP 状态码/异常信息分类，给出针对性提示而非笼统的"调用失败"
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "错误：API Key 无效或已过期，请检查 config.py 中的 MODELSCOPE_API_KEY"
        if "429" in error_msg:
            return "错误：API 调用频率超限，请稍后重试"
        if "404" in error_msg:
            return f"错误：模型 {LLM_MODEL_ID} 未找到，请确认模型 ID 是否正确并在 API 可用列表中"
        if "NoneType" in error_msg or "not subscriptable" in error_msg:
            return (
                f"错误：模型 {LLM_MODEL_ID} 返回空结果，该模型可能不支持 chat/completions 协议。\n"
                f"请换成已确认可用的模型，如 Qwen/Qwen3.5-27B 或 deepseek-ai/DeepSeek-V3.2"
            )
        return f"LLM 调用失败: {error_msg}"
