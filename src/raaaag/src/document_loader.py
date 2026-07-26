"""
文档加载与分块。
将知识库文本按语义边界切分为可嵌入的向量块，
每个块携带来源章节元数据，方便检索时溯源。

分两步：
1. 章节切分（parse_sections）：按行首关键词+冒号识别章节边界
2. 语义分块（chunk_document）：在章节内按段落/句子边界切小块
"""

import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_markdown(filepath: str) -> str:
    """读取 markdown 文件全部内容。

    为什么用 utf-8-sig 而非 utf-8：
    Windows 记事本保存的 UTF-8 文件会在开头插入 BOM（Byte Order Mark，\ufeff），
    用 utf-8 读取时 BOM 字符会混入第一行内容，导致：
    - 第一行的正则匹配失败（因为行首多了个不可见字符）
    - 第一个 chunk 的内容以乱码开头
    utf-8-sig 会自动剥离 BOM，跨平台兼容性更好。
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"知识库文件不存在: {filepath}")
    return path.read_text(encoding="utf-8-sig")


def parse_sections(text: str) -> list[dict]:
    """识别知识库中的章节边界，拆分为带标题的段落。

    知识库的实际格式（标题和内容在同一行，用冒号分隔）：
        太原理工大学创新学社秉承"敢立高标..."（开头简介，无标题，归入"学社简介"）
        报到与宿舍生活：开学报到前，要确认录取通知书上的时间和地点...
        校园美食：北餐有葛辉饺子、馄饨和自选菜...
        学习方法与考试/复习策略：课堂是获取知识和加深老师印象的重要场所...
        ...

    返回: [
        {"title": "学社简介", "content": "太原理工大学创新学社秉承..."},
        {"title": "校园美食", "content": "北餐有葛辉饺子..."},
        ...
    ]

    为什么不用 markdown 标题（## 语法）：
    知识库是纯文本段落，没有 markdown 语法。之前尝试用 ## 匹配导致
    0 个章节（所有内容被丢弃），因为知识库里根本没有任何 # 号。

    为什么用 \"行首 2-10 字符 + 冒号\" 这种 heuristic 而非更复杂的 NLP 方法：
    - 知识库格式高度规整，每个章节都是"关键词：正文"模式
    - 2-10 字符的范围覆盖了从"必备物品"（4字）到"学习方法与考试/复习策略"（10字）的所有标题
    - 限制 2-10 避免误匹配正文中偶然出现的冒号（如"时间：早上8点"这种不会出现在行首）
    - 如果用 NLP 分句/分段，反而可能把"勋章机制:创新学社通过..."误拆成两个段落

    为什么标题和内容在同一行也要正确拆分：
    初版代码假设标题行短（<=50 字符），内容在后续行。但知识库中标题和内容
    挤在同一行，整行数百字符，<=50 的过滤把真正标题全过滤掉了，
    导致全部内容归入"学社简介"一个章节（3812 字的一坨）。
    """
    lines = text.split("\n")
    sections = []
    # 第一个标题出现之前的所有内容归入"学社简介"。
    # 这是因为知识库开头是学社的整体介绍，没有明确的章节标题。
    current_title = "学社简介"
    current_lines = []

    # 匹配行首 2-10 个中文字符/字母/数字后紧跟冒号，并将冒号前后的文本分别捕获
    # group(1) = 标题部分（冒号前），group(2) = 内容部分（冒号后）
    # \\w 包含字母数字下划线，覆盖了"TYUT""NLP"等英文缩写标题的可能
    title_pattern = re.compile(r"^([\u4e00-\u9fa5\w]{2,10})[：:](.+)$")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue  # 空行跳过，不作为标题也不作为内容

        m = title_pattern.match(stripped)
        if m:
            # 发现新章节标题：先保存上一节，再开始新节
            if current_lines:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                })
            # 冒号前 = 标题，冒号后 = 该节第一行内容
            # 不丢弃冒号后的内容——它本身就是该节正文的一部分
            current_title = m.group(1).strip()
            current_lines = [m.group(2).strip()]
        else:
            # 非标题行：追加到当前章节的内容中
            current_lines.append(line)

    # 最后一节（不会再有下一个标题来触发保存，必须手动收尾）
    if current_lines:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines).strip(),
        })

    return sections


def chunk_document(filepath: str) -> list[dict]:
    """加载文档并按语义边界分块。

    返回: [
        {"content": "块文本...", "metadata": {"source": "校园美食", "chunk_index": 0}},
        {"content": "块文本...", "metadata": {"source": "校园美食", "chunk_index": 1}},
        ...
    ]

    分块策略（两步走）：
    第一步：parse_sections() 按行首关键词+冒号识别章节
        → 保证"校园美食"的 chunk 不会混入"报到与宿舍生活"的内容
        → 检索"食堂有什么"时不会因为某个词碰巧出现而召回无关章节
    第二步：RecursiveCharacterTextSplitter 在章节内切小块
        → 优先在段落边界（\\n\\n）断开，失败则试行边界（\\n），
          再失败则试中文标点（。！？），最差情况硬切单个字符
        → 保证了中文语义的完整性，不会被切成"北餐有葛辉饺"这种碎片

    为什么用 RecursiveCharacterTextSplitter 而非自己写 split：
    自己写 split 容易在以下边界情况出错：
    - 中英文混排（"需要带iPad、MacBook等设备"在"、"处断开会破坏语义）
    - 列表格式（用"- "开头的内容，按句号切会切成碎片）
    - 数字+单位（"90×190cm"中"。"不会被误认为句子结束）
    langchain 的实现已经在大量项目中被验证，处理了这些 corner case。
    """
    text = load_markdown(filepath)
    sections = parse_sections(text)

    # 分隔符按优先级排列：越靠前越优先使用
    # "\\n\\n" 最优先：段落边界是自然的语义断点
    # "" 兜底：前面所有分隔符都匹配不到时，只能按字符硬切（极端情况）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,  # 用字符数而非 token 数计量，中文场景更直观
    )

    chunks = []
    for section in sections:
        section_chunks = splitter.split_text(section["content"])
        for i, chunk_text in enumerate(section_chunks):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": section["title"],   # 章节名，检索时可溯源
                    "chunk_index": i,             # 章节内序号，同一章节多个 chunk 时区分
                },
            })

    return chunks
