"""
================================================================================
01_generate_data.py — 教师模型生成蒸馏训练数据
================================================================================

【问题背景】
    知识蒸馏需要 Teacher 先产出高质量回答，Student 再从这些回答里学。
    但 Qwen2.5-7B-Instruct 在面对"三角度解释"这种结构化 prompt 时，
    有三个鸡贼行为：

    1. Prompt 泄漏
       模型有时把用户指令当成回答的一部分复述出来，比如输出里混入
       "请从三个完全不同的角度解释【公平】这个概念" 这种指令文本。
       根本原因：裸文本拼接喂给 ChatML 训练出来的 instruct 模型，
       模型分不清哪里是"指令区"哪里是"回答区"，边界模糊。

    2. 角度一重复
       三个角度写完之后，模型"意犹未尽"又把角度一的内容重新说一遍。
       就像写完作文发现字数不够又抄一遍开头。

    3. 输出格式冗余
       老版本的输出是 "【角度一】用一个生活比喻解释..."，
       连格式描述一起输出。学生模型会学到的是"格式描述"而非"内容结构"。
       我们只想要干净的三段内容，让学生从内容本身感悟内在逻辑。

【解决方案】
    1. ChatML 格式：用 tokenizer.apply_chat_template() 生成完整 ChatML，
       system 消息强调格式纪律，边界一清二楚。
    2. Token ID 精确截取：new_tokens = out[0][inputs.input_ids.shape[1]:]，
       不靠字符串对齐，彻底杜绝 prompt 边界切割错误。
    3. 清洗管道：多层防护 —— 控制字符 → Markdown → 小一锚定 → 重复截断 →
       废话前缀清除 → 指令泄漏匹配。
    4. 输出标记简化：prompt 里保留三段内容要求（比喻/学术/排除），
       但输出格式只写 "一/二/三"，不附带描述文字。

【数据格式】
    {
        "自由": "一：...\n\n二：...\n\n三：...",
        "递归": "一：...\n\n二：...\n\n三：...",
        ...
    }
    200 条，保存到 data/teacher_train.json

【运行方式】
    conda activate pytorch_env
    python 01_generate_data.py
================================================================================
"""

import os
# ── bitsandbytes 版本兼容 ──
# Windows + PyTorch 13.2 环境下，bitsandbytes 的 CUDA 版本检测逻辑只能识别到 13.0，
# 不设这个环境变量会报 "CUDA version 13.2 not supported" 然后 fallback 到 CPU。
# 必须在 import transformers 之前设置，因为 bitsandbytes 在导入时读取这个值。
os.environ.setdefault("BNB_CUDA_VERSION", "130")

import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ════════════════════════════════════════════════════════════════
# 路径与数据配置
# ════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
TEACHER_MODEL = str(BASE_DIR / "models" / "Qwen2.5-7B-Instruct")
OUTPUT_FILE = str(BASE_DIR / "data" / "teacher_train.json")

# ── 200 个抽象概念 ──
# 选择标准：每个概念都有多义性，能从不同维度解释。
# 覆盖哲学（自由、因果）、数学（递归、熵）、社科（公正、偏见）、
# AI（推理、注意力）、日常生活（习惯、耐心）等领域。
# 同义/近义概念（如"推理"出现两次）刻意保留 —— 语义相同但语境不同时，
# Teacher 可能给出不同侧重点的回答，增加数据多样性。
CONCEPTS = [
    "自由", "递归", "熵", "时间", "因果", "涌现", "美", "随机", "语言", "智能",
    "公平", "习惯", "意义", "对称", "混乱", "秩序", "真理", "谎言", "信任", "恐惧",
    "希望", "记忆", "梦想", "意识", "命运", "正义", "权力", "责任", "勇气", "耐心",
    "孤独", "友谊", "爱", "恨", "嫉妒", "同理心", "尊严", "幽默", "创造力", "好奇心",
    "偏见", "宽容", "牺牲", "妥协", "坚持", "悖论", "无限", "虚无", "模仿", "平衡",
    "变化", "结构", "过程", "系统", "边界", "联系", "冲突", "和谐", "稳定", "动态",
    "选择", "决策", "风险", "机遇", "成长", "学习", "知识", "经验", "方法", "规律",
    "观察", "分析", "推理", "归纳", "演绎", "抽象", "具体", "比较", "分类", "评估",
    "优化", "效率", "资源", "成本", "价值", "目标", "动机", "意志", "欲望", "情绪",
    "认知", "感知", "注意", "想象", "理解", "表达", "交流", "沟通", "反馈", "协作",
    "竞争", "合作", "规则", "制度", "规范", "自由意志", "自我", "他者", "身份", "归属",
    "传统", "现代", "未来", "过去", "现在", "空间", "维度", "尺度", "方向", "距离",
    "速度", "力量", "能量", "物质", "信息", "数据", "代码", "算法", "模型", "训练",
    "推理", "预测", "分类", "识别", "生成", "翻译", "总结", "问答", "搜索", "检索",
    "隐私", "安全", "伦理", "责任", "正能量", "负能量", "平衡", "均衡", "差异", "相似",
    "相反", "互补", "对立", "依赖", "独立", "关联", "因果链", "条件", "假设", "证明",
    "复杂", "简单", "精确", "模糊", "清晰", "混乱", "秩序", "节奏", "节拍", "节省",
    "探索", "发现", "创新", "突破", "适应", "转变", "进化", "演化", "遗传", "环境",
    "适应性", "反馈机制", "控制", "调控", "调节", "失衡", "平衡点", "极限", "边际", "阈值",
    "隐喻", "比喻", "类比", "映射", "重构", "归因", "解释", "理解力", "记忆力", "注意力"
][:200]  # 只取前 200 个；列表留 220 个方便增减不破索引


# ════════════════════════════════════════════════════════════════
# Prompt 构建
# ════════════════════════════════════════════════════════════════

# ── System Prompt ──
# ChatML 的 system 角色在每次对话开头注入，不占 token 生成配额。
# 这里强调三件事：
#   1. 严谨 —— 禁止自由发挥
#   2. 格式纪律 —— 必须按用户指定的格式输出
#   3. 从第一个标记到最后一个标记，不要废话
# Qwen2.5 对 system prompt 的遵循度很高，这段是防止 prompt 泄漏的第一道防线。
SYSTEM_PROMPT = (
    "你是一个严谨的AI。你的回答必须严格遵循用户指定的格式。"
    "不要输出任何开场白、总结、复述指令、或任何额外文字。"
    "直接从第一个标记开始输出，到最后一个标记结束。"
)


def build_chatml_prompt(tokenizer, concept: str) -> str:
    """构建完整 ChatML 格式 prompt，末尾已包含 generation prompt。

    ── 为什么必须用 ChatML 而不是裸文本拼接？ ──
    Qwen2.5-7B-Instruct 的训练数据全部是 ChatML 格式：
        <|im_start|>system\n...<|im_end|>
        <|im_start|>user\n...<|im_end|>
        <|im_start|>assistant\n...<|im_end|>

    如果直接把裸文本 "请从三个角度解释..." 扔给模型，它没见过这种输入模式。
    模型的分界线是 <|im_start|>assistant，没这个标记它就不知道什么时候该开始回答，
    于是就"猜测"该从哪开始 —— 有时候猜对了、有时候把指令也输出一遍。
    用 apply_chat_template() 生成标准 ChatML 后，模型看到 <|im_start|>assistant\n
    就知道"现在轮到我说话了，从这里开始生成"，prompt 泄漏概率从 ~30% 降到接近 0。

    ── Prompt 设计哲学 ──
    用户消息分三层：
      1. 任务描述：写三句话，每句从不同、不重叠的角度解释
      2. 内容要求：一=生活比喻、二=学术定义、三=反向排除
         （这三段描述教 Teacher "怎么写"，但不会出现在输出格式里）
      3. 输出格式：只给三个空标记 "一：/二：/三："
         Teacher 看到空标记，知道该往里面填空，但不会把"生活比喻"这些描述也抄一遍

    学生模型将来看到的数据格式是：
        一：炒菜不放盐就像没有自由...
        二：自由是行动者在无外部强制下...
        三：自由不是为所欲为...

    学生从这三个句子的内在差异中自己领悟"比喻 vs 定义 vs 排除"的结构，
    而不是靠"【角度一】用一个生活比喻解释"这种显式标签。这才是真正的知识蒸馏。

    ── 参数说明 ──
    add_generation_prompt=True：
        在末尾追加 <|im_start|>assistant\n，告诉模型从这里开始生成。
        如果不加，模型会认为对话不完整，行为不可预测。
    """
    user_content = (
        f'对【{concept}】这个概念，写三句话，每句话从一个完全不同、互不重叠的角度解释它。\n'
        f'三句话之间不能有任何关联、递进、或呼应关系。\n\n'
        # ↓ 下面是"怎么写"的指导，不是输出格式的一部分
        f'内容要求：\n'
        f'一 — 用一个生活比喻解释【{concept}】\n'
        f'二 — 给出精确的学术定义或理论框架\n'
        f'三 — 用反向排除法说明【{concept}】不是什么\n\n'
        # ↓ 下面才是"输出应该长什么样"。只有空标记，没有描述
        f'输出格式（严格遵循，不要多说一个字）：\n\n'
        f'一：\n'
        f'二：\n'
        f'三：'
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,            # 返回字符串，交给 tokenizer 在生成时再编码
        add_generation_prompt=True # <|im_start|>assistant\n
    )


# ════════════════════════════════════════════════════════════════
# 清洗管道
# ════════════════════════════════════════════════════════════════

def clean_generated_answer(text: str) -> str:
    """多层清洗管道。按顺序依次过滤，每一层解决一类脏数据。

    ── 脏数据类型与对应处理 ──

    L1: 控制字符
        NULL(\\x00)、BEL(\\x07)、SO/SI(\\x0e/\\x0f) 等不可见字符混入。
        来源：tokenizer 解码时偶发。直接 strip。

    L2: Markdown 代码块
        Qwen2.5 有极低概率用 ```...``` 包裹输出。正则移除。

    L3: 小一锚定
        这是最关键的截断点。所有有效内容必须从第一个 "小一" 开始，
        之前的内容（无论是什么）全部丢弃。这一刀解决了 90% 的 prompt 泄漏。

    L4: 角度一重复截断
        模型写完小三后，可能重新从"小一"开始再说一遍。
        检测：在小三标记之后查找是否还有 "小一"，有就截断。
        为什么只截断小三级之后的内容而不是整段？因为前面三个角度是完整的，
        只是后面多了一截，切掉尾巴即可。

    L5: 废话前缀清除
        即使模型知道从"小一"开始，它也可能在"小一："后面加废话：
        "从生活比喻的角度来解释..." "第一个角度，我想从..." "让我从小一角度来说..."
        这些前缀用正则匹配 + 删除。

    L6: 指令泄漏模式匹配
        兜底：匹配常见的指令文本（"请从..."、"不要开场白"、"严格按以下格式"等），
        直接删除。这一层是安全网，正常情况下 ChatML 已经防止了泄漏。

    ── 参数 ──
    text: 模型生成的原始文本（已 strip 特殊 token）

    ── 返回 ──
    清洗后的干净文本，格式为：
        小一：[内容]\n\n小二：[内容]\n\n小三：[内容]
    """
    if not text:
        return ""

    # ═══ L1: 控制字符 + 换行规范化 ═══
    # range (0x00, 0x08) 是 NULL~BS，0x0b(垂直制表), 0x0c(换页), 0x0e-0x1f(各种控制)
    # 这些字符在 NLP 任务中无意义且会干扰后续正则匹配
    text = "".join(
        ch for ch in text
        if ch != "\ufffd"                                 # Unicode replacement char
        and not ("\u0000" <= ch <= "\u0008")              # C0 控制字符
        and not ("\u000e" <= ch <= "\u001f")              # 更多 C0 控制字符
        and ch not in "\u000b\u000c"                      # VT, FF
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # ═══ L2: Markdown 代码块移除 ═══
    # 模式：```...``` 或 ```python...```，(?is) 使 . 匹配换行 + 忽略大小写
    text = re.sub(r"(?is)```.*?```", "", text)

    # ═══ L3: 小一锚定 —— 从这里开始才是有效内容 ═══
    # 为什么要从"小一"开始截而不是从"小一："？
    # 因为有些输出格式可能是 "小一：" 或 "小一："（中英文冒号混用），
    # 用 "小一" 更鲁棒。后面 L5 会统一清理格式。
    m1 = re.search(r"小一[：:]", text)
    if not m1:
        # 连 "小一" 都没有，数据无意义，原样返回
        return text.strip()
    text = text[m1.start():]

    # ═══ L4: 角度一重复检测 ═══
    # 逻辑：
    #   1. 定位小三标记的位置
    #   2. 检查小三之后的内容中是否再次出现小一
    #   3. 如果出现，说明模型在重复，截断到第一次小三内容结束处
    # 注意：不直接用 index/rfind，因为正则更鲁棒（处理中英文冒号差异）
    m3 = re.search(r"小三[：:]", text)
    if m3:
        after_three = text[m3.end():]   # 小三标记之后的所有内容
        m1_again = re.search(r"小一[：:]", after_three)
        if m1_again:
            # 截断点 = 小三标记结束 + 小一再次出现的位置
            text = text[:m3.end() + m1_again.start()]

    # ═══ L5: 废话前缀清除 ═══
    # 场景：模型输出 "小一：从生活比喻的角度来解释，公平就像是..."
    # 我们需要的只是 "小一：公平就像是..."
    # 匹配模式分两类：
    #   A. "从XX角度..." 系列 —— 最常见
    #   B. "让我从..."、"我先从..."、"第一个角度..." 系列 —— 次常见
    #
    # 为什么用 re.sub 而不是 split+过滤？
    # 因为这些废话前缀可能出现在段落开头、也可能混在内容中间（虽然概率低）。
    # 正则替换比循环逐行处理更简洁且覆盖更多边界情况。
    text = re.sub(
        r'^(从[\u4e00-\u9fff]+角度[\u4e00-\u9fff]*(解释|说明|看|阐述|来看|来说|理解|拆解|分析)[：:，。,\.\s]*)+',
        '', text, flags=re.MULTILINE
    )
    text = re.sub(
        r'^(让我|我先|首先|第一个角度|第二个角度|第三个角度)[\u4e00-\u9fff，。,\.\s]*[：:，。,\.\s]*',
        '', text, flags=re.MULTILINE
    )

    # ═══ L6: 指令泄漏模式匹配（安全网） ═══
    # 这些是已知的 prompt 泄漏特征文本。如果前面 L1-L5 都没拦住，
    # 兜底用正则删除。每个模式匹配一种已知的泄漏形式。
    # 注意：这些匹配是全文搜索，不会漏掉藏在段落中间的泄漏。
    leak_patterns = [
        r'请从.{0,20}角度解释.{0,10}概念',   # "请从三个完全不同的角度解释公平这个概念"
        r'不要开场白',                         # "不要开场白、不要总结"
        r'不要总结',
        r'不要任何额外文字',
        r'严格按以下格式',
        r'输出格式',
        r'Human[：:]',                         # 裸文本模式的残留
        r'Assistant[：:]',                     # 同上
        r'你是一个严谨的AI',                    # system prompt 泄漏
    ]
    for pat in leak_patterns:
        text = re.sub(pat, '', text)

    # ═══ 最终整理 ═══
    # 连续 3 个以上空行压缩为 2 个空行（段间距保持一致）
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ════════════════════════════════════════════════════════════════
# 模型加载
# ════════════════════════════════════════════════════════════════

print("加载 Qwen2.5-7B tokenizer...")
tt = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
# Qwen2.5 的 tokenizer 默认没有 pad_token（ChatML 不需要 padding），
# 但 HuggingFace 的 generate() 在 batch>1 或某些采样模式下需要它。
# 设 eos_token 为 pad_token 是最小副作用的做法 —— 填充位和结束位重合不会产生额外 token。
if tt.pad_token is None:
    tt.pad_token = tt.eos_token

print("加载 Qwen2.5-7B 模型（4-bit）...")

# ── 4-bit 量化配置 ──
# nf4 (NormalFloat4)：专门为神经网络权重分布优化的 4-bit 量化格式，
#   比 fp4 精度更高。Qwen2.5-7B (16bit * 7B ≈ 14GB) → 4-bit ≈ 4-5GB。
# double_quant：对量化常数本身再做一次量化，再省约 0.4GB。
# bfloat16 compute：量化存储用 4bit，但计算时反量化到 bfloat16，精度损失极小。
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    quantization_config=quant,
    device_map="auto",         # 自动分配层到 GPU，OOM 时自动 offload 到 CPU
    trust_remote_code=True,    # Qwen2.5 模型有自定义代码，必须开
)
teacher.eval()

alloc = torch.cuda.memory_allocated() / 1024**3
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"显存占用: {alloc:.1f}GB / {total:.1f}GB")


# ════════════════════════════════════════════════════════════════
# 批量生成
# ════════════════════════════════════════════════════════════════

data = {}
failed = []   # (概念名, 失败原因)

for i, concept in enumerate(CONCEPTS, 1):
    # ── 1. 构建 ChatML prompt ──
    prompt = build_chatml_prompt(tt, concept)
    inputs = tt(prompt, return_tensors="pt").to(DEVICE)

    # ── 2. 自回归生成 ──
    # 参数选择理由：
    #   temperature=0.3：低温度让输出更确定。三角度解释需要创造性但不是随机性。
    #     0.1 太 greedy（所有概念输出趋向雷同），0.7 太高（格式容易崩）。
    #   top_p=0.9：nucleus sampling 的累积概率阈值。和低温度配合，
    #     既保持多样性又不放飞自我。
    #   max_new_tokens=512：三句话大约 300-500 中文字符（中文 token ratio ≈ 1.2:1），
    #     512 留有余量但不过多，防止模型没完没了。
    #   do_sample=True：必须开。不开就是 greedy decoding，所有概念输出千篇一律。
    with torch.no_grad():
        out = teacher.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tt.pad_token_id,
            eos_token_id=tt.eos_token_id,
        )

    # ── 3. Token ID 精确截取新生成内容 ──
    # 为什么不用 full_text[len(prompt):] 这种字符串方式？
    #   因为 decode(..., skip_special_tokens=True) 会把 ChatML 的 <|im_start|>、
    #   <|im_end|> 等特殊 token 去掉，导致 decoded 后的 prompt 和原始 prompt
    #   字符串不再对齐。用 token ID 长度做截断是唯一精确的方法。
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    answer = tt.decode(new_tokens, skip_special_tokens=True).strip()

    # ── 4. 清洗 ──
    answer = clean_generated_answer(answer)

    # ── 5. 校验 ──
    # 必须有三个标记。缺少任何一个说明生成质量不合格，计入失败但保留原始输出
    # 方便事后排查。
    has_one = "小一" in answer
    has_two = "小二" in answer
    has_three = "小三" in answer

    if has_one and has_two and has_three:
        data[concept] = answer
        status = "OK"
    else:
        missing = []
        if not has_one: missing.append("一")
        if not has_two: missing.append("二")
        if not has_three: missing.append("三")
        failed.append((concept, f"缺少角度: {missing}"))
        data[concept] = answer if answer else "(生成失败)"
        status = f"FAIL({','.join(missing)})"

    print(f"[{i:3d}/{len(CONCEPTS)}] {concept:6s}  {len(answer):4d}字符  {status}")

    # ── 6. 每 10 条保存一次 ──
    # 200 条 × 每条约 5s = ~17 分钟。如果跑到 150 条时崩溃没保存，心态炸裂。
    # 每隔 10 条 checkpoint 一次，最坏情况丢 9 条。
    if i % 10 == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════
# 最终保存 + 清理
# ════════════════════════════════════════════════════════════════

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

ok_count = len(data) - len(failed)
print(f"\n{'='*60}")
print(f"已保存: {OUTPUT_FILE}")
print(f"成功: {ok_count}/{len(data)}  失败: {len(failed)}/{len(data)}")
if failed:
    print(f"\n失败条目:")
    for concept, reason in failed:
        print(f"  - {concept}: {reason}")
print(f"{'='*60}")

# 释放显存。虽然脚本结束会自动释放，但养成好习惯且方便在交互环境中复用。
del teacher
torch.cuda.empty_cache()
