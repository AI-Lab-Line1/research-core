# 太原理工大学创新学社 · RAG 智能问答系统

基于检索增强生成（RAG）的智能问答系统，知识库为太原理工大学创新学社的社团资料，涵盖社团精神、校园生活、学习资源、勋章机制等内容。

## 项目结构

```
raaaag/
├── 知识库.md                  # 原始知识库（纯文本，关键词+冒号分隔章节）
├── config.py                  # 全局配置（模型、参数、路径）
├── main.py                    # CLI 入口（交互式/单次问答）
├── README.md                  # 本文件
├── vector_store/              # ChromaDB 向量库持久化目录（自动生成）
└── src/
    ├── __init__.py
    ├── document_loader.py     # 文档加载 + 章节识别 + 语义分块
    ├── embedder.py            # 文本向量化（BGE-large-zh-v1.5）
    ├── vector_store.py        # ChromaDB 向量存储与检索
    ├── llm.py                 # LLM 调用（ModelScope API / Qwen3.5-27B）
    └── rag_pipeline.py        # RAG 主流程（建索引 + 问答）
```

## 技术架构

```
用户提问
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  embedder.py │ ──► │  vector_store.py  │ ──► │   llm.py      │
│  问题→向量    │     │  向量→Top-K 文档块  │     │  文档块+问题   │
│  BGE-large   │     │  ChromaDB 检索     │     │  → 生成回答   │
└──────────────┘     └──────────────────┘     └───────────────┘
                                                     │
                                                     ▼
                                                CLI 输出回答
                                               + 参考来源追溯
```

**离线阶段（建索引）**：`知识库.md → 章节切分 → 语义分块 → BGE 嵌入 → ChromaDB 持久化`

**在线阶段（问答）**：`用户问题 → BGE 嵌入 → ChromaDB 检索 Top-K → LLM 生成回答`

## 组件详解

### 1. `config.py` — 全局配置

所有可调参数集中管理，修改后执行 `python main.py --rebuild` 使 embedding 相关改动生效。

| 配置项                 | 默认值                     | 说明                                                       |
| ---------------------- | -------------------------- | ---------------------------------------------------------- |
| `CHUNK_SIZE`         | 400                        | 每个文本块最大字符数。中文信息密度高，400 字 ≈ 250 tokens |
| `CHUNK_OVERLAP`      | 80                         | 相邻块重叠字符数（20% 重叠率），防止关键信息在边界被切断   |
| `EMBEDDING_MODEL_ID` | `BAAI/bge-large-zh-v1.5` | 中文语义检索 SOTA，C-MTEB 基准排名靠前                     |
| `EMBEDDING_DEVICE`   | `cpu`                    | embedding 在 CPU 推理，留 GPU 显存给 LLM                   |
| `LLM_MODEL_ID`       | `Qwen/Qwen3.5-27B`       | 通过 ModelScope API 调用的生成模型                         |
| `LLM_TEMPERATURE`    | 0.3                        | RAG 场景偏保守，减少幻觉                                   |
| `LLM_MAX_TOKENS`     | 1024                       | 生成回答的最大 token 数                                    |
| `RETRIEVAL_TOP_K`    | 5                          | 每次检索返回的最相关文档块数                               |

**实测可用的 LLM 模型**（通过 `curl https://api-inference.modelscope.cn/v1/models` 验证）：

| 模型 ID                       | 状态      | 说明                        |
| ----------------------------- | --------- | --------------------------- |
| `Qwen/Qwen3.5-27B`          | ✓ 可用   | 中文顶级，当前选择          |
| `deepseek-ai/DeepSeek-V3.2` | ✓ 可用   | DeepSeek 旗舰               |
| `Qwen/Qwen3-8B`             | ✗ 返回空 | HTTP 200 但 choices 为 None |
| `Qwen/Qwen3-4B`             | ✗ 不可用 | HTTP 400                    |
| `ZhipuAI/GLM-4.7-Flash`     | ✗ 返回空 | HTTP 200 但 choices 为 None |

### 2. `src/document_loader.py` — 文档加载与分块

**两步走策略**：

#### 第一步：章节切分（`parse_sections`）

知识库的实际格式是纯文本，标题和正文在同一行，用冒号分隔：

```
太原理工大学创新学社秉承"敢立高标..."（开头简介 → 归入"学社简介"）
报到与宿舍生活：开学报到前，要确认录取通知书上的时间和地点...
校园美食：北餐有葛辉饺子、馄饨和自选菜...
学习方法与考试/复习策略：课堂是获取知识和加深老师印象的重要场所...
```

用正则 `^([\u4e00-\u9fa5\w]{2,10})[：:](.+)$` 识别行首 2-10 个字符后紧跟冒号的行作为章节标题，冒号前的部分为标题，冒号后的部分为该节正文。

**设计考量**：

- 限制 2-10 字符避免误匹配正文中的冒号（如"时间：早上8点"不会出现在行首）
- 不用 markdown `##` 语法的原因是知识库里根本没有 markdown 标记
- 不用 NLP 分段器的原因是知识库格式高度规整，正则更可靠且零依赖

#### 第二步：语义分块（`RecursiveCharacterTextSplitter`）

每个章节内部用 langchain 的 `RecursiveCharacterTextSplitter` 按语义边界切小块。

分隔符优先级（越靠前越优先）：

```
\n\n   → 段落边界（最优，完整语义）
\n     → 行边界
。！？  → 中文句子标点
；，   → 中文分句标点
空格   → 单词边界
字符   → 兜底：硬切（极端情况）
```

**设计考量**：

- 不自己手写 split：中英文混排、列表格式、数字+单位等边界情况 langchain 已处理好
- chunk_size=400 + overlap=80：经验最佳实践，覆盖 25 个 chunk 的知识库

### 3. `src/embedder.py` — 文本向量化

**模型**：`BAAI/bge-large-zh-v1.5`（北京智源 BGE 系列）

- 向量维度：1024
- 模型大小：约 1.3GB
- 中文语义检索 SOTA（C-MTEB 基准）
- 专门为 retrieval 任务训练，非通用 embedding

**加载方式**：

1. `modelscope.hub.snapshot_download()` — 从 ModelScope Hub 下载模型文件到 `~/.cache/modelscope/`
2. `sentence_transformers.SentenceTransformer()` — 从本地路径加载模型

**为什么不直接用 modelscope pipeline**：

- modelscope pipeline 对底层 transformers 版本高度敏感
- 实测报错 `'BertConfig' object has no attribute 'is_decoder'`（版本不兼容）
- sentence-transformers 直接加载模型权重文件，绕过了这层脆弱的封装

**关键参数**：

- `normalize_embeddings=True`：向量归一化，使得内积 = 余弦相似度，与 ChromaDB 的 L2 距离等价
- `device='cpu'`：embedding 模型推理轻量，CPU 即可，留 GPU 给可能的本地 LLM

### 4. `src/vector_store.py` — 向量存储与检索

**数据库**：ChromaDB（PersistentClient，SQLite 持久化）

**选型理由**（对比 FAISS / Milvus）：

- FAISS：C++ 实现，Windows 需编译，且纯内存模式不持久化
- Milvus：需 Docker，Windows 上 Docker Desktop 资源占用大
- ChromaDB：纯 Python，`pip install` 即用，自带元数据存储和持久化

**核心操作**：

| 函数                           | 作用                 | 关键设计                                            |
| ------------------------------ | -------------------- | --------------------------------------------------- |
| `get_or_create_collection()` | 获取/创建 collection | get_or_create 一个调用覆盖两种情况，避免 try/except |
| `rebuild_collection()`       | 先删后建             | 保证幂等性——多次重建结果一致                      |
| `add_chunks_to_store()`      | 批量写入向量         | ID 用"章节名_序号"而非 UUID，保证可读性和幂等性     |
| `query_collection()`         | 检索 Top-K           | L2 距离排序，距离越小越相似                         |

**为什么需要 rebuild 而非追加**：

- 知识库修改后，追加旧向量仍在 → 用户看到过时/矛盾信息
- 删除的内容继续存在 → 向量库与知识库不一致
- 对 25 个 chunk 的小库，重建只需几秒，投入产出比极高

### 5. `src/llm.py` — LLM 调用

**模型**：`Qwen/Qwen3.5-27B`（通过 ModelScope 推理 API）

**通信协议**：OpenAI 兼容的 `/v1/chat/completions`，用 `openai` Python SDK 调用。

**Prompt 结构**：

```
system:  你是太原理工大学创新学社的智能助手...
         回答规则：只根据资料回答，不确定就说"未找到"

user:    参考资料：
           [来源1] ...（检索到的 chunk 1）
           [来源2] ...（检索到的 chunk 2）
         ---
         用户问题：xxx
         请根据以上参考资料回答问题。
```

**关键设计**：

- system 消息写死"只根据资料回答"——这是 RAG 的核心约束，防止 LLM 用训练数据编造
- temperature=0.3——RAG 场景偏保守，抑制"创造性发挥"
- 错误分类处理——401/429/404/空结果 分别给出针对性提示，而非笼统的"调用失败"

### 6. `src/rag_pipeline.py` — RAG 主流程

串联上述所有模块，提供两个入口：

**`build_index(force=False)`**：

```
知识库.md → chunk_document() → embed_texts() → rebuild_collection() → add_chunks_to_store()
```

- `force=False`：collection 非空时跳过（默认行为）
- `force=True`：无条件重建（`--rebuild` 触发）

**`query(question, top_k=5)`**：

```
问题 → embed_texts() → query_collection() → chat() → 回答 + 来源
```

**返回结构**特意包含 `sources` 字段（而不仅仅是 answer 字符串），因为 RAG 的核心价值在于**可验证性**——用户可以通过 sources 追溯原始资料，判断 LLM 回答是否可信。

### 7. `main.py` — CLI 入口

**三种使用模式**：

```bash
# 模式 1：交互式问答（启动后持续对话）
python main.py

# 模式 2：重建索引（知识库修改后）
python main.py --rebuild

# 模式 3：单次问答（输出后退出，适合脚本/测试）
python main.py -q "有哪些食堂推荐"
python main.py -k 10 -q "..."    # 指定检索 top-K=10
```

**交互模式下的特殊命令**：

| 输入         | 作用                       |
| ------------ | -------------------------- |
| `/exit`    | 退出程序                   |
| `/sources` | 查看上次检索的原始来源文档 |
| `Ctrl+C`   | 强制退出                   |

## 环境搭建

### 1. 创建 conda 环境

```bash
conda create -n rag python=3.10 -y
conda activate rag
```

（git-bash 下需先 `source /d/Users/33497/anaconda3/etc/profile.d/conda.sh`）

### 2. 安装依赖

```bash
cd /d D:\01\study\RAG\raaaag
pip install -r requirements.txt
```

### 3. 首次运行

```bash
python main.py --rebuild
```

首次运行会：

1. 下载 BGE-large-zh-v1.5 模型（约 1.3GB，仅首次，缓存到 `~/.cache/modelscope/`）
2. 将知识库分块并生成向量
3. 写入 ChromaDB 向量库（持久化到 `vector_store/` 目录）

后续启动 `python main.py` 直接进入问答，零等待。

## 使用示例

```
$ python main.py

============================================================
  太原理工大学创新学社 · RAG 问答系统
============================================================
输入问题即可查询，输入 /exit 退出，输入 /sources 查看上次来源

你的问题: 创新学社的精神是什么？

[检索] 正在检索与问题相关的资料 (top-5)...
[LLM] 正在生成回答...

回答:
太原理工大学创新学社的精神是"敢立高标擎日月，甘为前路破荆棘"，
面向那些不甘平庸、渴望进步的同学，强调实事求是、平等沟通和真诚合作。

参考来源 (5 条):
  [1] 学社简介 (距离=0.0123)
  [2] 人员要求 (距离=0.0266)
  ...

你的问题: /exit
再见！
```

## 常见问题

**Q: 修改了知识库后怎么更新？**

```bash
python main.py --rebuild
```

强制重建向量索引，使新内容生效。

**Q: 想切换 LLM 模型？**
修改 `config.py` 中的 `LLM_MODEL_ID`，无需改其他代码：

```python
LLM_MODEL_ID = "deepseek-ai/DeepSeek-V3.2"  # 换到 DeepSeek
```

**Q: 检索不准怎么办？**

- 检查 `RETRIEVAL_TOP_K`：调大到 8-10 看是否有改善
- 检查 `CHUNK_SIZE`：如果知识库新增了大量内容，考虑调整分块大小
- 检查 `EMBEDDING_MODEL_ID`：BGE-large 是当前最优选择，一般不需要换

**Q: LLM 返回"资料中未找到相关信息"？**

- 先用 `/sources` 查看检索到的文档块是否真的不包含相关信息
- 如果 sources 里确实没有 → 换种提问方式试试
- 如果 sources 里有但 LLM 说没找到 → 可能是模型对"只根据资料回答"这条指令执行过严，换 DeepSeek-V3.2 试试

**Q: 如何查看 ModelScope API 当前可用的模型列表？**

```bash
curl -s https://api-inference.modelscope.cn/v1/models \
  -H "Authorization: Bearer ms-1858aa66-bc09-4c0a-8d30-5ae1ba4f6015"
```

但并非列表中的所有模型都支持 `chat/completions` 协议，需逐个测试。
