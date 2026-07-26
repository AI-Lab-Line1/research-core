# Python 依赖建议

下面这份清单只做技术选型参考，不等于最终安装方案。

项目第一版的目标是把 RAG 流程完整跑通，所以依赖也按这个目标分层。

---

## 1. 第一版建议必需

### 1.1 Web 后端

- `fastapi`
- `uvicorn`

用途：

- 提供 API 接口
- 接收前端请求
- 返回切分、检索和问答结果

### 1.2 数据校验与配置

- `pydantic`
- `python-dotenv`

用途：

- 定义请求和响应结构
- 管理环境变量和配置项

### 1.3 文本处理

- `markdown`
- `beautifulsoup4`
- `lxml`

用途：

- 解析 Markdown 或网页文本
- 提取结构化内容

### 1.4 向量与数值

- `numpy`

用途：

- 存储和处理向量数据
- 处理相似度计算结果

### 1.5 切分与检索基础

- `scikit-learn`

用途：

- 可用于基础文本向量化对照
- 可用于一些简单检索或降维辅助

### 1.6 日志

- `loguru`，或者直接用标准库 `logging`

用途：

- 记录流程 trace
- 排查每一步中间结果

---

## 2. 第一版很可能会用到的 RAG 相关依赖

### 2.1 Embedding / 检索封装

这一类依赖要根据你选的模型和平台确定。

常见可能性包括：

- `sentence-transformers`
- `transformers`
- `torch`

用途：

- 生成文本向量
- 跑本地模型或开源模型

### 2.2 向量索引

可选：

- `faiss-cpu`
- `chromadb`

用途：

- 保存向量索引
- 做相似度检索

### 2.3 关键词检索

如果要做 BM25 或类似检索，可以考虑：

- `rank-bm25`

用途：

- 做关键词召回
- 作为混合检索的一部分

### 2.4 重排

如果第一版要加 rerank，可能还会引入：

- `sentence-transformers`
- `transformers`

用途：

- 对召回结果重新排序

---

## 3. 可视化和前端联调时可能需要

### 3.1 API 调试辅助

- `httpx`

用途：

- 后端调用外部服务
- 本地联调时做请求测试

### 3.2 数据展示

- `pandas`

用途：

- 处理表格型调试数据
- 方便日志分析和结果整理

### 3.3 图表或降维

- `matplotlib`
- `seaborn`
- `plotly`

用途：

- 可视化流程结果
- 如果你后面想看向量分布，可以再加降维展示

---

## 4. 后续增强可选

下面这些不是第一版必须，但如果你后面要扩展，可能会用到。

- `openai`
- `langchain`
- `llama-index`
- `rapidfuzz`
- `tiktoken`
- `orjson`
- `rich`

用途说明：

- `openai`：如果接 OpenAI 或兼容 API
- `langchain` / `llama-index`：如果想快速拼装 RAG 流程
- `rapidfuzz`：文本相似度辅助
- `tiktoken`：token 估算
- `orjson`：更快 JSON 处理
- `rich`：终端调试更方便

---

## 5. 我对第一版的判断

如果你想“先把流程跑通并且便于展示”，第一版不建议把依赖堆得太满。

推荐先有一个最小依赖集合：

- `fastapi`
- `uvicorn`
- `pydantic`
- `python-dotenv`
- `numpy`
- `scikit-learn`
- `markdown`
- `beautifulsoup4`
- `lxml`

如果再加上向量化和检索引擎，根据你最终选的方案再补。

---

## 6. 是否建议单独开虚拟环境

我的建议是：**建议单独开。**

原因：

1. 这是一个完整项目，不是一次性脚本
2. Python 依赖会比较多，尤其是 RAG、模型、检索、可视化相关包
3. 后面你很可能会反复切换依赖版本
4. 独立环境更方便复现和迁移

如果你只想做非常短平快的实验，不单开也能跑，但从项目化角度看，单独环境更稳。

---

## 7. 现在不做决定的内容

以下项先不拍板，等你确认：

- 用哪种 embedding 模型
- 用本地模型还是外部 API
- 向量库选 FAISS 还是 Chroma，或者别的
- 是否接 BM25 混合检索
- 是否第一版就加 rerank
- 是否使用 LangChain / LlamaIndex

