# RAG Demo Project

这是一个面向学习的小型 RAG Demo 项目，基于单份中文知识库 `data/source/知识库.md` 构建。

目标不是先把功能做满，而是先把整体框架搭稳：

- 前后端分离
- 浏览器可交互问答
- 支持多种 RAG 方法对比
- 支持可视化观察文本切分、向量化、索引写入和检索过程
- 支持运行状态检查与单次/对比 Markdown 报告导出
- 方便学习不同方法的优缺点

## 当前状态

第三版教学型完整流程已经可以运行：

```text
知识库.md
-> 结构化切分 / 固定窗口切分 / TF-IDF 语义凝聚度切分
-> jieba 分词
-> TF-IDF / BM25 / 加权混合检索
-> 可选查询词覆盖率重排
-> 上下文拼装
-> 结构化教学抽取 / LongCat-2.0 证据生成
-> 前端展示完整 Trace
-> 同题方法对比
```

生成阶段可以在本地结构化抽取与 LongCat-2.0 之间切换。两种方法复用完全相同的检索结果和上下文，因此可以单独观察生成策略带来的差异。稠密 Embedding 和 Cross-Encoder 仍保留为后续扩展点。

## 已实现的方法

| 阶段 | 方法 | 界面可观察内容 |
| --- | --- | --- |
| 切分 | 标题 / 段落结构切分 | 章节、位置、字符数 |
| 切分 | 固定长度 + overlap | 窗口参数、重复字符、语义截断 |
| 切分 | TF-IDF 语义凝聚度切分 | 相邻余弦相似度、边界原因、合并单元数、阈值与最大长度 |
| 检索 | TF-IDF | 词项权重、余弦相似度 |
| 检索 | BM25 | 命中词、BM25 相关度 |
| 检索 | TF-IDF + BM25 | 两个归一化分量与融合分数 |
| 重排 | 不重排 / 查询词覆盖率重排 | 初排、重排分数、前后名次 |
| 生成 | 结构化教学抽取 | 枚举列表、流程步骤、事实要点、选句原因、引用和无答案兜底 |
| 生成 | LongCat-2.0 | Prompt、自然语言综合回答、Token、完成原因、引用和失败回退 |

## 目录说明

- `data/source/`：原始知识库输入
- `backend/`：FastAPI 接口和 RAG 流程实现
- `backend/app/rag/`：加载、切分、索引、检索、上下文和生成模块
- `backend/tests/`：核心流程测试
- `frontend/`：React + TypeScript 可视化工作台
- `docs/`：项目规划、架构说明和设计文档
- `docs/architecture/`：架构层面的总体说明
- `docs/visualization/`：可视化思路和展示需求
- `docs/questions/`：需要你确认的设计问题
- `docs/architecture/第二版扩展说明.md`：本轮新增方法的原理和代码数据流
- `docs/architecture/结构化生成说明.md`：答案意图、证据选句和结构解析流程
- `docs/architecture/LongCat接入说明.md`：外部 LLM 调用、引用解析与回退流程

## 完整文档

- [`docs/RAG从零到全流程学习笔记.md`](docs/RAG从零到全流程学习笔记.md)：从零解释 RAG 的用途、发展历史、完整方法谱系、优缺点、评测、安全和本项目实践
- [`docs/RAG代码模块实现笔记.md`](docs/RAG代码模块实现笔记.md)：逐模块对应代码，详细解释切分、TF-IDF、BM25、混合检索、重排、生成、引用、Trace 和扩展入口
- [`docs/项目实现与展示文档.md`](docs/项目实现与展示文档.md)：项目架构、环境启动、页面功能、演示脚本、API、验证、故障排查与迭代路线

## LongCat 配置

项目只从根目录 `.env` 读取密钥，该文件已被 `.gitignore` 排除。配置模板见 `.env.example`：

```dotenv
LONGCAT_API_KEY=replace-with-your-api-key
LONGCAT_BASE_URL=https://api.longcat.chat/openai/v1
LONGCAT_MODEL=LongCat-2.0
LONGCAT_TIMEOUT_SECONDS=60
LONGCAT_MAX_TOKENS=900
```

如果 `LONGCAT_BASE_URL` 没有以 `/v1` 结尾，后端会自动补齐。密钥不会进入 Prompt、Trace 或 API 响应。

## 启动项目

确保所有命令都在 `blue` 环境执行。

### 一键启动与关闭

在项目根目录执行：

```bash
cd /home/blue/code/rag/rag-demo
./start.sh
```

脚本会检查 `blue` 环境和前端依赖，启动后端 `8001` 与前端 `5174`，等待两个服务通过健康检查，并将日志写入 `.runtime/logs/`。

关闭由该脚本启动的前后端：

```bash
cd /home/blue/code/rag/rag-demo
./stop.sh
```

脚本通过 `.runtime/*.pid` 只关闭本项目管理的进程，不会根据端口关闭其他程序。重复执行 `./start.sh` 不会重复启动服务。

可按需覆盖环境名或端口：

```bash
RAG_CONDA_ENV=blue RAG_BACKEND_PORT=8010 RAG_FRONTEND_PORT=5180 ./start.sh
```

### 手动启动

后端：

```bash
cd /home/blue/code/rag/rag-demo/backend
conda activate blue
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

前端另开一个终端：

```bash
cd /home/blue/code/rag/rag-demo/frontend
conda activate blue
VITE_API_BASE=http://127.0.0.1:8001/api npm run dev -- --host 127.0.0.1 --port 5174
```

浏览器打开 `http://127.0.0.1:5174`，API 文档位于 `http://127.0.0.1:8001/docs`。

如果端口已被占用，先访问对应的 `/api/health` 检查是否已有后端。需要备用端口时，后端端口和 `VITE_API_BASE` 必须同步修改。

```bash
VITE_API_BASE=http://127.0.0.1:8010/api npm run dev -- --port 5180
```

## API

- `POST /api/index/build`：按给定切分和检索方法重建索引
- `POST /api/query`：执行完整 RAG 流程
- `POST /api/compare`：隔离运行 2-6 套配置并返回并列结果
- `GET /api/runtime`：返回版本、知识库状态、LongCat 配置状态和方法能力摘要，不包含 API Key
- `GET /api/methods`：返回已实现和规划中的方法
- `GET /api/chunks`：查看当前索引中的 chunk

检索结果页可以导出单次问答 Markdown 报告；方法对比页可以导出多配置对比报告，便于课堂展示和实验留档。

## 验证

```bash
cd /home/blue/code/rag/rag-demo
conda run -n blue python -m unittest discover -s backend/tests -t backend -v
cd frontend
conda run -n blue npm run build
```

## 项目原则

- 每一个中间阶段都返回可展示的数据
- 已实现方法与规划方法明确区分
- 外部模型和服务在确认后再接入
- 尽量保留多方案对比，而不是只做单一路径
