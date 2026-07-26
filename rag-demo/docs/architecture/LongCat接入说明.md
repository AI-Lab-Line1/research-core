# LongCat 接入说明

## 1. 目标

LongCat-2.0 作为生成模块接入，不参与切分、索引和检索。它与本地结构化生成器共享完全相同的 RAG 上下文：

```text
问题
  -> 切分与索引
  -> Top-K 检索
  -> 可选重排
  -> 带编号的上下文 [1] [2] ...
  -> extractive 或 LongCat-2.0
  -> 回答、引用、生成元数据
```

这种设计允许在方法对比页中只改变生成方法，其他变量保持不变。

## 2. OpenAI 兼容调用

后端使用官方 `openai` Python SDK，客户端配置为：

```text
base_url = https://api.longcat.chat/openai/v1
model = LongCat-2.0
endpoint = /chat/completions
```

SDK 最终请求的是 `/v1/chat/completions`。温度固定为 `0.2`，默认最大输出为 900 Tokens。

## 3. Prompt 结构

系统消息规定：

- 只能使用检索资料
- 每条事实必须标注 `[n]`
- 枚举用列表，流程用步骤
- 资料不足时明确回答无法确定
- 不输出思考过程

用户消息由带编号上下文、问题和直接回答指令组成。前端“上下文”页面可以查看实际 Prompt 预览，但不会显示 API Key。

## 4. 引用映射

模型回答中的 `[2]` 不是文档全局编号，而是本次上下文的第二个 block。后端会：

1. 用正则提取回答中的引用编号。
2. 丢弃超出上下文范围的编号。
3. 将有效编号映射回 `chunk_id` 和章节。
4. 将带引用的列表行转换成 `AnswerPoint`。

如果模型返回回答但没有有效引用，页面会显示警告，不会伪造引用。

## 5. 生成元数据

`QueryResponse.generation_metadata` 包含：

```text
requested_method
effective_method
provider
model
prompt_tokens
completion_tokens
total_tokens
finish_reason
fallback_used
```

这些字段和生成耗时一起用于比较本地规则生成与外部 LLM 的成本和延迟。

## 6. 失败回退

以下情况会触发本地结构化抽取：

- `.env` 缺少 API Key
- 连接失败
- 请求超时
- 服务返回非成功 HTTP 状态
- 模型返回空内容

回退后：

- `requested_method` 仍为 `longcat`
- `effective_method` 变成 `extractive`
- `fallback_used` 为 `true`
- 页面显示经过清理的错误提示
- Trace 明确记录模型没有成功生成

后端不会把 API Key、认证头或服务端完整响应体放入错误信息。

## 7. GPU 说明

LongCat 是远程 API，调用它不使用本机 GPU。本机 RTX 4060 可用于后续本地 Embedding、语义切分或本地重排模型，这些属于检索与排序阶段。
