# 知识蒸馏 · Qwen2.5-7B → Qwen2-0.5B

让一个大模型（Teacher）把自己写"三角度解释"的能力压缩到一个 14 倍小的模型（Student）里。

## 概览

```
Teacher:  Qwen2.5-7B-Instruct  (4-bit, ~4GB 显存)
Student:  Qwen2-0.5B-Instruct  (float32, ~2GB 显存)
GPU:      RTX 5060, 8GB 显存
```

Teacher 生成 200 个抽象概念的三角度解释 → Student 通过 CE + Cosine 双损失学习。

## 文件结构

```
.
├── 01_generate_data.py       教师模型生成训练数据
├── 02_train_distillation.py  知识蒸馏训练
├── 03_inference_compare.py   交互式推理对比
├── data/
│   └── teacher_train.json    200 条三角度解释
├── output/
│   ├── distilled_student.pt         学生模型权重 (~2GB)
│   ├── projection.pt                投影层权重 3584→896 (~12MB)
│   ├── distilled_student_tokenizer/ 学生 tokenizer
│   ├── teacher_hidden_cache.pt      Teacher 预计算特征缓存
│   ├── loss_curve.png               训练损失曲线
│   └── distilled_student_loss_history.json
└── models/
    └── Qwen2.5-7B-Instruct/  教师模型（本地下载）
```

## 流水线

### 第一步：生成教师数据

```
conda activate pytorch_env
python 01_generate_data.py
```

用 Qwen2.5-7B 为 200 个抽象概念生成三角度解释。输出格式是干净的 `一:/二:/三:` 标记，不附带格式描述——学生以后从内容本身学结构。

每个概念的三句话逻辑：
- **一** — 生活比喻
- **二** — 学术定义
- **三** — 反向排除（不是什么）

关键实现细节：
- 使用 ChatML 格式（`apply_chat_template`）而非裸文本拼接，消除 prompt 泄漏
- 用 token ID 差值精确截取新生成内容，不靠字符串对齐
- 清洗管道含 6 层过滤（控制字符 → Markdown → 锚定 → 重复截断 → 废话清除 → 指令泄漏匹配）

### 第二步：蒸馏训练

```
python 02_train_distillation.py
```

#### 两个损失

| 损失 | 权重 | 作用 |
|---|---|---|
| **CE（交叉熵）** | 0.5 | 数据蒸馏。Student 做 next-token prediction，学习 Teacher 回答的文本格式和内容 |
| **Cosine（余弦相似度）** | 0.5 | 表示层对齐。Teacher 最后一层 hidden state (3584维) → 投影层 → Student 空间 (896维)，算 1-cos(θ) |

没有 KL 散度——Teacher 词表（152064）和 Student 词表（151646）不一致，无法在同一概率空间比较。

#### 训练流程

1. 预计算 Teacher 在所有数据上的 hidden state（均值池化），存为缓存
2. 卸载 Teacher，释放 ~4GB 显存
3. 训练 Student：CE + Cosine 双损失，10 epoch
4. 含 8 步 warmup（仅训练投影层），梯度累积 2 步

#### 显存优化（8GB 显卡适配）

- Teacher 4-bit 量化（nf4 + double quant）
- 预计算后卸载 Teacher
- Student 开启 gradient_checkpointing
- 前向使用 bf16 autocast

### 第三步：推理对比

```
python 03_inference_compare.py
```

输入概念名，原始 Qwen2-0.5B 和蒸馏学生同时生成，肉眼对比效果。

## 结果怎么看

对于同一个问题"请解释一下{概念}。"：

- 蒸馏学生的回答是否比原始模型更有条理？
- 是否学到了三角度格式（一: 比喻 / 二: 定义 / 三: 排除）？
- 回答风格是否更接近 Teacher 的训练数据？

## 训练结果（10 epoch, 969 step）

| 损失 | 最终值 | 说明 |
|---|---|---|
| 总损失 | **0.0039** | CE + Cosine 加权和 |
| CE | **0.0007** | Student 几乎完美预测 Teacher 文本 |
| Cosine | **0.0071** | 隐状态向量夹角 < 3°，表示空间高度对齐 |

## 依赖

- Python 3.10+
- PyTorch 2.x（CUDA 13.x）
- transformers
- bitsandbytes（4-bit 量化）
- matplotlib（损失曲线）
- tqdm（进度条）

## 注意事项

- 首次运行教师模型生成需要约 17 分钟（200 条）
- 训练约 55 分钟（194 batch × 10 epoch，~5.5s/batch）
- bitsandbytes 需要设置 `BNB_CUDA_VERSION=130`
- DataLoader 的 `num_workers=0`（Windows 兼容）
