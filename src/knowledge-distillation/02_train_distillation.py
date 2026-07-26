"""
================================================================================
02_train_distillation.py — 知识蒸馏训练（数据蒸馏 + 表示层对齐）
================================================================================

【架构】
    Teacher: Qwen2.5-7B-Instruct（4-bit 量化，仅 forward，不生成）
    Student: Qwen2-0.5B-Instruct（float32，训练）

【数据来源】
    data/teacher_train.json —— Teacher 生成的 200 条三角度解释

【训练流程（每个 batch）】
    1. 从 JSON 读取概念 + Teacher 的回答文本
    2. Student forward：在[简单 prompt + Teacher 回答]上做 next-token prediction
    3. Teacher forward：Teacher 在同一个回答文本上 forward（用自己 tokenizer）
    4. CE 损失：Student 学习 Teacher 的回答格式和内容
    5. Cosine 损失：Student 隐状态向 Teacher 对齐（跨架构蒸馏）

【两个损失】
    CE（Cross-Entropy）
        Student 预测下一个 token。
        让 Student 学会写出类似 Teacher 的回答格式和内容。
        这是"数据蒸馏"——Teacher 的知识压缩在文本中，Student 从文本学。

    Cosine（余弦相似度）
        对 Student 和 Teacher 最后一层 hidden states 做 mean pooling，
        然后算 1 - cos(θ)，让两个表示向量方向对齐。
        Teacher(Qwen2.5-7B, 3584维) → 投影层 → Student(Qwen2-0.5B, 896维)
        这是"表示层蒸馏"——Student 内部表示向 Teacher 靠拢。

【为什么没有 KL 散度】
    Teacher 词表（152064）和 Student 词表（151646）不同，
    KL 需要两个概率分布在同一词表上比较，所以不能算。

【为什么没有 student.generate()】
    Student 永远不需要自回归生成——数据已经准备好了。
    避免训练循环中 generate() 导致的 CUDA 崩溃。

【运行方式】
    conda activate pytorch_env
    python 02_train_distillation.py
================================================================================
"""

import os
# bitsandbytes CUDA 版本兼容：PyTorch 13.2 但 bitsandbytes 只支持到 13.0
os.environ.setdefault("BNB_CUDA_VERSION", "130")

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

# matplotlib 中文支持
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 训练时优先使用 float32，避免半精度下更新过小/NaN
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ================================================================
# 超参数（可调整）
# ================================================================

# ── 文件路径 ──
TRAIN_DATA_PATH = "data/teacher_train.json"   # Teacher 生成的训练数据
OUTPUT_DIR = "output"                          # 模型保存路径
OUTPUT_NAME = "distilled_student"              # 模型文件名

# ── 模型 ──
TEACHER_MODEL = "./models/Qwen2.5-7B-Instruct"     # Qwen2.5-7B（本地路径）
STUDENT_MODEL = "Qwen/Qwen2-0.5B-Instruct"         # Qwen2-0.5B（HF 缓存）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float32                        # 训练时使用 float32

# ── 训练参数 ──
EPOCHS = 10               # 训练轮数
BATCH_SIZE = 1            # 每次 1 个概念（显存限制）
GRAD_ACCUM_STEPS = 2      # 每 2 步更新一次参数
LEARNING_RATE = 3e-5      # AdamW 学习率
MAX_LENGTH = 256          # 增加长度，Qwen2.5 的回答更长（512 tokens 生成）
LOG_STEPS = 1             # 每 1 个梯度更新打印一次日志

# ── 损失权重 ──
CE_WEIGHT = 0.5           # CE 损失权重：Student 学写 Teacher 的文本
HIDDEN_WEIGHT = 0.5       # Cosine 损失权重：Student 隐状态向 Teacher 对齐

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"设备: {DEVICE}")
print(f"教师: Qwen2.5-7B-Instruct（4-bit，仅 forward）")
print(f"学生: Qwen2-0.5B-Instruct")
print(f"损失权重: CE={CE_WEIGHT}  HIDDEN={HIDDEN_WEIGHT}")


# ================================================================
# 投影层：Teacher 的 hidden_size(3584) → Student 的 hidden_size(896)
# ================================================================
# Teacher(Qwen2.5-7B) 的隐藏层维度是 3584，Student(Qwen2-0.5B) 是 896。
# 需要训练一个线性投影层把 Teacher 的 hidden states 映射到 Student 的空间，
# 才能做余弦相似度比较。
# ================================================================
class HiddenProjection(nn.Module):
    def __init__(self, teacher_hidden, student_hidden):
        super().__init__()
        # 线性层：3584 → 896
        self.proj = nn.Linear(teacher_hidden, student_hidden)

    def forward(self, x):
        return self.proj(x)


# ================================================================
# 数据集：每个样本包含概念名 + Teacher 的回答文本
# ================================================================
class DistillDataset(Dataset):
    def __init__(self, data: dict):
        self.items = [{"index": i, "concept": c, "teacher_text": t} for i, (c, t) in enumerate(data.items())]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def precompute_teacher_features(teacher, tt, dataset, max_length=MAX_LENGTH):
    """训练前只预计算一次教师 hidden states 的 pooled 特征，后续训练循环直接复用。"""
    cache = []
    teacher.eval()
    with torch.no_grad():
        for item in tqdm(dataset.items, desc="预计算 Teacher 特征"):
            teacher_text = item["teacher_text"]
            enc = tt(
                teacher_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            ti = enc["input_ids"].to(DEVICE)
            ta = enc["attention_mask"].to(DEVICE)
            t_out = teacher(
                input_ids=ti,
                attention_mask=ta,
                output_hidden_states=True,
            )
            t_hidden = t_out.hidden_states[-1]
            t_pooled = t_hidden.mean(dim=1).to(torch.float32).squeeze(0)
            cache.append(t_pooled.cpu())
    return cache


# ================================================================
# 训练函数（核心）
# ================================================================
def train(student, st, projection, dataset, teacher_cache):
    """
    参数：
        student:       Qwen2-0.5B（训练）
        st:            Student tokenizer
        projection:    投影层 3584→896
        dataset:       DistillDataset
        teacher_cache: 预计算好的教师 pooled hidden features 缓存
    """
    # ── 模式设置 ──
    student.train()
    projection.train()

    # ── 数据加载器 ──
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ── 优化器：同时优化 Student + 投影层 ──
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projection.parameters()),
        lr=LEARNING_RATE,
    )

    # ── 学习率调度器 ──
    total_steps = len(loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps,
    )

    # ── 日志记录 ──
    history = {"step": [], "total_loss": [], "ce_loss": [], "hidden_loss": []}
    gs = 0                  # global step
    at = ac = ah = 0.0      # 累积损失
    WARMUP_STEPS = 8        # 预热：前 8 步只训练投影层

    student_vs = student.lm_head.out_features  # Student 词表大小

    print(f"\n=== 蒸馏训练（数据蒸馏 + 表示层对齐）===")
    print(f"学生词表: {student_vs}")
    print(f"教师 hidden: 3584  学生 hidden: 896")

    # ════════════════════════════════════════════════════════════
    # 训练循环
    # ════════════════════════════════════════════════════════════
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in pbar:
            # ── 从 batch 中取出当前概念和 Teacher 的回答 ──
            item_idx = batch["index"][0]
            concept = batch["concept"][0]
            teacher_text = batch["teacher_text"][0]

            # ════════════════════════════════════════════════════
            # 第一步：构建 Student 的输入
            # ════════════════════════════════════════════════════
            student_input = f"请解释一下{concept}。{teacher_text}"
            enc = st(
                student_input,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            s_input = enc["input_ids"].to(DEVICE)
            s_mask = enc["attention_mask"].to(DEVICE)

            # ── Labels 设置 ──
            # prompt 部分设 -100，不计算损失
            prompt_len = len(st.encode(f"请解释一下{concept}。", add_special_tokens=False))
            labels = s_input.clone()
            labels[:, :prompt_len] = -100
            labels = labels[:, :MAX_LENGTH]
            s_input = s_input[:, :MAX_LENGTH]
            s_mask = s_mask[:, :MAX_LENGTH]

            # ════════════════════════════════════════════════════
            # 第二步：Student forward（bf16 autocast 省一半激活显存）
            # ════════════════════════════════════════════════════
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                s_out = student(
                    input_ids=s_input,
                    attention_mask=s_mask,
                    output_hidden_states=True,
                )
            s_logits = s_out.logits.float()  # 回到 float32 算 loss
            s_hidden = s_out.hidden_states[-1].float()  # [batch, seq_len, 896]
            del s_out

            # ════════════════════════════════════════════════════
            # 第三步：CE 损失（数据蒸馏）
            # ════════════════════════════════════════════════════
            shift_s = s_logits[..., :-1, :].contiguous()
            shift_l = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_s.view(-1, student_vs), shift_l.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

            # ════════════════════════════════════════════════════
            # 第四步：读取预计算好的 Teacher 特征
            # ════════════════════════════════════════════════════
            t_pooled = teacher_cache[item_idx].to(DEVICE, dtype=torch.float32)

            # ════════════════════════════════════════════════════
            # 第五步：Cosine 损失（隐状态对齐）
            # ════════════════════════════════════════════════════
            if HIDDEN_WEIGHT > 0:
                s_pooled = s_hidden.mean(dim=1).to(torch.float32)  # [batch, 896]

                # 预热阶段：Student detach，只训练投影层
                if gs < WARMUP_STEPS:
                    s_pooled = s_pooled.detach()

                # 投影：Teacher 的 3584 维 → Student 的 896 维
                t_projected = projection(t_pooled)  # [batch, 896]

                # 余弦相似度损失：1 - cos(θ)
                hidden_loss = 1 - F.cosine_similarity(s_pooled, t_projected, dim=-1).mean()
            else:
                hidden_loss = torch.tensor(0.0, device=DEVICE)

            # ════════════════════════════════════════════════════
            # 第六步：总损失 + 反向传播
            # ════════════════════════════════════════════════════
            total_loss = CE_WEIGHT * ce_loss + HIDDEN_WEIGHT * hidden_loss

            raw_total = total_loss.item()
            total_loss = total_loss / GRAD_ACCUM_STEPS
            total_loss.backward()

            at += raw_total
            ac += ce_loss.item()
            ah += hidden_loss.item()

            # ════════════════════════════════════════════════════
            # 第七步：梯度累积 → 参数更新
            # ════════════════════════════════════════════════════
            if (gs + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(projection.parameters()),
                    100.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # ── 日志 ──
                us = gs // GRAD_ACCUM_STEPS
                if us % LOG_STEPS == 0:
                    history["step"].append(us)
                    history["total_loss"].append(at / GRAD_ACCUM_STEPS)
                    history["ce_loss"].append(ac / GRAD_ACCUM_STEPS)
                    history["hidden_loss"].append(ah / GRAD_ACCUM_STEPS)
                    pbar.set_postfix({
                        "total": f"{at/GRAD_ACCUM_STEPS:.4f}",
                        "ce": f"{ac/GRAD_ACCUM_STEPS:.4f}",
                        "hidden": f"{ah/GRAD_ACCUM_STEPS:.6f}",
                    })
                at = ac = ah = 0.0

            gs += 1

        # ════════════════════════════════════════════════════════
        # 每个 epoch 结束：保存检查点（仅最后一轮）
        # ════════════════════════════════════════════════════════
        if epoch == EPOCHS:
            torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_epoch{epoch}_student.pt"))
            torch.save(projection.state_dict(), os.path.join(OUTPUT_DIR, f"projection_epoch{epoch}.pt"))

    # ════════════════════════════════════════════════════════════
    # 保存最终模型
    # ════════════════════════════════════════════════════════════
    torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.pt"))
    st.save_pretrained(os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_tokenizer"))
    torch.save(projection.state_dict(), os.path.join(OUTPUT_DIR, "projection.pt"))
    print(f"\n模型权重已保存: {OUTPUT_DIR}/")

    # 保存损失历史
    hist = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_loss_history.json")
    with open(hist, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"损失历史已保存: {hist}")
    return history


# ================================================================
# 主流程
# ================================================================
def main():
    # ── 1. 加载数据 ──
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # ── 2. 加载 tokenizer ──
    print("\n加载 tokenizer...")
    # Teacher tokenizer（Qwen2.5-7B，词表 152064）
    tt = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    # Student tokenizer（Qwen2-0.5B，词表 151646）
    st = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if st.pad_token is None:
        st.pad_token = st.eos_token

    print(f"Teacher tokenizer: {len(tt)}  Student tokenizer: {len(st)}")

    # ── 3. Teacher（Qwen2.5-7B 4-bit 量化） ──
    print(f"\n加载教师（4-bit）: {TEACHER_MODEL}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    teacher.eval()
    teacher_hidden = teacher.config.hidden_size  # 3584
    print(f"  Teacher hidden_size: {teacher_hidden}")

    # ── 4. Student（Qwen2-0.5B） ──
    print(f"\n加载学生: {STUDENT_MODEL}")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        trust_remote_code=True,
        dtype=MODEL_DTYPE,
        output_hidden_states=True,
    )
    student = student.to(DEVICE, dtype=torch.float32)
    student.gradient_checkpointing_enable()  # 用计算换显存：前向不存中间激活，反向时重算
    student_hidden = student.config.hidden_size  # 896
    print(f"  Student hidden_size: {student_hidden}")

    # ── 5. 投影层（3584 → 896） ──
    projection = HiddenProjection(teacher_hidden, student_hidden).to(DEVICE, dtype=torch.float32)

    # ── 显存信息 ──
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n显存: {alloc:.1f}GB / {total:.1f}GB")

    # ── 6. 数据集 ──
    dataset = DistillDataset(train_data)
    print(f"数据集: {len(dataset)} 条")

    # ── 7. 预计算 Teacher 特征缓存 ──
    cache_path = os.path.join(OUTPUT_DIR, "teacher_hidden_cache.pt")
    if os.path.exists(cache_path):
        teacher_cache = torch.load(cache_path, map_location="cpu")
        # 校验：数据重新生成后条目数可能变化，缓存失效必须重建
        if len(teacher_cache) != len(dataset):
            print(f"缓存条目数({len(teacher_cache)})≠数据集({len(dataset)})，重建中...")
            os.remove(cache_path)
            teacher_cache = precompute_teacher_features(teacher, tt, dataset, max_length=MAX_LENGTH)
            torch.save(teacher_cache, cache_path)
            print(f"教师缓存已重建: {cache_path}")
        else:
            print(f"加载教师缓存: {cache_path} ({len(teacher_cache)}条)")
    else:
        teacher_cache = precompute_teacher_features(teacher, tt, dataset, max_length=MAX_LENGTH)
        torch.save(teacher_cache, cache_path)
        print(f"教师缓存已保存: {cache_path}")

    # ── 8. 释放 Teacher 显存 ──
    # 预计算完成后 Teacher 不再需要。训练时 Student + AdamW 状态 ≈ 6GB，
    # Teacher 4-bit ≈ 4GB，同时存在超 8GB 显存。
    del teacher
    torch.cuda.empty_cache()
    print("Teacher 已卸载，释放显存给 Student 训练")

    # ── 9. 训练（KeyboardInterrupt 时自动保存当前权重） ──
    try:
        history = train(student, st, projection, dataset, teacher_cache)
    except KeyboardInterrupt:
        print("\n\n⚠ 训练被中断，保存当前权重...")
        torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_interrupted.pt"))
        torch.save(projection.state_dict(), os.path.join(OUTPUT_DIR, "projection_interrupted.pt"))
        print(f"中断权重已保存: {OUTPUT_DIR}/{OUTPUT_NAME}_interrupted.pt")
        raise

    # ── 10. 画损失曲线 ──
    def plot_curves(h, save_path=None):
        steps = h["step"]
        if not steps:
            return
        plt.figure(figsize=(12, 6))
        plt.plot(steps, h["total_loss"], label="总损失", linewidth=2)
        plt.plot(steps, h["ce_loss"], label="CE", linewidth=1.5, linestyle="--")
        plt.plot(steps, h["hidden_loss"], label="HIDDEN(cos)", linewidth=1.5, linestyle="-.")
        plt.xlabel("训练步数", fontsize=13)
        plt.ylabel("损失值", fontsize=13)
        plt.title("知识蒸馏损失曲线\nTeacher: Qwen2.5-7B → Student: Qwen2-0.5B", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"曲线已保存: {save_path}")
        plt.close()

    plot_curves(history, save_path=os.path.join(OUTPUT_DIR, "loss_curve.png"))

    # ── 11. 清理 ──
    del student
    torch.cuda.empty_cache()
    print("\n✓ 蒸馏训练完成")


if __name__ == "__main__":
    main()
