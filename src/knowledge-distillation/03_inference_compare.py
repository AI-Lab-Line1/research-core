"""
================================================================================
03_inference_compare.py — 交互式推理对比
================================================================================

【功能】
    加载原始 Qwen2-0.5B 和蒸馏训练后的学生模型，
    你输入一个概念，两个模型同时生成回答，直观对比蒸馏效果。

【对比对象】
    原始 Qwen2-0.5B（基线）：没经过蒸馏，来自 HuggingFace 的预训练权重
    蒸馏学生（我们的模型）：训练后的 distilled_student

【怎么看结果】
    对于同一个问题"请解释一下{概念}。"：
    - 蒸馏学生的回答是否比原始模型更清晰、更有条理？
    - 蒸馏学生是否学到了三角度格式？
    - 蒸馏学生的回答是否更接近训练数据的风格？

【运行方式】
    conda activate pytorch_env
    python 03_inference_compare.pyliub

    启动后输入概念名，两个模型各自生成回答。
    输入 q 退出。
================================================================================
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ================================================================
# 配置
# ================================================================

OUTPUT_DIR = "output"

# 原始 Qwen2-0.5B（基线）：HuggingFace 上的公开权重，未经任何微调
ORIGINAL_MODEL = "Qwen/Qwen2-0.5B-Instruct"

# 蒸馏训练后的学生模型：兼容两种保存方式
# 1) 旧版：输出目录 output/distilled_student/
# 2) 现版：权重文件 output/distilled_student.pt + tokenizer 目录 output/distilled_student_tokenizer/
DISTILLED_MODEL = os.path.join(OUTPUT_DIR, "distilled_student")
DISTILLED_WEIGHTS = os.path.join(OUTPUT_DIR, "distilled_student.pt")
DISTILLED_TOKENIZER = os.path.join(OUTPUT_DIR, "distilled_student_tokenizer")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 推理采样参数
MAX_NEW_TOKENS = 512      # 三句话中文回答 + 留余量
TEMPERATURE = 0.6         # 采样温度
TOP_P = 0.85              # nucleus sampling 的累积概率阈值

print(f"设备: {DEVICE}")


# ================================================================
# 推理函数
# ================================================================

@torch.no_grad()  # 推理不需要梯度，省显存
def generate(model, tokenizer, prompt: str) -> str:
    """
    用模型生成回答。

    流程：
        1. 把问句（prompt）编码成数字（token IDs）
        2. 调用 model.generate() 让模型自回归续写
        3. 把输出的数字解码回文字
        4. 去掉问句部分，只保留模型生成的内容

    参数：
        model:     HuggingFace 模型
        tokenizer: 对应的 tokenizer
        prompt:    用户提问，如"请解释一下自由。"
    返回：
        模型生成的回答（不含问句部分）
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,                # 采样模式（不是贪婪解码）
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full[len(prompt):].strip()
    return answer


# ================================================================
# 模型加载
# ================================================================

def load_model(name_or_path: str):
    """
    加载 HuggingFace 模型和对应的 tokenizer。

    参数：
        name_or_path: HF 模型名（自动下载）或本地路径
    返回：
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True,
                                              local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, trust_remote_code=True, torch_dtype=torch.float16,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def load_distilled_student():
    """兼容当前训练脚本保存的权重格式。"""
    if os.path.isdir(DISTILLED_MODEL):
        return load_model(DISTILLED_MODEL)

    if os.path.exists(DISTILLED_WEIGHTS) and os.path.isdir(DISTILLED_TOKENIZER):
        tokenizer = AutoTokenizer.from_pretrained(DISTILLED_TOKENIZER, trust_remote_code=True,
                                                  local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            ORIGINAL_MODEL, trust_remote_code=True, torch_dtype=torch.float16,
            local_files_only=True,
        ).to(DEVICE)
        state_dict = torch.load(DISTILLED_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, tokenizer

    return None, None


# ================================================================
# 主交互循环
# ================================================================

def main():
    # 1. 加载模型
    print(f"\n加载原始模型: {ORIGINAL_MODEL}")
    model_orig, tok_orig = load_model(ORIGINAL_MODEL)
    print("  ✓ 原始模型加载完成")

    print(f"\n加载蒸馏学生: {DISTILLED_MODEL} 或 {DISTILLED_WEIGHTS}")
    model_distill, tok_distill = load_distilled_student()
    if model_distill is not None:
        print("  ✓ 蒸馏学生加载完成")
    else:
        print("  ✗ 未找到蒸馏学生模型，请先运行 02_train_distillation.py")
        model_distill = None

    # 2. 交互循环
    print(f"\n{'='*60}")
    print("输入概念名（如 自由、递归、熵），输入 q 退出")
    print(f"{'='*60}")

    while True:
        concept = input("\n概念名 > ").strip()
        if concept.lower() in ("q", "quit", "exit"):
            break
        if not concept:
            continue

        prompt = f"请解释一下{concept}。"

        # 原始模型生成
        ans_orig = generate(model_orig, tok_orig, prompt)
        print(f"\n【原始 Qwen2-0.5B】")
        print(ans_orig)

        # 蒸馏学生生成
        if model_distill:
            ans_distill = generate(model_distill, tok_distill, prompt)
            print(f"\n【蒸馏学生】")
            print(ans_distill)
        else:
            print(f"\n【蒸馏学生】模型未加载")


if __name__ == "__main__":
    main()
