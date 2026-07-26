"""
太原理工大学创新学社 · RAG 智能问答系统（CLI 入口）。

用法:
    python main.py                      # 交互式问答（自动检查/构建索引）
    python main.py --rebuild            # 强制重建向量索引（知识库修改后）
    python main.py -q "有哪些食堂"       # 单次问答，输出后退出
    python main.py --rebuild -q "..."   # 重建索引 + 单次问答
    python main.py -k 10 -q "..."       # 指定检索 top-K=10

交互模式命令:
    /exit       退出
    /sources    查看上次检索的来源文档
    Ctrl+C      强制退出
"""

import sys
import argparse
from src.rag_pipeline import build_index, query


def interactive_mode():
    """交互式问答循环。

    为什么用简单的 while True + input() 而非复杂的 REPL 框架（如 cmd.Cmd、prompt_toolkit）：
    - RAG 场景不需要历史对话管理：每次查询独立检索，不依赖上一轮上下文
    - 减少外部依赖：prompt_toolkit 需要额外安装，Windows 上偶尔有编码问题
    - input() 足够用：支持中文输入、退格编辑，符合用户的最小惊讶原则

    /sources 命令的存在理由：
    用户看到 LLM 回答后可能怀疑准确性（"这真的是知识库里的内容吗？"），
    /sources 让用户直接看到原始检索结果，验证 LLM 是否基于资料回答。
    这是 RAG 相比纯 LLM 的核心信任机制。
    """
    print("=" * 60)
    print("  太原理工大学创新学社 · RAG 问答系统")
    print("=" * 60)
    print("输入问题即可查询，输入 /exit 退出，输入 /sources 查看上次来源\n")

    # 保存最近一次检索的来源，供 /sources 命令使用
    # 不保存到内存以外的原因是 sources 只在当前会话有意义，
    # 退出后无需保留，下次启动自然清空
    last_sources = []

    while True:
        try:
            user_input = input("\n你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            # EOFError: 用户按 Ctrl+Z (Windows) 或 Ctrl+D (Unix)
            # KeyboardInterrupt: 用户按 Ctrl+C
            # 两种都是正常的退出意图，友好告别而非打印 traceback
            print("\n再见！")
            break

        if not user_input:
            continue  # 空输入跳过，不浪费 API 调用

        if user_input.lower() == "/exit":
            print("再见！")
            break

        if user_input.lower() == "/sources":
            if not last_sources:
                print("还没有查询过，没有来源记录")
            else:
                print("\n上次查询的来源: ")
                for i, s in enumerate(last_sources, 1):
                    print(f"\n  [{i}] 章节: {s['source']} (距离: {s['distance']})")
                    # 不展示完整内容（可能很长），截取前 200 字符保持终端整洁
                    content_preview = s['content'][:200]
                    if len(s['content']) > 200:
                        content_preview += "..."
                    print(f"      内容: {content_preview}")
            continue

        print()  # 空行分隔输入和输出，视觉上更清晰
        result = query(user_input)
        last_sources = result["sources"]

        # 打印 LLM 回答
        print(f"\n回答:\n{result['answer']}")

        # 打印参考来源（即使 answer 是错误信息也打印，方便用户自行判断）
        if result["sources"]:
            print(f"\n参考来源 ({len(result['sources'])} 条):")
            for i, s in enumerate(result["sources"], 1):
                print(f"  [{i}] {s['source']} (距离={s['distance']})")


def main():
    parser = argparse.ArgumentParser(
        description="太原理工大学创新学社 RAG 问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                      # 交互式问答
  python main.py --rebuild            # 重建向量索引
  python main.py -q "有哪些食堂"       # 单次问答
  python main.py --rebuild -q "..."   # 重建后问答
        """,
    )
    parser.add_argument(
        "--rebuild", "-r",
        action="store_true",
        help="强制重建向量索引（知识库文件修改后必须执行）",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="单次问答模式：输入问题，输出回答后退出（不进入交互循环）",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="检索返回的文档块数，覆盖 config.py 中的 RETRIEVAL_TOP_K",
    )

    args = parser.parse_args()

    # ---- 索引构建 ----
    # build_index(force=False) 在已有索引时静默跳过，
    # build_index(force=True) 无条件重建。
    # 两种模式都先于问答执行，确保无论如何向量库都处于就绪状态。

    if args.rebuild:
        build_index(force=True)
        print()

    # 如果用户既没有 --rebuild 也没有 --query（即纯交互模式），
    # 仍需确保索引存在。build_index(force=False) 在库已存在时
    # 几乎零开销（仅一次 get_or_create_collection 调用）。
    if args.query is None and not args.rebuild:
        build_index(force=False)
        print()

    # ---- 单次问答模式 ----
    # 用户指定 -q 时，输出回答后立即退出，不进入交互循环。
    # 适用于脚本调用、快速测试等场景。
    if args.query:
        result = query(args.query, top_k=args.top_k)
        print(f"\n回答:\n{result['answer']}")
        if result["sources"]:
            print(f"\n参考来源 ({len(result['sources'])} 条):")
            for i, s in enumerate(result["sources"], 1):
                print(f"  [{i}] {s['source']} (距离={s['distance']})")
        return

    # ---- 交互模式 ----
    # 既没有 -q 也不是纯 --rebuild → 进入交互循环
    if not args.rebuild:
        interactive_mode()


if __name__ == "__main__":
    main()
