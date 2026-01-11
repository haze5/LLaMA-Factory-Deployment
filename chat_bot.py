#!/usr/bin/env python3
"""
RAG 对话机器人 - 主入口
基于 DeepSeek-R1-Distill-Qwen-1.5B 的检索增强生成
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.rag_pipeline import RAGPipeline


def main():
    """主函数"""
    print("=" * 50)
    print("  RAG 对话机器人")
    print("=" * 50)
    print()

    # 初始化 RAG 管道
    print("正在初始化 RAG 系统...")
    rag = RAGPipeline(
        model_path="/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B",
        knowledge_dir="/workspace/rag/knowledge",
        vector_db_dir="/workspace/rag/vector_db/chroma",
        top_k=3
    )
    print("✅ RAG 系统初始化完成！\n")

    # 交互式对话
    print("开始对话（输入 'quit' 或 'exit' 退出）")
    print("-" * 50)

    while True:
        try:
            # 获取用户输入
            user_input = input("\n你: ").strip()

            # 退出命令
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n再见！👋")
                break

            if not user_input:
                continue

            # 调用 RAG 生成回答
            print("\n正在思考...")
            response = rag.chat(user_input)
            print(f"\nAI: {response}")

        except KeyboardInterrupt:
            print("\n\n再见！👋")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            continue


if __name__ == "__main__":
    main()
