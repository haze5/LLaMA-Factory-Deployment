"""
RAG 管道核心模块
整合检索、排序、生成等组件
"""

import os
from pathlib import Path
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .document_loader import DocumentLoader
from .text_splitter import TextSplitter


class RAGPipeline:
    """RAG 检索增强生成管道"""

    def __init__(
        self,
        model_path: str,
        knowledge_dir: str,
        vector_db_dir: str,
        top_k: int = 3,
        embedding_model: str = "BAAI/bge-small-zh-v1.5"
    ):
        """
        初始化 RAG 管道

        Args:
            model_path: 大模型路径
            knowledge_dir: 知识库目录
            vector_db_dir: 向量数据库目录
            top_k: 检索返回的文档数量
            embedding_model: 嵌入模型名称
        """
        self.model_path = model_path
        self.knowledge_dir = Path(knowledge_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.top_k = top_k

        print(f"[1/5] 初始化嵌入模型: {embedding_model}")
        self.embedding_model = EmbeddingModel(model_name=embedding_model)

        print(f"[2/5] 初始化向量数据库: {vector_db_dir}")
        self.vector_store = VectorStore(
            persist_dir=vector_db_dir,
            embedding_model=self.embedding_model
        )

        print(f"[3/5] 初始化文档加载器和分块器")
        self.doc_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        print(f"[4/5] 检查并更新知识库")
        self._update_knowledge_base()

        print(f"[5/5] 加载大语言模型: {model_path}")
        self._load_llm()

        print("\nRAG 管道初始化完成 ✓")

    def _load_llm(self):
        """加载大语言模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def _update_knowledge_base(self):
        """更新知识库"""
        # 检查是否有新的知识文档
        if not self.knowledge_dir.exists():
            print(f"  知识库目录不存在，创建示例知识库...")
            self._create_sample_knowledge()
            return

        # 检查向量数据库是否需要重建
        if not self.vector_store.is_initialized():
            print(f"  向量数据库未初始化，开始构建索引...")
            self._build_vector_index()
        else:
            print(f"  使用现有向量数据库")

    def _create_sample_knowledge(self):
        """创建示例知识库"""
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        sample_text = """
# Python 学习指南

## 基础语法
Python 是一种简洁易学的编程语言。它的语法清晰，适合初学者。

### 变量和数据类型
```python
# 变量赋值
name = "张三"
age = 25
is_student = True

# 数据类型转换
num_str = str(123)
str_num = int("456")
```

### 控制流程
```python
# 条件语句
if age >= 18:
    print("成年人")
else:
    print("未成年")

# 循环
for i in range(5):
    print(i)

# while 循环
count = 0
while count < 3:
    print("Hello")
    count += 1
```

## 函数定义
```python
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b
```

## 常用数据结构

### 列表
```python
fruits = ["apple", "banana", "orange"]
fruits.append("grape")
fruits[0] = "pear"
```

### 字典
```python
person = {
    "name": "张三",
    "age": 25,
    "city": "北京"
}
person["age"] = 26
```

## 文件操作
```python
# 读取文件
with open("file.txt", "r") as f:
    content = f.read()

# 写入文件
with open("file.txt", "w") as f:
    f.write("Hello World")
```
        """.strip()

        # 保存示例文档
        sample_file = self.knowledge_dir / "python_guide.md"
        sample_file.write_text(sample_text, encoding='utf-8')
        print(f"  已创建示例文档: {sample_file}")

        # 添加更多示例
        chat_history = """
# 对话历史示例

## 编程学习对话

用户: 我是个编程小白，想学 Python，应该怎么开始？
AI: 恭喜你选择了 Python！作为入门语言，Python 是最好的选择之一。建议你从以下几个方面开始：

1. 安装 Python 环境：从 python.org 下载安装包
2. 学习基础语法：变量、数据类型、控制流程
3. 动手实践：从简单的打印语句开始
4. 学习函数和模块：理解代码组织和复用
5. 小项目实践：做一个计算器、猜数字游戏等

用户: 有没有推荐的学习资源？
AI: 当然！这里有一些高质量的学习资源：

在线教程：
- 廖雪峰的 Python 教程（中文，非常推荐）
- Python 官方文档（权威）
- Real Python（英文，深入浅出）

视频课程：
- B 站上的 Python 零基础教程
- Coursera 上的 Python 课程

练习平台：
- LeetCode（算法练习）
- HackerRank（编程挑战）
- 牛客网（中文练习平台）

用户: 学到什么程度可以开始做项目？
AI: 这个问题很好！一般来说，掌握以下内容就可以开始小项目了：

1. 基础语法（变量、数据类型、控制流程）✅
2. 函数定义和调用 ✅
3. 列表、字典等基本数据结构 ✅
4. 文件读写操作 ✅
5. 简单的模块导入 ✅

建议的第一个项目：
- 计算器：练习函数和逻辑
- 猜数字游戏：练习循环和条件
- 记事本程序：练习文件操作
- 天气查询：练习调用 API

用户: 学完基础之后呢？
AI: 学完基础后，你可以选择一个方向深入：

Web 开发：
- 学习 Flask 或 Django 框架
- 了解 HTML/CSS/JavaScript
- 做一个博客或小型网站

数据分析：
- 学习 NumPy、Pandas
- 学习 Matplotlib 数据可视化
- 分析公开数据集

人工智能：
- 学习机器学习基础
- 接触 TensorFlow 或 PyTorch
- 尝试简单的深度学习项目

记住：选择一个你感兴趣的方向深入，比什么都学一点点要好！
        """.strip()

        history_file = self.knowledge_dir / "chat_history.md"
        history_file.write_text(chat_history, encoding='utf-8')
        print(f"  已创建示例文档: {history_file}")

    def _build_vector_index(self):
        """构建向量索引"""
        print(f"  扫描知识库目录: {self.knowledge_dir}")

        all_documents = []
        all_metadata = []

        # 遍历知识库目录
        for file_path in self.knowledge_dir.rglob("*"):
            if file_path.is_file():
                # 跳过隐藏文件
                if file_path.name.startswith('.'):
                    continue

                # 加载文档
                try:
                    documents = self.doc_loader.load(str(file_path))
                    for doc in documents:
                        # 添加元数据
                        metadata = {
                            "source": str(file_path.relative_to(self.knowledge_dir)),
                            "file_type": file_path.suffix
                        }
                        all_documents.append(doc)
                        all_metadata.append(metadata)
                        print(f"    加载: {file_path.relative_to(self.knowledge_dir)}")

                except Exception as e:
                    print(f"    跳过 {file_path.name}: {e}")
                    continue

        if not all_documents:
            print(f"  未找到有效文档")
            return

        # 分块处理
        print(f"  分块处理 {len(all_documents)} 个文档...")
        chunks = []
        chunk_metadatas = []

        for doc, meta in zip(all_documents, all_metadata):
            doc_chunks = self.text_splitter.split(doc)
            chunks.extend(doc_chunks)
            chunk_metadatas.extend([meta] * len(doc_chunks))

        print(f"  生成 {len(chunks)} 个文本块的向量...")

        # 生成嵌入向量并存储
        embeddings = self.embedding_model.embed_documents(chunks)

        self.vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadatas
        )

        self.vector_store.persist()
        print(f"  向量索引构建完成")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回的文档数量

        Returns:
            相关文档列表
        """
        top_k = top_k or self.top_k

        # 生成查询向量
        query_embedding = self.embedding_model.embed_query(query)

        # 检索相似文档
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results

    def generate(self, query: str, context: str) -> str:
        """
        基于上下文生成回答

        Args:
            query: 用户查询
            context: 检索到的上下文

        Returns:
            生成的回答
        """
        # 构建提示
        prompt = f"""你是一个有帮助的 AI 助手。请根据以下参考信息回答用户的问题。

参考信息：
{context}

用户问题：{query}

请提供准确、有帮助的回答："""

        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # 减小输入长度
        ).to(self.model.device)

        input_length = inputs['input_ids'].shape[1]
        print(f"    [调试] 输入 token 数: {input_length}")

        # 生成回答
        print(f"    [调试] 开始生成回答...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,  # 减小生成长度
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        print(f"    [调试] 生成完成，输出 token 数: {outputs.shape[1] - input_length}")

        # 解码输出
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # 提取生成的部分（去掉 prompt）
        if prompt in response:
            response = response.replace(prompt, "").strip()

        return response

    def chat(self, user_input: str) -> str:
        """
        对话接口

        Args:
            user_input: 用户输入

        Returns:
            AI 回答
        """
        print(f"    [调试] 检索相关文档...")

        # 检索相关文档
        relevant_docs = self.retrieve(user_input)

        print(f"    [调试] 检索到 {len(relevant_docs)} 条相关文档")

        if not relevant_docs:
            # 没有检索到文档，直接生成回答
            print(f"    [调试] 无检索结果，使用通用回答")
            return self.generate(user_input, "没有找到相关的参考信息。")

        # 组合上下文
        context = "\n\n".join([
            f"[参考 {i+1}]: {doc[:200]}..."  # 限制每条文档长度
            for i, doc in enumerate(relevant_docs)
        ])

        # 生成回答
        return self.generate(user_input, context)
