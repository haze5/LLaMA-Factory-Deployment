# RAG 对话机器人

基于检索增强生成（RAG）技术的个人对话机器人，使用自己的聊天记录作为知识库。

## 项目结构

```
/workspace/
├── chat_bot.py              # 主入口文件
├── rag/                     # RAG 核心模块
│   ├── __init__.py
│   ├── rag_pipeline.py      # RAG 管道（主类）
│   ├── document_loader.py   # 文档加载器
│   ├── text_splitter.py     # 文本分块器
│   ├── embeddings.py        # 嵌入模型
│   └── vector_store.py      # 向量数据库
├── knowledge/               # 知识库目录
│   ├── raw/                 # 原始文档（你的聊天记录）
│   └── processed/           # 处理后的文档
└── vector_db/               # 向量数据库存储
    └── chroma/              # ChromaDB 数据
```

## 快速开始

### 1. 安装依赖

```bash
cd /workspace
pip install -r rag_requirements.txt
```

### 2. 运行机器人

```bash
python chat_bot.py
```

### 3. 开始对话

```
你: 我想学 Python，应该怎么开始？
AI: [基于知识库的回答]
```

## 使用自己的聊天记录

### 步骤 1: 准备聊天记录

将你的聊天记录保存到 `knowledge/raw/` 目录，支持以下格式：

- `.txt` - 纯文本
- `.md` - Markdown
- `.json` - JSON 格式（LLaMA-Factory ShareGPT 格式）

### 步骤 2: 更新知识库

删除旧的向量数据库，让系统重新构建：

```bash
rm -rf vector_db/chroma
python chat_bot.py
```

系统会自动重新扫描 `knowledge/` 目录并构建索引。

### 步骤 3: 对话测试

开始与机器人对话，它会基于你的聊天记录回答问题。

## 技术架构

```
用户查询
    ↓
[嵌入模型] → 转换为向量
    ↓
[向量数据库] → 检索相似文档 (Top-K)
    ↓
[大语言模型] → 基于检索到的上下文生成回答
    ↓
返回答案
```

## 组件说明

### RAGPipeline
- 核心管道类，协调各组件工作
- 负责模型加载、知识库更新、检索和生成

### DocumentLoader
- 加载多种格式的文档
- 支持 txt, md, json 等格式

### TextSplitter
- 将长文档分割成小块
- 支持多种分隔符，保持语义完整

### EmbeddingModel
- 使用 sentence-transformers 生成文本向量
- 支持 BGE 等中文优化模型

### VectorStore
- 使用 ChromaDB 存储和检索向量
- 支持持久化和元数据过滤

## 后续优化方向

1. **数据处理**: 添加聊天记录解析脚本（微信、Telegram 等）
2. **嵌入模型**: 尝试更大的嵌入模型提高检索质量
3. **重排序**: 添加重排序模块，提升检索准确性
4. **混合检索**: 结合关键词检索和向量检索
5. **缓存机制**: 缓存常见查询的答案
6. **记忆机制**: 记住对话历史，支持多轮对话
7. **微调集成**: 结合意图识别和微调模型
