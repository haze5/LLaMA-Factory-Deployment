# AI Assistant 项目

本项目包含两个核心模块：
- **LLaMA-Factory**: 大模型微调框架，提供零代码微调百余种大模型的完整解决方案
- **RAG 对话机器人**: 基于检索增强生成的个人对话机器人，使用自己的聊天记录作为知识库

## 🚀 项目特性

### LLaMA-Factory 模块

- **零代码微调**：提供 Web UI 和命令行两种操作方式
- **支持百余种大模型**：包括 LLaMA、Qwen、DeepSeek 等
- **多种微调方法**：支持 LoRA、QLoRA、全参数微调等
- **多模态支持**：支持文本、图像、音频、视频等多模态数据
- **分布式训练**：支持多 GPU、多节点训练
- **量化训练**：支持 INT8、INT4、GPTQ、AWQ 等量化方案

## 📋 环境要求

- **Python**: 3.11.1
- **CUDA**: 12.1 (推荐 GPU 环境)
- **内存**: 建议 16GB+ RAM
- **存储**: 建议 50GB+ 可用空间

## 🛠️ 快速开始

### 环境安装

```bash
# 克隆项目
git clone <your-repo-url>
cd <project-name>

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装 LLaMA-Factory 依赖
pip install -r requirements.txt

# 安装 RAG 额外依赖（如需使用 RAG 功能）
pip install -r rag_requirements.txt
```

### 2. 启动方式

#### Web UI 模式（推荐）

```bash
cd LLaMA-Factory
python src/webui.py
```

访问地址：`http://127.0.0.1:7860`

#### API 服务模式

```bash
cd LLaMA-Factory
python src/api.py
```

API 文档：`http://localhost:8000/docs`

#### 命令行训练模式

```bash
# 使用配置文件训练
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# 或直接使用参数训练
llamafactory-cli train \
  model_name_or_path=models/Qwen2.5-7B-Instruct \
  dataset=identity,alpaca_en_demo \
  finetuning_type=lora \
  output_dir=saves/qwen2.5-7b/lora/sft
```

## 📁 项目结构

```
├── LLaMA-Factory/           # 大模型微调框架
│   ├── src/                 # 源代码
│   │   ├── api.py          # API 服务入口
│   │   ├── train.py        # 训练入口
│   │   ├── webui.py        # Web UI 入口
│   │   └── llamafactory/   # 核心模块
│   ├── data/               # 数据集和配置
│   ├── examples/           # 示例配置文件
│   ├── scripts/            # 工具脚本
│   └── requirements.txt    # 依赖列表
├── rag/                    # RAG 对话机器人模块
│   ├── rag_pipeline.py      # RAG 管道
│   ├── document_loader.py   # 文档加载器
│   ├── text_splitter.py     # 文本分块器
│   ├── embeddings.py        # 向量嵌入
│   ├── vector_store.py      # 向量数据库
│   ├── knowledge/           # 知识库目录
│   └── README.md
├── intent_recognition/     # 意图识别模块
│   ├── config/              # 配置文件
│   ├── data/                # 数据集
│   ├── scripts/             # 工具脚本
│   └── README.md
├── models/                 # 预训练模型
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/
│   └── Qwen2.5-7B-Instruct/
├── frp/                    # 内网穿透工具
├── chat_bot.py            # RAG 终端入口
├── requirements.txt        # LLaMA-Factory 依赖
├── rag_requirements.txt   # RAG 额外依赖
└── README.md
```

## 🎯 预装模型

项目已预装以下模型：

1. **DeepSeek-R1-Distill-Qwen-1.5B** (3.31 GB)
   - 路径：`models/DeepSeek-R1-Distill-Qwen-1.5B/`
   - 适合快速测试和原型开发

2. **Qwen2.5-7B-Instruct** (14.18 GB)
   - 路径：`models/Qwen2.5-7B-Instruct/`
   - 适合生产环境使用

## 🔧 配置说明

### LLaMA-Factory 配置

### 环境配置

主要配置文件：`LLaMA-Factory/.env.local`

```env
# API 服务配置
API_HOST=0.0.0.0
API_PORT=8000

# Web UI 配置
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# 分布式训练配置
MASTER_ADDR=localhost
MASTER_PORT=29500
```

### 数据集配置

数据集配置文件：`LLaMA-Factory/data/dataset_info.json`

```json
{
  "train": {
    "file_name": "train_change.json",
    "formatting": "sharegpt"
  },
  "eval": {
    "file_name": "eval_change.json", 
    "formatting": "sharegpt"
  }
}
```

### RAG 配置

RAG 系统的配置可以在以下文件中调整：

| 文件 | 说明 |
|------|------|
| `chat_bot.py` | 主入口，配置模型路径和目录 |
| `rag/rag_pipeline.py` | RAG 管道，调整 top_k、chunk_size 等参数 |
| `rag/embeddings.py` | 嵌入模型选择 |
| `rag/vector_store.py` | 向量数据库配置 |

默认配置：
- 检索数量 (top_k): 3
- 文档块大小 (chunk_size): 500
- 文档块重叠 (chunk_overlap): 50
- 嵌入模型: BAAI/bge-small-zh-v1.5

## 🚀 内网穿透

如果需要从外网访问 Web UI，可以使用项目内置的 FRP 工具：

### 服务器端配置

```bash
cd frp/frp_0.65.0_linux_amd64/
./frps -c frps.toml
```

### 客户端配置

```bash
cd frp/frp_0.65.0_linux_amd64/
./frpc -c frpc.toml
```

## 🎨 功能特性

### 支持的微调方法

- **LoRA**: 低秩适应，参数高效微调
- **QLoRA**: 量化 LoRA，内存友好
- **全参数微调**: 完整模型训练
- **DPO**: 直接偏好优化
- **PPO**: 近端策略优化
- **KTO**: 卡尼曼-特沃斯基优化

### 支持的模型架构

- LLaMA 系列 (LLaMA-2, LLaMA-3, LLaMA-4)
- Qwen 系列
- DeepSeek 系列
- Mixtral
- Baichuan
- ChatGLM
- 其他 100+ 模型

### 高级功能

- **多模态支持**: 图文、音视频处理
- **分布式训练**: 支持多 GPU、多节点
- **量化训练**: INT8、INT4、GPTQ、AWQ
- **内存优化**: Flash Attention、Gradient Checkpointing
- **推理加速**: vLLM、SGLang 后端

### RAG 对话机器人模块
- **知识库检索**：从个人对话历史中快速找到相关信息
- **向量数据库**：使用 ChromaDB 存储和检索向量
- **智能分块**：保持语义完整的文档分割
- **中文优化**：BGE 嵌入模型，针对中文优化
- **灵活部署**：支持终端和 Web UI 两种模式

## 📚 使用示例

### 1. LoRA 微调示例

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

### 2. 使用本地模型

```bash
llamafactory-cli train \
  model_name_or_path=models/Qwen2.5-7B-Instruct \
  dataset=identity,alpaca_en_demo \
  finetuning_type=lora \
  output_dir=saves/qwen2.5-7b/lora/sft
```

### 3. 模型推理

```bash
llamafactory-cli chat \
  model_name_or_path=saves/qwen2.5-7b/lora/sft \
  template=qwen
```

---

## 🤖 RAG 对话机器人

### 快速启动

```bash
# 运行终端版对话机器人
python chat_bot.py
```

### 使用自己的聊天记录

1. 将聊天记录放入 `rag/knowledge/` 目录
2. 删除旧向量数据库：`rm -rf rag/vector_db/chroma`
3. 重新运行：`python chat_bot.py`

### RAG 架构

```
用户查询
    ↓
[嵌入模型] → 向量表示
    ↓
[向量数据库] → 检索 Top-K 文档
    ↓
[大语言模型] → 基于上下文生成回答
    ↓
返回答案
```

## 🛠️ 开发工具

项目提供了完整的开发工具链：

```bash
# 代码格式化和检查
ruff format .
ruff check .

# 运行测试
pytest

# 构建包
python -m build

# 预提交钩子
pre-commit install
```

## 📖 相关文档

### LLaMA-Factory
- [LLaMA-Factory 官方文档](https://llamafactory.readthedocs.io/)
- [API 接口文档](http://localhost:8000/docs)
- [配置文件示例](LLaMA-Factory/examples/)

### RAG 对话机器人
- [RAG 模块说明](rag/README.md)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [BGE 嵌入模型](https://github.com/FlagOpen/FlagEmbedding)

### 意图识别
- [意图识别模块说明](intent_recognition/README.md)

- [LLaMA-Factory 官方文档](https://llamafactory.readthedocs.io/)
- [API 接口文档](http://localhost:8000/docs)
- [配置文件示例](LLaMA-Factory/examples/)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🔗 相关链接

- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Hugging Face](https://huggingface.co/)
- [PyTorch](https://pytorch.org/)

---

## 🎉 项目进展

### ✅ 已完成

| 模块 | 状态 | 描述 |
|------|------|------|
| **LLaMA-Factory** | ✅ | 大模型微调框架已部署 |
| **预装模型** | ✅ | DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-7B-Instruct |
| **RAG 系统** | ✅ | 完整的检索增强生成管道 |
| **向量数据库** | ✅ | ChromaDB 向量存储 |
| **文档加载器** | ✅ | 支持 txt, md, json 格式 |
| **文本分块器** | ✅ | 智能分割保持语义完整 |
| **嵌入模型** | ✅ | BGE 中文优化模型 |
| **意图识别** | 🚧 | 基础框架已搭建 |

### 🚧 进行中

- **聊天记录解析** - 支持微信、ChatGPT 等多种格式
- **Web UI 界面** - Gradio 网页版对话界面

### 📋 待开发

| 优先级 | 功能 | 描述 |
|--------|------|------|
| 🔴 高 | 聊天记录导入 | 批量导入个人对话历史 |
| 🔴 高 | Web UI | 网页版对话界面 |
| 🟡 中 | LLaMA-Factory API 集成 | 替代本地推理，提升速度 |
| 🟡 中 | 意图识别 + RAG 融合 | 结合提升准确性 |
| 🟢 低 | 多轮对话记忆 | 记住对话上下文 |
| 🟢 低 | 知识库管理 | 可视化管理界面 |

## 🔄 开发路线图

### 阶段 1: 知识库建设
- [x] RAG 核心框架
- [ ] 聊天记录解析脚本
- [ ] 批量导入工具

### 阶段 2: 用户界面
- [ ] Web UI (Gradio)
- [ ] 知识库管理
- [ ] 对话历史查看

### 阶段 3: 性能优化
- [ ] LLaMA-Factory API 集成
- [ ] 模型推理加速
- [ ] 缓存机制

### 阶段 4: 功能增强
- [ ] 意图识别集成
- [ ] 多轮对话记忆
- [ ] 个性化风格微调

---

**注意**: 首次运行前请确保已正确安装所有依赖，并根据硬件配置调整相关参数。