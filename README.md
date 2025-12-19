# LLaMA-Factory 部署项目

这是一个基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的大模型微调框架部署项目，提供了零代码微调百余种大模型的完整解决方案。

## 🚀 项目特性

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

### 1. 环境安装

```bash
# 克隆项目
git clone <your-repo-url>
cd <project-name>

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
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
├── LLaMA-Factory/           # 主项目目录
│   ├── src/                 # 源代码
│   │   ├── api.py          # API 服务入口
│   │   ├── train.py        # 训练入口
│   │   ├── webui.py        # Web UI 入口
│   │   └── llamafactory/   # 核心模块
│   ├── data/               # 数据集和配置
│   ├── examples/           # 示例配置文件
│   ├── scripts/            # 工具脚本
│   └── requirements.txt    # 依赖列表
├── models/                 # 预训练模型
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/
│   └── Qwen2.5-7B-Instruct/
├── frp/                    # 内网穿透工具
└── README.md              # 项目说明
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

**注意**: 首次运行前请确保已正确安装所有依赖，并根据硬件配置调整相关参数。