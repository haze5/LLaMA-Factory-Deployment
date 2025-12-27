# 意图识别训练项目

基于 LLaMA-Factory 的意图识别模型训练项目，专门用于训练高效、轻量的意图识别模型。

## 🎯 项目目标

- **主要目标**：训练专门用于意图识别的轻量级模型
- **应用场景**：智能客服、语音助手、任务自动化
- **性能指标**：准确率 > 90%，响应时间 < 100ms
- **部署要求**：支持 CPU/GPU 推理，内存占用 < 2GB

## 🤖 模型选择

### 推荐模型：DeepSeek-R1-Distill-Qwen-1.5B
- **路径**：`/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B/`
- **大小**：3.31 GB
- **优势**：轻量级、中文理解能力强、已预装

### 备选模型：Qwen2.5-7B-Instruct
- **路径**：`/workspace/models/Qwen2.5-7B-Instruct/`
- **大小**：14.18 GB
- **适用场景**：需要更高精度的生产环境

## 📊 数据集

### 1. ATIS 数据集（航空领域）
- **样本数**：5,000 训练 + 1,000 测试
- **意图类型**：26 种
- **特点**：标准基准，便于对比

### 2. SNIPS 数据集（智能助手）
- **样本数**：13,084 训练
- **意图类型**：7 种核心意图
- **特点**：贴近实际应用

### 3. CrossWOZ 数据集（中文多领域）
- **样本数**：5K 对话，30K+ 语句
- **领域**：餐饮、电影、酒店等 5 个领域
- **特点**：中文场景覆盖

## 🛠️ 训练流程

### 阶段一：基础指令微调
```bash
# 使用 LoRA 微调
llamafactory-cli train config/model_config.yaml
```

### 阶段二：领域自适应
```bash
# 使用 QLoRA 微调，降低显存需求
llamafactory-cli train config/training_config.yaml
```

### 阶段三：推理测试
```bash
# 测试模型效果
llamafactory-cli chat model_name_or_path=outputs/models/intent_recognition_lora
```

## 📁 项目结构

```
intent_recognition/
├── README.md                    # 项目说明
├── config/                      # 配置文件
│   ├── model_config.yaml        # 模型配置
│   ├── dataset_config.yaml      # 数据集配置
│   └── training_config.yaml     # 训练配置
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后数据
│   └── dataset_info.json        # LLaMA-Factory 数据集配置
├── scripts/                     # 脚本目录
│   ├── download_dataset.py      # 数据下载
│   ├── preprocess_data.py       # 数据预处理
│   └── evaluate.py              # 模型评估
├── outputs/                     # 输出目录
│   ├── models/                  # 训练完成的模型
│   ├── logs/                    # 训练日志
│   └── evaluations/             # 评估结果
└── examples/                    # 示例目录
    ├── inference_example.py     # 推理示例
    └── api_example.py          # API 使用示例
```

## 🚀 快速开始

### 1. 环境准备
```bash
cd /workspace/intent_recognition
source /workspace/hf_venv/bin/activate
```

### 2. 数据准备
```bash
python scripts/download_dataset.py
python scripts/preprocess_data.py
```

### 3. 开始训练
```bash
cd /workspace/LLaMA-Factory
llamafactory-cli train /workspace/intent_recognition/config/model_config.yaml
```

### 4. 模型评估
```bash
python /workspace/intent_recognition/scripts/evaluate.py
```

## 📈 评估指标

- **准确率** (Accuracy) > 90%
- **精确率** (Precision) > 85%
- **召回率** (Recall) > 85%
- **F1 分数** (F1-Score) > 85%

## 🔧 技术栈

- **框架**：LLaMA-Factory
- **模型**：DeepSeek-R1-Distill-Qwen-1.5B
- **微调方法**：LoRA、QLoRA
- **数据处理**：pandas、jieba、NLTK
- **评估工具**：sklearn、transformers

## 📝 使用示例

### Python 推理示例
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("outputs/models/intent_recognition_lora")
model = AutoModelForCausalLM.from_pretrained("outputs/models/intent_recognition_lora")

def recognize_intent(text):
    prompt = f"请识别以下用户意图：{text}\n意图："
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("意图：")[-1].strip()

# 使用示例
intent = recognize_intent("帮我查询明天北京到上海的航班")
print(f"识别意图：{intent}")
```

### API 调用示例
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intent_recognition_lora",
    "messages": [
      {"role": "user", "content": "帮我查询明天北京到上海的航班"}
    ]
  }'
```

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/new-intent`)
3. 提交更改 (`git commit -am 'Add new intent type'`)
4. 推送到分支 (`git push origin feature/new-intent`)
5. 创建 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源。