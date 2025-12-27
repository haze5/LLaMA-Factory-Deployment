#!/usr/bin/env python3
"""
意图识别数据集下载脚本
支持下载 ATIS、SNIPS 和 CrossWOZ 数据集
"""

import os
import json
import requests
from typing import Dict, Any
import zipfile
from pathlib import Path

# 数据集下载链接
DATASETS = {
    "atis": {
        "url": "https://github.com/yvchen/JointSLU/raw/master/data/atis.train.w-intent.iob",
        "filename": "atis_train.json",
        "description": "ATIS 航空领域意图识别数据集"
    },
    "snips": {
        "url": "https://github.com/sonos/nlu-benchmark/releases/download/0.3.0/snips.tar.gz",
        "filename": "snips.tar.gz",
        "description": "SNIPS 智能助手意图识别数据集"
    },
    "crosswoz": {
        "url": "https://huggingface.co/datasets/thu-coai/CrossWOZ/resolve/main/CrossWOZ/data.json",
        "filename": "crosswoz.json",
        "description": "CrossWOZ 中文多领域对话数据集"
    }
}

def download_file(url: str, filepath: str) -> bool:
    """下载文件"""
    try:
        print(f"正在下载: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"下载完成: {filepath}")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def extract_archive(filepath: str, extract_to: str) -> bool:
    """解压文件"""
    try:
        if filepath.endswith('.tar.gz'):
            import tarfile
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            print(f"不支持的压缩格式: {filepath}")
            return False
        
        print(f"解压完成: {extract_to}")
        return True
        
    except Exception as e:
        print(f"解压失败: {e}")
        return False

def create_sample_dataset(raw_dir: str) -> None:
    """创建示例数据集用于快速测试"""
    
    # 意图识别示例数据
    sample_data = {
        "conversations": [
            {
                "from": "human",
                "value": "帮我查询明天北京到上海的航班"
            },
            {
                "from": "gpt", 
                "value": "query_flight"
            }
        ]
    }
    
    examples = [
        ("帮我查询明天北京到上海的航班", "query_flight"),
        ("预订一张去深圳的机票", "book_flight"),
        ("取消我的航班预订", "cancel_flight"),
        ("今天天气怎么样", "query_weather"),
        ("明天早上8点提醒我开会", "set_reminder"),
        ("播放周杰伦的歌", "play_music"),
        ("给张三发个消息", "send_message"),
        ("帮我打电话给李四", "make_call"),
        ("搜索一下人工智能的资料", "search_info"),
        ("导航到最近的加油站", "navigation"),
        ("预订一家意大利餐厅", "restaurant_booking"),
        ("买两张电影票", "movie_booking"),
        ("预订一个酒店房间", "hotel_booking"),
        ("在网上买一本书", "shopping"),
        ("把这句话翻译成英文", "translation"),
        ("帮我计算25乘以60", "calculator"),
        ("设置一个10分钟的计时器", "timer"),
        "明天早上6点叫醒我", "alarm"),
        ("记一下这个笔记", "note"),
        ("查看我明天的日程", "calendar"),
        ("写一封邮件给老板", "email"),
        ("删除这个文件", "file_management"),
        ("打开系统设置", "system_control"),
        ("你好，今天天气不错", "greeting"),
        ("再见，下次聊", "goodbye"),
        ("随便聊聊吧", "general_chat")
    ]
    
    # 生成训练数据
    train_data = []
    eval_data = []
    
    for i, (text, intent) in enumerate(examples):
        data_point = {
            "conversations": [
                {"from": "human", "value": f"请识别以下用户意图：{text}"},
                {"from": "gpt", "value": intent}
            ]
        }
        
        if i < len(examples) * 0.8:
            train_data.append(data_point)
        else:
            eval_data.append(data_point)
    
    # 保存示例数据
    os.makedirs(f"{raw_dir}/processed", exist_ok=True)
    
    with open(f"{raw_dir}/processed/intent_train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{raw_dir}/processed/intent_eval.json", 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"已创建示例数据集：{len(train_data)} 条训练样本，{len(eval_data)} 条验证样本")

def main():
    """主函数"""
    raw_dir = "/workspace/intent_recognition/data/raw"
    
    # 创建目录
    os.makedirs(raw_dir, exist_ok=True)
    
    print("=== 意图识别数据集下载脚本 ===")
    print(f"数据将保存到: {raw_dir}")
    print()
    
    # 询问用户下载哪些数据集
    print("可用数据集:")
    for i, (name, info) in enumerate(DATASETS.items(), 1):
        print(f"{i}. {name.upper()}: {info['description']}")
    
    print("4. 创建示例数据集（用于快速测试）")
    
    choice = input("请选择 (1-4，多个用逗号分隔): ").strip()
    
    if not choice:
        print("未选择任何数据集，将创建示例数据集...")
        create_sample_dataset(raw_dir)
        return
    
    selected = []
    for c in choice.split(','):
        c = c.strip()
        if c.isdigit():
            idx = int(c) - 1
            if 0 <= idx < len(DATASETS):
                selected.append(list(DATASETS.keys())[idx])
            elif idx == 3:  # 示例数据集
                create_sample_dataset(raw_dir)
                return
    
    # 下载数据集
    for name in selected:
        dataset_info = DATASETS[name]
        url = dataset_info["url"]
        filename = dataset_info["filename"]
        filepath = os.path.join(raw_dir, filename)
        
        if download_file(url, filepath):
            # 如果是压缩文件，解压
            if filename.endswith(('.tar.gz', '.zip')):
                extract_to = os.path.join(raw_dir, name)
                extract_archive(filepath, extract_to)
        else:
            print(f"下载 {name} 数据集失败，跳过...")
    
    print("\n=== 下载完成 ===")
    print("请运行预处理脚本：python preprocess_data.py")

if __name__ == "__main__":
    main()