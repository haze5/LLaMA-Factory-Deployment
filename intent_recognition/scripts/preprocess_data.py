#!/usr/bin/env python3
"""
意图识别数据预处理脚本
将原始数据转换为 LLaMA-Factory 支持的 ShareGPT 格式
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import random
from pathlib import Path
import re

# 意图类型映射
INTENT_MAPPING = {
    # 航空领域
    "atis_flight": "query_flight",
    "atis_airfare": "query_flight", 
    "atis_airline": "query_flight",
    "atis_ground_service": "query_flight",
    "atis_flight_time": "query_flight",
    
    # SNIPS 意图映射
    "GetWeather": "query_weather",
    "PlayMusic": "play_music", 
    "BookRestaurant": "restaurant_booking",
    "GetPlayMusic": "play_music",
    "SearchCreativeWork": "search_info",
    "RateBook": "general_chat",
    "SearchScreeningEvent": "movie_booking",
    
    # CrossWOZ 意图映射
    "预订-酒店": "hotel_booking",
    "预订-餐厅": "restaurant_booking", 
    "预订-电影": "movie_booking",
    "查询-天气": "query_weather",
    "导航": "navigation",
    "购买-产品": "shopping"
}

def clean_text(text: str) -> str:
    """清理文本"""
    if not isinstance(text, str):
        return ""
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 去除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）《》【】\-]', '', text)
    
    return text

def normalize_intent(intent: str) -> str:
    """标准化意图名称"""
    if not isinstance(intent, str):
        return "general_chat"
    
    # 直接映射
    if intent in INTENT_MAPPING:
        return INTENT_MAPPING[intent]
    
    # 模糊匹配
    intent_lower = intent.lower()
    if any(keyword in intent_lower for keyword in ['flight', '航班', '飞机']):
        return "query_flight"
    elif any(keyword in intent_lower for keyword in ['weather', '天气', '气温']):
        return "query_weather"
    elif any(keyword in intent_lower for keyword in ['book', '预订', '订票']):
        return "book_flight"
    elif any(keyword in intent_lower for keyword in ['cancel', '取消', '退票']):
        return "cancel_flight"
    elif any(keyword in intent_lower for keyword in ['music', '音乐', '歌曲']):
        return "play_music"
    elif any(keyword in intent_lower for keyword in ['restaurant', '餐厅', '吃饭']):
        return "restaurant_booking"
    
    return "general_chat"

def convert_to_sharegpt_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换为 ShareGPT 格式"""
    sharegpt_data = []
    
    for item in data:
        if 'text' in item and 'intent' in item:
            text = clean_text(item['text'])
            intent = normalize_intent(item['intent'])
            
            if text and intent:
                sharegpt_item = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"请识别以下用户意图：{text}"
                        },
                        {
                            "from": "gpt", 
                            "value": intent
                        }
                    ]
                }
                sharegpt_data.append(sharegpt_item)
    
    return sharegpt_data

def load_atis_data(data_path: str) -> List[Dict[str, Any]]:
    """加载 ATIS 数据"""
    data = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # ATIS 格式解析
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        text = parts[0]
                        intent = parts[1] if len(parts) > 1 else "general_chat"
                        data.append({"text": text, "intent": intent})
    
    except Exception as e:
        print(f"加载 ATIS 数据失败: {e}")
    
    return data

def load_snips_data(data_path: str) -> List[Dict[str, Any]]:
    """加载 SNIPS 数据"""
    data = []
    
    try:
        # SNIPS 数据通常在 JSON 文件中
        with open(data_path, 'r', encoding='utf-8') as f:
            snips_data = json.load(f)
        
        # 解析 SNIPS 数据结构
        if isinstance(snips_data, dict):
            for intent_name, utterances in snips_data.items():
                if isinstance(utterances, list):
                    for utterance in utterances:
                        if isinstance(utterance, dict) and 'text' in utterance:
                            data.append({
                                "text": utterance['text'],
                                "intent": intent_name
                            })
                        elif isinstance(utterance, str):
                            data.append({
                                "text": utterance,
                                "intent": intent_name
                            })
    
    except Exception as e:
        print(f"加载 SNIPS 数据失败: {e}")
    
    return data

def load_crosswoz_data(data_path: str) -> List[Dict[str, Any]]:
    """加载 CrossWOZ 数据"""
    data = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            crosswoz_data = json.load(f)
        
        # 解析对话数据，提取用户意图
        if isinstance(crosswoz_data, list):
            for dialogue in crosswoz_data:
                if isinstance(dialogue, dict) and 'dialogue' in dialogue:
                    for turn in dialogue['dialogue']:
                        if isinstance(turn, dict):
                            if turn.get('speaker') == 'user':
                                text = turn.get('utterance', '')
                                # 从系统响应或标注中提取意图
                                intent = extract_intent_from_context(turn)
                                data.append({"text": text, "intent": intent})
    
    except Exception as e:
        print(f"加载 CrossWOZ 数据失败: {e}")
    
    return data

def extract_intent_from_context(turn: Dict[str, Any]) -> str:
    """从对话上下文中提取意图"""
    # 简单的意图提取逻辑
    text = turn.get('utterance', '').lower()
    
    if any(keyword in text for keyword in ['预订', '订', '预订酒店', '预订餐厅']):
        if '酒店' in text:
            return "hotel_booking"
        elif '餐厅' in text or '吃饭' in text:
            return "restaurant_booking"
        elif '电影' in text or '票' in text:
            return "movie_booking"
    
    elif any(keyword in text for keyword in ['天气', '气温', '下雨']):
        return "query_weather"
    
    elif any(keyword in text for keyword in ['怎么走', '导航', '路线']):
        return "navigation"
    
    elif any(keyword in text for keyword in ['买', '购买', '购物']):
        return "shopping"
    
    return "general_chat"

def augment_data(data: List[Dict[str, Any]], augment_factor: int = 2) -> List[Dict[str, Any]]:
    """数据增强"""
    augmented_data = data.copy()
    
    # 同义词替换
    synonyms = {
        "查询": ["搜索", "查找", "看看", "查一下"],
        "预订": ["订", "预定", "预约"],
        "航班": ["飞机", "机票", "航班信息"],
        "天气": ["气温", "下雨", "晴天"],
        "音乐": ["歌曲", "歌", "听歌"],
        "餐厅": ["饭店", "吃饭的地方", "餐厅"],
        "电影": ["影片", "电影票", "看电影"],
        "导航": ["怎么走", "路线", "去"],
        "购买": ["买", "购物", "下单"]
    }
    
    for _ in range(augment_factor):
        for item in data:
            text = item['text']
            intent = item['intent']
            
            # 随机替换同义词
            for original, synonym_list in synonyms.items():
                if original in text:
                    synonym = random.choice(synonym_list)
                    augmented_text = text.replace(original, synonym, 1)
                    
                    if augmented_text != text:
                        augmented_data.append({
                            "text": augmented_text,
                            "intent": intent
                        })
    
    return augmented_data

def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8, eval_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """分割数据集"""
    random.shuffle(data)
    
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    eval_size = int(total_size * eval_ratio)
    
    train_data = data[:train_size]
    eval_data = data[train_size:train_size + eval_size]
    test_data = data[train_size + eval_size:]
    
    return train_data, eval_data, test_data

def save_processed_data(train_data: List[Dict], eval_data: List[Dict], test_data: List[Dict], output_dir: str):
    """保存处理后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为 ShareGPT 格式
    train_sharegpt = convert_to_sharegpt_format(train_data)
    eval_sharegpt = convert_to_sharegpt_format(eval_data)
    test_sharegpt = convert_to_sharegpt_format(test_data)
    
    # 保存数据
    with open(os.path.join(output_dir, 'intent_train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_sharegpt, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'intent_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(eval_sharegpt, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'intent_test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_sharegpt, f, ensure_ascii=False, indent=2)
    
    print(f"数据保存完成:")
    print(f"  训练集: {len(train_sharegpt)} 条")
    print(f"  验证集: {len(eval_sharegpt)} 条") 
    print(f"  测试集: {len(test_sharegpt)} 条")

def main():
    """主函数"""
    raw_dir = "/workspace/intent_recognition/data/raw"
    output_dir = "/workspace/intent_recognition/data/processed"
    
    print("=== 意图识别数据预处理脚本 ===")
    
    all_data = []
    
    # 加载各种数据集
    datasets = {
        "atis": load_atis_data,
        "snips": load_snips_data, 
        "crosswoz": load_crosswoz_data
    }
    
    for dataset_name, load_func in datasets.items():
        dataset_path = os.path.join(raw_dir, dataset_name)
        
        if os.path.exists(dataset_path):
            print(f"加载 {dataset_name} 数据集...")
            
            # 查找数据文件
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(('.json', '.txt', '.iob')):
                        file_path = os.path.join(root, file)
                        data = load_func(file_path)
                        all_data.extend(data)
                        print(f"  从 {file} 加载了 {len(data)} 条数据")
        else:
            print(f"未找到 {dataset_name} 数据集，跳过...")
    
    if not all_data:
        print("未找到任何数据，将使用示例数据进行演示...")
        # 创建示例数据
        sample_data = [
            {"text": "帮我查询明天北京到上海的航班", "intent": "query_flight"},
            {"text": "预订一张去深圳的机票", "intent": "book_flight"},
            {"text": "今天天气怎么样", "intent": "query_weather"},
            {"text": "播放周杰伦的歌", "intent": "play_music"},
            {"text": "预订一家意大利餐厅", "intent": "restaurant_booking"}
        ]
        all_data = sample_data * 100  # 复制增加数据量
    
    print(f"总共加载了 {len(all_data)} 条数据")
    
    # 数据增强
    print("进行数据增强...")
    all_data = augment_data(all_data, augment_factor=1)
    print(f"增强后数据量: {len(all_data)} 条")
    
    # 分割数据集
    print("分割数据集...")
    train_data, eval_data, test_data = split_data(all_data)
    
    # 保存处理后的数据
    save_processed_data(train_data, eval_data, test_data, output_dir)
    
    print("\n=== 预处理完成 ===")
    print("数据已保存到:", output_dir)
    print("现在可以开始训练模型了！")

if __name__ == "__main__":
    main()