#!/usr/bin/env python3
"""
意图识别推理示例脚本
演示如何使用训练好的模型进行意图识别
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import List, Dict, Any

class IntentRecognitionModel:
    """意图识别模型类"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化模型
        
        Args:
            model_path: 模型路径
            device: 设备类型 ("cuda", "cpu", "auto")
        """
        self.model_path = model_path
        
        print(f"正在加载模型: {model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        print(f"模型加载完成，使用设备: {self.device}")
    
    def predict_intent(self, text: str, max_new_tokens: int = 64, temperature: float = 0.1) -> str:
        """
        预测文本意图
        
        Args:
            text: 输入文本
            max_new_tokens: 最大生成token数
            temperature: 温度参数，控制随机性
            
        Returns:
            识别的意图
        """
        prompt = f"请识别以下用户意图：{text}\n意图："
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.model.device)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # 提取意图
        intent = self.extract_intent_from_response(generated_text)
        return intent
    
    def extract_intent_from_response(self, response: str) -> str:
        """从响应中提取意图"""
        # 清理响应文本
        intent = response.split('\n')[0].strip()
        
        # 预定义的意图类型
        valid_intents = [
            "query_flight", "book_flight", "cancel_flight", "query_weather",
            "set_reminder", "play_music", "send_message", "make_call",
            "search_info", "navigation", "restaurant_booking", "movie_booking",
            "hotel_booking", "shopping", "translation", "calculator",
            "timer", "alarm", "note", "calendar", "email", "file_management",
            "system_control", "general_chat", "greeting", "goodbye"
        ]
        
        # 检查是否是有效意图
        if intent in valid_intents:
            return intent
        
        # 模糊匹配
        for valid_intent in valid_intents:
            if valid_intent in intent or intent in valid_intent:
                return valid_intent
        
        # 如果都不匹配，返回通用意图
        return "general_chat"
    
    def batch_predict(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        批量预测意图
        
        Args:
            texts: 输入文本列表
            batch_size: 批处理大小
            
        Returns:
            意图列表
        """
        intents = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_intents = []
            
            for text in batch_texts:
                intent = self.predict_intent(text)
                batch_intents.append(intent)
            
            intents.extend(batch_intents)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"已处理 {min(i + batch_size, len(texts))}/{len(texts)} 条")
        
        return intents

def main():
    """主函数 - 演示使用"""
    # 模型路径（请根据实际情况修改）
    model_path = "/workspace/intent_recognition/outputs/models/intent_recognition_lora"
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请先训练模型或修改模型路径")
        return
    
    # 初始化模型
    try:
        model = IntentRecognitionModel(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 示例文本
    test_texts = [
        "帮我查询明天北京到上海的航班",
        "预订一张去深圳的机票",
        "取消我的航班预订",
        "今天天气怎么样",
        "明天早上8点提醒我开会",
        "播放周杰伦的歌",
        "给张三发个消息",
        "帮我打电话给李四",
        "搜索一下人工智能的资料",
        "导航到最近的加油站",
        "预订一家意大利餐厅",
        "买两张电影票",
        "预订一个酒店房间",
        "在网上买一本书",
        "把这句话翻译成英文",
        "帮我计算25乘以60",
        "设置一个10分钟的计时器",
        "明天早上6点叫醒我",
        "记一下这个笔记",
        "查看我明天的日程",
        "写一封邮件给老板",
        "删除这个文件",
        "打开系统设置",
        "你好，今天天气不错",
        "再见，下次聊",
        "随便聊聊吧"
    ]
    
    print("\n=== 意图识别演示 ===\n")
    
    # 单个预测示例
    print("1. 单个文本预测:")
    for i, text in enumerate(test_texts[:5], 1):
        intent = model.predict_intent(text)
        print(f"{i}. 文本: {text}")
        print(f"   意图: {intent}\n")
    
    # 批量预测示例
    print("2. 批量文本预测:")
    batch_intents = model.batch_predict(test_texts)
    
    for text, intent in zip(test_texts, batch_intents):
        print(f"文本: {text:<30} -> 意图: {intent}")
    
    # 交互式示例
    print("\n3. 交互式测试 (输入 'quit' 退出):")
    while True:
        user_input = input("\n请输入文本: ").strip()
        if user_input.lower() in ['quit', 'exit', '退出']:
            break
        
        if not user_input:
            continue
        
        intent = model.predict_intent(user_input)
        print(f"识别意图: {intent}")
    
    print("\n=== 演示结束 ===")

if __name__ == "__main__":
    main()