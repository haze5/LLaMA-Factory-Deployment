#!/usr/bin/env python3
"""
意图识别 API 使用示例
演示如何通过 API 调用进行意图识别
"""

import requests
import json
import time
from typing import List, Dict, Any, Optional

class IntentRecognitionAPI:
    """意图识别 API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        初始化 API 客户端
        
        Args:
            base_url: API 基础地址
            api_key: API 密钥（如果需要）
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # 设置请求头
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}"
            })
    
    def health_check(self) -> bool:
        """检查 API 服务健康状态"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def recognize_intent(self, text: str, model: str = "intent_recognition_lora", 
                        temperature: float = 0.1, max_tokens: int = 64) -> Dict[str, Any]:
        """
        识别单个文本的意图
        
        Args:
            text: 输入文本
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            API 响应结果
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"请识别以下用户意图：{text}"
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/chat",
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"API 请求失败: {str(e)}"
            }
    
    def batch_recognize(self, texts: List[str], model: str = "intent_recognition_lora",
                       batch_delay: float = 0.1) -> List[Dict[str, Any]]:
        """
        批量识别意图
        
        Args:
            texts: 文本列表
            model: 模型名称
            batch_delay: 批次间延迟（秒）
            
        Returns:
            批量识别结果
        """
        results = []
        
        for i, text in enumerate(texts):
            result = self.recognize_intent(text, model=model)
            results.append(result)
            
            # 添加延迟避免过快请求
            if i < len(texts) - 1:
                time.sleep(batch_delay)
            
            # 进度提示
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(texts)} 条")
        
        return results
    
    def extract_intent_from_response(self, response: Dict[str, Any]) -> str:
        """从 API 响应中提取意图"""
        if response.get("error"):
            return "error"
        
        try:
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                
                # 提取意图（假设模型直接返回意图名称）
                intent = content.strip().split('\n')[0].strip()
                
                # 验证意图是否有效
                valid_intents = [
                    "query_flight", "book_flight", "cancel_flight", "query_weather",
                    "set_reminder", "play_music", "send_message", "make_call",
                    "search_info", "navigation", "restaurant_booking", "movie_booking",
                    "hotel_booking", "shopping", "translation", "calculator",
                    "timer", "alarm", "note", "calendar", "email", "file_management",
                    "system_control", "general_chat", "greeting", "goodbye"
                ]
                
                if intent in valid_intents:
                    return intent
                
                # 模糊匹配
                for valid_intent in valid_intents:
                    if valid_intent in intent or intent in valid_intent:
                        return valid_intent
                
                return "general_chat"
            
        except Exception as e:
            print(f"解析响应失败: {e}")
        
        return "error"

def main():
    """主函数 - API 使用演示"""
    
    print("=== 意图识别 API 使用演示 ===\n")
    
    # 初始化 API 客户端
    api = IntentRecognitionAPI("http://localhost:8000")
    
    # 检查服务状态
    print("1. 检查 API 服务状态...")
    if api.health_check():
        print("✅ API 服务正常")
    else:
        print("❌ API 服务不可用，请确保 LLaMA-Factory API 服务已启动")
        print("启动命令: cd /workspace/LLaMA-Factory && python src/api.py")
        return
    
    # 示例文本
    test_texts = [
        "帮我查询明天北京到上海的航班",
        "预订一张去深圳的机票",
        "今天天气怎么样",
        "播放周杰伦的歌",
        "预订一家意大利餐厅",
        "设置一个10分钟的计时器",
        "你好，今天天气不错",
        "随便聊聊吧"
    ]
    
    print("\n2. 单个文本识别演示:")
    for i, text in enumerate(test_texts[:3], 1):
        print(f"\n示例 {i}:")
        print(f"输入: {text}")
        
        # 调用 API
        response = api.recognize_intent(text)
        
        if response.get("error"):
            print(f"错误: {response['message']}")
        else:
            intent = api.extract_intent_from_response(response)
            print(f"识别结果: {intent}")
            
            # 显示完整的 API 响应（可选）
            print(f"完整响应: {json.dumps(response, ensure_ascii=False, indent=2)}")
    
    print("\n3. 批量识别演示:")
    start_time = time.time()
    
    batch_results = api.batch_recognize(test_texts, batch_delay=0.05)
    
    end_time = time.time()
    
    print(f"\n批量识别完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每条: {(end_time - start_time) / len(test_texts):.3f} 秒")
    
    print("\n识别结果:")
    for i, (text, result) in enumerate(zip(test_texts, batch_results), 1):
        intent = api.extract_intent_from_response(result)
        status = "✅" if intent != "error" else "❌"
        print(f"{i:2d}. {status} {text:<25} -> {intent}")
    
    print("\n4. 性能测试:")
    performance_test_texts = test_texts * 3  # 重复测试
    
    start_time = time.time()
    perf_results = api.batch_recognize(performance_test_texts, batch_delay=0.01)
    end_time = time.time()
    
    total_requests = len(performance_test_texts)
    successful_requests = sum(1 for r in perf_results if not r.get("error"))
    success_rate = successful_requests / total_requests * 100
    
    print(f"总请求数: {total_requests}")
    print(f"成功请求数: {successful_requests}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"平均响应时间: {(end_time - start_time) / total_requests:.3f} 秒")
    print(f"QPS (每秒请求数): {total_requests / (end_time - start_time):.1f}")
    
    print("\n=== 演示结束 ===")

if __name__ == "__main__":
    main()