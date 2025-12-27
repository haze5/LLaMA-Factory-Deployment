#!/usr/bin/env python3
"""
意图识别模型评估脚本
支持多种评估指标和可视化
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm

class IntentEvaluator:
    """意图识别评估器"""
    
    def __init__(self, model_path: str, tokenizer_path: str = None):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径
            tokenizer_path: 分词器路径，如果为None则使用model_path
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
        print(f"加载模型: {model_path}")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        
        # 意图类型定义
        self.intent_types = [
            "query_flight", "book_flight", "cancel_flight", "query_weather",
            "set_reminder", "play_music", "send_message", "make_call",
            "search_info", "navigation", "restaurant_booking", "movie_booking",
            "hotel_booking", "shopping", "translation", "calculator",
            "timer", "alarm", "note", "calendar", "email", "file_management",
            "system_control", "general_chat", "greeting", "goodbye"
        ]
    
    def predict_intent(self, text: str, max_new_tokens: int = 64) -> str:
        """
        预测意图
        
        Args:
            text: 输入文本
            max_new_tokens: 最大生成token数
            
        Returns:
            预测的意图
        """
        prompt = f"请识别以下用户意图：{text}\n意图："
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # 清理输出，只返回意图
        intent = generated_text.split('\n')[0].strip()
        
        # 验证意图是否有效
        if intent not in self.intent_types:
            # 尝试从完整输出中提取意图
            for valid_intent in self.intent_types:
                if valid_intent in intent:
                    return valid_intent
            # 如果没有找到有效意图，返回默认值
            return "general_chat"
        
        return intent
    
    def evaluate_dataset(self, test_data_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        评估整个数据集
        
        Args:
            test_data_path: 测试数据路径
            output_dir: 输出目录
            
        Returns:
            评估结果
        """
        # 加载测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        true_intents = []
        pred_intents = []
        
        print(f"开始评估 {len(test_data)} 条数据...")
        
        for item in tqdm(test_data):
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:
                # 提取用户文本和真实意图
                user_text = conversations[0]['value'].replace("请识别以下用户意图：", "")
                true_intent = conversations[1]['value']
                
                # 预测意图
                pred_intent = self.predict_intent(user_text)
                
                true_intents.append(true_intent)
                pred_intents.append(pred_intent)
        
        # 计算评估指标
        results = self.calculate_metrics(true_intents, pred_intents)
        
        # 保存结果
        if output_dir:
            self.save_results(results, true_intents, pred_intents, output_dir)
        
        return results
    
    def calculate_metrics(self, true_intents: List[str], pred_intents: List[str]) -> Dict[str, Any]:
        """计算评估指标"""
        
        # 基本指标
        accuracy = accuracy_score(true_intents, pred_intents)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_intents, pred_intents, average='weighted', zero_division=0
        )
        
        # 详细分类报告
        report = classification_report(true_intents, pred_intents, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(true_intents, pred_intents, labels=self.intent_types)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'support': support.tolist() if hasattr(support, 'tolist') else support
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], true_intents: List[str], 
                    pred_intents: List[str], output_dir: str):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数值结果
        results_to_save = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'classification_report': results['classification_report']
        }
        
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
        
        # 保存预测结果
        predictions = []
        for i, (true, pred) in enumerate(zip(true_intents, pred_intents)):
            predictions.append({
                'index': i,
                'true_intent': true,
                'pred_intent': pred,
                'correct': true == pred
            })
        
        with open(os.path.join(output_dir, 'predictions.json'), 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        # 生成可视化图表
        self.plot_confusion_matrix(results['confusion_matrix'], output_dir)
        self.plot_metrics(results, output_dir)
        
        print(f"评估结果已保存到: {output_dir}")
    
    def plot_confusion_matrix(self, cm: List[List[int]], output_dir: str):
        """绘制混淆矩阵"""
        plt.figure(figsize=(15, 12))
        
        # 过滤掉空行空列
        non_empty_indices = []
        for i, intent in enumerate(self.intent_types):
            if i < len(cm) and sum(cm[i]) > 0:
                non_empty_indices.append(i)
        
        if not non_empty_indices:
            print("警告：混淆矩阵为空，跳过绘图")
            return
        
        filtered_cm = np.array(cm)[np.ix_(non_empty_indices, non_empty_indices)]
        filtered_labels = [self.intent_types[i] for i in non_empty_indices]
        
        sns.heatmap(
            filtered_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=filtered_labels,
            yticklabels=filtered_labels
        )
        
        plt.title('意图识别混淆矩阵')
        plt.xlabel('预测意图')
        plt.ylabel('真实意图')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics(self, results: Dict[str, Any], output_dir: str):
        """绘制评估指标"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [results[metric] for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'wheat'])
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.title('意图识别评估指标')
        plt.ylabel('分数')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def interactive_test(self):
        """交互式测试"""
        print("=== 意图识别交互式测试 ===")
        print("输入文本进行测试，输入 'quit' 退出")
        
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            intent = self.predict_intent(text)
            print(f"识别意图: {intent}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="意图识别模型评估")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据路径")
    parser.add_argument("--output_dir", type=str, default="/workspace/intent_recognition/outputs/evaluations", help="输出目录")
    parser.add_argument("--interactive", action="store_true", help="启动交互式测试")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = IntentEvaluator(args.model_path)
    
    if args.interactive:
        # 交互式测试
        evaluator.interactive_test()
    else:
        # 数据集评估
        results = evaluator.evaluate_dataset(args.test_data, args.output_dir)
        
        print("\n=== 评估结果 ===")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1 分数: {results['f1_score']:.4f}")
        print(f"\n详细报告:")
        print(results['classification_report'])

if __name__ == "__main__":
    main()