#!/usr/bin/env python3
"""
意图识别项目快速开始脚本
一键完成数据准备、训练和评估
"""

import os
import sys
import subprocess
import json
from pathlib import Path

class IntentRecognitionQuickStart:
    """意图识别快速开始类"""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = workspace_root
        self.project_root = f"{workspace_root}/intent_recognition"
        self.llamafactory_root = f"{workspace_root}/LLaMA-Factory"
        
        # 检查必要路径
        self.check_environment()
    
    def check_environment(self):
        """检查环境是否就绪"""
        print("=== 环境检查 ===")
        
        # 检查 LLaMA-Factory
        if not os.path.exists(self.llamafactory_root):
            print("❌ LLaMA-Factory 不存在")
            return False
        
        # 检查模型
        model_path = f"{self.workspace_root}/models/DeepSeek-R1-Distill-Qwen-1.5B"
        if not os.path.exists(model_path):
            print("❌ DeepSeek-R1-Distill-Qwen-1.5B 模型不存在")
            return False
        
        print("✅ 环境检查通过")
        return True
    
    def step1_prepare_data(self):
        """步骤1: 数据准备"""
        print("\n=== 步骤1: 数据准备 ===")
        
        try:
            # 运行数据下载脚本
            print("下载示例数据集...")
            subprocess.run([
                sys.executable, 
                f"{self.project_root}/scripts/download_dataset.py"
            ], check=True, input="4\n", text=True)  # 选择4，创建示例数据
            
            # 运行数据预处理脚本
            print("预处理数据...")
            subprocess.run([
                sys.executable,
                f"{self.project_root}/scripts/preprocess_data.py"
            ], check=True)
            
            print("✅ 数据准备完成")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 数据准备失败: {e}")
            return False
    
    def step2_train_model(self):
        """步骤2: 模型训练"""
        print("\n=== 步骤2: 模型训练 ===")
        
        try:
            # 切换到 LLaMA-Factory 目录并运行训练
            config_path = f"{self.project_root}/config/model_config.yaml"
            
            print("开始训练模型...")
            cmd = [
                sys.executable, 
                "-m", "llamafactory.cli.train",
                config_path
            ]
            
            # 设置环境变量并运行
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.llamafactory_root}:{env.get('PYTHONPATH', '')}"
            
            result = subprocess.run(
                cmd, 
                cwd=self.llamafactory_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            if result.returncode == 0:
                print("✅ 模型训练完成")
                return True
            else:
                print(f"❌ 模型训练失败:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ 训练超时（超过1小时）")
            return False
        except Exception as e:
            print(f"❌ 训练过程中出错: {e}")
            return False
    
    def step3_evaluate_model(self):
        """步骤3: 模型评估"""
        print("\n=== 步骤3: 模型评估 ===")
        
        try:
            # 检查模型是否训练完成
            model_path = f"{self.project_root}/outputs/models/intent_recognition_lora"
            if not os.path.exists(model_path):
                print("❌ 训练完成的模型不存在")
                return False
            
            # 运行评估脚本
            test_data_path = f"{self.project_root}/data/processed/intent_test.json"
            output_dir = f"{self.project_root}/outputs/evaluations"
            
            cmd = [
                sys.executable,
                f"{self.project_root}/scripts/evaluate.py",
                "--model_path", model_path,
                "--test_data", test_data_path,
                "--output_dir", output_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 模型评估完成")
                print(result.stdout)
                
                # 显示评估结果文件
                results_file = f"{output_dir}/evaluation_results.json"
                if os.path.exists(results_file):
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    print(f"\n📊 评估摘要:")
                    print(f"准确率: {results['accuracy']:.4f}")
                    print(f"精确率: {results['precision']:.4f}")
                    print(f"召回率: {results['recall']:.4f}")
                    print(f"F1分数: {results['f1_score']:.4f}")
                
                return True
            else:
                print(f"❌ 模型评估失败:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 评估过程中出错: {e}")
            return False
    
    def step4_test_inference(self):
        """步骤4: 推理测试"""
        print("\n=== 步骤4: 推理测试 ===")
        
        try:
            # 运行推理示例
            subprocess.run([
                sys.executable,
                f"{self.project_root}/examples/inference_example.py"
            ], check=True)
            
            print("✅ 推理测试完成")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 推理测试失败: {e}")
            return False
    
    def run_all_steps(self):
        """运行所有步骤"""
        print("🚀 开始意图识别项目快速启动")
        print("这将依次执行：数据准备 -> 模型训练 -> 模型评估 -> 推理测试")
        
        steps = [
            ("数据准备", self.step1_prepare_data),
            ("模型训练", self.step2_train_model),
            ("模型评估", self.step3_evaluate_model),
            ("推理测试", self.step4_test_inference)
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            try:
                results[step_name] = step_func()
            except Exception as e:
                print(f"❌ {step_name}步骤出现异常: {e}")
                results[step_name] = False
            
            if not results[step_name]:
                print(f"\n⚠️  {step_name}失败，终止后续步骤")
                break
        
        # 显示最终结果
        self.show_final_results(results)
    
    def show_final_results(self, results: dict):
        """显示最终结果"""
        print("\n" + "="*50)
        print("📋 执行结果汇总:")
        print("="*50)
        
        for step_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            print(f"{step_name:<15}: {status}")
        
        # 如果所有步骤都成功，显示使用说明
        if all(results.values()):
            print(f"\n🎉 恭喜！所有步骤执行成功！")
            print(f"\n📁 项目文件位置:")
            print(f"  项目目录: {self.project_root}")
            print(f"  训练模型: {self.project_root}/outputs/models/intent_recognition_lora")
            print(f"  评估结果: {self.project_root}/outputs/evaluations")
            print(f"  数据文件: {self.project_root}/data/processed")
            
            print(f"\n🔧 下一步可以:")
            print(f"1. 查看评估报告: {self.project_root}/outputs/evaluations/evaluation_results.json")
            print(f"2. 运行交互式测试: python {self.project_root}/examples/inference_example.py")
            print(f"3. 启动API服务: cd {self.llamafactory_root} && python src/api.py")
            print(f"4. 使用API测试: python {self.project_root}/examples/api_example.py")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="意图识别快速启动脚本")
    parser.add_argument("--step", choices=["1", "2", "3", "4", "all"], default="all",
                       help="执行特定步骤，默认执行所有步骤")
    
    args = parser.parse_args()
    
    quick_start = IntentRecognitionQuickStart()
    
    if args.step == "all":
        quick_start.run_all_steps()
    else:
        step_map = {
            "1": ("数据准备", quick_start.step1_prepare_data),
            "2": ("模型训练", quick_start.step2_train_model),
            "3": ("模型评估", quick_start.step3_evaluate_model),
            "4": ("推理测试", quick_start.step4_test_inference)
        }
        
        if args.step in step_map:
            step_name, step_func = step_map[args.step]
            print(f"🚀 执行步骤{args.step}: {step_name}")
            
            try:
                success = step_func()
                if success:
                    print(f"✅ {step_name}完成")
                else:
                    print(f"❌ {step_name}失败")
            except Exception as e:
                print(f"❌ {step_name}出现异常: {e}")

if __name__ == "__main__":
    main()