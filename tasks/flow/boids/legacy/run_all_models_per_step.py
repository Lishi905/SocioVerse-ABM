#!/usr/bin/env python3
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
同时运行Qwen、OpenAI、DeepSeek三个模型的per-step测试
Created: 2025-04-12
"""
import subprocess
import sys
import time
from datetime import datetime

def run_model_test(model_name: str, args: list):
    """运行单个模型的测试"""
    print("\n" + "="*80)
    print(f"开始运行 {model_name} 模型测试")
    print("="*80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"命令: python evaluate_boids_llm_per_step.py {' '.join(args)}")
    print("-"*80)
    
    start_time = time.time()
    
    try:
        # 运行测试
        result = subprocess.run(
            ["python", "evaluate_boids_llm_per_step.py"] + args,
            check=True,
            capture_output=False,  # 显示实时输出
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "-"*80)
        print(f"✅ {model_name} 测试完成")
        print(f"   耗时: {elapsed_time/60:.1f} 分钟")
        print("="*80)
        
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print("\n" + "-"*80)
        print(f"❌ {model_name} 测试失败")
        print(f"   错误代码: {e.returncode}")
        print(f"   耗时: {elapsed_time/60:.1f} 分钟")
        print("="*80)
        return False, elapsed_time
    
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print("\n" + "-"*80)
        print(f"⚠️  {model_name} 测试被用户中断")
        print(f"   耗时: {elapsed_time/60:.1f} 分钟")
        print("="*80)
        return False, elapsed_time

def main():
    """主函数：依次运行三个模型的测试"""
    print("="*80)
    print("多模型Per-Step LLM测试")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n将依次运行以下模型:")
    print("  1. Qwen")
    print("  2. OpenAI")
    print("  3. DeepSeek")
    print("\n每个模型将运行:")
    print("  - 1次运行")
    print("  - 100步")
    print("  - batch_size=45")
    print("\n预计总时间: 约25-30分钟")
    print("="*80)
    
    # 测试配置
    runs = 1
    steps = 100
    batch_size = 45
    
    models = [
        {
            "name": "Qwen",
            "args": ["--qwen", "--runs", str(runs), "--steps", str(steps), "--batch-size", str(batch_size)]
        },
        {
            "name": "OpenAI",
            "args": ["--runs", str(runs), "--steps", str(steps), "--batch-size", str(batch_size)]
        },
        {
            "name": "DeepSeek",
            "args": ["--deepseek", "--runs", str(runs), "--steps", str(steps), "--batch-size", str(batch_size)]
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for idx, model in enumerate(models, start=1):
        print(f"\n\n{'='*80}")
        print(f"进度: {idx}/{len(models)} - {model['name']}")
        print(f"{'='*80}")
        
        success, elapsed = run_model_test(model['name'], model['args'])
        
        results.append({
            'model': model['name'],
            'success': success,
            'time_minutes': elapsed / 60
        })
        
        # 如果不是最后一个，等待一下再继续
        if idx < len(models):
            print(f"\n等待3秒后继续下一个模型...")
            time.sleep(3)
    
    # 总结
    total_time = time.time() - total_start_time
    
    print("\n\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print("\n各模型结果:")
    print("-"*80)
    
    for result in results:
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"{result['model']:10s}: {status:8s} ({result['time_minutes']:.1f} 分钟)")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"\n成功: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("\n🎉 所有模型测试完成！")
        print("\n可以运行以下命令分析结果:")
        print("  python analyze_speed_alignment_comparison.py")
    else:
        print(f"\n⚠️  有 {len(results) - success_count} 个模型测试失败")
    
    print("="*80)
    
    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(1)

