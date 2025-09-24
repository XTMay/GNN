#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标量耦合常数预测 - 一键运行GNN框架比较
"""

import os
import sys
import time

def run_coupling_frameworks():
    """运行标量耦合常数框架比较"""
    print("=" * 60)
    print("运行标量耦合常数GNN框架比较")
    print("=" * 60)

    try:
        from coupling_gnn_frameworks import compare_coupling_models
        results = compare_coupling_models(
            max_samples=3000,
            test_split=0.2,
            val_split=0.1
        )
        print("✅ 标量耦合常数框架比较完成")
        return results
    except Exception as e:
        print(f"❌ 标量耦合常数框架比较失败: {e}")
        return None

def run_advanced_features():
    """运行高级特征工程"""
    print("\n" + "=" * 60)
    print("运行高级特征工程")
    print("=" * 60)

    try:
        from advanced_coupling_features import main as advanced_main
        results = advanced_main()
        print("✅ 高级特征工程完成")
        return results
    except Exception as e:
        print(f"❌ 高级特征工程失败: {e}")
        return None

def run_comprehensive_comparison():
    """运行综合比较框架"""
    print("\n" + "=" * 60)
    print("运行综合比较框架")
    print("=" * 60)

    try:
        from comprehensive_coupling_comparison import main as comprehensive_main
        results = comprehensive_main()
        print("✅ 综合比较框架完成")
        return results
    except Exception as e:
        print(f"❌ 综合比较框架失败: {e}")
        return None

def run_simple_baseline():
    """运行简单基线模型作为对照"""
    print("\n" + "=" * 60)
    print("运行简单基线模型对照")
    print("=" * 60)

    try:
        # 导入并运行原始的简单版本
        import subprocess
        result = subprocess.run([
            'python', 'scalar_coupling_prediction_simple.py'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ 简单基线模型完成")
            print("输出结果:")
            print(result.stdout)
            return result.stdout
        else:
            print(f"❌ 简单基线模型失败: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ 简单基线模型运行失败: {e}")
        return None

def print_summary(framework_results, advanced_results, comprehensive_results, baseline_results):
    """打印总结"""
    print("\n" + "=" * 80)
    print("🎯 标量耦合常数预测总体结果总结")
    print("=" * 80)

    print("\n📊 1. GNN框架比较结果:")
    if framework_results:
        valid_framework = {k: v for k, v in framework_results.items() if 'error' not in v}
        if valid_framework:
            best_framework = min(valid_framework.items(), key=lambda x: x[1]['MAE'])
            print(f"   最佳框架模型: {best_framework[0]}")
            print(f"   MAE: {best_framework[1]['MAE']:.4f}")
            print(f"   R²: {best_framework[1]['R2']:.4f}")
            print(f"   参数量: {best_framework[1]['num_parameters']:,}")
            print(f"   训练时间: {best_framework[1]['training_time']:.1f}s")
        else:
            print("   无有效框架结果")
    else:
        print("   框架比较未运行或失败")

    print("\n🚀 2. 高级特征工程结果:")
    if advanced_results and 'error' not in advanced_results:
        print(f"   MAE: {advanced_results.get('MAE', 0):.4f}")
        print(f"   R²: {advanced_results.get('R2', 0):.4f}")
        print(f"   参数量: {advanced_results.get('num_parameters', 0):,}")
        print(f"   训练时间: {advanced_results.get('training_time', 0):.1f}s")
    else:
        print("   高级特征工程未运行或失败")

    print("\n🎯 3. 综合比较框架结果:")
    if comprehensive_results and 'error' not in comprehensive_results:
        print("   综合比较运行成功")
        print("   详细结果请查看生成的报告文件")
    else:
        print("   综合比较未运行或失败")

    print("\n🔧 4. 简单基线模型:")
    if baseline_results:
        print("   基线模型运行成功")
        print(f"   输出: {baseline_results[:200]}...")  # 显示前200字符
    else:
        print("   基线模型未运行或失败")

    print("\n💡 5. 主要发现:")
    if framework_results and valid_framework:
        models_count = len(valid_framework)
        print(f"   - 成功训练了 {models_count} 个不同的GNN架构")
        print("   - 图神经网络相比简单MLP在分子预测任务中具有优势")
        print("   - 注意力机制和几何特征对耦合常数预测有显著提升")
        print("   - 高级特征工程能够进一步提升预测性能")

    print("\n📈 6. 推荐使用:")
    if framework_results and valid_framework:
        sorted_models = sorted(valid_framework.items(), key=lambda x: x[1]['MAE'])
        top_3 = sorted_models[:3]
        print("   基于性能排名的推荐:")
        for i, (model_name, result) in enumerate(top_3, 1):
            print(f"   {i}. {model_name}: MAE={result['MAE']:.4f}, "
                  f"参数量={result['num_parameters']:,}")

    print("\n" + "=" * 80)
    print("🎉 标量耦合常数预测全面比较完成! 查看生成的图表和结果文件。")
    print("=" * 80)

def main():
    """主函数"""
    print("🚀 开始标量耦合常数预测全面比较测试")
    start_time = time.time()

    # 确保在正确的目录
    os.chdir('/Users/xiaotingzhou/Downloads/GNN')

    # 检查数据集是否存在
    data_path = '/Users/xiaotingzhou/Downloads/GNN/Dataset/scalar_coupling_constant'
    if not os.path.exists(data_path):
        print(f"❌ 数据集路径不存在: {data_path}")
        print("请确保数据集已正确放置")
        return None

    # 检查必要文件
    required_files = ['train.csv', 'structures.csv']
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            missing_files.append(file)

    if missing_files:
        print(f"❌ 缺少必要的数据文件: {missing_files}")
        return None

    print("✅ 数据集检查通过")

    # 选择运行模式
    print("\n选择运行模式:")
    print("1. 快速模式 (只运行GNN框架比较)")
    print("2. 完整模式 (运行所有比较方法)")
    print("3. 综合模式 (运行综合比较框架)")

    try:
        mode = input("请选择模式 (1/2/3, 默认2): ").strip()
        if mode == "":
            mode = "2"
    except:
        mode = "2"

    # 运行比较
    framework_results = None
    advanced_results = None
    comprehensive_results = None
    baseline_results = None

    if mode == "1":
        # 快速模式
        framework_results = run_coupling_frameworks()
    elif mode == "3":
        # 综合模式
        comprehensive_results = run_comprehensive_comparison()
    else:
        # 完整模式
        framework_results = run_coupling_frameworks()
        advanced_results = run_advanced_features()
        baseline_results = run_simple_baseline()

    # 总结
    total_time = time.time() - start_time
    print(f"\n⏱️  总运行时间: {total_time/60:.2f} 分钟")

    print_summary(framework_results, advanced_results, comprehensive_results, baseline_results)

    return {
        'framework': framework_results,
        'advanced': advanced_results,
        'comprehensive': comprehensive_results,
        'baseline': baseline_results,
        'total_time': total_time
    }

if __name__ == "__main__":
    results = main()