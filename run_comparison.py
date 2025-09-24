#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行GNN模型比较的脚本 - 增强版
生成统一的综合报告，包含所有框架的详细结果
"""

import os
import sys
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def run_basic_comparison():
    """运行基础模型比较"""
    print("="*60)
    print("运行基础GNN框架比较")
    print("="*60)

    try:
        from gnn_comparison_framework import compare_models
        results = compare_models()
        print("✅ 基础比较完成")
        return results
    except Exception as e:
        print(f"❌ 基础比较失败: {e}")
        return None

def run_advanced_features():
    """运行高级特征提取"""
    print("\n" + "="*60)
    print("运行高级特征提取GNN")
    print("="*60)

    try:
        from advanced_feature_gnn import main as advanced_main
        results = advanced_main()
        print("✅ 高级特征比较完成")
        return results
    except Exception as e:
        print(f"❌ 高级特征比较失败: {e}")
        return None

def run_comprehensive():
    """运行综合比较"""
    print("\n" + "="*60)
    print("运行综合GNN框架比较")
    print("="*60)

    try:
        from comprehensive_gnn_comparison import main as comprehensive_main
        results = comprehensive_main()
        print("✅ 综合比较完成")
        return results
    except Exception as e:
        print(f"❌ 综合比较失败: {e}")
        return None

def generate_unified_report(basic_results, advanced_results, comprehensive_results, total_time):
    """生成统一的综合报告"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"GNN_Unified_Report_{timestamp}.html"

    print(f"\n📝 生成统一报告: {report_filename}")

    # 收集所有有效结果
    all_results = {}

    # 处理基础模型结果
    if basic_results:
        for model_name, result in basic_results.items():
            if 'error' not in result and 'eval_results' in result:
                all_results[f"Basic-{model_name}"] = {
                    'category': '基础模型',
                    'model_name': model_name,
                    'MAE': result['eval_results']['MAE'],
                    'RMSE': result['eval_results']['RMSE'],
                    'R2': result['eval_results']['R2'],
                    'num_parameters': result.get('num_parameters', 0),
                    'training_time': result.get('training_time', 0),
                    'description': f"基础{model_name}模型"
                }

    # 处理高级特征结果
    if advanced_results and 'error' not in advanced_results:
        all_results['Advanced-Features'] = {
            'category': '高级特征',
            'model_name': '高级特征GNN',
            'MAE': advanced_results.get('mae', 0),
            'RMSE': advanced_results.get('rmse', 0),
            'R2': advanced_results.get('r2', 0),
            'num_parameters': advanced_results.get('num_parameters', 0),
            'training_time': advanced_results.get('training_time', 0),
            'description': "集成RDKit描述符和高级特征的GNN模型"
        }

    # 处理综合框架结果
    if comprehensive_results:
        for model_name, result in comprehensive_results.items():
            if 'error' not in result:
                all_results[f"Comprehensive-{model_name}"] = {
                    'category': '综合框架',
                    'model_name': model_name,
                    'MAE': result.get('MAE', 0),
                    'RMSE': result.get('RMSE', 0),
                    'R2': result.get('R2', 0),
                    'num_parameters': result.get('num_parameters', 0),
                    'training_time': result.get('training_time', 0),
                    'description': f"综合框架中的{model_name}模型"
                }

    # 生成HTML报告
    html_content = generate_html_report(all_results, total_time)

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # 生成CSV数据文件
    csv_filename = f"GNN_Results_Data_{timestamp}.csv"
    generate_csv_report(all_results, csv_filename)

    # 生成可视化图表
    plot_filename = f"GNN_Comparison_Plots_{timestamp}.png"
    generate_comparison_plots(all_results, plot_filename)

    # 生成详细文本报告
    txt_filename = f"GNN_Detailed_Report_{timestamp}.txt"
    generate_text_report(all_results, total_time, txt_filename)

    print(f"✅ 报告生成完成:")
    print(f"   📄 HTML报告: {report_filename}")
    print(f"   📊 CSV数据: {csv_filename}")
    print(f"   📈 可视化图表: {plot_filename}")
    print(f"   📝 详细文本报告: {txt_filename}")

    return all_results


def generate_html_report(all_results, total_time):
    """生成HTML格式的报告"""

    if not all_results:
        return "<html><body><h1>没有有效的结果可以报告</h1></body></html>"

    # 按MAE排序
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['MAE'])

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GNN框架综合比较报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .best-model {{
            background-color: #e8f5e8;
        }}
        .category-basic {{
            background-color: #fff3cd;
        }}
        .category-advanced {{
            background-color: #d1ecf1;
        }}
        .category-comprehensive {{
            background-color: #f8d7da;
        }}
        .metric {{
            font-weight: bold;
            color: #2980b9;
        }}
        .highlight {{
            background-color: #ffffcc;
            padding: 2px 5px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 GNN框架综合比较报告</h1>
            <h2>QM9分子属性预测任务</h2>
            <p><strong>生成时间:</strong> {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
        </div>

        <div class="summary">
            <h2>📊 实验概况</h2>
            <p><strong>总运行时间:</strong> <span class="highlight">{total_time/60:.2f} 分钟</span></p>
            <p><strong>成功训练的模型数量:</strong> <span class="highlight">{len(all_results)} 个</span></p>
            <p><strong>最佳模型:</strong> <span class="highlight">{sorted_results[0][1]['model_name']} (MAE: {sorted_results[0][1]['MAE']:.4f})</span></p>
        </div>

        <h2>🏆 模型性能排行榜</h2>
        <table>
            <thead>
                <tr>
                    <th>排名</th>
                    <th>模型名称</th>
                    <th>类别</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R²</th>
                    <th>参数量</th>
                    <th>训练时间(s)</th>
                    <th>描述</th>
                </tr>
            </thead>
            <tbody>
"""

    for i, (key, result) in enumerate(sorted_results, 1):
        row_class = "best-model" if i == 1 else ""
        category_class = f"category-{result['category'].replace('高级特征', 'advanced').replace('综合框架', 'comprehensive').replace('基础模型', 'basic')}"

        html += f"""
                <tr class="{row_class} {category_class}">
                    <td class="metric">{i}</td>
                    <td><strong>{result['model_name']}</strong></td>
                    <td>{result['category']}</td>
                    <td>{result['MAE']:.4f}</td>
                    <td>{result['RMSE']:.4f}</td>
                    <td>{result['R2']:.4f}</td>
                    <td>{result['num_parameters']:,}</td>
                    <td>{result['training_time']:.1f}</td>
                    <td>{result['description']}</td>
                </tr>
"""

    html += f"""
            </tbody>
        </table>

        <h2>📈 关键发现</h2>
        <div class="summary">
            <h3>性能分析:</h3>
            <ul>
                <li><strong>最佳性能:</strong> {sorted_results[0][1]['model_name']} (MAE: {sorted_results[0][1]['MAE']:.4f})</li>
                <li><strong>最快训练:</strong> {min(all_results.values(), key=lambda x: x['training_time'])['model_name']} ({min(all_results.values(), key=lambda x: x['training_time'])['training_time']:.1f}s)</li>
                <li><strong>参数最少:</strong> {min(all_results.values(), key=lambda x: x['num_parameters'])['model_name']} ({min(all_results.values(), key=lambda x: x['num_parameters'])['num_parameters']:,}参数)</li>
            </ul>

            <h3>类别表现:</h3>
            <ul>
                <li><strong>基础模型:</strong> 平均MAE {np.mean([r['MAE'] for r in all_results.values() if r['category'] == '基础模型']):.4f}</li>
                <li><strong>高级特征:</strong> {'包含高级特征工程的模型' if any(r['category'] == '高级特征' for r in all_results.values()) else '未运行'}</li>
                <li><strong>综合框架:</strong> {'包含多种高级架构' if any(r['category'] == '综合框架' for r in all_results.values()) else '未运行'}</li>
            </ul>
        </div>

        <h2>💡 建议</h2>
        <div class="summary">
            <h3>根据不同需求的推荐:</h3>
            <ul>
                <li><strong>追求极致性能:</strong> {sorted_results[0][1]['model_name']}</li>
                <li><strong>平衡性能和效率:</strong> 建议选择MAE在前3名且训练时间适中的模型</li>
                <li><strong>快速原型开发:</strong> 推荐参数量少、训练快的基础模型</li>
            </ul>
        </div>

        <footer style="text-align: center; margin-top: 50px; color: #7f8c8d;">
            <p>报告由GNN框架比较脚本自动生成</p>
            <p>数据集: QM9 | 任务: HOMO-LUMO Gap预测</p>
        </footer>
    </div>
</body>
</html>
"""

    return html


def generate_csv_report(all_results, filename):
    """生成CSV格式的数据报告"""
    if not all_results:
        return

    df_data = []
    for key, result in all_results.items():
        df_data.append({
            'Model_ID': key,
            'Model_Name': result['model_name'],
            'Category': result['category'],
            'MAE': result['MAE'],
            'RMSE': result['RMSE'],
            'R2': result['R2'],
            'Num_Parameters': result['num_parameters'],
            'Training_Time_s': result['training_time'],
            'Description': result['description']
        })

    df = pd.DataFrame(df_data)
    df = df.sort_values('MAE')
    df.to_csv(filename, index=False, encoding='utf-8')


def generate_comparison_plots(all_results, filename):
    """生成比较可视化图表"""
    if len(all_results) < 2:
        print("结果数量太少，跳过可视化")
        return

    # 设置图表样式
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GNN框架综合性能比较', fontsize=16, fontweight='bold')

    models = [r['model_name'] for r in all_results.values()]
    maes = [r['MAE'] for r in all_results.values()]
    r2s = [r['R2'] for r in all_results.values()]
    params = [r['num_parameters'] for r in all_results.values()]
    times = [r['training_time'] for r in all_results.values()]
    categories = [r['category'] for r in all_results.values()]

    # 创建颜色映射
    unique_categories = list(set(categories))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}
    model_colors = [color_map[cat] for cat in categories]

    # 1. MAE对比
    bars1 = axes[0, 0].bar(range(len(models)), maes, color=model_colors, alpha=0.7)
    axes[0, 0].set_title('平均绝对误差 (MAE) 对比')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')

    # 2. R²对比
    bars2 = axes[0, 1].bar(range(len(models)), r2s, color=model_colors, alpha=0.7)
    axes[0, 1].set_title('决定系数 (R²) 对比')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')

    # 3. 参数量vs性能散点图
    scatter = axes[1, 0].scatter(params, maes, c=[color_map[cat] for cat in categories],
                                s=100, alpha=0.7, edgecolors='black')
    axes[1, 0].set_xlabel('参数数量')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('模型复杂度 vs 性能')

    # 添加模型标签
    for i, model in enumerate(models):
        axes[1, 0].annotate(model, (params[i], maes[i]), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)

    # 4. 训练时间对比
    bars4 = axes[1, 1].bar(range(len(models)), times, color=model_colors, alpha=0.7)
    axes[1, 1].set_title('训练时间对比')
    axes[1, 1].set_ylabel('训练时间 (秒)')
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')

    # 添加图例
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[cat], alpha=0.7, label=cat)
                      for cat in unique_categories]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def generate_text_report(all_results, total_time, filename):
    """生成详细的文本报告"""
    if not all_results:
        return

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['MAE'])

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GNN框架综合比较详细报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"总运行时间: {total_time/60:.2f} 分钟\n")
        f.write(f"成功训练模型数: {len(all_results)} 个\n\n")

        f.write("模型性能排行榜 (按MAE排序):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'排名':<4} {'模型名称':<20} {'类别':<10} {'MAE':<8} {'R²':<8} {'参数量':<10} {'训练时间':<8}\n")
        f.write("-" * 80 + "\n")

        for i, (key, result) in enumerate(sorted_results, 1):
            f.write(f"{i:<4} {result['model_name']:<20} {result['category']:<10} "
                   f"{result['MAE']:<8.4f} {result['R2']:<8.4f} "
                   f"{result['num_parameters']:<10,} {result['training_time']:<8.1f}s\n")

        f.write("\n\n详细结果:\n")
        f.write("=" * 50 + "\n")

        for i, (key, result) in enumerate(sorted_results, 1):
            f.write(f"\n{i}. {result['model_name']} ({result['category']})\n")
            f.write("-" * 30 + "\n")
            f.write(f"   MAE: {result['MAE']:.6f}\n")
            f.write(f"   RMSE: {result['RMSE']:.6f}\n")
            f.write(f"   R²: {result['R2']:.6f}\n")
            f.write(f"   参数量: {result['num_parameters']:,}\n")
            f.write(f"   训练时间: {result['training_time']:.1f}s\n")
            f.write(f"   描述: {result['description']}\n")

        # 统计分析
        f.write(f"\n\n统计分析:\n")
        f.write("=" * 30 + "\n")

        avg_mae = np.mean([r['MAE'] for r in all_results.values()])
        best_mae = min([r['MAE'] for r in all_results.values()])
        avg_time = np.mean([r['training_time'] for r in all_results.values()])

        f.write(f"平均MAE: {avg_mae:.4f}\n")
        f.write(f"最佳MAE: {best_mae:.4f}\n")
        f.write(f"平均训练时间: {avg_time:.1f}s\n")

        # 按类别统计
        categories = {}
        for result in all_results.values():
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['MAE'])

        f.write(f"\n按类别统计:\n")
        for cat, maes in categories.items():
            f.write(f"{cat}: 平均MAE = {np.mean(maes):.4f}, 模型数量 = {len(maes)}\n")


def print_summary(basic_results, advanced_results, comprehensive_results):
    """打印总结并生成统一报告"""
    print("\n" + "="*80)
    print("🎯 总体结果总结")
    print("="*80)

    print("\n📊 1. 基础模型比较:")
    if basic_results:
        valid_basic = {k: v for k, v in basic_results.items() if 'error' not in v}
        if valid_basic:
            best_basic = min(valid_basic.items(), key=lambda x: x[1]['eval_results']['MAE'])
            print(f"   最佳模型: {best_basic[0]}")
            print(f"   MAE: {best_basic[1]['eval_results']['MAE']:.4f}")
            print(f"   R²: {best_basic[1]['eval_results']['R2']:.4f}")
        else:
            print("   无有效结果")
    else:
        print("   未运行或失败")

    print("\n🚀 2. 高级特征模型:")
    if advanced_results:
        print(f"   MAE: {advanced_results.get('mae', 0):.4f}")
        print(f"   R²: {advanced_results.get('r2', 0):.4f}")
        print(f"   参数量: {advanced_results.get('num_parameters', 0):,}")
    else:
        print("   未运行或失败")

    print("\n🔥 3. 综合框架比较:")
    if comprehensive_results:
        valid_comp = {k: v for k, v in comprehensive_results.items() if 'error' not in v}
        if valid_comp:
            best_comp = min(valid_comp.items(), key=lambda x: x[1]['MAE'])
            print(f"   最佳模型: {best_comp[0]}")
            print(f"   MAE: {best_comp[1]['MAE']:.4f}")
            print(f"   R²: {best_comp[1]['R2']:.4f}")
            print(f"   参数量: {best_comp[1]['num_parameters']:,}")
        else:
            print("   无有效结果")
    else:
        print("   未运行或失败")

    print("\n" + "="*80)
    print("🎉 所有比较完成! 正在生成统一报告...")
    print("="*80)

def load_saved_results():
    """从已保存的结果文件加载数据"""
    print("🔍 搜索已保存的结果文件...")

    # 搜索结果文件
    result_files = []
    for file in os.listdir('.'):
        if file.startswith('GNN_Results_Data_') and file.endswith('.csv'):
            result_files.append(file)

    if not result_files:
        print("❌ 未找到已保存的结果文件")
        return None, None, None

    # 选择最新的结果文件
    latest_file = max(result_files, key=lambda x: os.path.getmtime(x))
    print(f"📄 找到结果文件: {latest_file}")

    try:
        df = pd.read_csv(latest_file)
        print(f"✅ 成功加载 {len(df)} 个模型的结果")

        # 转换回原始格式
        basic_results = {}
        advanced_results = {}
        comprehensive_results = {}

        for _, row in df.iterrows():
            model_data = {
                'MAE': row['MAE'],
                'RMSE': row['RMSE'],
                'R2': row['R2'],
                'num_parameters': row['Num_Parameters'],
                'training_time': row['Training_Time_s']
            }

            if row['Category'] == '基础模型':
                basic_results[row['Model_Name']] = {
                    'eval_results': model_data,
                    **model_data
                }
            elif row['Category'] == '高级特征':
                advanced_results = {
                    'mae': row['MAE'],
                    'rmse': row['RMSE'],
                    'r2': row['R2'],
                    'num_parameters': row['Num_Parameters'],
                    'training_time': row['Training_Time_s']
                }
            elif row['Category'] == '综合框架':
                comprehensive_results[row['Model_Name']] = model_data

        return basic_results, advanced_results, comprehensive_results

    except Exception as e:
        print(f"❌ 加载结果文件失败: {e}")
        return None, None, None


def main():
    """主函数 - 增强版，生成统一报告"""
    print("🚀 GNN框架全面比较工具")
    print("="*60)

    # 确保在正确的目录
    os.chdir('/Users/xiaotingzhou/Downloads/GNN')

    # 选择运行模式
    print("\n选择运行模式:")
    print("1. 🔥 完整运行 (运行所有模型比较 + 生成报告)")
    print("2. 📊 仅生成报告 (从已保存结果生成报告)")
    print("3. ⚡ 快速测试 (仅运行基础比较)")

    try:
        choice = input("\n请选择 (1/2/3, 默认1): ").strip()
        if choice == "":
            choice = "1"
    except:
        choice = "1"

    start_time = time.time()

    if choice == "2":
        # 仅生成报告模式
        print("\n📄 从已保存结果生成报告...")
        basic_results, advanced_results, comprehensive_results = load_saved_results()

        if basic_results is None and advanced_results is None and comprehensive_results is None:
            print("❌ 无法加载已保存的结果，请先运行模型比较")
            return None

        total_time = 0  # 因为没有实际训练

    elif choice == "3":
        # 快速测试模式
        print("\n⚡ 快速测试模式...")
        basic_results = run_basic_comparison()
        advanced_results = None
        comprehensive_results = None
        total_time = time.time() - start_time

    else:
        # 完整运行模式
        print("\n🔥 完整运行模式...")
        basic_results = run_basic_comparison()
        advanced_results = run_advanced_features()
        comprehensive_results = run_comprehensive()
        total_time = time.time() - start_time

    print(f"\n⏱️  总运行时间: {total_time/60:.2f} 分钟")

    # 打印基础总结
    print_summary(basic_results, advanced_results, comprehensive_results)

    # 生成统一报告
    try:
        unified_results = generate_unified_report(
            basic_results,
            advanced_results,
            comprehensive_results,
            total_time
        )

        print(f"\n🎊 统一报告生成成功! 共收集了 {len(unified_results)} 个模型的结果")

        if unified_results:
            best_model = min(unified_results.items(), key=lambda x: x[1]['MAE'])
            print(f"🏆 总体最佳模型: {best_model[1]['model_name']} (MAE: {best_model[1]['MAE']:.4f})")

            # 显示前3名
            sorted_models = sorted(unified_results.items(), key=lambda x: x[1]['MAE'])
            print(f"\n🥇 前三名模型:")
            for i, (key, result) in enumerate(sorted_models[:3], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                print(f"{emoji} {i}. {result['model_name']} - MAE: {result['MAE']:.4f}, "
                      f"R²: {result['R2']:.4f}")

    except Exception as e:
        print(f"❌ 生成统一报告时出错: {e}")
        unified_results = {}

    print(f"\n{'='*80}")
    print("📋 报告文件已生成:")
    print("   📄 HTML报告: 可在浏览器中查看详细结果")
    print("   📊 CSV数据: 可用于进一步分析")
    print("   📈 可视化图表: PNG格式的性能对比图")
    print("   📝 文本报告: 详细的文本格式报告")
    print(f"{'='*80}")

    return {
        'basic': basic_results,
        'advanced': advanced_results,
        'comprehensive': comprehensive_results,
        'unified_results': unified_results,
        'total_time': total_time,
        'mode': choice
    }

if __name__ == "__main__":
    results = main()