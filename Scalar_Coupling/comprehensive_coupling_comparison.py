#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标量耦合常数预测 - 综合比较框架
集成所有方法的全面比较：简单MLP、GNN框架、高级特征工程
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# 导入我们创建的各个模块
try:
    from coupling_gnn_frameworks import compare_coupling_models as gnn_compare
    from advanced_coupling_features import main as advanced_main
    GNN_FRAMEWORKS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GNN框架导入失败: {e}")
    GNN_FRAMEWORKS_AVAILABLE = False

try:
    from advanced_coupling_features import AdvancedCouplingDataset, AdvancedCouplingMLP
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 高级特征模块导入失败: {e}")
    ADVANCED_FEATURES_AVAILABLE = False


class CouplingComparisonSuite:
    """标量耦合常数预测综合比较套件"""

    def __init__(self, data_path, max_samples=3000):
        self.data_path = data_path
        self.max_samples = max_samples
        self.results = {}

        print("🚀 初始化标量耦合常数综合比较框架")
        print(f"数据路径: {data_path}")
        print(f"最大样本数: {max_samples}")

    def run_simple_baseline(self):
        """运行简单基线模型"""
        print(f"\n{'='*60}")
        print("📊 运行简单基线MLP模型")
        print(f"{'='*60}")

        try:
            # 简单的MLP实现
            from torch.utils.data import DataLoader, Dataset
            from sklearn.preprocessing import StandardScaler, LabelEncoder

            class SimpleDataset(Dataset):
                def __init__(self, data_path, max_samples):
                    # 加载数据
                    train_df = pd.read_csv(os.path.join(data_path, 'train.csv')).head(max_samples)
                    structures_df = pd.read_csv(os.path.join(data_path, 'structures.csv'))

                    features, targets = self._process_data(train_df, structures_df)

                    self.scaler = StandardScaler()
                    self.features = torch.tensor(self.scaler.fit_transform(features), dtype=torch.float32)
                    self.targets = torch.tensor(targets, dtype=torch.float32)

                def _process_data(self, train_df, structures_df):
                    features = []
                    targets = []

                    for _, row in train_df.iterrows():
                        mol_name = row['molecule_name']
                        atom_0 = row['atom_index_0']
                        atom_1 = row['atom_index_1']
                        coupling_type = row['type']
                        target = row['scalar_coupling_constant']

                        mol_struct = structures_df[structures_df['molecule_name'] == mol_name]
                        if len(mol_struct) <= max(atom_0, atom_1):
                            continue

                        # 简单特征
                        coords_0 = mol_struct.iloc[atom_0][['x', 'y', 'z']].values
                        coords_1 = mol_struct.iloc[atom_1][['x', 'y', 'z']].values
                        distance = np.linalg.norm(coords_1 - coords_0)

                        atom_0_type = mol_struct.iloc[atom_0]['atom']
                        atom_1_type = mol_struct.iloc[atom_1]['atom']

                        # 编码特征
                        atom_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
                        type_map = {'1JHC': 0, '2JHH': 1, '3JHH': 2, '1JCC': 3,
                                   '2JHC': 4, '2JCH': 5, '3JHC': 6}

                        feature_vec = [
                            distance,
                            coords_1[0] - coords_0[0],
                            coords_1[1] - coords_0[1],
                            coords_1[2] - coords_0[2],
                            atom_map.get(atom_0_type, 0),
                            atom_map.get(atom_1_type, 0),
                            type_map.get(coupling_type, 0),
                            len(mol_struct)
                        ]

                        features.append(feature_vec)
                        targets.append(target)

                    return np.array(features), np.array(targets)

                def __len__(self):
                    return len(self.features)

                def __getitem__(self, idx):
                    return self.features[idx], self.targets[idx]

            # 简单MLP模型
            class SimpleMLP(nn.Module):
                def __init__(self, input_dim=8):
                    super().__init__()
                    self.mlp = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 1)
                    )

                def forward(self, x):
                    return self.mlp(x)

            # 训练简单模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dataset = SimpleDataset(self.data_path, self.max_samples)

            # 数据划分
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size

            train_data, val_data, test_data = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )

            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

            model = SimpleMLP().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # 训练
            start_time = time.time()
            best_val_loss = float('inf')

            for epoch in range(30):
                model.train()
                for features, targets in train_loader:
                    features, targets = features.to(device), targets.to(device)
                    optimizer.zero_grad()
                    pred = model(features).squeeze()
                    loss = criterion(pred, targets)
                    loss.backward()
                    optimizer.step()

                # 验证
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for features, targets in val_loader:
                        features, targets = features.to(device), targets.to(device)
                        pred = model(features).squeeze()
                        val_loss += criterion(pred, targets).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict().copy()

            # 测试
            model.load_state_dict(best_model)
            model.eval()
            predictions = []
            targets_list = []

            with torch.no_grad():
                for features, targets in test_loader:
                    features, targets = features.to(device), targets.to(device)
                    pred = model(features).squeeze()
                    predictions.extend(pred.cpu().numpy())
                    targets_list.extend(targets.cpu().numpy())

            predictions = np.array(predictions)
            targets_array = np.array(targets_list)

            training_time = time.time() - start_time

            # 计算指标
            mae = mean_absolute_error(targets_array, predictions)
            mse = mean_squared_error(targets_array, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(targets_array, predictions)

            result = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'training_time': training_time,
                'description': '简单基线MLP',
                'predictions': predictions,
                'targets': targets_array
            }

            print(f"✅ 简单基线完成:")
            print(f"   MAE: {mae:.4f}")
            print(f"   R²: {r2:.4f}")
            print(f"   参数量: {result['num_parameters']:,}")
            print(f"   训练时间: {training_time:.1f}s")

            return result

        except Exception as e:
            print(f"❌ 简单基线失败: {e}")
            return {'error': str(e)}

    def run_gnn_frameworks(self):
        """运行GNN框架比较"""
        print(f"\n{'='*60}")
        print("🔥 运行GNN框架比较")
        print(f"{'='*60}")

        if not GNN_FRAMEWORKS_AVAILABLE:
            print("❌ GNN框架不可用")
            return {'error': 'GNN框架不可用'}

        try:
            results = gnn_compare(
                data_path=self.data_path,
                max_samples=self.max_samples,
                test_split=0.2,
                val_split=0.1
            )
            print("✅ GNN框架比较完成")
            return results
        except Exception as e:
            print(f"❌ GNN框架比较失败: {e}")
            return {'error': str(e)}

    def run_advanced_features(self):
        """运行高级特征工程"""
        print(f"\n{'='*60}")
        print("🚀 运行高级特征工程")
        print(f"{'='*60}")

        if not ADVANCED_FEATURES_AVAILABLE:
            print("❌ 高级特征模块不可用")
            return {'error': '高级特征模块不可用'}

        try:
            # 修改数据路径并运行
            original_path = '/Users/xiaotingzhou/Downloads/GNN/Dataset/scalar_coupling_constant'

            # 临时修改模块中的数据路径
            import sys
            import importlib

            # 重新加载模块并设置路径
            if 'advanced_coupling_features' in sys.modules:
                importlib.reload(sys.modules['advanced_coupling_features'])

            from advanced_coupling_features import main as advanced_main

            # 运行高级特征版本
            result = advanced_main()
            print("✅ 高级特征工程完成")
            return result
        except Exception as e:
            print(f"❌ 高级特征工程失败: {e}")
            return {'error': str(e)}

    def run_comprehensive_comparison(self):
        """运行综合比较"""
        print(f"\n{'='*80}")
        print("🎯 开始综合比较")
        print(f"{'='*80}")

        start_time = time.time()

        # 运行所有方法
        simple_result = self.run_simple_baseline()
        gnn_results = self.run_gnn_frameworks()
        advanced_result = self.run_advanced_features()

        total_time = time.time() - start_time

        # 整理结果
        self.results = {
            'Simple_Baseline': simple_result,
            'GNN_Frameworks': gnn_results,
            'Advanced_Features': advanced_result,
            'total_comparison_time': total_time
        }

        # 生成综合报告
        self._generate_comprehensive_report()

        return self.results

    def _generate_comprehensive_report(self):
        """生成综合比较报告"""
        print(f"\n{'='*80}")
        print("📋 标量耦合常数预测方法综合比较报告")
        print(f"{'='*80}")

        # 收集所有有效结果
        all_methods = {}

        # 简单基线
        if 'error' not in self.results['Simple_Baseline']:
            all_methods['简单基线MLP'] = self.results['Simple_Baseline']

        # GNN框架结果
        if 'error' not in self.results['GNN_Frameworks']:
            gnn_results = self.results['GNN_Frameworks']
            for method_name, result in gnn_results.items():
                if 'error' not in result:
                    all_methods[f"GNN-{method_name}"] = result

        # 高级特征结果
        if 'error' not in self.results['Advanced_Features']:
            all_methods['高级特征MLP'] = self.results['Advanced_Features']

        if not all_methods:
            print("❌ 没有成功的方法可以比较")
            return

        # 性能排序
        sorted_methods = sorted(all_methods.items(), key=lambda x: x[1].get('MAE', float('inf')))

        print(f"\n🏆 方法性能排名 (按MAE):")
        print("-" * 100)
        print(f"{'排名':<4} {'方法':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'参数量':<12} {'训练时间':<10}")
        print("-" * 100)

        for i, (method_name, result) in enumerate(sorted_methods, 1):
            mae = result.get('MAE', 0)
            rmse = result.get('RMSE', 0)
            r2 = result.get('R2', 0)
            params = result.get('num_parameters', 0)
            time_s = result.get('training_time', 0)

            print(f"{i:<4} {method_name:<25} {mae:<10.4f} {rmse:<10.4f} "
                  f"{r2:<10.4f} {params:<12,} {time_s:<10.1f}s")

        # 生成可视化
        self._create_comprehensive_plots(all_methods)

        # 保存详细结果
        self._save_comprehensive_results(all_methods)

        print(f"\n⏱️ 总比较时间: {self.results['total_comparison_time']/60:.2f} 分钟")
        print("\n💾 详细结果已保存到 comprehensive_coupling_results.txt")
        print("📊 可视化图表已保存为 comprehensive_coupling_comparison.png")

    def _create_comprehensive_plots(self, all_methods):
        """创建综合比较可视化"""
        if len(all_methods) < 2:
            print("方法太少，跳过可视化")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('标量耦合常数预测方法综合比较', fontsize=16, fontweight='bold')

        methods = list(all_methods.keys())
        maes = [all_methods[m].get('MAE', 0) for m in methods]
        r2s = [all_methods[m].get('R2', 0) for m in methods]
        params = [all_methods[m].get('num_parameters', 0) for m in methods]
        times = [all_methods[m].get('training_time', 0) for m in methods]
        rmses = [all_methods[m].get('RMSE', 0) for m in methods]

        # 1. MAE对比
        axes[0, 0].bar(methods, maes, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('平均绝对误差 (MAE) 比较')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. R²对比
        axes[0, 1].bar(methods, r2s, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('决定系数 (R²) 比较')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. RMSE对比
        axes[0, 2].bar(methods, rmses, color='orange', alpha=0.7)
        axes[0, 2].set_title('均方根误差 (RMSE) 比较')
        axes[0, 2].set_ylabel('RMSE')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # 4. 参数量对比
        axes[1, 0].bar(methods, params, color='salmon', alpha=0.7)
        axes[1, 0].set_title('模型参数量比较')
        axes[1, 0].set_ylabel('参数数量')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 5. 训练时间对比
        axes[1, 1].bar(methods, times, color='plum', alpha=0.7)
        axes[1, 1].set_title('训练时间比较')
        axes[1, 1].set_ylabel('时间 (秒)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 6. 性能效率散点图
        scatter = axes[1, 2].scatter(params, maes, c=times, s=100, alpha=0.7, cmap='viridis')
        axes[1, 2].set_xlabel('参数量')
        axes[1, 2].set_ylabel('MAE')
        axes[1, 2].set_title('性能 vs 复杂度 (颜色=训练时间)')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=axes[1, 2])
        cbar.set_label('训练时间 (秒)')

        # 添加方法标签
        for i, method in enumerate(methods):
            axes[1, 2].annotate(method.replace('GNN-', '').replace('Coupling', ''),
                              (params[i], maes[i]), xytext=(5, 5),
                              textcoords='offset points', fontsize=8)

        plt.tight_layout()
        plt.savefig('comprehensive_coupling_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _save_comprehensive_results(self, all_methods):
        """保存综合比较结果"""
        with open('comprehensive_coupling_results.txt', 'w', encoding='utf-8') as f:
            f.write("标量耦合常数预测方法综合比较结果\n")
            f.write("=" * 60 + "\n\n")

            f.write("实验设置:\n")
            f.write(f"- 数据路径: {self.data_path}\n")
            f.write(f"- 最大样本数: {self.max_samples}\n")
            f.write(f"- 总比较时间: {self.results['total_comparison_time']/60:.2f} 分钟\n\n")

            # 按性能排序
            sorted_methods = sorted(all_methods.items(), key=lambda x: x[1].get('MAE', float('inf')))

            f.write("方法性能排名 (按MAE):\n")
            f.write("-" * 60 + "\n")

            for i, (method_name, result) in enumerate(sorted_methods, 1):
                f.write(f"{i}. {method_name}\n")
                f.write(f"   MAE: {result.get('MAE', 0):.6f}\n")
                f.write(f"   RMSE: {result.get('RMSE', 0):.6f}\n")
                f.write(f"   R²: {result.get('R2', 0):.6f}\n")
                f.write(f"   参数量: {result.get('num_parameters', 0):,}\n")
                f.write(f"   训练时间: {result.get('training_time', 0):.1f}s\n")
                if 'description' in result:
                    f.write(f"   描述: {result['description']}\n")
                f.write("\n")

            f.write("主要结论:\n")
            f.write("-" * 30 + "\n")
            if sorted_methods:
                best_method, best_result = sorted_methods[0]
                f.write(f"• 最佳性能方法: {best_method}\n")
                f.write(f"• 最佳MAE: {best_result.get('MAE', 0):.6f}\n")

                # 效率分析
                efficient_methods = [m for m in sorted_methods if m[1].get('training_time', 0) < 60]
                if efficient_methods:
                    f.write(f"• 最高效方法: {efficient_methods[0][0]}\n")

                # 平衡方法
                balanced_methods = sorted(sorted_methods,
                                        key=lambda x: x[1].get('MAE', 1) * x[1].get('num_parameters', 1))
                if balanced_methods:
                    f.write(f"• 平衡性能/复杂度方法: {balanced_methods[0][0]}\n")


def main():
    """主函数"""
    print("🎯 标量耦合常数预测综合比较框架")

    # 数据路径
    data_path = '/Users/xiaotingzhou/Downloads/GNN/Dataset/scalar_coupling_constant'

    # 检查数据集
    if not os.path.exists(data_path):
        print(f"❌ 数据集路径不存在: {data_path}")
        return None

    # 创建比较套件
    comparison_suite = CouplingComparisonSuite(data_path, max_samples=3000)

    # 运行综合比较
    results = comparison_suite.run_comprehensive_comparison()

    print("\n🎉 综合比较完成!")

    return results


if __name__ == "__main__":
    results = main()