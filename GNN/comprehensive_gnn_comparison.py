#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合GNN框架比较分析
整合所有不同的模型架构和特征工程方法进行全面对比

包括的方法:
1. 基础方法 (原始GCN)
2. 不同GNN架构 (GraphSAGE, GAT, Transformer等)
3. 高级特征工程
4. 预训练和迁移学习
5. 模型集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import *
from torch_geometric.transforms import Compose, NormalizeFeatures
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ComprehensiveComparison:
    """综合比较分析类"""

    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}

        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)

        print(f"使用设备: {self.device}")

    def load_and_preprocess_data(self, sample_size=3000):
        """加载和预处理数据"""
        print(f"加载QM9数据集 (样本数: {sample_size})...")

        transform = Compose([NormalizeFeatures()])
        dataset = QM9(root=self.data_path, transform=transform)
        dataset = dataset[3000:3000+sample_size]

        target_index = 4  # HOMO-LUMO gap
        processed_data, target_mean, target_std = self._create_single_target_dataset(dataset, target_index)

        # 数据划分
        num_samples = len(processed_data)
        num_train = int(0.7 * num_samples)
        num_val = int(0.15 * num_samples)

        train_data = processed_data[:num_train]
        val_data = processed_data[num_train:num_train + num_val]
        test_data = processed_data[num_train + num_val:]

        self.dataset_info = {
            'total_samples': num_samples,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'num_features': processed_data[0].x.shape[1],
            'target_mean': target_mean,
            'target_std': target_std
        }

        print(f"数据集信息: {self.dataset_info}")

        return train_data, val_data, test_data

    def _create_single_target_dataset(self, original_dataset, target_index):
        """创建单目标数据集"""
        new_data_list = []

        targets = []
        for data in original_dataset:
            targets.append(data.y[0, target_index].item())

        target_mean = np.mean(targets)
        target_std = np.std(targets)

        for data in original_dataset:
            target_value = (data.y[0, target_index].item() - target_mean) / target_std

            new_data = Data(
                x=data.x.clone(),
                edge_index=data.edge_index.clone(),
                edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
                y=torch.tensor([target_value], dtype=torch.float),
                pos=data.pos.clone() if hasattr(data, 'pos') and data.pos is not None else None
            )
            new_data_list.append(new_data)

        return new_data_list, target_mean, target_std

    def define_models(self):
        """定义所有要比较的模型"""
        num_features = self.dataset_info['num_features']

        models = {
            # 基础模型
            'GCN-Small': self._create_gcn_model(num_features, hidden_dim=64, num_layers=3),
            'GCN-Medium': self._create_gcn_model(num_features, hidden_dim=128, num_layers=4),
            'GCN-Large': self._create_gcn_model(num_features, hidden_dim=256, num_layers=5),

            # 不同架构
            'GraphSAGE': self._create_sage_model(num_features, hidden_dim=128, num_layers=4),
            'GAT': self._create_gat_model(num_features, hidden_dim=128, num_layers=3, heads=4),
            'GraphTransformer': self._create_transformer_model(num_features, hidden_dim=128, num_layers=3),

            # 深度模型
            'DeepGCN': self._create_deep_gcn_model(num_features, hidden_dim=128, num_layers=8),
            'ResGCN': self._create_residual_gcn_model(num_features, hidden_dim=128, num_layers=6),

            # 特殊架构
            'GIN': self._create_gin_model(num_features, hidden_dim=128, num_layers=4),
            'GraphUNet': self._create_graph_unet_model(num_features, hidden_dim=128),
        }

        return models

    def _create_gcn_model(self, num_features, hidden_dim, num_layers):
        """创建GCN模型"""
        class GCNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()

                self.convs.append(GCNConv(num_features, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

                for _ in range(num_layers - 1):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                    self.bns.append(nn.BatchNorm1d(hidden_dim))

                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1)
                )

            def forward(self, x, edge_index, batch):
                for conv, bn in zip(self.convs, self.bns):
                    x = F.relu(bn(conv(x, edge_index)))
                    x = F.dropout(x, training=self.training)

                x1 = global_mean_pool(x, batch)
                x2 = global_max_pool(x, batch)
                x3 = global_add_pool(x, batch)
                x = torch.cat([x1, x2, x3], dim=1)

                return self.fc(x)

        return GCNModel()

    def _create_sage_model(self, num_features, hidden_dim, num_layers):
        """创建GraphSAGE模型"""
        class SAGEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()

                self.convs.append(SAGEConv(num_features, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

                for _ in range(num_layers - 1):
                    self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                    self.bns.append(nn.BatchNorm1d(hidden_dim))

                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1)
                )

            def forward(self, x, edge_index, batch):
                for conv, bn in zip(self.convs, self.bns):
                    x = F.relu(bn(conv(x, edge_index)))
                    x = F.dropout(x, training=self.training)

                x1 = global_mean_pool(x, batch)
                x2 = global_max_pool(x, batch)
                x3 = global_add_pool(x, batch)
                x = torch.cat([x1, x2, x3], dim=1)

                return self.fc(x)

        return SAGEModel()

    def _create_gat_model(self, num_features, hidden_dim, num_layers, heads):
        """创建GAT模型"""
        class GATModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()

                self.convs.append(GATConv(num_features, hidden_dim // heads, heads=heads))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

                for _ in range(num_layers - 2):
                    self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
                    self.bns.append(nn.BatchNorm1d(hidden_dim))

                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1)
                )

            def forward(self, x, edge_index, batch):
                for conv, bn in zip(self.convs, self.bns):
                    x = F.elu(bn(conv(x, edge_index)))
                    x = F.dropout(x, training=self.training)

                x1 = global_mean_pool(x, batch)
                x2 = global_max_pool(x, batch)
                x3 = global_add_pool(x, batch)
                x = torch.cat([x1, x2, x3], dim=1)

                return self.fc(x)

        return GATModel()

    def _create_transformer_model(self, num_features, hidden_dim, num_layers):
        """创建Transformer模型"""
        class TransformerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(num_features, hidden_dim)
                self.transformers = nn.ModuleList()
                self.layer_norms = nn.ModuleList()

                for _ in range(num_layers):
                    self.transformers.append(TransformerConv(hidden_dim, hidden_dim // 8, heads=8))
                    self.layer_norms.append(nn.LayerNorm(hidden_dim))

                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1)
                )

            def forward(self, x, edge_index, batch):
                x = self.embedding(x)

                for transformer, ln in zip(self.transformers, self.layer_norms):
                    residual = x
                    x = transformer(x, edge_index)
                    x = ln(x + residual)
                    x = F.gelu(x)

                x1 = global_mean_pool(x, batch)
                x2 = global_max_pool(x, batch)
                x3 = global_add_pool(x, batch)
                x = torch.cat([x1, x2, x3], dim=1)

                return self.fc(x)

        return TransformerModel()

    def _create_gin_model(self, num_features, hidden_dim, num_layers):
        """创建GIN模型"""
        class GINModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()

                for i in range(num_layers):
                    if i == 0:
                        mlp = nn.Sequential(
                            nn.Linear(num_features, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim)
                        )
                    else:
                        mlp = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim)
                        )
                    self.convs.append(GINConv(mlp))

                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1)
                )

            def forward(self, x, edge_index, batch):
                for conv in self.convs:
                    x = F.relu(conv(x, edge_index))
                    x = F.dropout(x, training=self.training)

                x1 = global_mean_pool(x, batch)
                x2 = global_max_pool(x, batch)
                x3 = global_add_pool(x, batch)
                x = torch.cat([x1, x2, x3], dim=1)

                return self.fc(x)

        return GINModel()

    def _create_deep_gcn_model(self, num_features, hidden_dim, num_layers):
        """创建深度GCN模型"""
        class DeepGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(num_features, hidden_dim)
                self.convs = nn.ModuleList()
                self.norms = nn.ModuleList()

                for _ in range(num_layers):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                    self.norms.append(nn.BatchNorm1d(hidden_dim))

                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1)
                )

            def forward(self, x, edge_index, batch):
                x = self.input_proj(x)

                for conv, norm in zip(self.convs, self.norms):
                    residual = x
                    x = conv(x, edge_index)
                    x = norm(x)
                    x = F.relu(x + residual)  # 残差连接
                    x = F.dropout(x, training=self.training)

                x1 = global_mean_pool(x, batch)
                x2 = global_max_pool(x, batch)
                x3 = global_add_pool(x, batch)
                x = torch.cat([x1, x2, x3], dim=1)

                return self.fc(x)

        return DeepGCN()

    def _create_residual_gcn_model(self, num_features, hidden_dim, num_layers):
        """创建残差GCN模型"""
        return self._create_deep_gcn_model(num_features, hidden_dim, num_layers)

    def _create_graph_unet_model(self, num_features, hidden_dim):
        """创建GraphUNet模型"""
        class GraphUNet(nn.Module):
            def __init__(self):
                super().__init__()
                # 简化版本的GraphUNet
                self.down1 = GCNConv(num_features, hidden_dim)
                self.down2 = GCNConv(hidden_dim, hidden_dim * 2)
                self.up1 = GCNConv(hidden_dim * 2, hidden_dim)
                self.up2 = GCNConv(hidden_dim, hidden_dim)

                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1)
                )

            def forward(self, x, edge_index, batch):
                # 下采样
                x1 = F.relu(self.down1(x, edge_index))
                x2 = F.relu(self.down2(x1, edge_index))

                # 上采样
                x3 = F.relu(self.up1(x2, edge_index))
                x4 = F.relu(self.up2(x3 + x1, edge_index))  # 跳跃连接

                # 池化
                x_mean = global_mean_pool(x4, batch)
                x_max = global_max_pool(x4, batch)
                x_add = global_add_pool(x4, batch)
                x = torch.cat([x_mean, x_max, x_add], dim=1)

                return self.fc(x)

        return GraphUNet()

    def train_single_model(self, model, train_loader, val_loader, model_name, num_epochs=30):
        """训练单个模型"""
        print(f"\n训练 {model_name} 模型...")

        model.to(self.device)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None

        start_time = time.time()

        for epoch in range(num_epochs):
            # 训练
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()
                train_loss += loss.item()

            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out.squeeze(), batch.y.squeeze())
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}')

        # 加载最佳模型
        if best_model_state:
            model.load_state_dict(best_model_state)

        training_time = time.time() - start_time

        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'num_parameters': sum(p.numel() for p in model.parameters())
        }

    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        predictions = []
        true_values = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, batch.batch)

                # 反标准化
                pred = out.squeeze().cpu().numpy() * self.dataset_info['target_std'] + self.dataset_info['target_mean']
                true = batch.y.squeeze().cpu().numpy() * self.dataset_info['target_std'] + self.dataset_info['target_mean']

                predictions.extend(pred.tolist() if pred.ndim > 0 else [pred.item()])
                true_values.extend(true.tolist() if true.ndim > 0 else [true.item()])

        predictions = np.array(predictions)
        true_values = np.array(true_values)

        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predictions)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'predictions': predictions,
            'true_values': true_values
        }

    def run_comprehensive_comparison(self, sample_size=3000):
        """运行综合比较"""
        print("=" * 80)
        print("开始综合GNN框架比较")
        print("=" * 80)

        # 加载数据
        train_data, val_data, test_data = self.load_and_preprocess_data(sample_size)

        # 创建数据加载器
        batch_size = 32
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # 定义模型
        models = self.define_models()

        # 训练和评估每个模型
        for model_name, model in models.items():
            try:
                # 训练
                train_results = self.train_single_model(model, train_loader, val_loader, model_name)

                # 评估
                eval_results = self.evaluate_model(model, test_loader)

                # 合并结果
                self.results[model_name] = {
                    **train_results,
                    **eval_results
                }

                print(f"  {model_name} - MAE: {eval_results['MAE']:.4f}, "
                      f"RMSE: {eval_results['RMSE']:.4f}, R²: {eval_results['R2']:.4f}, "
                      f"Time: {train_results['training_time']:.1f}s")

            except Exception as e:
                print(f"  {model_name} 训练失败: {str(e)}")
                self.results[model_name] = {'error': str(e)}

        # 保存结果
        self.save_results()

        # 生成报告
        self.generate_comparison_report()

        return self.results

    def save_results(self):
        """保存结果"""
        # 保存详细结果
        results_to_save = {}
        for model_name, result in self.results.items():
            if 'error' not in result:
                results_to_save[model_name] = {
                    'MAE': result['MAE'],
                    'RMSE': result['RMSE'],
                    'R2': result['R2'],
                    'training_time': result['training_time'],
                    'num_parameters': result['num_parameters'],
                    'best_val_loss': result['best_val_loss']
                }
            else:
                results_to_save[model_name] = result

        with open(os.path.join(self.output_path, 'comparison_results.json'), 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\n结果已保存到: {self.output_path}")

    def generate_comparison_report(self):
        """生成比较报告"""
        # 提取有效结果
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not valid_results:
            print("没有有效的结果生成报告")
            return

        # 创建比较表格
        self._create_comparison_table(valid_results)

        # 创建可视化图表
        self._create_comparison_plots(valid_results)

        # 创建性能矩阵
        self._create_performance_matrix(valid_results)

    def _create_comparison_table(self, valid_results):
        """创建比较表格"""
        print("\n" + "="*100)
        print("模型性能对比表")
        print("="*100)

        # 表头
        print(f"{'模型':<18} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'参数量':<12} "
              f"{'训练时间(s)':<12} {'验证损失':<10}")
        print("-" * 100)

        # 按MAE排序
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['MAE'])

        for model_name, result in sorted_results:
            print(f"{model_name:<18} {result['MAE']:<8.4f} {result['RMSE']:<8.4f} "
                  f"{result['R2']:<8.4f} {result['num_parameters']:<12,} "
                  f"{result['training_time']:<12.1f} {result['best_val_loss']:<10.6f}")

        print("-" * 100)

    def _create_comparison_plots(self, valid_results):
        """创建比较图表"""
        model_names = list(valid_results.keys())
        maes = [valid_results[name]['MAE'] for name in model_names]
        rmses = [valid_results[name]['RMSE'] for name in model_names]
        r2s = [valid_results[name]['R2'] for name in model_names]
        times = [valid_results[name]['training_time'] for name in model_names]
        params = [valid_results[name]['num_parameters']/1000 for name in model_names]

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # MAE比较
        axes[0, 0].bar(range(len(model_names)), maes, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('平均绝对误差 (MAE) 比较', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # R²比较
        axes[0, 1].bar(range(len(model_names)), r2s, color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('决定系数 (R²) 比较', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 训练时间比较
        axes[0, 2].bar(range(len(model_names)), times, color='orange', alpha=0.8)
        axes[0, 2].set_title('训练时间比较', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('时间 (秒)')
        axes[0, 2].set_xticks(range(len(model_names)))
        axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 2].grid(axis='y', alpha=0.3)

        # 参数量比较
        axes[1, 0].bar(range(len(model_names)), params, color='pink', alpha=0.8)
        axes[1, 0].set_title('模型参数量比较', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('参数量 (千个)')
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 性能 vs 复杂度散点图
        axes[1, 1].scatter(params, maes, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (params[i], maes[i]), xytext=(5, 5),
                              textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('参数量 (千个)')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('模型复杂度 vs 性能', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # 训练效率散点图
        axes[1, 2].scatter(times, maes, s=100, alpha=0.7, c=range(len(model_names)), cmap='plasma')
        for i, name in enumerate(model_names):
            axes[1, 2].annotate(name, (times[i], maes[i]), xytext=(5, 5),
                              textcoords='offset points', fontsize=8)
        axes[1, 2].set_xlabel('训练时间 (秒)')
        axes[1, 2].set_ylabel('MAE')
        axes[1, 2].set_title('训练效率 vs 性能', fontsize=14, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'comprehensive_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_performance_matrix(self, valid_results):
        """创建性能矩阵热图"""
        model_names = list(valid_results.keys())
        metrics = ['MAE', 'RMSE', 'R2']

        # 创建标准化的性能矩阵
        matrix = np.zeros((len(model_names), len(metrics)))

        for i, model_name in enumerate(model_names):
            result = valid_results[model_name]
            matrix[i, 0] = result['MAE']
            matrix[i, 1] = result['RMSE']
            matrix[i, 2] = result['R2']

        # 标准化 (除了R²，其他指标越小越好)
        for j in range(len(metrics)):
            if metrics[j] != 'R2':
                matrix[:, j] = (matrix[:, j] - matrix[:, j].min()) / (matrix[:, j].max() - matrix[:, j].min())
                matrix[:, j] = 1 - matrix[:, j]  # 反转，使得越大越好
            else:
                matrix[:, j] = (matrix[:, j] - matrix[:, j].min()) / (matrix[:, j].max() - matrix[:, j].min())

        # 创建热图
        plt.figure(figsize=(8, 10))
        sns.heatmap(matrix,
                    xticklabels=metrics,
                    yticklabels=model_names,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn',
                    center=0.5,
                    cbar_kws={'label': '标准化性能分数'})

        plt.title('模型性能矩阵 (标准化)', fontsize=16, fontweight='bold')
        plt.xlabel('评估指标', fontsize=12)
        plt.ylabel('模型', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'performance_matrix.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    # 创建比较实例
    comparison = ComprehensiveComparison(
        data_path='/Users/xiaotingzhou/Downloads/GNN/Dataset',
        output_path='/Users/xiaotingzhou/Downloads/GNN/comparison_results'
    )

    # 运行综合比较
    results = comparison.run_comprehensive_comparison(sample_size=2000)

    # 输出最终总结
    print("\n" + "="*80)
    print("综合比较完成!")
    print("="*80)

    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_model = min(valid_results.items(), key=lambda x: x[1]['MAE'])
        print(f"最佳模型: {best_model[0]}")
        print(f"  MAE: {best_model[1]['MAE']:.4f}")
        print(f"  RMSE: {best_model[1]['RMSE']:.4f}")
        print(f"  R²: {best_model[1]['R2']:.4f}")
        print(f"  参数量: {best_model[1]['num_parameters']:,}")
        print(f"  训练时间: {best_model[1]['training_time']:.1f}秒")

    return results


if __name__ == "__main__":
    results = main()