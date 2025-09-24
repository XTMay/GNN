#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN分子属性预测框架比较
包含多种不同复杂度的GNN模型架构用于QM9数据集

模型包括:
1. 基础GCN模型 (原始版本)
2. GraphSAGE模型
3. 图注意力网络 (GAT)
4. 图Transformer
5. SchNet模型 (3D坐标)
6. 集成模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GCNConv, SAGEConv, GATConv, TransformerConv,
                               global_mean_pool, global_max_pool, global_add_pool,
                               BatchNorm, LayerNorm, GraphNorm)
from torch_geometric.transforms import Compose, NormalizeFeatures
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. 基础GCN模型 (原始版本)
# =============================================================================
class BasicGCN(nn.Module):
    """基础GCN模型"""

    def __init__(self, num_features, hidden_dim=128, num_layers=4, dropout=0.2):
        super(BasicGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 输入层
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.batch_norms.append(BatchNorm(hidden_dim))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        # GCN卷积
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x3 = global_add_pool(x, batch)
        x = torch.cat([x1, x2, x3], dim=1)

        return self.fc(x)


# =============================================================================
# 2. GraphSAGE模型
# =============================================================================
class GraphSAGEModel(nn.Module):
    """GraphSAGE模型 - 使用采样和聚合机制"""

    def __init__(self, num_features, hidden_dim=128, num_layers=4, dropout=0.2):
        super(GraphSAGEModel, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 输入层
        self.convs.append(SAGEConv(num_features, hidden_dim))
        self.batch_norms.append(BatchNorm(hidden_dim))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):
        # GraphSAGE卷积
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 多种池化组合
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x3 = global_add_pool(x, batch)
        x = torch.cat([x1, x2, x3], dim=1)

        return self.fc(x)


# =============================================================================
# 3. 图注意力网络 (GAT)
# =============================================================================
class GATModel(nn.Module):
    """图注意力网络模型"""

    def __init__(self, num_features, hidden_dim=128, num_layers=4, heads=4, dropout=0.2):
        super(GATModel, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 输入层
        self.convs.append(GATConv(num_features, hidden_dim // heads, heads=heads, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_dim))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # 最后一层 - 单头输出
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_dim))

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):
        # GAT卷积
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)  # GAT通常使用ELU激活
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x3 = global_add_pool(x, batch)
        x = torch.cat([x1, x2, x3], dim=1)

        return self.fc(x)


# =============================================================================
# 4. 图Transformer模型
# =============================================================================
class GraphTransformer(nn.Module):
    """图Transformer模型"""

    def __init__(self, num_features, hidden_dim=128, num_layers=4, heads=8, dropout=0.2):
        super(GraphTransformer, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 输入嵌入
        self.input_embedding = nn.Linear(num_features, hidden_dim)

        # Transformer层
        self.transformers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.transformers.append(
                TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            )
            self.layer_norms.append(LayerNorm(hidden_dim))

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):
        # 输入嵌入
        x = self.input_embedding(x)

        # Transformer层
        for i in range(self.num_layers):
            residual = x
            x = self.transformers[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.gelu(x)  # Transformer通常使用GELU
            x = x + residual  # 残差连接
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x3 = global_add_pool(x, batch)
        x = torch.cat([x1, x2, x3], dim=1)

        return self.fc(x)


# =============================================================================
# 5. SchNet模型 (利用3D坐标)
# =============================================================================
class SchNetModel(nn.Module):
    """SchNet模型 - 利用3D坐标信息"""

    def __init__(self, num_features, hidden_dim=128, num_layers=4, cutoff=5.0, dropout=0.2):
        super(SchNetModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.dropout = dropout

        # 原子嵌入
        self.atom_embedding = nn.Linear(num_features, hidden_dim)

        # 距离展开
        self.distance_expansion = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # SchNet交互层
        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            self.interactions.append(SchNetInteraction(hidden_dim))

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch, pos=None):
        # 如果没有位置信息，使用基础GCN
        if pos is None:
            return self._forward_without_pos(x, edge_index, batch)

        # 原子嵌入
        x = self.atom_embedding(x)

        # 计算距离
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)

        # 距离展开
        edge_attr = self.distance_expansion(dist)

        # SchNet交互
        for interaction in self.interactions:
            x = interaction(x, edge_index, edge_attr)

        # 全局池化
        x = global_add_pool(x, batch)

        return self.fc(x)

    def _forward_without_pos(self, x, edge_index, batch):
        """没有3D坐标时的前向传播"""
        x = self.atom_embedding(x)

        # 简单的图卷积替代
        for _ in range(self.num_layers):
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        return self.fc(x)


class SchNetInteraction(nn.Module):
    """SchNet交互层"""

    def __init__(self, hidden_dim):
        super(SchNetInteraction, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Softplus()
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index

        # 计算交互
        edge_feat = self.mlp(edge_attr) * self.conv(x[row])

        # 聚合
        out = torch.zeros_like(x)
        out = out.index_add_(0, col, edge_feat)

        # 更新
        out = self.act(x + self.lin(out))

        return out


# =============================================================================
# 6. 深度残差网络
# =============================================================================
class ResidualGCN(nn.Module):
    """带残差连接的深度GCN"""

    def __init__(self, num_features, hidden_dim=128, num_layers=8, dropout=0.2):
        super(ResidualGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 输入投影
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # 残差块
        self.res_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.res_blocks.append(ResidualBlock(hidden_dim, dropout))

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):
        # 输入投影
        x = self.input_proj(x)

        # 残差块
        for block in self.res_blocks:
            x = block(x, edge_index) + x  # 残差连接

        # 图级池化
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x3 = global_add_pool(x, batch)
        x = torch.cat([x1, x2, x3], dim=1)

        return self.fc(x)


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, hidden_dim, dropout):
        super(ResidualBlock, self).__init__()

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index)
        out = self.bn1(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        out = self.conv2(out, edge_index)
        out = self.bn2(out)

        return out


# =============================================================================
# 7. 集成模型
# =============================================================================
class EnsembleModel(nn.Module):
    """集成多个不同的GNN模型"""

    def __init__(self, num_features, hidden_dim=128):
        super(EnsembleModel, self).__init__()

        # 不同的子模型
        self.gcn = BasicGCN(num_features, hidden_dim, num_layers=4)
        self.sage = GraphSAGEModel(num_features, hidden_dim, num_layers=4)
        self.gat = GATModel(num_features, hidden_dim, num_layers=3, heads=4)

        # 元学习器
        self.meta_learner = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x, edge_index, batch):
        # 获取各个子模型的预测
        pred1 = self.gcn(x, edge_index, batch)
        pred2 = self.sage(x, edge_index, batch)
        pred3 = self.gat(x, edge_index, batch)

        # 堆叠预测结果
        ensemble_input = torch.cat([pred1, pred2, pred3], dim=1)

        # 元学习器组合预测
        final_pred = self.meta_learner(ensemble_input)

        return final_pred


# =============================================================================
# 数据处理函数
# =============================================================================
def create_single_target_dataset(original_dataset, target_index):
    """创建单目标数据集"""
    new_data_list = []

    # 计算统计信息
    targets = []
    for data in original_dataset:
        targets.append(data.y[0, target_index].item())

    target_mean = np.mean(targets)
    target_std = np.std(targets)

    print(f"目标属性统计信息:")
    print(f"  均值: {target_mean:.4f}")
    print(f"  标准差: {target_std:.4f}")

    # 创建新数据对象
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


# =============================================================================
# 训练和评估函数
# =============================================================================
def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    """训练模型"""
    model.to(device)

    # 选择合适的优化器
    if isinstance(model, GraphTransformer):
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 根据模型类型调用不同的前向传播
            if isinstance(model, SchNetModel):
                out = model(batch.x, batch.edge_index, batch.batch, getattr(batch, 'pos', None))
            else:
                out = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(out.squeeze(), batch.y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                if isinstance(model, SchNetModel):
                    out = model(batch.x, batch.edge_index, batch.batch, getattr(batch, 'pos', None))
                else:
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
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return train_losses, val_losses, best_val_loss


def evaluate_model(model, test_loader, device, target_std, target_mean):
    """评估模型"""
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            if isinstance(model, SchNetModel):
                out = model(batch.x, batch.edge_index, batch.batch, getattr(batch, 'pos', None))
            else:
                out = model(batch.x, batch.edge_index, batch.batch)

            # 反标准化
            pred = out.squeeze().cpu().numpy() * target_std + target_mean
            true = batch.y.squeeze().cpu().numpy() * target_std + target_mean

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


# =============================================================================
# 模型比较函数
# =============================================================================
def compare_models():
    """比较不同模型的性能"""

    print("=" * 80)
    print("GNN模型框架比较实验")
    print("=" * 80)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据加载
    print("\n加载QM9数据集...")
    transform = Compose([NormalizeFeatures()])
    dataset = QM9(root='/Users/xiaotingzhou/Downloads/GNN/Dataset', transform=transform)
    dataset = dataset[3000:8000]  # 使用5000个样本进行快速比较

    target_index = 4  # HOMO-LUMO gap
    processed_data, target_mean, target_std = create_single_target_dataset(dataset, target_index)

    # 数据划分
    num_samples = len(processed_data)
    num_train = int(0.7 * num_samples)
    num_val = int(0.15 * num_samples)

    train_data = processed_data[:num_train]
    val_data = processed_data[num_train:num_train + num_val]
    test_data = processed_data[num_train + num_val:]

    print(f"数据划分: 训练={len(train_data)}, 验证={len(val_data)}, 测试={len(test_data)}")

    # 数据加载器
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    num_features = processed_data[0].x.shape[1]
    print(f"节点特征维度: {num_features}")

    # 模型定义
    models = {
        'BasicGCN': BasicGCN(num_features, hidden_dim=64, num_layers=4),
        'GraphSAGE': GraphSAGEModel(num_features, hidden_dim=64, num_layers=4),
        'GAT': GATModel(num_features, hidden_dim=64, num_layers=3, heads=4),
        'GraphTransformer': GraphTransformer(num_features, hidden_dim=64, num_layers=3, heads=4),
        'SchNet': SchNetModel(num_features, hidden_dim=64, num_layers=3),
        'ResidualGCN': ResidualGCN(num_features, hidden_dim=64, num_layers=6),
        'Ensemble': EnsembleModel(num_features, hidden_dim=64)
    }

    results = {}

    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n" + "="*50)
        print(f"训练 {model_name} 模型")
        print("="*50)

        start_time = time.time()

        try:
            # 训练模型
            train_losses, val_losses, best_val_loss = train_model(
                model, train_loader, val_loader, device, num_epochs=30, lr=0.001
            )

            # 评估模型
            eval_results = evaluate_model(model, test_loader, device, target_std, target_mean)

            training_time = time.time() - start_time

            results[model_name] = {
                'eval_results': eval_results,
                'training_time': training_time,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'num_parameters': sum(p.numel() for p in model.parameters())
            }

            print(f"训练完成! 用时: {training_time:.2f}秒")
            print(f"测试结果 - MAE: {eval_results['MAE']:.4f}, RMSE: {eval_results['RMSE']:.4f}, R²: {eval_results['R2']:.4f}")

        except Exception as e:
            print(f"训练 {model_name} 时出错: {str(e)}")
            results[model_name] = {'error': str(e)}

    # 结果对比
    print("\n" + "="*80)
    print("模型性能对比")
    print("="*80)

    # 创建对比表格
    print(f"{'模型':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'参数量':<10} {'训练时间(s)':<12}")
    print("-" * 80)

    for model_name, result in results.items():
        if 'error' not in result:
            eval_res = result['eval_results']
            print(f"{model_name:<15} {eval_res['MAE']:<8.4f} {eval_res['RMSE']:<8.4f} "
                  f"{eval_res['R2']:<8.4f} {result['num_parameters']:<10,} {result['training_time']:<12.2f}")

    # 保存结果
    torch.save(results, '/Users/xiaotingzhou/Downloads/GNN/model_comparison_results.pth')

    # 可视化结果
    plot_comparison_results(results)

    return results


def plot_comparison_results(results):
    """可视化比较结果"""

    # 提取有效结果
    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    if not valid_results:
        print("没有有效的结果用于可视化")
        return

    # 准备数据
    model_names = list(valid_results.keys())
    maes = [valid_results[name]['eval_results']['MAE'] for name in model_names]
    rmses = [valid_results[name]['eval_results']['RMSE'] for name in model_names]
    r2s = [valid_results[name]['eval_results']['R2'] for name in model_names]
    times = [valid_results[name]['training_time'] for name in model_names]
    params = [valid_results[name]['num_parameters'] for name in model_names]

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # MAE比较
    axes[0, 0].bar(model_names, maes, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('模型MAE比较')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # R²比较
    axes[0, 1].bar(model_names, r2s, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('模型R²比较')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 训练时间比较
    axes[1, 0].bar(model_names, times, color='orange', alpha=0.7)
    axes[1, 0].set_title('训练时间比较')
    axes[1, 0].set_ylabel('时间 (秒)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 参数量比较
    axes[1, 1].bar(model_names, [p/1000 for p in params], color='pink', alpha=0.7)
    axes[1, 1].set_title('模型参数量比较')
    axes[1, 1].set_ylabel('参数量 (千个)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('/Users/xiaotingzhou/Downloads/GNN/model_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 性能 vs 复杂度散点图
    plt.figure(figsize=(10, 6))
    plt.scatter([p/1000 for p in params], maes, s=100, alpha=0.7)

    for i, name in enumerate(model_names):
        plt.annotate(name, (params[i]/1000, maes[i]),
                    xytext=(5, 5), textcoords='offset points')

    plt.xlabel('模型参数量 (千个)')
    plt.ylabel('MAE')
    plt.title('模型复杂度 vs 性能')
    plt.grid(True, alpha=0.3)
    plt.savefig('/Users/xiaotingzhou/Downloads/GNN/complexity_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    results = compare_models()