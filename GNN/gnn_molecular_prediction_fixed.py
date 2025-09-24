#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于图神经网络(GNN)的QM9分子属性预测 - 修复版本
使用PyTorch Geometric实现分子能量、带隙等属性的回归预测任务

主要功能：
1. 自动下载和预处理QM9数据集
2. 构建GNN模型进行分子级别预测
3. 训练、验证和测试流程
4. 支持GPU加速和模型保存
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.transforms import Compose, NormalizeFeatures
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import time


class GNNModel(nn.Module):
    """
    图神经网络模型，用于分子属性预测

    架构：
    - 多层GNN卷积层 (GCN)
    - 图级别池化 (global pooling)
    - 全连接层输出预测值
    """

    def __init__(self, num_features, hidden_dim=128, num_layers=3, num_targets=1, dropout=0.2):
        """
        初始化GNN模型

        Args:
            num_features: 节点特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            num_targets: 预测目标数量
            dropout: dropout概率
        """
        super(GNNModel, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GNN卷积层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层：输入特征 -> 隐藏维度
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 中间层：隐藏维度 -> 隐藏维度
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 最后一层：隐藏维度 -> 隐藏维度
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 全连接层用于最终预测
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3倍是因为使用了3种池化方式
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )

    def forward(self, x, edge_index, batch):
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, num_features]
            edge_index: 边索引 [2, num_edges]
            batch: 批处理索引，用于标识每个节点属于哪个图

        Returns:
            预测值 [batch_size, num_targets]
        """

        # 多层GNN卷积
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级别池化，将节点特征聚合为图特征
        x1 = global_mean_pool(x, batch)  # 平均池化
        x2 = global_max_pool(x, batch)   # 最大池化
        x3 = global_add_pool(x, batch)   # 求和池化

        # 连接不同池化结果
        x = torch.cat([x1, x2, x3], dim=1)

        # 通过全连接层输出最终预测
        x = self.fc_layers(x)

        return x


def create_single_target_dataset(original_dataset, target_index):
    """
    创建单目标数据集

    Args:
        original_dataset: 原始多目标数据集
        target_index: 目标属性索引

    Returns:
        处理后的单目标数据集
    """
    new_data_list = []

    # 首先计算目标值的统计信息
    targets = []
    for data in original_dataset:
        targets.append(data.y[0, target_index].item())

    target_mean = np.mean(targets)
    target_std = np.std(targets)

    print(f"目标属性统计信息:")
    print(f"  均值: {target_mean:.4f}")
    print(f"  标准差: {target_std:.4f}")
    print(f"  最小值: {np.min(targets):.4f}")
    print(f"  最大值: {np.max(targets):.4f}")

    # 创建新的数据对象列表
    for data in original_dataset:
        # 标准化目标值
        target_value = (data.y[0, target_index].item() - target_mean) / target_std

        # 创建新的Data对象，只包含单个目标
        new_data = Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            y=torch.tensor([target_value], dtype=torch.float),
            pos=data.pos.clone() if hasattr(data, 'pos') and data.pos is not None else None
        )
        new_data_list.append(new_data)

    return new_data_list, target_mean, target_std


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个epoch

    Args:
        model: GNN模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备 (CPU/GPU)

    Returns:
        平均训练损失
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 前向传播
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out.squeeze(), batch.y.squeeze())

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """
    验证模型

    Args:
        model: GNN模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备 (CPU/GPU)

    Returns:
        平均验证损失
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y.squeeze())
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def test_model(model, test_loader, device, target_std, target_mean):
    """
    测试模型并计算评估指标

    Args:
        model: 训练好的GNN模型
        test_loader: 测试数据加载器
        device: 设备 (CPU/GPU)
        target_std: 目标值标准差
        target_mean: 目标值均值

    Returns:
        mae: 平均绝对误差
        mse: 均方误差
        predictions: 预测值列表
        true_values: 真实值列表
    """
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
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

    return mae, mse, predictions, true_values


def plot_training_history(train_losses, val_losses):
    """
    绘制训练历史曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练过程中的损失变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/xiaotingzhou/Downloads/GNN/training_history_fixed.png')
    plt.show()


def plot_predictions(true_values, predictions, target_name):
    """
    绘制预测值与真实值对比图
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predictions, alpha=0.5, s=10)

    # 绘制理想预测线 (y=x)
    min_val = min(np.min(true_values), np.min(predictions))
    max_val = max(np.max(true_values), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')

    plt.xlabel(f'真实值 ({target_name})')
    plt.ylabel(f'预测值 ({target_name})')
    plt.title(f'{target_name} - 预测值 vs 真实值')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/xiaotingzhou/Downloads/GNN/predictions_comparison_fixed.png')
    plt.show()


def main():
    """
    主函数：完整的训练和测试流程
    """
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置预测目标 (可以修改为0-18中的任意值)
    target_index = 4  # HOMO-LUMO能隙

    # 目标属性名称
    target_names = [
        'dipole_moment', 'isotropic_polarizability', 'homo', 'lumo', 'gap',
        'electronic_spatial_extent', 'zero_point_vibrational_energy',
        'internal_energy_0K', 'internal_energy_298K', 'enthalpy_298K',
        'free_energy_298K', 'heat_capacity', 'atomization_energy',
        'atomization_enthalpy', 'atomization_free_energy',
        'rotational_constant_A', 'rotational_constant_B', 'rotational_constant_C',
        'vibrational_frequencies'
    ]

    target_name = target_names[target_index]
    print(f"预测目标: {target_name}")

    # 数据预处理变换
    transform = Compose([NormalizeFeatures()])

    # 加载数据
    print("=" * 50)
    print("开始数据加载和预处理...")

    dataset = QM9(root='/Users/xiaotingzhou/Downloads/GNN/Dataset', transform=transform)
    dataset = dataset[3000:]  # 移除前3000个不稳定样本

    print(f"原始数据集大小: {len(dataset)}")
    print(f"节点特征维度: {dataset[0].x.shape[1]}")
    print(f"原始目标属性数量: {dataset[0].y.shape[1]}")

    # 创建单目标数据集
    processed_data, target_mean, target_std = create_single_target_dataset(dataset, target_index)

    # 数据集划分
    num_samples = len(processed_data)
    num_train = int(0.8 * num_samples)
    num_val = int(0.1 * num_samples)

    train_data = processed_data[:num_train]
    val_data = processed_data[num_train:num_train + num_val]
    test_data = processed_data[num_train + num_val:]

    print(f"数据划分:")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    print(f"  测试集: {len(test_data)}")

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 验证数据维度
    sample_batch = next(iter(train_loader))
    print(f"批处理节点特征: {sample_batch.x.shape}")
    print(f"批处理目标维度: {sample_batch.y.shape}")

    # 创建模型
    print("=" * 50)
    print("创建GNN模型...")
    model = GNNModel(num_features=sample_batch.x.shape[1],
                    hidden_dim=128,
                    num_layers=4,
                    num_targets=1,
                    dropout=0.2)

    model = model.to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 设置训练参数
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=0.8, patience=10,
                                                          min_lr=1e-6)

    # 训练循环
    print("=" * 50)
    print("开始训练...")
    num_epochs = 100
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), '/Users/xiaotingzhou/Downloads/GNN/best_model_fixed.pth')
        else:
            patience_counter += 1

        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}] | '
                  f'Train Loss: {train_loss:.6f} | '
                  f'Val Loss: {val_loss:.6f} | '
                  f'Best Val Loss: {best_val_loss:.6f} | '
                  f'LR: {current_lr:.8f}')

        # 早停
        if patience_counter >= patience:
            print(f"早停触发！验证损失连续{patience}个epoch没有改善")
            break

    training_time = time.time() - start_time
    print(f"训练完成！总用时: {training_time/60:.2f} 分钟")

    # 加载最佳模型
    model.load_state_dict(torch.load('/Users/xiaotingzhou/Downloads/GNN/best_model_fixed.pth'))

    # 测试模型
    print("=" * 50)
    print("在测试集上评估模型...")
    mae, mse, predictions, true_values = test_model(model, test_loader, device, target_std, target_mean)

    print(f"测试结果 ({target_name}):")
    print(f"  平均绝对误差 (MAE): {mae:.4f}")
    print(f"  均方误差 (MSE): {mse:.4f}")
    print(f"  均方根误差 (RMSE): {np.sqrt(mse):.4f}")

    # 显示几个预测样例
    print("\n预测样例:")
    for i in range(min(10, len(predictions))):
        print(f"  样本 {i+1}: 真实值={true_values[i]:.4f}, 预测值={predictions[i]:.4f}, "
              f"误差={abs(true_values[i] - predictions[i]):.4f}")

    # 绘制训练历史和预测对比图
    print("=" * 50)
    print("生成可视化图表...")
    plot_training_history(train_losses, val_losses)
    plot_predictions(true_values, predictions, target_name)

    # 保存最终结果
    results = {
        'target_name': target_name,
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'training_epochs': len(train_losses),
        'training_time': training_time,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }

    np.save('/Users/xiaotingzhou/Downloads/GNN/results_fixed.npy', results)
    print("结果已保存到 results_fixed.npy")

    print("=" * 50)
    print("程序执行完成！")


if __name__ == "__main__":
    main()