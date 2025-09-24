#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版标量耦合常数预测 - 用于测试和调试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import time


class SimpleCouplingGNN(nn.Module):
    """
    简化的GNN模型用于标量耦合常数预测
    """

    def __init__(self, num_features=10, hidden_dim=128):
        super(SimpleCouplingGNN, self).__init__()

        # 简化的特征处理层
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, pair_features):
        """
        简化的前向传播，直接处理原子对特征
        """
        return self.feature_mlp(pair_features)


class SimpleCouplingDataset(Dataset):
    """
    简化的数据集类
    """

    def __init__(self, data_path, max_samples=5000):
        self.data_path = data_path
        self.max_samples = max_samples

        # 加载数据
        print(f"加载数据，最大样本数: {max_samples}")
        self.train_df = pd.read_csv(os.path.join(data_path, 'train.csv')).head(max_samples)
        self.structures_df = pd.read_csv(os.path.join(data_path, 'structures.csv'))

        # 编码器
        self.type_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        print(f"实际加载样本数: {len(self.train_df)}")

        # 预处理数据
        self._preprocess_data()

    def _preprocess_data(self):
        """
        预处理数据
        """
        print("预处理数据...")

        # 合并结构信息
        atom0_info = self.structures_df.rename(columns={
            'atom': 'atom_0', 'x': 'x_0', 'y': 'y_0', 'z': 'z_0'
        })
        atom1_info = self.structures_df.rename(columns={
            'atom': 'atom_1', 'x': 'x_1', 'y': 'y_1', 'z': 'z_1'
        })

        # 合并原子0信息
        merged_df = self.train_df.merge(
            atom0_info[['molecule_name', 'atom_index', 'atom_0', 'x_0', 'y_0', 'z_0']],
            left_on=['molecule_name', 'atom_index_0'],
            right_on=['molecule_name', 'atom_index']
        ).drop('atom_index', axis=1)

        # 合并原子1信息
        merged_df = merged_df.merge(
            atom1_info[['molecule_name', 'atom_index', 'atom_1', 'x_1', 'y_1', 'z_1']],
            left_on=['molecule_name', 'atom_index_1'],
            right_on=['molecule_name', 'atom_index']
        ).drop('atom_index', axis=1)

        print(f"合并后数据大小: {len(merged_df)}")

        # 计算距离特征
        merged_df['distance'] = np.sqrt(
            (merged_df['x_1'] - merged_df['x_0'])**2 +
            (merged_df['y_1'] - merged_df['y_0'])**2 +
            (merged_df['z_1'] - merged_df['z_0'])**2
        )

        # 编码原子类型
        all_atoms = list(merged_df['atom_0']) + list(merged_df['atom_1'])
        self.type_encoder.fit(all_atoms)

        merged_df['atom_0_encoded'] = self.type_encoder.transform(merged_df['atom_0'])
        merged_df['atom_1_encoded'] = self.type_encoder.transform(merged_df['atom_1'])

        # 创建特征矩阵
        feature_cols = [
            'atom_index_0', 'atom_index_1', 'atom_0_encoded', 'atom_1_encoded',
            'distance', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1'
        ]

        self.features = merged_df[feature_cols].values.astype(np.float32)
        self.labels = merged_df['scalar_coupling_constant'].values.astype(np.float32)

        # 标准化特征
        self.features = self.scaler.fit_transform(self.features)

        print(f"特征维度: {self.features.shape}")
        print(f"标签统计: 均值={np.mean(self.labels):.4f}, 标准差={np.std(self.labels):.4f}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(features)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    验证模型
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)
            loss = criterion(predictions, labels)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def test_model(model, test_loader, device):
    """
    测试模型
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)

            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    return np.array(all_predictions), np.array(all_labels)


def main():
    """
    主函数
    """
    print("=" * 60)
    print("简化版标量耦合常数预测")

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    data_path = '/Users/xiaotingzhou/Downloads/GNN/Dataset/scalar_coupling_constant'

    # 创建数据集（使用很少的样本进行快速测试）
    dataset = SimpleCouplingDataset(data_path, max_samples=2000)

    # 数据划分
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))

    print(f"数据划分:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    num_features = dataset.features.shape[1]
    model = SimpleCouplingGNN(num_features=num_features, hidden_dim=64).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练设置
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

    # 训练循环
    print("=" * 60)
    print("开始训练...")

    num_epochs = 20
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/Users/xiaotingzhou/Downloads/GNN/best_simple_coupling_model.pth')

        # 打印进度
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}] | '
                  f'Train Loss: {train_loss:.6f} | '
                  f'Val Loss: {val_loss:.6f} | '
                  f'LR: {current_lr:.8f}')

    training_time = time.time() - start_time
    print(f"训练完成！用时: {training_time/60:.2f} 分钟")

    # 加载最佳模型并测试
    model.load_state_dict(torch.load('/Users/xiaotingzhou/Downloads/GNN/best_simple_coupling_model.pth'))
    predictions, true_values = test_model(model, test_loader, device)

    # 计算指标
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)

    print("=" * 60)
    print("测试结果:")
    print(f"  平均绝对误差 (MAE): {mae:.4f}")
    print(f"  均方误差 (MSE): {mse:.4f}")
    print(f"  均方根误差 (RMSE): {rmse:.4f}")

    # 显示样例
    print("\n预测样例:")
    for i in range(min(10, len(predictions))):
        print(f"  样本 {i+1}: 真实值={true_values[i]:.4f}, 预测值={predictions[i]:.4f}, "
              f"误差={abs(true_values[i] - predictions[i]):.4f}")

    # 简单可视化
    plt.figure(figsize=(12, 4))

    # 训练历史
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练历史')
    plt.legend()
    plt.grid(True)

    # 预测结果
    plt.subplot(1, 2, 2)
    plt.scatter(true_values, predictions, alpha=0.5, s=10)
    min_val = min(np.min(true_values), np.min(predictions))
    max_val = max(np.max(true_values), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测结果')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('/Users/xiaotingzhou/Downloads/GNN/simple_coupling_results.png', dpi=300)
    plt.show()

    print("=" * 60)
    print("简化版测试完成！")


if __name__ == "__main__":
    main()