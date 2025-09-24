#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版本的GNN测试，用来验证修复
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.transforms import Compose, NormalizeFeatures
import numpy as np
import os

def main():
    print("测试GNN模型修复...")

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据预处理变换
    transform = Compose([NormalizeFeatures()])

    # 加载数据集
    print("加载QM9数据集...")
    dataset = QM9(root='/Users/xiaotingzhou/Downloads/GNN/Dataset', transform=transform)

    # 只使用前1000个样本进行测试
    dataset = dataset[:1000]

    print(f"测试数据集大小: {len(dataset)}")
    print(f"节点特征维度: {dataset[0].x.shape[1]}")
    print(f"目标属性数量: {dataset[0].y.shape[1]}")

    # 选择目标属性（HOMO-LUMO gap，索引4）
    target_index = 4

    # 提取目标值并计算统计信息
    targets = []
    for data in dataset:
        targets.append(data.y[0, target_index].item())
    targets = np.array(targets)

    target_mean = np.mean(targets)
    target_std = np.std(targets)

    print(f"目标属性统计信息:")
    print(f"  均值: {target_mean:.4f}")
    print(f"  标准差: {target_std:.4f}")

    # 标准化目标值
    print("标准化目标值...")
    for data in dataset:
        target_value = (data.y[0, target_index].item() - target_mean) / target_std
        data.y = torch.tensor([target_value], dtype=torch.float)

    # 验证数据维度
    sample_batch = [dataset[0], dataset[1]]
    print(f"单个样本目标维度: {dataset[0].y.shape}")

    # 创建数据加载器
    train_loader = DataLoader(dataset[:800], batch_size=32, shuffle=True)

    # 测试批处理
    sample_batch = next(iter(train_loader))
    print(f"批处理节点特征维度: {sample_batch.x.shape}")
    print(f"批处理目标维度: {sample_batch.y.shape}")

    # 简单的GNN模型
    class SimpleGNN(nn.Module):
        def __init__(self, num_features, hidden_dim=64):
            super(SimpleGNN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            x = self.fc(x)
            return x

    # 创建模型
    num_features = dataset[0].x.shape[1]
    model = SimpleGNN(num_features).to(device)

    # 测试前向传播
    sample_batch = sample_batch.to(device)
    with torch.no_grad():
        output = model(sample_batch.x, sample_batch.edge_index, sample_batch.batch)

    print(f"模型输出维度: {output.shape}")
    print(f"目标维度: {sample_batch.y.shape}")

    # 测试损失计算
    criterion = nn.MSELoss()
    loss = criterion(output.squeeze(), sample_batch.y.squeeze())
    print(f"损失值: {loss.item():.6f}")

    print("测试通过！维度匹配正确。")

if __name__ == "__main__":
    main()