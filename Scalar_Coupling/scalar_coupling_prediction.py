#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于图神经网络(GNN)的分子标量耦合常数预测
使用PyTorch Geometric实现原子对之间标量耦合常数的回归预测任务

主要功能：
1. 加载和预处理标量耦合常数数据集
2. 构建分子图结构，计算原子对特征
3. 训练GNN模型进行耦合常数预测
4. 支持GPU加速、模型保存和可视化分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import time
from collections import defaultdict
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')


class ScalarCouplingGNN(nn.Module):
    """
    用于标量耦合常数预测的图神经网络模型

    架构：
    - 多层图卷积网络处理分子结构
    - 原子对特征融合模块
    - 全连接层输出耦合常数预测
    """

    def __init__(self, num_atom_features, num_edge_features, hidden_dim=256, num_layers=4, dropout=0.2):
        """
        初始化模型

        Args:
            num_atom_features: 原子特征维度
            num_edge_features: 边特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            dropout: dropout概率
        """
        super(ScalarCouplingGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 原子特征预处理层
        self.atom_embedding = nn.Sequential(
            nn.Linear(num_atom_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GNN卷积层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 原子对特征融合层
        self.pair_feature_dim = hidden_dim * 2 + num_edge_features + 10  # 额外的距离和角度特征

        self.pair_mlp = nn.Sequential(
            nn.Linear(self.pair_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch, pair_indices, pair_features):
        """
        前向传播

        Args:
            x: 原子特征 [num_atoms, num_atom_features]
            edge_index: 边索引 [2, num_edges]
            batch: 批处理索引
            pair_indices: 原子对索引 [num_pairs, 2]
            pair_features: 原子对附加特征 [num_pairs, feature_dim]

        Returns:
            预测的标量耦合常数 [num_pairs, 1]
        """
        # 原子特征嵌入
        x = self.atom_embedding(x)

        # 多层图卷积
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 提取原子对特征
        atom0_features = x[pair_indices[:, 0]]  # 第一个原子的特征
        atom1_features = x[pair_indices[:, 1]]  # 第二个原子的特征

        # 拼接原子对特征
        pair_embedding = torch.cat([
            atom0_features,
            atom1_features,
            pair_features
        ], dim=1)

        # 通过MLP预测耦合常数
        coupling_constant = self.pair_mlp(pair_embedding)

        return coupling_constant


class ScalarCouplingDataset:
    """
    标量耦合常数数据集处理类
    负责数据加载、预处理、特征工程和数据集划分
    """

    def __init__(self, data_path):
        """
        初始化数据集

        Args:
            data_path: 数据集文件夹路径
        """
        self.data_path = data_path
        self.atom_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # 原子类型到特征的映射
        self.atom_features = {
            'H': [1, 1.008, 1, 0, 0, 0],      # 氢：原子序数，原子质量，价电子数，周期，族，电负性
            'C': [6, 12.01, 4, 2, 14, 2.55],  # 碳
            'N': [7, 14.01, 5, 2, 15, 3.04],  # 氮
            'O': [8, 16.00, 6, 2, 16, 3.44],  # 氧
            'F': [9, 19.00, 7, 2, 17, 3.98],  # 氟
        }

    def load_data(self):
        """
        加载所有数据文件

        Returns:
            train_df, test_df, structures_df, contributions_df
        """
        print("加载数据文件...")

        # 加载训练数据
        train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))

        # 加载测试数据
        test_df = pd.read_csv(os.path.join(self.data_path, 'test.csv'))

        # 加载分子结构数据
        structures_df = pd.read_csv(os.path.join(self.data_path, 'structures.csv'))

        # 加载耦合贡献数据
        contributions_df = pd.read_csv(os.path.join(self.data_path, 'scalar_coupling_contributions.csv'))

        print(f"训练样本数: {len(train_df):,}")
        print(f"测试样本数: {len(test_df):,}")
        print(f"分子结构数据: {len(structures_df):,}")
        print(f"耦合贡献数据: {len(contributions_df):,}")

        return train_df, test_df, structures_df, contributions_df

    def calculate_distance(self, pos1, pos2):
        """
        计算两个原子之间的距离

        Args:
            pos1, pos2: 原子坐标 (x, y, z)

        Returns:
            欧氏距离
        """
        return np.sqrt(np.sum((pos1 - pos2) ** 2))

    def calculate_angle(self, pos1, pos2, pos3):
        """
        计算三个原子之间的夹角

        Args:
            pos1, pos2, pos3: 原子坐标

        Returns:
            夹角（弧度）
        """
        v1 = pos1 - pos2
        v2 = pos3 - pos2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def extract_molecular_features(self, structures_df, molecule_name):
        """
        提取单个分子的特征

        Args:
            structures_df: 结构数据
            molecule_name: 分子名称

        Returns:
            atom_features, positions, adjacency_list
        """
        mol_structures = structures_df[structures_df['molecule_name'] == molecule_name].copy()
        mol_structures = mol_structures.sort_values('atom_index').reset_index(drop=True)

        # 提取原子特征
        atom_features = []
        positions = []

        for _, row in mol_structures.iterrows():
            atom = row['atom']
            pos = np.array([row['x'], row['y'], row['z']])

            # 获取原子基础特征
            base_features = self.atom_features.get(atom, [0, 0, 0, 0, 0, 0])
            atom_features.append(base_features)
            positions.append(pos)

        atom_features = np.array(atom_features, dtype=np.float32)
        positions = np.array(positions, dtype=np.float32)

        # 构建邻接关系（基于距离阈值）
        num_atoms = len(positions)
        adjacency_list = []
        edge_features = []

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = self.calculate_distance(positions[i], positions[j])

                # 如果距离小于阈值，认为是相连的
                if dist < 2.0:  # 可以根据化学键长度调整
                    adjacency_list.append([i, j])
                    adjacency_list.append([j, i])  # 无向图
                    edge_features.extend([dist, dist])

        if len(adjacency_list) == 0:
            # 如果没有边，创建自环
            for i in range(num_atoms):
                adjacency_list.append([i, i])
                edge_features.append(0.0)

        edge_index = np.array(adjacency_list).T
        edge_features = np.array(edge_features, dtype=np.float32).reshape(-1, 1)

        return atom_features, positions, edge_index, edge_features

    def create_pair_features(self, coupling_df, structures_df):
        """
        为每个原子对创建特征

        Args:
            coupling_df: 耦合数据
            structures_df: 结构数据

        Returns:
            配对特征和标签
        """
        print("创建原子对特征...")

        all_graph_data = []
        molecule_groups = coupling_df.groupby('molecule_name')

        for mol_name, group in molecule_groups:
            if len(all_graph_data) % 1000 == 0:
                print(f"处理分子: {len(all_graph_data)}")

            # 获取分子的图特征
            try:
                atom_features, positions, edge_index, edge_features = self.extract_molecular_features(
                    structures_df, mol_name)
            except:
                continue

            # 为每个原子对创建特征
            pair_indices = []
            pair_features = []
            labels = []
            coupling_types = []

            for _, row in group.iterrows():
                atom_idx_0 = row['atom_index_0']
                atom_idx_1 = row['atom_index_1']
                coupling_type = row['type']

                if atom_idx_0 >= len(positions) or atom_idx_1 >= len(positions):
                    continue

                # 计算原子对之间的距离
                distance = self.calculate_distance(positions[atom_idx_0], positions[atom_idx_1])

                # 计算相对位置特征
                relative_pos = positions[atom_idx_1] - positions[atom_idx_0]

                # 构建原子对特征
                pair_feature = np.concatenate([
                    [distance],                    # 距离
                    relative_pos,                  # 相对位置 (3维)
                    [len(positions)],             # 分子大小
                    atom_features[atom_idx_0][:3], # 原子0的前3个特征
                    atom_features[atom_idx_1][:3], # 原子1的前3个特征
                ])

                pair_indices.append([atom_idx_0, atom_idx_1])
                pair_features.append(pair_feature)
                coupling_types.append(coupling_type)

                if 'scalar_coupling_constant' in row:
                    labels.append(row['scalar_coupling_constant'])
                else:
                    labels.append(0.0)  # 测试集没有标签

            if len(pair_indices) > 0:
                graph_data = {
                    'molecule_name': mol_name,
                    'atom_features': torch.tensor(atom_features, dtype=torch.float),
                    'edge_index': torch.tensor(edge_index, dtype=torch.long),
                    'edge_features': torch.tensor(edge_features, dtype=torch.float),
                    'pair_indices': torch.tensor(pair_indices, dtype=torch.long),
                    'pair_features': torch.tensor(pair_features, dtype=torch.float),
                    'labels': torch.tensor(labels, dtype=torch.float),
                    'coupling_types': coupling_types
                }
                all_graph_data.append(graph_data)

        print(f"成功处理 {len(all_graph_data)} 个分子")
        return all_graph_data

    def create_data_loaders(self, train_df, test_df, structures_df, batch_size=32, val_split=0.2):
        """
        创建数据加载器

        Args:
            train_df: 训练数据
            test_df: 测试数据
            structures_df: 结构数据
            batch_size: 批次大小
            val_split: 验证集比例

        Returns:
            train_loader, val_loader, test_loader
        """
        # 创建训练数据图
        train_graphs = self.create_pair_features(train_df, structures_df)

        # 创建测试数据图
        test_graphs = self.create_pair_features(test_df, structures_df)

        # 划分训练集和验证集
        np.random.shuffle(train_graphs)
        split_idx = int(len(train_graphs) * (1 - val_split))
        train_graphs_split = train_graphs[:split_idx]
        val_graphs = train_graphs[split_idx:]

        print(f"数据划分:")
        print(f"  训练集: {len(train_graphs_split)} 个分子")
        print(f"  验证集: {len(val_graphs)} 个分子")
        print(f"  测试集: {len(test_graphs)} 个分子")

        # 创建自定义数据加载器
        def collate_graphs(graphs):
            """批处理图数据的collate函数"""
            batch_atom_features = []
            batch_edge_indices = []
            batch_edge_features = []
            batch_pair_indices = []
            batch_pair_features = []
            batch_labels = []
            batch_info = []

            atom_offset = 0
            pair_offset = 0

            for graph in graphs:
                num_atoms = graph['atom_features'].shape[0]
                num_pairs = graph['pair_indices'].shape[0]

                # 原子特征
                batch_atom_features.append(graph['atom_features'])

                # 边索引（需要加上偏移量）
                edge_index = graph['edge_index'] + atom_offset
                batch_edge_indices.append(edge_index)
                batch_edge_features.append(graph['edge_features'])

                # 原子对索引（需要加上偏移量）
                pair_indices = graph['pair_indices'] + atom_offset
                batch_pair_indices.append(pair_indices)
                batch_pair_features.append(graph['pair_features'])

                batch_labels.append(graph['labels'])

                # 记录批处理信息（每个原子对属于哪个分子）
                batch_info.extend([len(batch_atom_features) - 1] * num_pairs)

                atom_offset += num_atoms
                pair_offset += num_pairs

            return {
                'atom_features': torch.cat(batch_atom_features, dim=0),
                'edge_index': torch.cat(batch_edge_indices, dim=1),
                'edge_features': torch.cat(batch_edge_features, dim=0),
                'pair_indices': torch.cat(batch_pair_indices, dim=0),
                'pair_features': torch.cat(batch_pair_features, dim=0),
                'labels': torch.cat(batch_labels, dim=0),
                'batch_info': torch.tensor(batch_info, dtype=torch.long)
            }

        # 创建数据加载器
        train_loader = DataLoader(train_graphs_split, batch_size=batch_size,
                                shuffle=True, collate_fn=collate_graphs)
        val_loader = DataLoader(val_graphs, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_graphs)
        test_loader = DataLoader(test_graphs, batch_size=batch_size,
                               shuffle=False, collate_fn=collate_graphs)

        return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个epoch

    Args:
        model: GNN模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备

    Returns:
        平均训练损失
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        # 将数据移到设备上
        atom_features = batch['atom_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        pair_indices = batch['pair_indices'].to(device)
        pair_features = batch['pair_features'].to(device)
        labels = batch['labels'].to(device)
        batch_info = batch['batch_info'].to(device)

        optimizer.zero_grad()

        # 前向传播
        predictions = model(atom_features, edge_index, batch_info,
                          pair_indices, pair_features)

        # 计算损失
        loss = criterion(predictions.squeeze(), labels)

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
        device: 设备

    Returns:
        平均验证损失
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            atom_features = batch['atom_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            pair_indices = batch['pair_indices'].to(device)
            pair_features = batch['pair_features'].to(device)
            labels = batch['labels'].to(device)
            batch_info = batch['batch_info'].to(device)

            predictions = model(atom_features, edge_index, batch_info,
                              pair_indices, pair_features)

            loss = criterion(predictions.squeeze(), labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def test_model(model, test_loader, device):
    """
    测试模型并计算评估指标

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备

    Returns:
        predictions, true_values
    """
    model.eval()
    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for batch in test_loader:
            atom_features = batch['atom_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            pair_indices = batch['pair_indices'].to(device)
            pair_features = batch['pair_features'].to(device)
            labels = batch['labels'].to(device)
            batch_info = batch['batch_info'].to(device)

            predictions = model(atom_features, edge_index, batch_info,
                              pair_indices, pair_features)

            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_true_values.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_true_values)


def plot_training_history(train_losses, val_losses, save_path):
    """
    绘制训练历史
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='blue', alpha=0.8)
    plt.plot(val_losses, label='验证损失', color='red', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练过程')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label='训练损失 (log)', color='blue', alpha=0.8)
    plt.semilogy(val_losses, label='验证损失 (log)', color='red', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('损失 (log scale)')
    plt.title('训练过程 (对数坐标)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions(true_values, predictions, save_path):
    """
    绘制预测结果
    """
    plt.figure(figsize=(12, 5))

    # 散点图
    plt.subplot(1, 2, 1)
    plt.scatter(true_values, predictions, alpha=0.5, s=10)

    # 理想预测线
    min_val = min(np.min(true_values), np.min(predictions))
    max_val = max(np.max(true_values), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')

    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测结果对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 误差分布
    plt.subplot(1, 2, 2)
    errors = predictions - true_values
    plt.hist(errors, bins=50, alpha=0.7, color='green')
    plt.xlabel('预测误差')
    plt.ylabel('频次')
    plt.title('预测误差分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    主函数：完整的训练和测试流程
    """
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    data_path = '/Users/xiaotingzhou/Downloads/GNN/Dataset/scalar_coupling_constant'

    # 初始化数据集
    print("=" * 60)
    print("初始化数据集...")
    dataset = ScalarCouplingDataset(data_path)

    # 加载数据
    train_df, test_df, structures_df, contributions_df = dataset.load_data()

    # 数据预处理和创建数据加载器
    print("=" * 60)
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = dataset.create_data_loaders(
        train_df, test_df, structures_df, batch_size=16
    )

    # 获取特征维度
    sample_batch = next(iter(train_loader))
    num_atom_features = sample_batch['atom_features'].shape[1]
    num_edge_features = sample_batch['edge_features'].shape[1] if len(sample_batch['edge_features']) > 0 else 1
    num_pair_features = sample_batch['pair_features'].shape[1]

    print(f"特征维度信息:")
    print(f"  原子特征维度: {num_atom_features}")
    print(f"  边特征维度: {num_edge_features}")
    print(f"  原子对特征维度: {num_pair_features}")

    # 创建模型
    print("=" * 60)
    print("创建模型...")
    model = ScalarCouplingGNN(
        num_atom_features=num_atom_features,
        num_edge_features=num_pair_features,  # 使用原子对特征维度
        hidden_dim=256,
        num_layers=4,
        dropout=0.2
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")

    # 训练配置
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
    )

    # 训练循环
    print("=" * 60)
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

        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), '/Users/xiaotingzhou/Downloads/GNN/best_coupling_model.pth')
        else:
            patience_counter += 1

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
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
    model.load_state_dict(torch.load('/Users/xiaotingzhou/Downloads/GNN/best_coupling_model.pth'))

    # 测试模型
    print("=" * 60)
    print("在验证集上评估模型...")
    predictions, true_values = test_model(model, val_loader, device)

    # 计算评估指标
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)

    print(f"验证集结果:")
    print(f"  平均绝对误差 (MAE): {mae:.4f}")
    print(f"  均方误差 (MSE): {mse:.4f}")
    print(f"  均方根误差 (RMSE): {rmse:.4f}")

    # 显示预测样例
    print("\n预测样例:")
    for i in range(min(10, len(predictions))):
        print(f"  样本 {i+1}: 真实值={true_values[i]:.4f}, 预测值={predictions[i]:.4f}, "
              f"误差={abs(true_values[i] - predictions[i]):.4f}")

    # 可视化结果
    print("=" * 60)
    print("生成可视化图表...")

    # 绘制训练历史
    plot_training_history(train_losses, val_losses,
                         '/Users/xiaotingzhou/Downloads/GNN/coupling_training_history.png')

    # 绘制预测结果
    plot_predictions(true_values, predictions,
                    '/Users/xiaotingzhou/Downloads/GNN/coupling_predictions.png')

    # 保存结果
    results = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'training_epochs': len(train_losses),
        'training_time': training_time,
        'model_parameters': total_params,
        'best_val_loss': best_val_loss
    }

    np.save('/Users/xiaotingzhou/Downloads/GNN/coupling_results.npy', results)
    print("结果已保存到 coupling_results.npy")

    print("=" * 60)
    print("标量耦合常数预测任务完成！")


if __name__ == "__main__":
    main()