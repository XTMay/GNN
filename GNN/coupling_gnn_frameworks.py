#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标量耦合常数预测 - 多种GNN框架比较
基于原子对级别的预测任务，使用不同复杂度的图神经网络架构

包含的模型：
1. 基础原子对MLP (Simple)
2. 图卷积网络 (GCN)
3. 图注意力网络 (GAT)
4. Graph Transformer
5. 消息传递网络 (MPNN)
6. 集成学习方法
7. 3D几何增强模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import (GCNConv, GATConv, TransformerConv, NNConv,
                               global_mean_pool, global_max_pool, global_add_pool,
                               BatchNorm, LayerNorm)
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
import seaborn as sns
import os
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')


# =============================================================================
# 1. 基础原子对MLP模型 (简化版本)
# =============================================================================
class SimpleAtomPairMLP(nn.Module):
    """基础原子对MLP模型"""

    def __init__(self, num_features=10, hidden_dim=64):
        super(SimpleAtomPairMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, pair_features):
        return self.mlp(pair_features)


# =============================================================================
# 2. 图卷积网络 (GCN) 模型
# =============================================================================
class CouplingGCN(nn.Module):
    """基于GCN的标量耦合常数预测模型"""

    def __init__(self, num_atom_features, num_pair_features, hidden_dim=128, num_layers=3):
        super(CouplingGCN, self).__init__()

        self.num_layers = num_layers

        # 原子特征处理
        self.atom_embedding = nn.Linear(num_atom_features, hidden_dim)

        # GCN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # 原子对特征融合
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_pair_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, atom_features, edge_index, pair_indices, pair_features):
        # 原子特征嵌入
        x = self.atom_embedding(atom_features)

        # GCN层
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # 提取原子对特征
        atom_pair_0 = x[pair_indices[:, 0]]  # 第一个原子
        atom_pair_1 = x[pair_indices[:, 1]]  # 第二个原子

        # 拼接原子对特征
        combined_features = torch.cat([atom_pair_0, atom_pair_1, pair_features], dim=1)

        # 预测耦合常数
        return self.pair_mlp(combined_features)


# =============================================================================
# 3. 图注意力网络 (GAT) 模型
# =============================================================================
class CouplingGAT(nn.Module):
    """基于GAT的标量耦合常数预测模型"""

    def __init__(self, num_atom_features, num_pair_features, hidden_dim=128, num_layers=3, heads=4):
        super(CouplingGAT, self).__init__()

        self.num_layers = num_layers
        self.heads = heads

        # 原子特征处理
        self.atom_embedding = nn.Linear(num_atom_features, hidden_dim)

        # GAT层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层
        self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.2))
        self.batch_norms.append(BatchNorm(hidden_dim))

        # 中间层
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.2))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # 最后一层
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.2))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # 原子对预测层
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_pair_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, atom_features, edge_index, pair_indices, pair_features):
        x = self.atom_embedding(atom_features)

        # GAT层
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)  # GAT通常使用ELU
            x = F.dropout(x, p=0.2, training=self.training)

        # 原子对特征
        atom_pair_0 = x[pair_indices[:, 0]]
        atom_pair_1 = x[pair_indices[:, 1]]

        combined_features = torch.cat([atom_pair_0, atom_pair_1, pair_features], dim=1)

        return self.pair_mlp(combined_features)


# =============================================================================
# 4. Graph Transformer 模型
# =============================================================================
class CouplingTransformer(nn.Module):
    """基于Transformer的标量耦合常数预测模型"""

    def __init__(self, num_atom_features, num_pair_features, hidden_dim=128, num_layers=3, heads=8):
        super(CouplingTransformer, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 原子特征嵌入
        self.atom_embedding = nn.Linear(num_atom_features, hidden_dim)

        # Transformer层
        self.transformers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.transformers.append(
                TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.2)
            )
            self.layer_norms.append(LayerNorm(hidden_dim))

        # 原子对预测层
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_pair_features, hidden_dim * 2),
            nn.GELU(),  # Transformer通常使用GELU
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, atom_features, edge_index, pair_indices, pair_features):
        x = self.atom_embedding(atom_features)

        # Transformer层
        for i in range(self.num_layers):
            residual = x
            x = self.transformers[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.gelu(x + residual)  # 残差连接
            x = F.dropout(x, p=0.2, training=self.training)

        # 原子对特征
        atom_pair_0 = x[pair_indices[:, 0]]
        atom_pair_1 = x[pair_indices[:, 1]]

        combined_features = torch.cat([atom_pair_0, atom_pair_1, pair_features], dim=1)

        return self.pair_mlp(combined_features)


# =============================================================================
# 5. 消息传递神经网络 (MPNN)
# =============================================================================
class CouplingMPNN(nn.Module):
    """基于MPNN的标量耦合常数预测模型"""

    def __init__(self, num_atom_features, num_pair_features, hidden_dim=128, num_layers=3):
        super(CouplingMPNN, self).__init__()

        self.num_layers = num_layers

        # 原子特征嵌入
        self.atom_embedding = nn.Linear(num_atom_features, hidden_dim)

        # 边特征网络
        edge_network = nn.Sequential(
            nn.Linear(1, hidden_dim),  # 假设边特征是1维(距离)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        # MPNN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_network))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # 原子对预测层
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_pair_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, atom_features, edge_index, edge_attr, pair_indices, pair_features):
        x = self.atom_embedding(atom_features)

        # MPNN层
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # 原子对特征
        atom_pair_0 = x[pair_indices[:, 0]]
        atom_pair_1 = x[pair_indices[:, 1]]

        combined_features = torch.cat([atom_pair_0, atom_pair_1, pair_features], dim=1)

        return self.pair_mlp(combined_features)


# =============================================================================
# 6. 3D几何增强模型
# =============================================================================
class Coupling3DGCN(nn.Module):
    """基于3D几何信息增强的GCN模型"""

    def __init__(self, num_atom_features, num_pair_features, hidden_dim=128, num_layers=3):
        super(Coupling3DGCN, self).__init__()

        self.num_layers = num_layers

        # 原子特征处理
        self.atom_embedding = nn.Linear(num_atom_features, hidden_dim)

        # 3D几何特征处理
        self.geometry_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # 3D坐标
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        # GCN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            input_dim = hidden_dim + hidden_dim // 4 if i == 0 else hidden_dim
            self.convs.append(GCNConv(input_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # 原子对预测层
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_pair_features + 6, hidden_dim * 2),  # +6 for 3D coords
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, atom_features, atom_coords, edge_index, pair_indices, pair_features, pair_coords):
        x = self.atom_embedding(atom_features)

        # 处理3D坐标
        geom_features = self.geometry_mlp(atom_coords)

        # 第一层：拼接原子特征和几何特征
        x = torch.cat([x, geom_features], dim=1)

        # GCN层
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # 原子对特征
        atom_pair_0 = x[pair_indices[:, 0]]
        atom_pair_1 = x[pair_indices[:, 1]]

        # 拼接原子对特征和3D坐标
        combined_features = torch.cat([
            atom_pair_0, atom_pair_1, pair_features, pair_coords.flatten(1)
        ], dim=1)

        return self.pair_mlp(combined_features)


# =============================================================================
# 7. 集成学习模型
# =============================================================================
class CouplingEnsemble(nn.Module):
    """集成多个模型的预测器"""

    def __init__(self, num_atom_features, num_pair_features, hidden_dim=128):
        super(CouplingEnsemble, self).__init__()

        # 不同的子模型
        self.gcn_model = CouplingGCN(num_atom_features, num_pair_features, hidden_dim, num_layers=3)
        self.gat_model = CouplingGAT(num_atom_features, num_pair_features, hidden_dim, num_layers=2, heads=4)

        # 元学习器
        self.meta_learner = nn.Sequential(
            nn.Linear(2, 32),  # 2个子模型的输出
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, atom_features, edge_index, pair_indices, pair_features):
        # 获取子模型预测
        pred1 = self.gcn_model(atom_features, edge_index, pair_indices, pair_features)
        pred2 = self.gat_model(atom_features, edge_index, pair_indices, pair_features)

        # 元学习器组合
        ensemble_input = torch.cat([pred1, pred2], dim=1)
        final_pred = self.meta_learner(ensemble_input)

        return final_pred


# =============================================================================
# 数据处理类
# =============================================================================
class CouplingGraphDataset(Dataset):
    """用于图神经网络的标量耦合常数数据集"""

    def __init__(self, data_path, max_samples=5000, use_3d=False):
        self.data_path = data_path
        self.max_samples = max_samples
        self.use_3d = use_3d

        # 编码器和缩放器
        self.type_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.coord_scaler = StandardScaler()

        # 原子特征映射
        self.atom_features = {
            'H': [1, 1.008, 1, 1, 2.20, 0.31],  # 原子序数, 质量, 价电子, 周期, 电负性, 半径
            'C': [6, 12.01, 4, 2, 2.55, 0.76],
            'N': [7, 14.01, 5, 2, 3.04, 0.71],
            'O': [8, 16.00, 6, 2, 3.44, 0.66],
            'F': [9, 19.00, 7, 2, 3.98, 0.57],
        }

        print(f"加载标量耦合常数数据，样本数: {max_samples}")
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """加载和预处理数据"""
        # 加载数据
        train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv')).head(self.max_samples)
        structures_df = pd.read_csv(os.path.join(self.data_path, 'structures.csv'))

        print(f"加载 {len(train_df)} 个耦合样本")

        # 预处理分子图数据
        self.graph_data = self._create_graph_data(train_df, structures_df)

        print(f"成功创建 {len(self.graph_data)} 个图数据")

    def _create_graph_data(self, train_df, structures_df):
        """创建图数据"""
        graph_data_list = []

        # 按分子分组
        molecule_groups = train_df.groupby('molecule_name')

        processed_molecules = 0
        for mol_name, coupling_df in molecule_groups:
            if processed_molecules % 1000 == 0:
                print(f"处理分子: {processed_molecules}")

            # 获取分子结构
            mol_structure = structures_df[structures_df['molecule_name'] == mol_name]
            if len(mol_structure) == 0:
                continue

            try:
                graph_data = self._process_single_molecule(mol_name, coupling_df, mol_structure)
                if graph_data:
                    graph_data_list.extend(graph_data)
                processed_molecules += 1
            except Exception as e:
                continue

        return graph_data_list

    def _process_single_molecule(self, mol_name, coupling_df, mol_structure):
        """处理单个分子"""
        mol_structure = mol_structure.sort_values('atom_index').reset_index(drop=True)

        # 创建原子特征
        atom_features = []
        atom_coords = []

        for _, atom_row in mol_structure.iterrows():
            atom_type = atom_row['atom']
            coords = [atom_row['x'], atom_row['y'], atom_row['z']]

            # 获取原子特征
            features = self.atom_features.get(atom_type, [0, 0, 0, 0, 0, 0])
            atom_features.append(features)
            atom_coords.append(coords)

        atom_features = torch.tensor(atom_features, dtype=torch.float)
        atom_coords = torch.tensor(atom_coords, dtype=torch.float)

        # 创建边 (简单的距离阈值连接)
        edge_index, edge_attr = self._create_edges(atom_coords)

        # 处理每个原子对的耦合常数
        graph_data_list = []
        for _, coupling_row in coupling_df.iterrows():
            atom_idx_0 = coupling_row['atom_index_0']
            atom_idx_1 = coupling_row['atom_index_1']
            coupling_constant = coupling_row['scalar_coupling_constant']
            coupling_type = coupling_row['type']

            if atom_idx_0 >= len(atom_features) or atom_idx_1 >= len(atom_features):
                continue

            # 创建原子对特征
            pair_features = self._create_pair_features(
                atom_idx_0, atom_idx_1, coupling_type, atom_coords, mol_structure
            )

            # 创建数据对象
            data = {
                'molecule_name': mol_name,
                'atom_features': atom_features,
                'atom_coords': atom_coords,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'pair_indices': torch.tensor([[atom_idx_0, atom_idx_1]], dtype=torch.long),
                'pair_features': pair_features,
                'pair_coords': torch.stack([atom_coords[atom_idx_0], atom_coords[atom_idx_1]]).unsqueeze(0),
                'coupling_constant': torch.tensor([coupling_constant], dtype=torch.float),
                'coupling_type': coupling_type
            }

            graph_data_list.append(data)

        return graph_data_list

    def _create_edges(self, atom_coords, cutoff=2.0):
        """创建边连接"""
        num_atoms = len(atom_coords)
        edge_list = []
        edge_distances = []

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = torch.norm(atom_coords[i] - atom_coords[j]).item()
                if dist < cutoff:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # 无向图
                    edge_distances.extend([dist, dist])

        if len(edge_list) == 0:
            # 如果没有边，创建自环
            for i in range(num_atoms):
                edge_list.append([i, i])
                edge_distances.append(0.0)

        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)

        return edge_index, edge_attr

    def _create_pair_features(self, atom_idx_0, atom_idx_1, coupling_type, atom_coords, mol_structure):
        """创建原子对特征"""
        # 距离特征
        distance = torch.norm(atom_coords[atom_idx_0] - atom_coords[atom_idx_1]).item()

        # 相对位置
        rel_pos = atom_coords[atom_idx_1] - atom_coords[atom_idx_0]

        # 原子类型编码
        atom_0_type = mol_structure.iloc[atom_idx_0]['atom']
        atom_1_type = mol_structure.iloc[atom_idx_1]['atom']

        # 简单的原子类型编码
        atom_type_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        atom_0_encoded = atom_type_map.get(atom_0_type, 0)
        atom_1_encoded = atom_type_map.get(atom_1_type, 0)

        # 耦合类型编码
        type_map = {'1JHC': 0, '2JHH': 1, '3JHH': 2, '1JCC': 3, '2JHC': 4, '2JCH': 5, '3JHC': 6}
        coupling_type_encoded = type_map.get(coupling_type, 0)

        # 组合特征
        features = [
            distance,
            rel_pos[0].item(), rel_pos[1].item(), rel_pos[2].item(),  # 相对位置
            atom_0_encoded, atom_1_encoded,  # 原子类型
            coupling_type_encoded,  # 耦合类型
            len(mol_structure)  # 分子大小
        ]

        return torch.tensor(features, dtype=torch.float).unsqueeze(0)

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx]


def custom_collate_fn(batch):
    """自定义批处理函数"""
    # 由于每个样本都是单个原子对，我们需要合并它们

    all_atom_features = []
    all_atom_coords = []
    all_edge_indices = []
    all_edge_attrs = []
    all_pair_indices = []
    all_pair_features = []
    all_pair_coords = []
    all_coupling_constants = []

    atom_offset = 0

    for data in batch:
        num_atoms = data['atom_features'].shape[0]

        # 原子特征和坐标
        all_atom_features.append(data['atom_features'])
        all_atom_coords.append(data['atom_coords'])

        # 边索引需要加偏移
        edge_index = data['edge_index'] + atom_offset
        all_edge_indices.append(edge_index)
        all_edge_attrs.append(data['edge_attr'])

        # 原子对索引需要加偏移
        pair_indices = data['pair_indices'] + atom_offset
        all_pair_indices.append(pair_indices)

        all_pair_features.append(data['pair_features'])
        all_pair_coords.append(data['pair_coords'])
        all_coupling_constants.append(data['coupling_constant'])

        atom_offset += num_atoms

    # 合并所有数据
    batched_data = {
        'atom_features': torch.cat(all_atom_features, dim=0),
        'atom_coords': torch.cat(all_atom_coords, dim=0),
        'edge_index': torch.cat(all_edge_indices, dim=1),
        'edge_attr': torch.cat(all_edge_attrs, dim=0),
        'pair_indices': torch.cat(all_pair_indices, dim=0),
        'pair_features': torch.cat(all_pair_features, dim=0),
        'pair_coords': torch.cat(all_pair_coords, dim=0),
        'coupling_constants': torch.cat(all_coupling_constants, dim=0),
        'batch_size': len(batch)
    }

    return batched_data


# =============================================================================
# 训练和评估函数
# =============================================================================
def train_coupling_model(model, train_loader, val_loader, device, num_epochs=30, model_name="Model"):
    """训练耦合常数预测模型"""
    print(f"训练 {model_name} 模型...")

    model.to(device)
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
        train_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()

            atom_features = batch['atom_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            pair_indices = batch['pair_indices'].to(device)
            pair_features = batch['pair_features'].to(device)
            targets = batch['coupling_constants'].to(device)

            # 根据模型类型调用不同的前向传播
            if isinstance(model, SimpleAtomPairMLP):
                predictions = model(pair_features)
            elif isinstance(model, CouplingMPNN):
                edge_attr = batch['edge_attr'].to(device)
                predictions = model(atom_features, edge_index, edge_attr, pair_indices, pair_features)
            elif isinstance(model, Coupling3DGCN):
                atom_coords = batch['atom_coords'].to(device)
                pair_coords = batch['pair_coords'].to(device)
                predictions = model(atom_features, atom_coords, edge_index, pair_indices, pair_features, pair_coords)
            else:
                predictions = model(atom_features, edge_index, pair_indices, pair_features)

            loss = criterion(predictions.squeeze(), targets)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item() * len(targets)
            train_samples += len(targets)

        # 验证
        model.eval()
        val_loss = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                atom_features = batch['atom_features'].to(device)
                edge_index = batch['edge_index'].to(device)
                pair_indices = batch['pair_indices'].to(device)
                pair_features = batch['pair_features'].to(device)
                targets = batch['coupling_constants'].to(device)

                if isinstance(model, SimpleAtomPairMLP):
                    predictions = model(pair_features)
                elif isinstance(model, CouplingMPNN):
                    edge_attr = batch['edge_attr'].to(device)
                    predictions = model(atom_features, edge_index, edge_attr, pair_indices, pair_features)
                elif isinstance(model, Coupling3DGCN):
                    atom_coords = batch['atom_coords'].to(device)
                    pair_coords = batch['pair_coords'].to(device)
                    predictions = model(atom_features, atom_coords, edge_index, pair_indices, pair_features, pair_coords)
                else:
                    predictions = model(atom_features, edge_index, pair_indices, pair_features)

                loss = criterion(predictions.squeeze(), targets)
                val_loss += loss.item() * len(targets)
                val_samples += len(targets)

        train_loss /= train_samples
        val_loss /= val_samples

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')

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


def evaluate_coupling_model(model, test_loader, device):
    """评估耦合常数预测模型"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            atom_features = batch['atom_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            pair_indices = batch['pair_indices'].to(device)
            pair_features = batch['pair_features'].to(device)
            targets = batch['coupling_constants'].to(device)

            if isinstance(model, SimpleAtomPairMLP):
                predictions = model(pair_features)
            elif isinstance(model, CouplingMPNN):
                edge_attr = batch['edge_attr'].to(device)
                predictions = model(atom_features, edge_index, edge_attr, pair_indices, pair_features)
            elif isinstance(model, Coupling3DGCN):
                atom_coords = batch['atom_coords'].to(device)
                pair_coords = batch['pair_coords'].to(device)
                predictions = model(atom_features, atom_coords, edge_index, pair_indices, pair_features, pair_coords)
            else:
                predictions = model(atom_features, edge_index, pair_indices, pair_features)

            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'predictions': predictions,
        'targets': targets
    }


# =============================================================================
# 比较框架主函数
# =============================================================================
def compare_coupling_models(data_path='/Users/xiaotingzhou/Downloads/GNN/Dataset/scalar_coupling_constant',
                          max_samples=3000, test_split=0.2, val_split=0.1):
    """比较不同的标量耦合常数预测模型"""

    print("=" * 80)
    print("🚀 开始标量耦合常数GNN框架比较")
    print("=" * 80)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集
    print("\n📊 加载数据集...")
    dataset = CouplingGraphDataset(data_path, max_samples=max_samples)

    # 数据划分
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    print(f"数据划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)

    # 获取特征维度
    sample_data = dataset[0]
    num_atom_features = sample_data['atom_features'].shape[1]
    num_pair_features = sample_data['pair_features'].shape[1]

    print(f"原子特征维度: {num_atom_features}, 原子对特征维度: {num_pair_features}")

    # 定义模型配置
    models_config = {
        'SimpleAtomPairMLP': {
            'model': SimpleAtomPairMLP(num_features=num_pair_features, hidden_dim=64),
            'description': '基础原子对MLP'
        },
        'CouplingGCN': {
            'model': CouplingGCN(num_atom_features, num_pair_features, hidden_dim=128, num_layers=3),
            'description': '图卷积网络'
        },
        'CouplingGAT': {
            'model': CouplingGAT(num_atom_features, num_pair_features, hidden_dim=128, num_layers=3, heads=4),
            'description': '图注意力网络'
        },
        'CouplingTransformer': {
            'model': CouplingTransformer(num_atom_features, num_pair_features, hidden_dim=128, num_layers=2, heads=8),
            'description': 'Graph Transformer'
        },
        'CouplingMPNN': {
            'model': CouplingMPNN(num_atom_features, num_pair_features, hidden_dim=128, num_layers=3),
            'description': '消息传递神经网络'
        },
        'Coupling3DGCN': {
            'model': Coupling3DGCN(num_atom_features, num_pair_features, hidden_dim=128, num_layers=3),
            'description': '3D几何增强GCN'
        },
        'CouplingEnsemble': {
            'model': CouplingEnsemble(num_atom_features, num_pair_features, hidden_dim=96),
            'description': '集成学习模型'
        }
    }

    # 训练和评估所有模型
    results = {}

    for model_name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"🔥 训练模型: {model_name} - {config['description']}")
        print(f"{'='*60}")

        try:
            # 训练模型
            training_result = train_coupling_model(
                config['model'], train_loader, val_loader, device,
                num_epochs=25, model_name=model_name
            )

            # 评估模型
            eval_result = evaluate_coupling_model(training_result['model'], test_loader, device)

            # 合并结果
            results[model_name] = {
                **training_result,
                **eval_result,
                'description': config['description']
            }

            print(f"✅ {model_name} 完成!")
            print(f"   MAE: {eval_result['MAE']:.4f}")
            print(f"   R²: {eval_result['R2']:.4f}")
            print(f"   参数量: {training_result['num_parameters']:,}")
            print(f"   训练时间: {training_result['training_time']:.1f}s")

        except Exception as e:
            print(f"❌ {model_name} 训练失败: {str(e)}")
            results[model_name] = {'error': str(e)}

    # 生成比较报告
    print(f"\n{'='*80}")
    print("📋 标量耦合常数GNN模型比较结果")
    print(f"{'='*80}")

    # 过滤有效结果
    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    if valid_results:
        # 按MAE排序
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['MAE'])

        print(f"\n🏆 模型性能排名 (按MAE):")
        print("-" * 80)
        print(f"{'排名':<4} {'模型':<20} {'MAE':<8} {'R²':<8} {'参数量':<10} {'训练时间':<10} {'描述':<15}")
        print("-" * 80)

        for i, (model_name, result) in enumerate(sorted_results, 1):
            print(f"{i:<4} {model_name:<20} {result['MAE']:<8.4f} {result['R2']:<8.4f} "
                  f"{result['num_parameters']:<10,} {result['training_time']:<10.1f}s "
                  f"{result['description']:<15}")

        # 保存详细结果
        detailed_results = {}
        for model_name, result in valid_results.items():
            detailed_results[model_name] = {
                'MAE': result['MAE'],
                'RMSE': result['RMSE'],
                'R2': result['R2'],
                'num_parameters': result['num_parameters'],
                'training_time': result['training_time'],
                'description': result['description']
            }

        # 创建可视化
        create_coupling_comparison_plots(detailed_results, valid_results)

        print(f"\n💾 结果已保存到 coupling_comparison_results.txt")
        save_coupling_results(detailed_results)

    else:
        print("❌ 没有模型成功训练完成")

    return results


def create_coupling_comparison_plots(results, full_results):
    """创建标量耦合常数比较可视化"""

    if len(results) < 2:
        print("结果太少，跳过可视化")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('标量耦合常数GNN模型比较', fontsize=16, fontweight='bold')

    models = list(results.keys())
    maes = [results[m]['MAE'] for m in models]
    r2s = [results[m]['R2'] for m in models]
    params = [results[m]['num_parameters'] for m in models]
    times = [results[m]['training_time'] for m in models]

    # 1. MAE对比
    axes[0, 0].bar(models, maes, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('平均绝对误差 (MAE) 比较')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. R²对比
    axes[0, 1].bar(models, r2s, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('决定系数 (R²) 比较')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. 参数量对比
    axes[1, 0].bar(models, params, color='orange', alpha=0.7)
    axes[1, 0].set_title('模型参数量比较')
    axes[1, 0].set_ylabel('参数数量')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. 训练时间对比
    axes[1, 1].bar(models, times, color='salmon', alpha=0.7)
    axes[1, 1].set_title('训练时间比较')
    axes[1, 1].set_ylabel('时间 (秒)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('coupling_gnn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 创建性能-效率散点图
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scatter = ax.scatter(params, maes, c=times, s=100, alpha=0.7, cmap='viridis')

    for i, model in enumerate(models):
        ax.annotate(model, (params[i], maes[i]), xytext=(5, 5),
                   textcoords='offset points', fontsize=9)

    ax.set_xlabel('模型参数量')
    ax.set_ylabel('MAE')
    ax.set_title('模型性能 vs 复杂度 (颜色表示训练时间)')

    cbar = plt.colorbar(scatter)
    cbar.set_label('训练时间 (秒)')

    plt.tight_layout()
    plt.savefig('coupling_performance_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 预测效果可视化
    if 'CouplingGAT' in full_results:
        create_prediction_scatter_plot(full_results['CouplingGAT'])


def create_prediction_scatter_plot(result):
    """创建预测vs真实值散点图"""
    predictions = result['predictions']
    targets = result['targets']

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(targets, predictions, alpha=0.6, s=20)

    # 绘制理想线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测')

    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title(f'GAT模型预测效果 (R² = {result["R2"]:.4f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig('coupling_prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_coupling_results(results):
    """保存比较结果到文件"""
    with open('coupling_comparison_results.txt', 'w', encoding='utf-8') as f:
        f.write("标量耦合常数GNN模型比较结果\n")
        f.write("=" * 50 + "\n\n")

        # 按MAE排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['MAE'])

        f.write("模型性能排名 (按MAE):\n")
        f.write("-" * 50 + "\n")

        for i, (model_name, result) in enumerate(sorted_results, 1):
            f.write(f"{i}. {model_name} ({result['description']})\n")
            f.write(f"   MAE: {result['MAE']:.6f}\n")
            f.write(f"   RMSE: {result['RMSE']:.6f}\n")
            f.write(f"   R²: {result['R2']:.6f}\n")
            f.write(f"   参数量: {result['num_parameters']:,}\n")
            f.write(f"   训练时间: {result['training_time']:.1f}s\n\n")


if __name__ == "__main__":
    # 运行比较
    results = compare_coupling_models(
        max_samples=3000,  # 减少样本数量以加快速度
        test_split=0.2,
        val_split=0.1
    )