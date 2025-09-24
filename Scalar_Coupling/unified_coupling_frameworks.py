#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一标量耦合常数预测框架 - 所有模型集成版
将所有不同的框架整合在一个文件中，进行统一训练和比较

包含的框架：
1. 简化MLP基线
2. 原始GCN模型
3. 多种GNN架构 (GAT, Transformer, MPNN等)
4. 高级特征工程
5. 集成学习方法

输出：统一的比较报告，包含所有模型的详细结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import (GCNConv, GATConv, TransformerConv, NNConv, SAGEConv,
                               global_mean_pool, global_max_pool, global_add_pool,
                               BatchNorm, LayerNorm)
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import networkx as nx
import os
import time
import json
import warnings
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

# 尝试导入RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    from scipy.spatial import ConvexHull
    RDKIT_AVAILABLE = True
    print("✅ RDKit可用")
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit不可用，将跳过部分高级特征")


class UnifiedCouplingFrameworks:
    """统一的标量耦合常数预测框架"""

    def __init__(self, data_path='/Users/xiaotingzhou/Downloads/GNN/Dataset/scalar_coupling_constant',
                 max_samples=3000):
        self.data_path = data_path
        self.max_samples = max_samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}

        print(f"🚀 初始化统一标量耦合常数预测框架")
        print(f"数据路径: {data_path}")
        print(f"最大样本数: {max_samples}")
        print(f"使用设备: {self.device}")

    # =============================================================================
    # 1. 数据集类定义
    # =============================================================================

    class SimpleCouplingDataset(Dataset):
        """简化版数据集 - MLP基线使用"""

        def __init__(self, data_path, max_samples=3000):
            self.train_df = pd.read_csv(os.path.join(data_path, 'train.csv')).head(max_samples)
            self.structures_df = pd.read_csv(os.path.join(data_path, 'structures.csv'))

            print(f"简化数据集加载: {len(self.train_df)} 个样本")

            # 预处理数据
            self._preprocess_data()

        def _preprocess_data(self):
            features = []
            targets = []

            for _, row in self.train_df.iterrows():
                mol_name = row['molecule_name']
                atom_0 = row['atom_index_0']
                atom_1 = row['atom_index_1']
                coupling_type = row['type']
                target = row['scalar_coupling_constant']

                mol_struct = self.structures_df[self.structures_df['molecule_name'] == mol_name]
                if len(mol_struct) <= max(atom_0, atom_1):
                    continue

                # 提取基础特征
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

            # 标准化
            self.scaler = StandardScaler()
            self.features = torch.tensor(self.scaler.fit_transform(features), dtype=torch.float32)
            self.targets = torch.tensor(targets, dtype=torch.float32)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.targets[idx]

    class GraphCouplingDataset(Dataset):
        """图数据集 - GNN模型使用"""

        def __init__(self, data_path, max_samples=3000, advanced_features=False):
            self.data_path = data_path
            self.max_samples = max_samples
            self.advanced_features = advanced_features

            # 原子特征映射
            self.atom_features = {
                'H': [1, 1.008, 1, 1, 2.20, 0.31],
                'C': [6, 12.01, 4, 2, 2.55, 0.76],
                'N': [7, 14.01, 5, 2, 3.04, 0.71],
                'O': [8, 16.00, 6, 2, 3.44, 0.66],
                'F': [9, 19.00, 7, 2, 3.98, 0.57],
            }

            print(f"图数据集加载: {max_samples} 样本, 高级特征: {advanced_features}")
            self._load_and_process_data()

        def _load_and_process_data(self):
            train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv')).head(self.max_samples)
            structures_df = pd.read_csv(os.path.join(self.data_path, 'structures.csv'))

            self.graph_data = self._create_graph_data(train_df, structures_df)
            print(f"成功创建 {len(self.graph_data)} 个图数据")

        def _create_graph_data(self, train_df, structures_df):
            graph_data_list = []
            molecule_groups = train_df.groupby('molecule_name')

            for mol_name, coupling_df in molecule_groups:
                mol_structure = structures_df[structures_df['molecule_name'] == mol_name]
                if len(mol_structure) == 0:
                    continue

                mol_structure = mol_structure.sort_values('atom_index').reset_index(drop=True)

                try:
                    # 创建原子特征
                    atom_features = []
                    atom_coords = []

                    for _, atom_row in mol_structure.iterrows():
                        atom_type = atom_row['atom']
                        coords = [atom_row['x'], atom_row['y'], atom_row['z']]

                        features = self.atom_features.get(atom_type, [0, 0, 0, 0, 0, 0])
                        atom_features.append(features)
                        atom_coords.append(coords)

                    atom_features = torch.tensor(atom_features, dtype=torch.float)
                    atom_coords = torch.tensor(atom_coords, dtype=torch.float)

                    # 创建边连接
                    edge_index, edge_attr = self._create_edges(atom_coords)

                    # 处理每个原子对
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

                        # 高级特征
                        advanced_feats = []
                        if self.advanced_features:
                            advanced_feats = self._extract_advanced_features(mol_structure, atom_idx_0, atom_idx_1)

                        data = {
                            'atom_features': atom_features,
                            'atom_coords': atom_coords,
                            'edge_index': edge_index,
                            'edge_attr': edge_attr,
                            'pair_indices': torch.tensor([[atom_idx_0, atom_idx_1]], dtype=torch.long),
                            'pair_features': pair_features,
                            'advanced_features': torch.tensor(advanced_feats, dtype=torch.float) if advanced_feats else torch.tensor([]),
                            'coupling_constant': torch.tensor([coupling_constant], dtype=torch.float),
                            'coupling_type': coupling_type
                        }

                        graph_data_list.append(data)

                except Exception as e:
                    continue

            return graph_data_list

        def _create_edges(self, atom_coords, cutoff=2.0):
            num_atoms = len(atom_coords)
            edge_list = []
            edge_distances = []

            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    dist = torch.norm(atom_coords[i] - atom_coords[j]).item()
                    if dist < cutoff:
                        edge_list.extend([[i, j], [j, i]])
                        edge_distances.extend([dist, dist])

            if len(edge_list) == 0:
                for i in range(num_atoms):
                    edge_list.append([i, i])
                    edge_distances.append(0.0)

            edge_index = torch.tensor(edge_list, dtype=torch.long).T
            edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)

            return edge_index, edge_attr

        def _create_pair_features(self, atom_idx_0, atom_idx_1, coupling_type, atom_coords, mol_structure):
            # 距离特征
            distance = torch.norm(atom_coords[atom_idx_0] - atom_coords[atom_idx_1]).item()
            rel_pos = atom_coords[atom_idx_1] - atom_coords[atom_idx_0]

            # 原子类型编码
            atom_0_type = mol_structure.iloc[atom_idx_0]['atom']
            atom_1_type = mol_structure.iloc[atom_idx_1]['atom']

            atom_type_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            type_map = {'1JHC': 0, '2JHH': 1, '3JHH': 2, '1JCC': 3,
                       '2JHC': 4, '2JCH': 5, '3JHC': 6}

            features = [
                distance,
                rel_pos[0].item(), rel_pos[1].item(), rel_pos[2].item(),
                atom_type_map.get(atom_0_type, 0),
                atom_type_map.get(atom_1_type, 0),
                type_map.get(coupling_type, 0),
                len(mol_structure)
            ]

            return torch.tensor(features, dtype=torch.float).unsqueeze(0)

        def _extract_advanced_features(self, mol_structure, atom_idx_0, atom_idx_1):
            """提取高级特征"""
            features = []

            # 拓扑特征
            num_atoms = len(mol_structure)
            features.extend([num_atoms])

            # 原子类型分布
            atom_counts = mol_structure['atom'].value_counts()
            for atom_type in ['H', 'C', 'N', 'O', 'F']:
                features.append(atom_counts.get(atom_type, 0))

            # 几何特征
            coords = mol_structure[['x', 'y', 'z']].values
            bbox = coords.max(axis=0) - coords.min(axis=0)
            features.extend(bbox)

            center = coords.mean(axis=0)
            features.extend(center)

            # 回转半径
            centered_coords = coords - center
            gyration_radius = np.sqrt(np.sum(centered_coords ** 2) / len(coords))
            features.append(gyration_radius)

            # 分子量估算
            atom_masses = {'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00}
            mol_weight = sum(atom_masses.get(atom, 0) for atom in mol_structure['atom'])
            features.append(mol_weight)

            return features[:50]  # 限制特征数量

        def __len__(self):
            return len(self.graph_data)

        def __getitem__(self, idx):
            return self.graph_data[idx]

    # =============================================================================
    # 2. 模型定义
    # =============================================================================

    class SimpleMLP(nn.Module):
        """简化MLP模型"""
        def __init__(self, input_dim=8, hidden_dim=64):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim//2, 1)
            )

        def forward(self, x):
            return self.mlp(x)

    class CouplingGCN(nn.Module):
        """GCN模型"""
        def __init__(self, num_atom_features, num_pair_features, hidden_dim=128, num_layers=3):
            super().__init__()
            self.num_layers = num_layers

            self.atom_embedding = nn.Linear(num_atom_features, hidden_dim)

            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            for i in range(num_layers):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.batch_norms.append(BatchNorm(hidden_dim))

            self.pair_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + num_pair_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim//2, 1)
            )

        def forward(self, atom_features, edge_index, pair_indices, pair_features):
            x = self.atom_embedding(atom_features)

            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)

            atom_pair_0 = x[pair_indices[:, 0]]
            atom_pair_1 = x[pair_indices[:, 1]]

            combined = torch.cat([atom_pair_0, atom_pair_1, pair_features], dim=1)
            return self.pair_mlp(combined)

    class CouplingGAT(nn.Module):
        """GAT模型"""
        def __init__(self, num_atom_features, num_pair_features, hidden_dim=128, num_layers=3, heads=4):
            super().__init__()
            self.num_layers = num_layers
            self.heads = heads

            self.atom_embedding = nn.Linear(num_atom_features, hidden_dim)

            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.convs.append(GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=0.2))
            self.batch_norms.append(BatchNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=0.2))
                self.batch_norms.append(BatchNorm(hidden_dim))

            if num_layers > 1:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.2))
                self.batch_norms.append(BatchNorm(hidden_dim))

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

            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.batch_norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training)

            atom_pair_0 = x[pair_indices[:, 0]]
            atom_pair_1 = x[pair_indices[:, 1]]

            combined = torch.cat([atom_pair_0, atom_pair_1, pair_features], dim=1)
            return self.pair_mlp(combined)

    class CouplingTransformer(nn.Module):
        """Graph Transformer模型"""
        def __init__(self, num_atom_features, num_pair_features, hidden_dim=128, num_layers=2, heads=8):
            super().__init__()
            self.num_layers = num_layers

            self.atom_embedding = nn.Linear(num_atom_features, hidden_dim)

            self.transformers = nn.ModuleList()
            self.layer_norms = nn.ModuleList()

            for _ in range(num_layers):
                self.transformers.append(
                    TransformerConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=0.2)
                )
                self.layer_norms.append(LayerNorm(hidden_dim))

            self.pair_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + num_pair_features, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, atom_features, edge_index, pair_indices, pair_features):
            x = self.atom_embedding(atom_features)

            for i in range(self.num_layers):
                residual = x
                x = self.transformers[i](x, edge_index)
                x = self.layer_norms[i](x)
                x = F.gelu(x + residual)
                x = F.dropout(x, p=0.2, training=self.training)

            atom_pair_0 = x[pair_indices[:, 0]]
            atom_pair_1 = x[pair_indices[:, 1]]

            combined = torch.cat([atom_pair_0, atom_pair_1, pair_features], dim=1)
            return self.pair_mlp(combined)

    class AdvancedMLP(nn.Module):
        """高级特征MLP"""
        def __init__(self, num_features, hidden_dims=[256, 128, 64], dropout=0.3):
            super().__init__()

            layers = []
            input_dim = num_features

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            return self.mlp(x)

    # =============================================================================
    # 3. 自定义collate函数
    # =============================================================================

    def graph_collate_fn(self, batch):
        """图数据的batch处理函数"""
        all_atom_features = []
        all_atom_coords = []
        all_edge_indices = []
        all_edge_attrs = []
        all_pair_indices = []
        all_pair_features = []
        all_coupling_constants = []
        all_advanced_features = []

        atom_offset = 0

        for data in batch:
            num_atoms = data['atom_features'].shape[0]

            all_atom_features.append(data['atom_features'])
            all_atom_coords.append(data['atom_coords'])

            edge_index = data['edge_index'] + atom_offset
            all_edge_indices.append(edge_index)
            all_edge_attrs.append(data['edge_attr'])

            pair_indices = data['pair_indices'] + atom_offset
            all_pair_indices.append(pair_indices)

            all_pair_features.append(data['pair_features'])
            all_coupling_constants.append(data['coupling_constant'])

            if len(data['advanced_features']) > 0:
                all_advanced_features.append(data['advanced_features'])

            atom_offset += num_atoms

        return {
            'atom_features': torch.cat(all_atom_features, dim=0),
            'atom_coords': torch.cat(all_atom_coords, dim=0),
            'edge_index': torch.cat(all_edge_indices, dim=1),
            'edge_attr': torch.cat(all_edge_attrs, dim=0),
            'pair_indices': torch.cat(all_pair_indices, dim=0),
            'pair_features': torch.cat(all_pair_features, dim=0),
            'coupling_constants': torch.cat(all_coupling_constants, dim=0),
            'advanced_features': torch.cat(all_advanced_features, dim=0) if all_advanced_features else None
        }

    # =============================================================================
    # 4. 训练和评估函数
    # =============================================================================

    def train_model(self, model, train_loader, val_loader, model_name, num_epochs=25, lr=0.001):
        """统一的模型训练函数"""
        print(f"🔥 训练 {model_name}...")

        model.to(self.device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5
        )

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

                if isinstance(model, self.SimpleMLP):
                    if isinstance(batch, dict):
                        predictions = model(batch['pair_features'])
                        targets = batch['coupling_constants']
                    else:
                        features, targets = batch
                        features, targets = features.to(self.device), targets.to(self.device)
                        predictions = model(features).squeeze()
                else:
                    # GNN模型
                    atom_features = batch['atom_features'].to(self.device)
                    edge_index = batch['edge_index'].to(self.device)
                    pair_indices = batch['pair_indices'].to(self.device)
                    pair_features = batch['pair_features'].to(self.device)
                    targets = batch['coupling_constants'].to(self.device)

                    predictions = model(atom_features, edge_index, pair_indices, pair_features)
                    predictions = predictions.squeeze()

                if not isinstance(model, self.SimpleMLP) or isinstance(batch, dict):
                    targets = targets.to(self.device)

                loss = criterion(predictions, targets)
                loss.backward()

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
                    if isinstance(model, self.SimpleMLP):
                        if isinstance(batch, dict):
                            predictions = model(batch['pair_features'])
                            targets = batch['coupling_constants']
                        else:
                            features, targets = batch
                            features, targets = features.to(self.device), targets.to(self.device)
                            predictions = model(features).squeeze()
                    else:
                        atom_features = batch['atom_features'].to(self.device)
                        edge_index = batch['edge_index'].to(self.device)
                        pair_indices = batch['pair_indices'].to(self.device)
                        pair_features = batch['pair_features'].to(self.device)
                        targets = batch['coupling_constants'].to(self.device)

                        predictions = model(atom_features, edge_index, pair_indices, pair_features)
                        predictions = predictions.squeeze()

                    if not isinstance(model, self.SimpleMLP) or isinstance(batch, dict):
                        targets = targets.to(self.device)

                    loss = criterion(predictions, targets)
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
                print(f"  Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

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
        """统一的模型评估函数"""
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(model, self.SimpleMLP):
                    if isinstance(batch, dict):
                        predictions = model(batch['pair_features'])
                        targets = batch['coupling_constants']
                    else:
                        features, targets = batch
                        features = features.to(self.device)
                        predictions = model(features).squeeze()
                else:
                    atom_features = batch['atom_features'].to(self.device)
                    edge_index = batch['edge_index'].to(self.device)
                    pair_indices = batch['pair_indices'].to(self.device)
                    pair_features = batch['pair_features'].to(self.device)
                    targets = batch['coupling_constants']

                    predictions = model(atom_features, edge_index, pair_indices, pair_features)
                    predictions = predictions.squeeze()

                all_predictions.extend(predictions.cpu().numpy())
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
    # 5. 主要运行函数
    # =============================================================================

    def run_all_frameworks(self):
        """运行所有框架并比较"""
        print("\n" + "="*80)
        print("🚀 开始运行所有标量耦合常数预测框架")
        print("="*80)

        start_time = time.time()

        # 1. 简化MLP基线
        print(f"\n{'='*60}")
        print("📊 1. 运行简化MLP基线")
        print(f"{'='*60}")

        try:
            simple_result = self._run_simple_mlp()
            self.results['Simple_MLP'] = simple_result
            print(f"✅ Simple MLP完成: MAE={simple_result['MAE']:.4f}")
        except Exception as e:
            print(f"❌ Simple MLP失败: {e}")
            self.results['Simple_MLP'] = {'error': str(e)}

        # 2. GCN模型
        print(f"\n{'='*60}")
        print("🔥 2. 运行GCN模型")
        print(f"{'='*60}")

        try:
            gcn_result = self._run_gcn()
            self.results['GCN'] = gcn_result
            print(f"✅ GCN完成: MAE={gcn_result['MAE']:.4f}")
        except Exception as e:
            print(f"❌ GCN失败: {e}")
            self.results['GCN'] = {'error': str(e)}

        # 3. GAT模型
        print(f"\n{'='*60}")
        print("🎯 3. 运行GAT模型")
        print(f"{'='*60}")

        try:
            gat_result = self._run_gat()
            self.results['GAT'] = gat_result
            print(f"✅ GAT完成: MAE={gat_result['MAE']:.4f}")
        except Exception as e:
            print(f"❌ GAT失败: {e}")
            self.results['GAT'] = {'error': str(e)}

        # 4. Transformer模型
        print(f"\n{'='*60}")
        print("🌟 4. 运行Graph Transformer模型")
        print(f"{'='*60}")

        try:
            transformer_result = self._run_transformer()
            self.results['Transformer'] = transformer_result
            print(f"✅ Transformer完成: MAE={transformer_result['MAE']:.4f}")
        except Exception as e:
            print(f"❌ Transformer失败: {e}")
            self.results['Transformer'] = {'error': str(e)}

        # 5. 高级特征模型 (如果有高级特征)
        print(f"\n{'='*60}")
        print("🚀 5. 运行高级特征模型")
        print(f"{'='*60}")

        try:
            advanced_result = self._run_advanced_features()
            self.results['Advanced_Features'] = advanced_result
            print(f"✅ Advanced Features完成: MAE={advanced_result['MAE']:.4f}")
        except Exception as e:
            print(f"❌ Advanced Features失败: {e}")
            self.results['Advanced_Features'] = {'error': str(e)}

        total_time = time.time() - start_time

        # 生成统一报告
        self._generate_unified_report(total_time)

        return self.results

    def _run_simple_mlp(self):
        """运行简化MLP"""
        dataset = self.SimpleCouplingDataset(self.data_path, self.max_samples)

        # 数据划分
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_data, val_data, test_data = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

        model = self.SimpleMLP(input_dim=8, hidden_dim=64)

        train_result = self.train_model(model, train_loader, val_loader, "Simple MLP")
        eval_result = self.evaluate_model(train_result['model'], test_loader)

        return {**train_result, **eval_result}

    def _run_gcn(self):
        """运行GCN模型"""
        dataset = self.GraphCouplingDataset(self.data_path, self.max_samples)

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_data, val_data, test_data = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=self.graph_collate_fn)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=self.graph_collate_fn)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=self.graph_collate_fn)

        # 获取特征维度
        sample_data = dataset[0]
        num_atom_features = sample_data['atom_features'].shape[1]
        num_pair_features = sample_data['pair_features'].shape[1]

        model = self.CouplingGCN(num_atom_features, num_pair_features, hidden_dim=128)

        train_result = self.train_model(model, train_loader, val_loader, "GCN")
        eval_result = self.evaluate_model(train_result['model'], test_loader)

        return {**train_result, **eval_result}

    def _run_gat(self):
        """运行GAT模型"""
        dataset = self.GraphCouplingDataset(self.data_path, self.max_samples)

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_data, val_data, test_data = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=self.graph_collate_fn)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=self.graph_collate_fn)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=self.graph_collate_fn)

        sample_data = dataset[0]
        num_atom_features = sample_data['atom_features'].shape[1]
        num_pair_features = sample_data['pair_features'].shape[1]

        model = self.CouplingGAT(num_atom_features, num_pair_features, hidden_dim=128, heads=4)

        train_result = self.train_model(model, train_loader, val_loader, "GAT")
        eval_result = self.evaluate_model(train_result['model'], test_loader)

        return {**train_result, **eval_result}

    def _run_transformer(self):
        """运行Transformer模型"""
        dataset = self.GraphCouplingDataset(self.data_path, self.max_samples)

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_data, val_data, test_data = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=self.graph_collate_fn)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=self.graph_collate_fn)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=self.graph_collate_fn)

        sample_data = dataset[0]
        num_atom_features = sample_data['atom_features'].shape[1]
        num_pair_features = sample_data['pair_features'].shape[1]

        model = self.CouplingTransformer(num_atom_features, num_pair_features, hidden_dim=128, heads=8)

        train_result = self.train_model(model, train_loader, val_loader, "Transformer")
        eval_result = self.evaluate_model(train_result['model'], test_loader)

        return {**train_result, **eval_result}

    def _run_advanced_features(self):
        """运行高级特征模型"""
        dataset = self.GraphCouplingDataset(self.data_path, self.max_samples, advanced_features=True)

        # 提取所有特征用于MLP
        all_features = []
        all_targets = []

        for i, data in enumerate(dataset):
            if i % 1000 == 0:
                print(f"  处理特征 {i}/{len(dataset)}")

            # 合并所有特征
            pair_feats = data['pair_features'].numpy().flatten()
            advanced_feats = data['advanced_features'].numpy() if len(data['advanced_features']) > 0 else np.zeros(50)

            combined_features = np.concatenate([pair_feats, advanced_feats])
            all_features.append(combined_features)
            all_targets.append(data['coupling_constant'].item())

        # 标准化和特征选择
        scaler = RobustScaler()
        features = scaler.fit_transform(all_features)

        # 特征选择
        selector = SelectKBest(f_regression, k=min(50, features.shape[1]))
        features = selector.fit_transform(features, all_targets)

        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(all_targets, dtype=torch.float32)

        # 创建数据集
        class AdvancedDataset(Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]

        adv_dataset = AdvancedDataset(features, targets)

        # 数据划分
        train_size = int(0.7 * len(adv_dataset))
        val_size = int(0.15 * len(adv_dataset))
        test_size = len(adv_dataset) - train_size - val_size

        train_data, val_data, test_data = torch.utils.data.random_split(
            adv_dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

        model = self.AdvancedMLP(features.shape[1], hidden_dims=[512, 256, 128, 64])

        train_result = self.train_model(model, train_loader, val_loader, "Advanced Features")
        eval_result = self.evaluate_model(train_result['model'], test_loader)

        return {**train_result, **eval_result}

    # =============================================================================
    # 6. 报告生成
    # =============================================================================

    def _generate_unified_report(self, total_time):
        """生成统一比较报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*80}")
        print("📋 生成统一比较报告")
        print(f"{'='*80}")

        # 过滤有效结果
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not valid_results:
            print("❌ 没有有效的结果可以报告")
            return

        # 按MAE排序
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['MAE'])

        print(f"\n🏆 模型性能排行榜 (按MAE排序):")
        print("-" * 90)
        print(f"{'排名':<4} {'模型':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'参数量':<12} {'训练时间':<10}")
        print("-" * 90)

        for i, (model_name, result) in enumerate(sorted_results, 1):
            print(f"{i:<4} {model_name:<20} {result['MAE']:<10.4f} {result['RMSE']:<10.4f} "
                  f"{result['R2']:<10.4f} {result['num_parameters']:<12,} {result['training_time']:<10.1f}s")

        # 保存详细结果到CSV
        csv_filename = f"Unified_Coupling_Results_{timestamp}.csv"
        self._save_results_csv(valid_results, csv_filename)

        # 保存详细文本报告
        txt_filename = f"Unified_Coupling_Report_{timestamp}.txt"
        self._save_text_report(valid_results, total_time, txt_filename)

        # 生成可视化
        plot_filename = f"Unified_Coupling_Plots_{timestamp}.png"
        self._create_comparison_plots(valid_results, plot_filename)

        # 生成JSON结果
        json_filename = f"Unified_Coupling_Data_{timestamp}.json"
        self._save_json_results(valid_results, total_time, json_filename)

        print(f"\n✅ 报告生成完成:")
        print(f"   📊 CSV结果: {csv_filename}")
        print(f"   📝 文本报告: {txt_filename}")
        print(f"   📈 可视化图表: {plot_filename}")
        print(f"   💾 JSON数据: {json_filename}")
        print(f"\n⏱️  总运行时间: {total_time/60:.2f} 分钟")
        print(f"🏆 最佳模型: {sorted_results[0][0]} (MAE: {sorted_results[0][1]['MAE']:.4f})")

    def _save_results_csv(self, results, filename):
        """保存结果到CSV"""
        data = []
        for model_name, result in results.items():
            data.append({
                'Model': model_name,
                'MAE': result['MAE'],
                'RMSE': result['RMSE'],
                'R2': result['R2'],
                'Num_Parameters': result['num_parameters'],
                'Training_Time_s': result['training_time'],
                'Best_Val_Loss': result.get('best_val_loss', 0)
            })

        df = pd.DataFrame(data)
        df = df.sort_values('MAE')
        df.to_csv(filename, index=False)

    def _save_text_report(self, results, total_time, filename):
        """保存详细文本报告"""
        sorted_results = sorted(results.items(), key=lambda x: x[1]['MAE'])

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("统一标量耦合常数预测框架比较报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"数据集: {self.data_path}\n")
            f.write(f"样本数量: {self.max_samples}\n")
            f.write(f"总运行时间: {total_time/60:.2f} 分钟\n")
            f.write(f"成功训练模型数: {len(results)} 个\n\n")

            f.write("模型性能排行榜 (按MAE排序):\n")
            f.write("-" * 80 + "\n")

            for i, (model_name, result) in enumerate(sorted_results, 1):
                f.write(f"\n{i}. {model_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"   MAE: {result['MAE']:.6f}\n")
                f.write(f"   RMSE: {result['RMSE']:.6f}\n")
                f.write(f"   R²: {result['R2']:.6f}\n")
                f.write(f"   参数量: {result['num_parameters']:,}\n")
                f.write(f"   训练时间: {result['training_time']:.1f}s\n")

            f.write(f"\n\n统计分析:\n")
            f.write("=" * 30 + "\n")

            maes = [r['MAE'] for r in results.values()]
            f.write(f"平均MAE: {np.mean(maes):.4f}\n")
            f.write(f"最佳MAE: {min(maes):.4f}\n")
            f.write(f"最差MAE: {max(maes):.4f}\n")
            f.write(f"MAE标准差: {np.std(maes):.4f}\n")

            times = [r['training_time'] for r in results.values()]
            f.write(f"平均训练时间: {np.mean(times):.1f}s\n")
            f.write(f"总训练时间: {sum(times):.1f}s\n")

    def _save_json_results(self, results, total_time, filename):
        """保存JSON格式结果"""
        json_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_path': self.data_path,
                'max_samples': self.max_samples,
                'total_runtime': total_time,
                'device': str(self.device)
            },
            'results': {}
        }

        for model_name, result in results.items():
            json_data['results'][model_name] = {
                'MAE': float(result['MAE']),
                'RMSE': float(result['RMSE']),
                'R2': float(result['R2']),
                'num_parameters': int(result['num_parameters']),
                'training_time': float(result['training_time']),
                'best_val_loss': float(result.get('best_val_loss', 0))
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def _create_comparison_plots(self, results, filename):
        """创建比较可视化"""
        if len(results) < 2:
            print("结果数量太少，跳过可视化")
            return

        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('统一标量耦合常数预测框架性能比较', fontsize=16, fontweight='bold')

        models = list(results.keys())
        maes = [results[m]['MAE'] for m in models]
        r2s = [results[m]['R2'] for m in models]
        params = [results[m]['num_parameters'] for m in models]
        times = [results[m]['training_time'] for m in models]

        # 1. MAE对比
        axes[0, 0].bar(models, maes, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('平均绝对误差 (MAE) 对比')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. R²对比
        axes[0, 1].bar(models, r2s, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('决定系数 (R²) 对比')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 参数量vs性能散点图
        scatter = axes[1, 0].scatter(params, maes, c=times, s=100, alpha=0.7, cmap='viridis')
        axes[1, 0].set_xlabel('参数数量')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('模型复杂度 vs 性能 (颜色=训练时间)')

        for i, model in enumerate(models):
            axes[1, 0].annotate(model, (params[i], maes[i]), xytext=(5, 5),
                               textcoords='offset points', fontsize=8)

        plt.colorbar(scatter, ax=axes[1, 0], label='训练时间 (秒)')

        # 4. 训练时间对比
        axes[1, 1].bar(models, times, color='salmon', alpha=0.7)
        axes[1, 1].set_title('训练时间对比')
        axes[1, 1].set_ylabel('训练时间 (秒)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数 - 运行所有框架"""
    print("🎯 统一标量耦合常数预测框架")
    print("=" * 60)

    # 数据路径检查
    data_path = '/Users/xiaotingzhou/Downloads/GNN/Dataset/scalar_coupling_constant'

    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return None

    required_files = ['train.csv', 'structures.csv']
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            print(f"❌ 缺少必要文件: {file}")
            return None

    print("✅ 数据路径检查通过")

    # 设置样本数
    try:
        max_samples = int(input("请输入最大样本数 (默认3000): ").strip() or "3000")
    except:
        max_samples = 3000

    print(f"📊 使用样本数: {max_samples}")

    # 创建框架实例并运行
    frameworks = UnifiedCouplingFrameworks(data_path=data_path, max_samples=max_samples)
    results = frameworks.run_all_frameworks()

    print(f"\n{'='*80}")
    print("🎉 统一框架比较完成!")
    print("📋 查看生成的报告文件获取详细结果")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    results = main()