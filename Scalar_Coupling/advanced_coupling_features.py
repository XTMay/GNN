#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标量耦合常数预测 - 高级特征工程
集成RDKit分子描述符、拓扑特征、几何特征等高级特征提取方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, BatchNorm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings('ignore')

# RDKit导入
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Fragments
    from rdkit.Chem import rdmolops, rdmolfiles
    RDKIT_AVAILABLE = True
    print("✅ RDKit可用")
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit不可用，将跳过RDKit描述符")


class AdvancedCouplingFeatureExtractor:
    """高级特征提取器"""

    def __init__(self):
        self.feature_scalers = {}
        self.feature_selectors = {}

    def extract_topological_features(self, mol_structure):
        """提取拓扑特征"""
        features = []

        # 基础统计
        num_atoms = len(mol_structure)
        features.append(num_atoms)

        # 原子类型分布
        atom_types = mol_structure['atom'].value_counts()
        for atom in ['H', 'C', 'N', 'O', 'F']:
            features.append(atom_types.get(atom, 0))

        # 创建邻接矩阵用于图分析
        coords = mol_structure[['x', 'y', 'z']].values
        distances = np.sqrt(np.sum((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2, axis=2))
        adjacency = (distances < 2.0) & (distances > 0)

        # 度数分布
        degrees = adjacency.sum(axis=1)
        features.extend([
            degrees.mean(),
            degrees.std(),
            degrees.max(),
            degrees.min()
        ])

        # 图密度
        possible_edges = num_atoms * (num_atoms - 1) // 2
        actual_edges = adjacency.sum() // 2
        density = actual_edges / possible_edges if possible_edges > 0 else 0
        features.append(density)

        return np.array(features)

    def extract_geometric_features(self, mol_structure):
        """提取3D几何特征"""
        coords = mol_structure[['x', 'y', 'z']].values

        features = []

        # 分子尺寸
        bbox = coords.max(axis=0) - coords.min(axis=0)
        features.extend(bbox)  # x, y, z方向的跨度

        # 几何中心
        center = coords.mean(axis=0)
        features.extend(center)

        # 惯性矩相关特征
        centered_coords = coords - center
        inertia_tensor = np.dot(centered_coords.T, centered_coords)
        eigenvals = np.linalg.eigvals(inertia_tensor)
        eigenvals = np.sort(eigenvals)[::-1]  # 降序排列

        features.extend(eigenvals)  # 主惯性矩

        # 回转半径
        gyration_radius = np.sqrt(np.sum(centered_coords ** 2) / len(coords))
        features.append(gyration_radius)

        # 原子间距离统计
        distances = np.sqrt(np.sum((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2, axis=2))
        # 只考虑非零距离
        non_zero_distances = distances[distances > 0]
        if len(non_zero_distances) > 0:
            features.extend([
                non_zero_distances.mean(),
                non_zero_distances.std(),
                non_zero_distances.max(),
                non_zero_distances.min()
            ])
        else:
            features.extend([0, 0, 0, 0])

        # 质心到各原子的距离统计
        center_distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))
        features.extend([
            center_distances.mean(),
            center_distances.std(),
            center_distances.max()
        ])

        return np.array(features)

    def extract_rdkit_features(self, mol_structure):
        """提取RDKit分子描述符"""
        if not RDKIT_AVAILABLE:
            return np.zeros(50)  # 返回零向量

        try:
            # 从结构创建分子对象（简化版本）
            # 注意: 实际应用中需要更复杂的分子重建逻辑
            mol_features = []

            # 原子数和键数的估算
            num_atoms = len(mol_structure)
            mol_features.append(num_atoms)

            # 原子类型统计
            atom_counts = mol_structure['atom'].value_counts()
            mol_features.append(atom_counts.get('C', 0))  # 碳原子数
            mol_features.append(atom_counts.get('H', 0))  # 氢原子数
            mol_features.append(atom_counts.get('N', 0))  # 氮原子数
            mol_features.append(atom_counts.get('O', 0))  # 氧原子数
            mol_features.append(atom_counts.get('F', 0))  # 氟原子数

            # 分子量估算
            atom_masses = {'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00}
            mol_weight = sum(atom_masses.get(atom, 0) for atom in mol_structure['atom'])
            mol_features.append(mol_weight)

            # 基于3D坐标的简单描述符
            coords = mol_structure[['x', 'y', 'z']].values

            # 分子体积估算（凸包近似）
            if len(coords) >= 4:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(coords)
                    volume = hull.volume
                except:
                    volume = 0
            else:
                volume = 0
            mol_features.append(volume)

            # 表面积估算
            if len(coords) >= 4:
                try:
                    surface_area = hull.area
                except:
                    surface_area = 0
            else:
                surface_area = 0
            mol_features.append(surface_area)

            # 填充到50维
            while len(mol_features) < 50:
                mol_features.append(0.0)

            return np.array(mol_features[:50])

        except Exception as e:
            print(f"RDKit特征提取失败: {e}")
            return np.zeros(50)

    def extract_coupling_specific_features(self, atom_idx_0, atom_idx_1, coupling_type,
                                         mol_structure, topological_features, geometric_features):
        """提取耦合特异性特征"""
        features = []

        # 原子对基础信息
        atom_0 = mol_structure.iloc[atom_idx_0]
        atom_1 = mol_structure.iloc[atom_idx_1]

        coords_0 = np.array([atom_0['x'], atom_0['y'], atom_0['z']])
        coords_1 = np.array([atom_1['x'], atom_1['y'], atom_1['z']])

        # 原子对距离
        distance = np.linalg.norm(coords_1 - coords_0)
        features.append(distance)

        # 相对位置向量
        rel_pos = coords_1 - coords_0
        features.extend(rel_pos)

        # 距离的对数和倒数
        features.append(np.log(distance + 1e-6))
        features.append(1.0 / (distance + 1e-6))

        # 原子类型编码
        atom_type_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        atom_0_type = atom_type_map.get(atom_0['atom'], 0)
        atom_1_type = atom_type_map.get(atom_1['atom'], 0)
        features.extend([atom_0_type, atom_1_type])

        # 耦合类型编码
        coupling_type_map = {'1JHC': 0, '2JHH': 1, '3JHH': 2, '1JCC': 3,
                           '2JHC': 4, '2JCH': 5, '3JHC': 6}
        coupling_encoded = coupling_type_map.get(coupling_type, 0)
        features.append(coupling_encoded)

        # 原子在分子中的位置（相对于质心的距离）
        mol_center = mol_structure[['x', 'y', 'z']].mean().values
        dist_to_center_0 = np.linalg.norm(coords_0 - mol_center)
        dist_to_center_1 = np.linalg.norm(coords_1 - mol_center)
        features.extend([dist_to_center_0, dist_to_center_1])

        # 角度特征（如果分子中有三个以上原子）
        if len(mol_structure) >= 3:
            other_atoms = mol_structure[~mol_structure.index.isin([atom_idx_0, atom_idx_1])]
            if len(other_atoms) > 0:
                # 找到最近的第三个原子
                other_coords = other_atoms[['x', 'y', 'z']].values
                distances_to_0 = np.linalg.norm(other_coords - coords_0[np.newaxis, :], axis=1)
                nearest_idx = distances_to_0.argmin()
                coords_2 = other_coords[nearest_idx]

                # 计算角度 (atom_0 - atom_1 - atom_2)
                vec1 = coords_0 - coords_1
                vec2 = coords_2 - coords_1
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                features.append(angle)
            else:
                features.append(0)
        else:
            features.append(0)

        # 加入分子级别特征的摘要
        features.append(topological_features.mean())
        features.append(geometric_features.mean())

        return np.array(features)


class AdvancedCouplingDataset(Dataset):
    """高级特征的标量耦合常数数据集"""

    def __init__(self, data_path, max_samples=5000, feature_selection=True):
        self.data_path = data_path
        self.max_samples = max_samples
        self.feature_selection = feature_selection

        self.feature_extractor = AdvancedCouplingFeatureExtractor()
        self.scaler = RobustScaler()  # 使用RobustScaler处理异常值

        print(f"加载高级特征标量耦合常数数据，样本数: {max_samples}")
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """加载和预处理数据"""
        # 加载数据
        train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv')).head(self.max_samples)
        structures_df = pd.read_csv(os.path.join(self.data_path, 'structures.csv'))

        print(f"加载 {len(train_df)} 个耦合样本")

        # 提取高级特征
        self._extract_advanced_features(train_df, structures_df)

    def _extract_advanced_features(self, train_df, structures_df):
        """提取高级特征"""
        print("提取高级特征...")

        all_features = []
        all_targets = []

        # 按分子分组处理
        molecule_groups = train_df.groupby('molecule_name')
        processed_molecules = 0

        for mol_name, coupling_df in molecule_groups:
            if processed_molecules % 500 == 0:
                print(f"处理分子: {processed_molecules}/{len(molecule_groups)}")

            # 获取分子结构
            mol_structure = structures_df[structures_df['molecule_name'] == mol_name]
            if len(mol_structure) == 0:
                continue

            mol_structure = mol_structure.sort_values('atom_index').reset_index(drop=True)

            try:
                # 提取分子级别特征
                topo_features = self.feature_extractor.extract_topological_features(mol_structure)
                geom_features = self.feature_extractor.extract_geometric_features(mol_structure)
                rdkit_features = self.feature_extractor.extract_rdkit_features(mol_structure)

                # 处理每个原子对的耦合常数
                for _, coupling_row in coupling_df.iterrows():
                    atom_idx_0 = coupling_row['atom_index_0']
                    atom_idx_1 = coupling_row['atom_index_1']
                    coupling_constant = coupling_row['scalar_coupling_constant']
                    coupling_type = coupling_row['type']

                    if atom_idx_0 >= len(mol_structure) or atom_idx_1 >= len(mol_structure):
                        continue

                    # 提取耦合特异性特征
                    coupling_features = self.feature_extractor.extract_coupling_specific_features(
                        atom_idx_0, atom_idx_1, coupling_type, mol_structure,
                        topo_features, geom_features
                    )

                    # 合并所有特征
                    combined_features = np.concatenate([
                        coupling_features,
                        topo_features,
                        geom_features,
                        rdkit_features
                    ])

                    all_features.append(combined_features)
                    all_targets.append(coupling_constant)

                processed_molecules += 1

            except Exception as e:
                continue

        # 转换为numpy数组
        self.features = np.array(all_features)
        self.targets = np.array(all_targets)

        print(f"提取特征完成: {len(self.features)} 个样本, {self.features.shape[1]} 个特征")

        # 特征缩放
        self.features = self.scaler.fit_transform(self.features)

        # 特征选择
        if self.feature_selection and self.features.shape[1] > 50:
            print("进行特征选择...")
            selector = SelectKBest(f_regression, k=min(50, self.features.shape[1]))
            self.features = selector.fit_transform(self.features, self.targets)
            self.feature_selector = selector
            print(f"特征选择后: {self.features.shape[1]} 个特征")

        # 转换为tensor
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class AdvancedCouplingMLP(nn.Module):
    """高级特征的耦合常数预测模型"""

    def __init__(self, num_features, hidden_dims=[256, 128, 64], dropout=0.3):
        super(AdvancedCouplingMLP, self).__init__()

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

        # 输出层
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def train_advanced_model(model, train_loader, val_loader, device, num_epochs=50):
    """训练高级特征模型"""
    print("训练高级特征模型...")

    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=7, min_lr=1e-6
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

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(features).squeeze()
            loss = criterion(predictions, targets)
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
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features).squeeze()
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


def evaluate_advanced_model(model, test_loader, device):
    """评估高级特征模型"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features).squeeze()

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


def create_advanced_comparison_plots(train_result, eval_result):
    """创建高级特征比较可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('高级特征标量耦合常数预测结果', fontsize=16)

    # 1. 训练曲线
    epochs = range(1, len(train_result['train_losses']) + 1)
    axes[0, 0].plot(epochs, train_result['train_losses'], label='训练损失')
    axes[0, 0].plot(epochs, train_result['val_losses'], label='验证损失')
    axes[0, 0].set_title('训练曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. 预测 vs 真实值
    predictions = eval_result['predictions']
    targets = eval_result['targets']

    axes[0, 1].scatter(targets, predictions, alpha=0.6, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_title(f'预测效果 (R² = {eval_result["R2"]:.4f})')
    axes[0, 1].set_xlabel('真实值')
    axes[0, 1].set_ylabel('预测值')

    # 3. 残差分析
    residuals = predictions - targets
    axes[1, 0].scatter(predictions, residuals, alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('残差分析')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差')

    # 4. 误差分布
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('误差分布')
    axes[1, 1].set_xlabel('残差')
    axes[1, 1].set_ylabel('频次')

    plt.tight_layout()
    plt.savefig('advanced_coupling_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("🚀 开始高级特征标量耦合常数预测")

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    data_path = '/Users/xiaotingzhou/Downloads/GNN/Dataset/scalar_coupling_constant'

    # 创建数据集
    dataset = AdvancedCouplingDataset(data_path, max_samples=4000, feature_selection=True)

    # 数据划分
    total_size = len(dataset)
    test_size = int(total_size * 0.2)
    val_size = int(total_size * 0.1)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 创建模型
    num_features = dataset.features.shape[1]
    model = AdvancedCouplingMLP(num_features, hidden_dims=[512, 256, 128, 64], dropout=0.3)

    print(f"模型特征维度: {num_features}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    train_result = train_advanced_model(model, train_loader, val_loader, device, num_epochs=60)

    # 评估模型
    eval_result = evaluate_advanced_model(train_result['model'], test_loader, device)

    # 打印结果
    print(f"\n{'='*60}")
    print("🎯 高级特征模型结果:")
    print(f"{'='*60}")
    print(f"MAE: {eval_result['MAE']:.6f}")
    print(f"RMSE: {eval_result['RMSE']:.6f}")
    print(f"R²: {eval_result['R2']:.6f}")
    print(f"参数量: {train_result['num_parameters']:,}")
    print(f"训练时间: {train_result['training_time']:.1f}s")

    # 创建可视化
    create_advanced_comparison_plots(train_result, eval_result)

    # 保存结果
    results = {
        **train_result,
        **eval_result
    }

    return results


if __name__ == "__main__":
    results = main()