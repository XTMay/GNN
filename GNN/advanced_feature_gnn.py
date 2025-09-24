#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征提取GNN模型
探索不同的特征工程和预处理方法

特征工程方法包括:
1. 基于RDKit的分子描述符
2. 量子化学特征
3. 拓扑特征
4. 3D几何特征
5. 图结构特征
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
from torch_geometric.utils import to_networkx, degree
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import networkx as nx
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("警告: RDKit未安装，将跳过基于RDKit的特征")


class AdvancedFeatureExtractor:
    """高级特征提取器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 保留95%的方差
        self.feature_selector = SelectKBest(f_regression, k=50)

    def extract_topological_features(self, data):
        """提取拓扑特征"""
        edge_index = data.edge_index.numpy()
        num_nodes = data.x.shape[0]

        # 构建NetworkX图
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_index.T)

        features = []

        # 基础图统计
        features.append(G.number_of_nodes())  # 节点数
        features.append(G.number_of_edges())  # 边数
        features.append(2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0)  # 平均度

        # 连通性特征
        features.append(1 if nx.is_connected(G) else 0)  # 是否连通
        features.append(len(list(nx.connected_components(G))))  # 连通分量数

        # 中心性特征
        if G.number_of_nodes() > 0:
            degree_centrality = list(nx.degree_centrality(G).values())
            features.extend([
                np.mean(degree_centrality),  # 平均度中心性
                np.std(degree_centrality),   # 度中心性标准差
                np.max(degree_centrality),   # 最大度中心性
            ])

            # 聚类系数
            clustering = list(nx.clustering(G).values())
            features.extend([
                np.mean(clustering),  # 平均聚类系数
                np.std(clustering),   # 聚类系数标准差
            ])

            # 路径长度特征
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
                diameter = nx.diameter(G)
                radius = nx.radius(G)
                features.extend([avg_path_length, diameter, radius])
            else:
                features.extend([0, 0, 0])

        else:
            features.extend([0] * 8)

        # 环结构特征
        cycles = nx.cycle_basis(G)
        features.append(len(cycles))  # 环的数量

        # 环的长度分布
        cycle_lengths = [len(cycle) for cycle in cycles]
        features.extend([
            len([l for l in cycle_lengths if l == 3]),  # 三元环数量
            len([l for l in cycle_lengths if l == 4]),  # 四元环数量
            len([l for l in cycle_lengths if l == 5]),  # 五元环数量
            len([l for l in cycle_lengths if l == 6]),  # 六元环数量
            len([l for l in cycle_lengths if l > 6]),   # 大环数量
        ])

        return np.array(features, dtype=np.float32)

    def extract_3d_geometric_features(self, data):
        """提取3D几何特征"""
        if not hasattr(data, 'pos') or data.pos is None:
            return np.zeros(20, dtype=np.float32)  # 返回零特征

        pos = data.pos.numpy()
        features = []

        # 分子尺寸特征
        min_coords = np.min(pos, axis=0)
        max_coords = np.max(pos, axis=0)
        size = max_coords - min_coords

        features.extend([
            np.linalg.norm(size),      # 分子大小
            size[0], size[1], size[2]  # x, y, z方向的尺寸
        ])

        # 质心特征
        centroid = np.mean(pos, axis=0)
        features.extend(centroid)  # 质心坐标

        # 惯性矩特征
        centered_pos = pos - centroid
        inertia_tensor = np.dot(centered_pos.T, centered_pos)
        eigenvalues = np.linalg.eigvals(inertia_tensor)
        eigenvalues = np.sort(eigenvalues)[::-1]
        features.extend(eigenvalues)  # 主惯性矩

        # 回转半径
        gyration_radius = np.sqrt(np.mean(np.sum(centered_pos**2, axis=1)))
        features.append(gyration_radius)

        # 偏心率
        if eigenvalues[0] > 0:
            eccentricity = eigenvalues[2] / eigenvalues[0]
            features.append(eccentricity)
        else:
            features.append(0)

        # 原子间距离统计
        distances = []
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                dist = np.linalg.norm(pos[i] - pos[j])
                distances.append(dist)

        if distances:
            features.extend([
                np.mean(distances),    # 平均距离
                np.std(distances),     # 距离标准差
                np.min(distances),     # 最小距离
                np.max(distances),     # 最大距离
            ])
        else:
            features.extend([0, 0, 0, 0])

        return np.array(features, dtype=np.float32)

    def extract_rdkit_features(self, smiles):
        """提取RDKit分子描述符"""
        if not RDKIT_AVAILABLE or smiles is None:
            return np.zeros(50, dtype=np.float32)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(50, dtype=np.float32)

            features = []

            # 基础分子性质
            features.extend([
                Descriptors.MolWt(mol),                    # 分子量
                Descriptors.MolLogP(mol),                  # LogP
                Descriptors.NumHDonors(mol),               # 氢键供体数
                Descriptors.NumHAcceptors(mol),            # 氢键受体数
                rdMolDescriptors.CalcTPSA(mol),            # 拓扑极性表面积
                Descriptors.NumRotatableBonds(mol),        # 可旋转键数
                Descriptors.NumAromaticRings(mol),         # 芳香环数
                Descriptors.NumAliphaticRings(mol),        # 脂肪环数
                Descriptors.RingCount(mol),                # 总环数
                Descriptors.FractionCsp3(mol),             # sp3碳比例
            ])

            # 原子计数
            features.extend([
                mol.GetNumAtoms(),                         # 原子数
                mol.GetNumHeavyAtoms(),                    # 重原子数
                mol.GetNumBonds(),                         # 键数
                len([x for x in mol.GetAtoms() if x.GetSymbol() == 'C']),  # 碳原子数
                len([x for x in mol.GetAtoms() if x.GetSymbol() == 'N']),  # 氮原子数
                len([x for x in mol.GetAtoms() if x.GetSymbol() == 'O']),  # 氧原子数
            ])

            # 连通性指数
            features.extend([
                rdMolDescriptors.BertzCT(mol),             # Bertz复杂度
                rdMolDescriptors.CalcHallKierAlpha(mol),   # Hall-Kier alpha
                rdMolDescriptors.CalcKappa1(mol),          # Kappa1形状指数
                rdMolDescriptors.CalcKappa2(mol),          # Kappa2形状指数
                rdMolDescriptors.CalcKappa3(mol),          # Kappa3形状指数
            ])

            # 电子特征
            features.extend([
                rdMolDescriptors.CalcNumHBA(mol),          # 氢键受体数(另一种计算)
                rdMolDescriptors.CalcNumHBD(mol),          # 氢键供体数(另一种计算)
                rdMolDescriptors.CalcFractionCsp3(mol),    # sp3碳比例
                rdMolDescriptors.CalcNumRings(mol),        # 环数
                rdMolDescriptors.CalcNumAromaticRings(mol), # 芳香环数
            ])

            # Lipinski参数
            features.extend([
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
            ])

            # Crippen参数
            logp, mr = Crippen.CrippenLogPAndMR(mol)
            features.extend([logp, mr])

            # 分子指纹相关
            features.extend([
                rdMolDescriptors.CalcExactMolWt(mol),      # 精确分子量
                rdMolDescriptors.CalcLabuteASA(mol),       # Labute表面积
                rdMolDescriptors.CalcPBF(mol),             # 平面键分数
                rdMolDescriptors.CalcPMI1(mol),            # 主惯性矩1
                rdMolDescriptors.CalcPMI2(mol),            # 主惯性矩2
                rdMolDescriptors.CalcPMI3(mol),            # 主惯性矩3
                rdMolDescriptors.CalcNPR1(mol),            # 归一化主惯性矩比1
                rdMolDescriptors.CalcNPR2(mol),            # 归一化主惯性矩比2
                rdMolDescriptors.CalcRadiusOfGyration(mol), # 回转半径
                rdMolDescriptors.CalcInertialShapeFactor(mol), # 惯性形状因子
                rdMolDescriptors.CalcEccentricity(mol),     # 偏心率
                rdMolDescriptors.CalcAsphericity(mol),      # 非球面性
                rdMolDescriptors.CalcSpherocityIndex(mol),  # 球形指数
            ])

            # 补齐到50维
            while len(features) < 50:
                features.append(0.0)

            return np.array(features[:50], dtype=np.float32)

        except Exception as e:
            print(f"RDKit特征提取错误: {e}")
            return np.zeros(50, dtype=np.float32)

    def extract_graph_structure_features(self, data):
        """提取图结构特征"""
        features = []

        # 节点度数统计
        edge_index = data.edge_index
        node_degrees = degree(edge_index[0], num_nodes=data.x.shape[0])

        features.extend([
            float(torch.mean(node_degrees)),      # 平均度数
            float(torch.std(node_degrees)),       # 度数标准差
            float(torch.max(node_degrees)),       # 最大度数
            float(torch.min(node_degrees)),       # 最小度数
        ])

        # 边数与节点数比例
        num_nodes = data.x.shape[0]
        num_edges = edge_index.shape[1] // 2  # 无向图，边数除以2
        features.append(num_edges / num_nodes if num_nodes > 0 else 0)

        # 图的稀疏性
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        sparsity = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        features.append(sparsity)

        # 节点特征统计
        node_features = data.x
        features.extend([
            float(torch.mean(node_features)),     # 节点特征平均值
            float(torch.std(node_features)),      # 节点特征标准差
            float(torch.sum(node_features)),      # 节点特征总和
        ])

        # 原子类型分布 (假设前5维是原子类型的one-hot编码)
        if node_features.shape[1] >= 5:
            atom_counts = torch.sum(node_features[:, :5], dim=0)
            features.extend([float(count) for count in atom_counts])

        return np.array(features, dtype=np.float32)

    def extract_all_features(self, data, smiles=None):
        """提取所有特征"""
        features = []

        # 基础节点特征
        node_features = data.x
        graph_features = torch.cat([
            global_mean_pool(node_features, torch.zeros(node_features.shape[0], dtype=torch.long)),
            global_max_pool(node_features, torch.zeros(node_features.shape[0], dtype=torch.long)),
            global_add_pool(node_features, torch.zeros(node_features.shape[0], dtype=torch.long))
        ], dim=1)
        features.extend(graph_features.squeeze().numpy())

        # 拓扑特征
        topo_features = self.extract_topological_features(data)
        features.extend(topo_features)

        # 3D几何特征
        geom_features = self.extract_3d_geometric_features(data)
        features.extend(geom_features)

        # RDKit特征
        rdkit_features = self.extract_rdkit_features(smiles)
        features.extend(rdkit_features)

        # 图结构特征
        graph_struct_features = self.extract_graph_structure_features(data)
        features.extend(graph_struct_features)

        return np.array(features, dtype=np.float32)


class AdvancedFeatureGNN(nn.Module):
    """基于高级特征的GNN模型"""

    def __init__(self, num_node_features, num_graph_features, hidden_dim=128, num_layers=4, dropout=0.2):
        super(AdvancedFeatureGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 节点特征处理分支
        self.node_convs = nn.ModuleList()
        self.node_bns = nn.ModuleList()

        self.node_convs.append(GCNConv(num_node_features, hidden_dim))
        self.node_bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.node_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.node_bns.append(nn.BatchNorm1d(hidden_dim))

        # 图级特征处理分支
        self.graph_mlp = nn.Sequential(
            nn.Linear(num_graph_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 特征融合层
        fusion_dim = hidden_dim * 4  # 3种池化 + 图级特征
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch, graph_features):
        # 节点特征处理
        for i in range(self.num_layers):
            x = self.node_convs[i](x, edge_index)
            x = self.node_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x3 = global_add_pool(x, batch)

        # 图级特征处理
        graph_feat = self.graph_mlp(graph_features)

        # 特征融合
        fused_features = torch.cat([x1, x2, x3, graph_feat], dim=1)

        return self.fusion_layers(fused_features)


def create_advanced_dataset(original_dataset, target_index, feature_extractor):
    """创建包含高级特征的数据集"""
    print("提取高级特征...")

    new_data_list = []
    graph_features_list = []

    # 计算目标值统计信息
    targets = []
    for data in original_dataset:
        targets.append(data.y[0, target_index].item())

    target_mean = np.mean(targets)
    target_std = np.std(targets)

    print(f"目标属性统计信息:")
    print(f"  均值: {target_mean:.4f}")
    print(f"  标准差: {target_std:.4f}")

    for i, data in enumerate(original_dataset):
        if i % 500 == 0:
            print(f"处理样本: {i}/{len(original_dataset)}")

        # 标准化目标值
        target_value = (data.y[0, target_index].item() - target_mean) / target_std

        # 提取高级特征
        graph_features = feature_extractor.extract_all_features(data)
        graph_features_list.append(graph_features)

        # 创建新数据对象
        new_data = Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            y=torch.tensor([target_value], dtype=torch.float),
            pos=data.pos.clone() if hasattr(data, 'pos') and data.pos is not None else None
        )
        new_data_list.append(new_data)

    # 标准化图级特征
    graph_features_array = np.array(graph_features_list)
    feature_extractor.scaler.fit(graph_features_array)
    graph_features_scaled = feature_extractor.scaler.transform(graph_features_array)

    # 特征选择 (可选)
    if graph_features_scaled.shape[1] > 100:
        print("执行特征选择...")
        targets_array = np.array(targets)
        feature_extractor.feature_selector.fit(graph_features_scaled, targets_array)
        graph_features_selected = feature_extractor.feature_selector.transform(graph_features_scaled)
        print(f"特征维度: {graph_features_array.shape[1]} -> {graph_features_selected.shape[1]}")
    else:
        graph_features_selected = graph_features_scaled

    # 添加图级特征到数据对象
    for i, data in enumerate(new_data_list):
        data.graph_features = torch.tensor(graph_features_selected[i], dtype=torch.float)

    return new_data_list, target_mean, target_std, graph_features_selected.shape[1]


def advanced_feature_collate(batch):
    """自定义批处理函数"""
    from torch_geometric.data import Batch

    # 使用PyG默认的批处理
    batched_data = Batch.from_data_list(batch)

    # 提取图级特征
    graph_features = torch.stack([data.graph_features for data in batch])

    return batched_data, graph_features


def train_advanced_model(model, train_loader, val_loader, device, num_epochs=50):
    """训练高级特征模型"""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        for batched_data, graph_features in train_loader:
            batched_data = batched_data.to(device)
            graph_features = graph_features.to(device)

            optimizer.zero_grad()
            out = model(batched_data.x, batched_data.edge_index, batched_data.batch, graph_features)
            loss = criterion(out.squeeze(), batched_data.y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batched_data, graph_features in val_loader:
                batched_data = batched_data.to(device)
                graph_features = graph_features.to(device)

                out = model(batched_data.x, batched_data.edge_index, batched_data.batch, graph_features)
                loss = criterion(out.squeeze(), batched_data.y.squeeze())
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


def main():
    """主函数"""
    print("=" * 80)
    print("高级特征GNN分子属性预测")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载QM9数据集...")
    transform = Compose([NormalizeFeatures()])
    dataset = QM9(root='/Users/xiaotingzhou/Downloads/GNN/Dataset', transform=transform)
    dataset = dataset[3000:5000]  # 使用2000个样本进行测试

    target_index = 4  # HOMO-LUMO gap

    # 创建特征提取器
    feature_extractor = AdvancedFeatureExtractor()

    # 创建高级特征数据集
    processed_data, target_mean, target_std, num_graph_features = create_advanced_dataset(
        dataset, target_index, feature_extractor
    )

    print(f"图级特征维度: {num_graph_features}")

    # 数据划分
    num_samples = len(processed_data)
    num_train = int(0.7 * num_samples)
    num_val = int(0.15 * num_samples)

    train_data = processed_data[:num_train]
    val_data = processed_data[num_train:num_train + num_val]
    test_data = processed_data[num_train + num_val:]

    print(f"数据划分: 训练={len(train_data)}, 验证={len(val_data)}, 测试={len(test_data)}")

    # 数据加载器
    from functools import partial
    collate_fn = partial(advanced_feature_collate)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 创建模型
    num_node_features = processed_data[0].x.shape[1]
    model = AdvancedFeatureGNN(
        num_node_features=num_node_features,
        num_graph_features=num_graph_features,
        hidden_dim=128,
        num_layers=4
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("开始训练...")
    start_time = time.time()

    train_losses, val_losses, best_val_loss = train_advanced_model(
        model, train_loader, val_loader, device, num_epochs=50
    )

    training_time = time.time() - start_time
    print(f"训练完成! 用时: {training_time:.2f}秒")

    # 评估模型
    print("评估模型...")
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batched_data, graph_features in test_loader:
            batched_data = batched_data.to(device)
            graph_features = graph_features.to(device)

            out = model(batched_data.x, batched_data.edge_index, batched_data.batch, graph_features)

            # 反标准化
            pred = out.squeeze().cpu().numpy() * target_std + target_mean
            true = batched_data.y.squeeze().cpu().numpy() * target_std + target_mean

            predictions.extend(pred.tolist() if pred.ndim > 0 else [pred.item()])
            true_values.extend(true.tolist() if true.ndim > 0 else [true.item()])

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # 计算指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)

    print(f"\n测试结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")

    # 可视化结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练历史')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(true_values, predictions, alpha=0.6)
    min_val = min(np.min(true_values), np.min(predictions))
    max_val = max(np.max(true_values), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测结果')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('/Users/xiaotingzhou/Downloads/GNN/advanced_feature_results.png', dpi=300)
    plt.show()

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'training_time': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }


if __name__ == "__main__":
    results = main()