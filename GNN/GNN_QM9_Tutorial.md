# 基于图神经网络(GNN)的QM9分子属性预测教程

## 📋 目录

1. [概述](#概述)
2. [QM9数据集介绍](#qm9数据集介绍)
3. [任务定义](#任务定义)
4. [图神经网络基础](#图神经网络基础)
5. [数据预处理流程](#数据预处理流程)
6. [模型架构详解](#模型架构详解)
7. [训练流程分析](#训练流程分析)
8. [代码实现详解](#代码实现详解)
9. [结果分析与可视化](#结果分析与可视化)
10. [扩展与优化](#扩展与优化)

---

## 📚 概述

本教程详细介绍如何使用图神经网络(Graph Neural Networks, GNN)预测分子属性。我们以QM9数据集为例，构建一个端到端的分子属性预测系统，专门预测分子的HOMO-LUMO能隙。

### 🎯 学习目标

- 理解分子数据的图表示方法
- 掌握GNN在分子属性预测中的应用
- 学会处理化学数据集的预处理技巧
- 熟练使用PyTorch Geometric进行图神经网络建模

---

## 🧪 QM9数据集介绍

### 数据集概述

QM9是一个包含约134,000个小分子的大规模量子化学数据集，每个分子包含最多9个重原子(C、N、O、F，除了氢原子)。

### 数据集特征

```python
# 数据集基本信息
- 样本数量: ~134,000个小分子
- 原子类型: H(氢)、C(碳)、N(氮)、O(氧)、F(氟)
- 最大原子数: 29个原子
- 目标属性: 19种量子化学属性
```

### 19种目标属性

| 索引 | 属性名称 | 单位 | 描述 |
|------|----------|------|------|
| 0 | dipole_moment | Debye | 偶极矩 |
| 1 | isotropic_polarizability | Bohr³ | 各向同性极化率 |
| 2 | homo | Hartree | HOMO轨道能级 |
| 3 | lumo | Hartree | LUMO轨道能级 |
| **4** | **gap** | **Hartree** | **HOMO-LUMO能隙(本教程目标)** |
| 5 | electronic_spatial_extent | Bohr² | 电子空间扩展 |
| 6 | zero_point_vibrational_energy | Hartree | 零点振动能 |
| 7 | internal_energy_0K | Hartree | 0K内能 |
| 8 | internal_energy_298K | Hartree | 298K内能 |
| 9 | enthalpy_298K | Hartree | 298K焓 |
| 10 | free_energy_298K | Hartree | 298K自由能 |
| 11 | heat_capacity | cal/(mol·K) | 热容 |
| 12 | atomization_energy | Hartree | 原子化能 |
| 13 | atomization_enthalpy | Hartree | 原子化焓 |
| 14 | atomization_free_energy | Hartree | 原子化自由能 |
| 15 | rotational_constant_A | GHz | 转动常数A |
| 16 | rotational_constant_B | GHz | 转动常数B |
| 17 | rotational_constant_C | GHz | 转动常数C |
| 18 | vibrational_frequencies | cm⁻¹ | 振动频率 |

### HOMO-LUMO能隙的重要性

**HOMO-LUMO能隙**是分子中最高占用分子轨道(HOMO)和最低未占用分子轨道(LUMO)之间的能量差，它决定了：

- 🔬 **分子稳定性**: 能隙越大，分子越稳定
- ⚡ **导电性**: 能隙小的分子更容易导电
- 🌈 **光学性质**: 能隙决定分子的吸收光谱
- 💊 **药物设计**: 影响分子的反应活性

---

## 🎯 任务定义

### 问题描述

给定一个分子的结构信息，预测其HOMO-LUMO能隙值。这是一个**回归任务**。

```python
# 输入: 分子图G = (V, E)
# V: 原子集合 (节点)
# E: 化学键集合 (边)
#
# 输出: HOMO-LUMO能隙 (标量值)
```

### 挑战

1. **结构多样性**: 分子大小和形状差异很大
2. **特征复杂性**: 需要捕获原子级和分子级特征
3. **长程依赖**: 原子间的相互作用可能跨越多个键
4. **数据规模**: 需要高效处理大规模数据

---

## 🕸️ 图神经网络基础

### 为什么使用图表示分子？

分子天然具有图结构：

```
    H         H
    |         |
H—C—C—C—C—H  (丁烷分子)
|   |   |   |
H   H   H   H

节点(原子): C, H
边(化学键): C-C, C-H
```

### 图神经网络的优势

1. **排列不变性**: 原子顺序不影响预测结果
2. **局部性**: 化学键的局部相互作用
3. **可扩展性**: 处理不同大小的分子
4. **可解释性**: 可以分析重要的原子和键

### GNN的核心思想

```python
# 消息传递框架
for layer in range(num_layers):
    # 1. 消息计算
    messages = compute_messages(node_features, edge_features, edge_index)

    # 2. 消息聚合
    aggregated = aggregate_messages(messages, edge_index)

    # 3. 节点更新
    node_features = update_nodes(node_features, aggregated)

# 4. 图级预测
graph_representation = global_pooling(node_features)
prediction = mlp(graph_representation)
```

---

## 🔄 数据预处理流程

### 1. 数据下载与加载

```python
# 自动下载QM9数据集
dataset = QM9(root='/path/to/dataset', transform=transform)
```

QM9数据集的每个样本包含：
- `x`: 节点特征矩阵 [num_atoms, 11]
- `edge_index`: 边索引 [2, num_edges]
- `edge_attr`: 边特征矩阵 [num_edges, 4]
- `y`: 目标属性向量 [1, 19]
- `pos`: 原子3D坐标 [num_atoms, 3]

### 2. 节点特征详解

每个原子(节点)有11维特征：

| 维度 | 特征名称 | 描述 |
|------|----------|------|
| 0-4 | 原子类型 | H, C, N, O, F的one-hot编码 |
| 5 | 度数 | 原子的连接数 |
| 6 | 形式电荷 | 原子的形式电荷 |
| 7 | 手性 | 手性标签 |
| 8 | 杂化类型 | sp, sp2, sp3等 |
| 9 | 芳香性 | 是否为芳香原子 |
| 10 | 氢原子数 | 连接的氢原子数量 |

### 3. 边特征详解

每条边(化学键)有4维特征：

| 维度 | 特征名称 | 描述 |
|------|----------|------|
| 0-3 | 键类型 | 单键、双键、三键、芳香键的one-hot编码 |

### 4. 数据标准化

```python
def create_single_target_dataset(original_dataset, target_index):
    # 提取目标值
    targets = [data.y[0, target_index].item() for data in original_dataset]

    # 计算统计信息
    target_mean = np.mean(targets)
    target_std = np.std(targets)

    # 标准化: z = (x - μ) / σ
    for data in original_dataset:
        target_value = (data.y[0, target_index].item() - target_mean) / target_std
        data.y = torch.tensor([target_value], dtype=torch.float)
```

**标准化的重要性**:
- 加速收敛
- 数值稳定性
- 便于设置学习率

### 5. 数据集划分

```python
# 80% 训练 / 10% 验证 / 10% 测试
train_data = processed_data[:train_size]
val_data = processed_data[train_size:train_size + val_size]
test_data = processed_data[train_size + val_size:]
```

---

## 🏗️ 模型架构详解

### 整体架构

```
输入分子图
    ↓
原子特征嵌入
    ↓
多层GCN卷积
    ↓
图级池化
    ↓
全连接层
    ↓
HOMO-LUMO能隙预测
```

### 1. 原子特征嵌入层

```python
self.atom_embedding = nn.Sequential(
    nn.Linear(num_features, hidden_dim),     # 11 → 128
    nn.BatchNorm1d(hidden_dim),              # 批归一化
    nn.ReLU(),                               # ReLU激活
    nn.Dropout(dropout)                      # Dropout正则化
)
```

**作用**: 将原始原子特征映射到高维空间，便于GNN处理。

### 2. 多层图卷积网络

```python
# 4层GCN卷积
self.convs = nn.ModuleList()
self.batch_norms = nn.ModuleList()

for i in range(num_layers):
    self.convs.append(GCNConv(hidden_dim, hidden_dim))
    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
```

**GCN卷积公式**:
```
h_i^(l+1) = σ(W^(l) · ∑_{j∈N(i)∪{i}} h_j^(l) / √(d_i · d_j))
```

其中：
- `h_i^(l)`: 第l层中节点i的特征
- `N(i)`: 节点i的邻居
- `W^(l)`: 第l层的权重矩阵
- `d_i`: 节点i的度数

### 3. 图级池化层

```python
# 三种池化方式的组合
x1 = global_mean_pool(x, batch)    # 平均池化
x2 = global_max_pool(x, batch)     # 最大池化
x3 = global_add_pool(x, batch)     # 求和池化

# 拼接不同池化结果
x = torch.cat([x1, x2, x3], dim=1)  # [batch_size, hidden_dim*3]
```

**池化的作用**:
- **平均池化**: 捕获分子的整体平均特性
- **最大池化**: 关注最显著的特征
- **求和池化**: 保持特征的总量信息

### 4. 全连接预测层

```python
self.fc_layers = nn.Sequential(
    nn.Linear(hidden_dim * 3, hidden_dim),      # 384 → 128
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),     # 128 → 64
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, num_targets)     # 64 → 1
)
```

**设计原则**:
- 逐层降维
- ReLU激活保持非线性
- Dropout防止过拟合

---

## 🎓 训练流程分析

### 1. 损失函数

```python
criterion = nn.MSELoss()  # 均方误差损失
loss = criterion(predictions, targets)
```

**MSE适用于回归任务**:
```
MSE = (1/n) × ∑(y_pred - y_true)²
```

### 2. 优化器配置

```python
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

**Adam优化器优势**:
- 自适应学习率
- 处理稀疏梯度
- 快速收敛

### 3. 学习率调度

```python
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                             factor=0.8, patience=10)
```

**作用**: 验证损失停止下降时，自动降低学习率。

### 4. 早停机制

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    torch.save(model.state_dict(), 'best_model.pth')
else:
    patience_counter += 1

if patience_counter >= patience:
    print("早停触发！")
    break
```

**防止过拟合**: 验证损失不再改善时停止训练。

### 5. 训练循环详解

```python
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(predictions.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_loss = validate(model, val_loader, criterion, device)

    # 学习率调度
    scheduler.step(val_loss)
```

---

## 💻 代码实现详解

### 核心类结构

#### 1. GNNModel类

```python
class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_layers=3,
                 num_targets=1, dropout=0.2):
        super(GNNModel, self).__init__()

        # 层定义
        self.atom_embedding = ...
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.fc_layers = ...

    def forward(self, x, edge_index, batch):
        # 前向传播逻辑
        pass
```

#### 2. create_single_target_dataset函数

```python
def create_single_target_dataset(original_dataset, target_index):
    """
    将多目标数据集转换为单目标数据集

    Args:
        original_dataset: 原始19维目标数据集
        target_index: 目标属性索引(0-18)

    Returns:
        new_data_list: 单目标数据列表
        target_mean: 目标均值(用于反标准化)
        target_std: 目标标准差(用于反标准化)
    """
```

**关键创新点**:
- 避免维度不匹配问题
- 正确处理标准化
- 保持数据结构完整性

### 前向传播流程

```python
def forward(self, x, edge_index, batch):
    # 步骤1: 原子特征嵌入
    x = self.atom_embedding(x)  # [num_atoms, hidden_dim]

    # 步骤2: 多层图卷积
    for i in range(self.num_layers):
        x = self.convs[i](x, edge_index)
        x = self.batch_norms[i](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

    # 步骤3: 图级池化
    x1 = global_mean_pool(x, batch)
    x2 = global_max_pool(x, batch)
    x3 = global_add_pool(x, batch)
    x = torch.cat([x1, x2, x3], dim=1)  # [batch_size, hidden_dim*3]

    # 步骤4: 最终预测
    x = self.fc_layers(x)  # [batch_size, 1]

    return x
```

### 批处理机制

```python
# batch参数的作用
batch = [0, 0, 0, 1, 1, 1, 1, 2, 2]
#        |--mol0--| |--mol1--| |-mol2-|

# global_mean_pool会根据batch自动分组计算
# 分子0: mean(node_features[0:3])
# 分子1: mean(node_features[3:7])
# 分子2: mean(node_features[7:9])
```

---

## 📊 结果分析与可视化

### 1. 训练历史可视化

```python
def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练过程中的损失变化')
    plt.legend()
    plt.grid(True)
```

**分析要点**:
- 训练损失持续下降 ✅
- 验证损失下降后趋于稳定 ✅
- 两者差距不大(无过拟合) ✅

### 2. 预测结果分析

```python
def plot_predictions(true_values, predictions, target_name):
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predictions, alpha=0.5, s=10)

    # 理想预测线 (y=x)
    min_val = min(np.min(true_values), np.min(predictions))
    max_val = max(np.max(true_values), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--',
             linewidth=2, label='理想预测')

    plt.xlabel(f'真实值 ({target_name})')
    plt.ylabel(f'预测值 ({target_name})')
    plt.title(f'{target_name} - 预测值 vs 真实值')
```

### 3. 评估指标

```python
# 平均绝对误差
mae = mean_absolute_error(true_values, predictions)

# 均方误差
mse = mean_squared_error(true_values, predictions)

# 均方根误差
rmse = np.sqrt(mse)

# R²决定系数
from sklearn.metrics import r2_score
r2 = r2_score(true_values, predictions)
```

**指标解释**:
- **MAE**: 平均预测偏差，单位与目标相同
- **RMSE**: 对大误差更敏感，单位与目标相同
- **R²**: 模型解释方差的比例，越接近1越好

### 4. 典型结果示例

```
测试结果 (gap):
  平均绝对误差 (MAE): 0.1234
  均方误差 (MSE): 0.0456
  均方根误差 (RMSE): 0.2136
  R² 决定系数: 0.8956

预测样例:
  样本 1: 真实值=5.6789, 预测值=5.7123, 误差=0.0334
  样本 2: 真实值=7.8901, 预测值=7.8456, 误差=0.0445
  ...
```

---

## 🚀 扩展与优化

### 1. 模型架构优化

#### 使用注意力机制

```python
from torch_geometric.nn import GATConv

# 图注意力网络
self.convs = nn.ModuleList([
    GATConv(hidden_dim, hidden_dim, heads=4, dropout=0.2)
    for _ in range(num_layers)
])
```

#### 残差连接

```python
def forward(self, x, edge_index, batch):
    x = self.atom_embedding(x)

    for conv, bn in zip(self.convs, self.batch_norms):
        # 残差连接
        residual = x
        x = conv(x, edge_index)
        x = bn(x)
        x = F.relu(x + residual)  # 加上残差
```

#### 多尺度特征融合

```python
# 收集不同层的特征
layer_outputs = [x]
for conv in self.convs:
    x = conv(x, edge_index)
    layer_outputs.append(x)

# 拼接多层特征
x = torch.cat(layer_outputs, dim=1)
```

### 2. 数据增强策略

#### 分子构象增强

```python
# 添加随机噪声到原子坐标
def augment_molecule(data, noise_level=0.1):
    if hasattr(data, 'pos') and data.pos is not None:
        noise = torch.randn_like(data.pos) * noise_level
        data.pos = data.pos + noise
    return data
```

#### 化学等效性增强

```python
# 利用分子对称性生成等效结构
def symmetric_augmentation(mol_data):
    # 应用分子对称操作
    # 旋转、镜像等变换
    pass
```

### 3. 多任务学习

```python
class MultiTaskGNN(nn.Module):
    def __init__(self, num_tasks=19):
        super().__init__()
        # 共享的GNN骨干网络
        self.backbone = GNNModel(...)

        # 任务特定的头部
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x, edge_index, batch):
        # 共享特征提取
        features = self.backbone.get_features(x, edge_index, batch)

        # 多任务预测
        predictions = [head(features) for head in self.task_heads]
        return torch.cat(predictions, dim=1)
```

### 4. 模型集成

```python
class EnsembleGNN(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, edge_index, batch):
        predictions = []
        for model in self.models:
            pred = model(x, edge_index, batch)
            predictions.append(pred)

        # 平均集成
        return torch.mean(torch.stack(predictions), dim=0)
```

### 5. 超参数优化

```python
import optuna

def objective(trial):
    # 搜索超参数
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    # 训练模型并返回验证误差
    model = GNNModel(hidden_dim=hidden_dim, num_layers=num_layers,
                     dropout=dropout)
    val_error = train_and_validate(model, learning_rate)
    return val_error

# 运行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

---

## 🔬 高级技巧

### 1. 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. 学习率预热

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0

    def step(self):
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * (self.step_count / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.step_count += 1
```

### 3. 模型解释性

```python
# 使用GNNExplainer分析重要特征
from torch_geometric.explain import GNNExplainer

explainer = GNNExplainer(model, epochs=100)
explanation = explainer(x, edge_index, batch_index=0)

# 可视化重要的原子和键
important_atoms = explanation.node_mask
important_bonds = explanation.edge_mask
```

### 4. 不确定性量化

```python
# Monte Carlo Dropout估计预测不确定性
def predict_with_uncertainty(model, data, n_samples=100):
    model.train()  # 启用dropout
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(data.x, data.edge_index, data.batch)
            predictions.append(pred.cpu().numpy())

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)

    return mean_pred, uncertainty
```

---

## 📝 总结

### 关键学习要点

1. **图表示的重要性**: 分子的图结构天然适合GNN处理
2. **数据预处理**: 标准化、单目标转换是成功的关键
3. **模型设计**: 嵌入→卷积→池化→预测的经典流程
4. **训练技巧**: 早停、学习率调度、正则化防过拟合
5. **评估方法**: 多种指标综合评估模型性能

### 实际应用场景

- 🏥 **药物发现**: 预测化合物的ADMET属性
- 🔋 **材料科学**: 设计新型电池材料
- 🧪 **催化剂设计**: 预测催化活性
- 🌱 **农药开发**: 预测生物活性和毒性

### 进一步学习方向

1. **高级GNN架构**: Transformer、Graph Transformer
2. **化学信息学**: RDKit、分子描述符
3. **量子化学**: DFT计算、电子结构理论
4. **深度学习**: 注意力机制、自监督学习

---

## 📚 参考资料

### 论文推荐

1. **Quantum chemistry structures and properties of 134 kilo molecules** - QM9数据集原始论文
2. **Neural Message Passing for Quantum Chemistry** - MPNN在分子预测中的应用
3. **Graph Attention Networks** - 注意力机制在图神经网络中的应用

### 开源项目

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/): 图神经网络库
- [RDKit](https://www.rdkit.org/): 化学信息学工具包
- [DeepChem](https://deepchem.io/): 深度学习化学工具包

### 在线资源

- [Graph Neural Networks Course](https://web.stanford.edu/class/cs224w/)
- [Molecular Machine Learning](https://dmol.pub/)
- [Chemical Space Blog](https://www.chemicalspace.com/blog)

---

*本教程旨在提供GNN分子属性预测的全面指南。如有问题或建议，欢迎交流讨论！* 🚀