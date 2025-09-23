# DeepChem 教学文档

## 1. 项目介绍

### 项目背景和发展历史
DeepChem 是一个面向药物发现、材料科学和生物学应用的深度学习开源平台，最初由斯坦福大学的研究团队于2016年创建。该项目旨在降低将机器学习应用于化学和生物学研究的门槛，提供了完整的从数据处理到模型训练的工具链。DeepChem已被全球数百个研究机构和制药公司采用。

### 主要目标和应用领域
- **主要目标**：
  - 民主化化学信息学中的深度学习应用
  - 提供标准化的分子机器学习流程
  - 构建可重现的科学研究平台
  - 促进化学和生物学的AI创新

- **应用领域**：
  - **药物发现**：分子毒性预测、药物-靶点相互作用、ADMET性质预测
  - **材料科学**：材料性质预测、催化剂设计、聚合物性能预测
  - **生物学**：蛋白质折叠预测、基因表达分析、生物标志物发现
  - **环境科学**：污染物检测、环境毒性评估

### 官方链接和重要资源
- **官方网站**：https://deepchem.io
- **GitHub仓库**：https://github.com/deepchem/deepchem
- **文档**：https://deepchem.readthedocs.io
- **教程**：https://github.com/deepchem/deepchem/tree/master/examples/tutorials
- **论文**：Deep Learning for the Life Sciences (O'Reilly, 2019)

## 2. 核心概念

### 项目涉及的基础理论

#### 分子表示学习
```
分子 → 特征化 → 数值向量 → 机器学习模型 → 性质预测
```

**主要分子表示方法**：
- **SMILES**：分子的字符串表示
- **分子指纹**：基于子结构的二进制向量
- **分子描述符**：计算得出的物理化学性质
- **图表示**：原子作为节点，化学键作为边
- **3D构象**：分子的三维空间结构

#### 深度学习在化学中的应用
```
传统方法: 人工特征工程 → 机器学习算法
深度学习: 原始数据 → 自动特征学习 → 端到端预测
```

**核心深度学习架构**：
- **多层感知机（MLP）**：处理分子指纹等固定长度特征
- **卷积神经网络（CNN）**：处理分子图像或网格化表示
- **循环神经网络（RNN/LSTM）**：处理SMILES序列
- **图神经网络（GNN）**：直接处理分子图结构
- **Transformer**：处理分子序列，捕获长距离依赖

#### 化学信息学基础
```
分子 → 结构分析 → 性质关系 → 预测模型 → 新分子设计
```

### 初学者需要掌握的相关知识点
1. **化学基础**：原子、化学键、分子结构、化学反应
2. **Python编程**：NumPy、Pandas、scikit-learn基础
3. **深度学习基础**：神经网络原理、反向传播、优化算法
4. **机器学习概念**：监督学习、交叉验证、模型评估
5. **化学信息学**：SMILES表示法、分子描述符概念

## 3. 技术框架

### 使用的算法和模型

#### 经典机器学习模型
- **随机森林（Random Forest）**
- **支持向量机（SVM）**
- **梯度提升树（XGBoost, LightGBM）**

#### 深度学习模型
- **图卷积网络（Graph Convolution）**
- **消息传递神经网络（MPNN）**
- **注意力机制模型（AttentiveFP）**
- **Transformer架构（ChemBERTa）**

#### DeepChem特有模型
```
MultitaskClassifier: 多任务分类器
    ↓
GraphConvModel: 图卷积模型
    ↓
WeaveModel: Weave架构（处理分子图）
    ↓
TextCNN: 处理SMILES文本的CNN
```

### 数据结构和数据格式要求

#### 输入数据格式
```python
# CSV格式示例
# compound_id,smiles,target_1,target_2,target_3
# mol_001,CCO,0.5,1.2,0
# mol_002,CC(C)O,0.8,0.9,1
```

#### 支持的分子表示
- **SMILES字符串**：'CCO', 'CC(=O)O'
- **SDF文件**：包含3D结构信息
- **分子指纹**：二进制向量或计数向量
- **分子描述符**：连续值特征向量

### 编程语言和依赖库
- **核心语言**：Python 3.7+
- **深度学习框架**：TensorFlow 2.x, PyTorch
- **科学计算**：NumPy, SciPy, Pandas
- **化学信息学**：RDKit, OpenEye (可选)
- **可视化**：Matplotlib, Seaborn
- **其他**：scikit-learn, XGBoost

## 4. 核心模块/组件

### 数据加载与预处理

```python
import deepchem as dc
import pandas as pd
import numpy as np

# 1. 加载CSV数据
def load_dataset(csv_file, smiles_col='smiles', target_cols=['target']):
    """加载和预处理分子数据集"""
    df = pd.read_csv(csv_file)
    
    # 创建DeepChem数据集
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(
        tasks=target_cols,
        smiles_field=smiles_col,
        featurizer=featurizer
    )
    
    dataset = loader.featurize(csv_file)
    return dataset

# 2. 数据集分割
def split_dataset(dataset, split_type='scaffold'):
    """分割数据集为训练/验证/测试集"""
    if split_type == 'scaffold':
        splitter = dc.splits.ScaffoldSplitter()
    elif split_type == 'random':
        splitter = dc.splits.RandomSplitter()
    else:
        splitter = dc.splits.ButinaSplitter()
    
    train, valid, test = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1
    )
    
    return train, valid, test

# 3. 数据标准化
def normalize_dataset(train, valid, test):
    """标准化数据集"""
    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train
        )
    ]
    
    for transformer in transformers:
        train = transformer.transform(train)
        valid = transformer.transform(valid)
        test = transformer.transform(test)
    
    return train, valid, test, transformers

# 完整的数据预处理流程
def preprocess_molecular_data(csv_file):
    """完整的分子数据预处理流程"""
    print("加载数据集...")
    dataset = load_dataset(csv_file)
    
    print("分割数据集...")
    train, valid, test = split_dataset(dataset)
    
    print("标准化数据...")
    train, valid, test, transformers = normalize_dataset(train, valid, test)
    
    print(f"训练集大小: {len(train)}")
    print(f"验证集大小: {len(valid)}")
    print(f"测试集大小: {len(test)}")
    
    return train, valid, test, transformers
```

### 模型训练

```python
# 1. 多任务神经网络模型
def create_multitask_model(train_dataset, model_dir='./model_checkpoint'):
    """创建多任务分类/回归模型"""
    n_tasks = len(train_dataset.get_task_names())
    n_features = train_dataset.get_data_shape()[0]
    
    model = dc.models.MultitaskClassifier(
        n_tasks=n_tasks,
        n_features=n_features,
        layer_sizes=[1000, 500, 100],
        dropouts=[0.25, 0.25, 0.25],
        learning_rate=0.001,
        model_dir=model_dir
    )
    
    return model

# 2. 图神经网络模型
def create_graph_conv_model(train_dataset, model_dir='./graph_model'):
    """创建图卷积模型"""
    n_tasks = len(train_dataset.get_task_names())
    
    model = dc.models.GraphConvModel(
        n_tasks=n_tasks,
        graph_conv_layers=[64, 64],
        dense_layer_size=128,
        dropout=0.25,
        model_dir=model_dir
    )
    
    return model

# 3. 训练函数
def train_model(model, train_dataset, valid_dataset, nb_epoch=50):
    """训练模型"""
    # 定义评估指标
    train_task_type = train_dataset.get_task_names()[0]
    if 'classification' in str(type(model)).lower():
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
    else:
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(nb_epoch):
        # 训练一个epoch
        model.fit(train_dataset, nb_epoch=1)
        
        # 评估
        if epoch % 10 == 0:
            train_score = model.evaluate(train_dataset, [metric])
            valid_score = model.evaluate(valid_dataset, [metric])
            
            print(f"Epoch {epoch}:")
            print(f"  训练集评分: {train_score}")
            print(f"  验证集评分: {valid_score}")
    
    return model

# 4. 高级训练设置
def train_with_callbacks(model, train_dataset, valid_dataset):
    """使用回调函数训练模型"""
    # 早停回调
    early_stopping = dc.models.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 学习率调度
    lr_scheduler = dc.models.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    )
    
    # 训练
    model.fit(
        train_dataset,
        nb_epoch=100,
        validation_data=valid_dataset,
        callbacks=[early_stopping, lr_scheduler]
    )
    
    return model
```

### 推理/预测

```python
# 1. 基本预测
def make_predictions(model, test_dataset):
    """进行预测"""
    predictions = model.predict(test_dataset)
    
    # 获取真实标签
    y_true = test_dataset.y
    
    return predictions, y_true

# 2. 批量预测新分子
def predict_new_molecules(model, smiles_list, featurizer):
    """预测新分子的性质"""
    # 特征化新分子
    features = featurizer.featurize(smiles_list)
    
    # 创建临时数据集
    temp_dataset = dc.data.NumpyDataset(X=features)
    
    # 预测
    predictions = model.predict(temp_dataset)
    
    results = []
    for i, smiles in enumerate(smiles_list):
        result = {
            'smiles': smiles,
            'predictions': predictions[i].tolist()
        }
        results.append(result)
    
    return results

# 3. 不确定性量化
def predict_with_uncertainty(model, dataset, n_predictions=10):
    """使用蒙特卡洛dropout进行不确定性估计"""
    predictions_list = []
    
    # 启用训练模式以保持dropout
    for _ in range(n_predictions):
        preds = model.predict(dataset, training=True)
        predictions_list.append(preds)
    
    # 计算统计量
    predictions = np.array(predictions_list)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred

# 4. 模型解释性分析
def analyze_feature_importance(model, dataset, feature_names=None):
    """分析特征重要性"""
    try:
        # 尝试获取特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # 使用置换重要性
            from sklearn.inspection import permutation_importance
            
            # 这需要将DeepChem模型包装成sklearn兼容的形式
            result = permutation_importance(
                model, dataset.X, dataset.y, 
                n_repeats=10, random_state=42
            )
            importances = result.importances_mean
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # 创建重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        print(f"特征重要性分析失败: {e}")
        return None
```

### 可视化模块

```python
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import plotly.graph_objects as go
import plotly.express as px

# 1. 分子结构可视化
def visualize_molecules(smiles_list, predictions=None, n_cols=4):
    """可视化分子结构和预测结果"""
    n_mols = len(smiles_list)
    n_rows = (n_mols - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, smiles in enumerate(smiles_list):
        row, col = i // n_cols, i % n_cols
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(200, 200))
            axes[row, col].imshow(img)
            
            title = f"Mol {i+1}"
            if predictions is not None:
                title += f"\nPred: {predictions[i]:.3f}"
            
            axes[row, col].set_title(title, fontsize=10)
        
        axes[row, col].axis('off')
    
    # 隐藏空的子图
    for i in range(n_mols, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# 2. 预测结果可视化
def plot_predictions_vs_actual(y_true, y_pred, task_names=None):
    """绘制预测值vs真实值散点图"""
    n_tasks = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if n_tasks == 1:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值')
        
        # 计算相关系数
        corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        plt.text(0.05, 0.95, f'相关系数: {corr:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.show()
    else:
        # 多任务情况
        fig, axes = plt.subplots(2, (n_tasks + 1) // 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(n_tasks):
            axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
            axes[i].plot([y_true[:, i].min(), y_true[:, i].max()], 
                        [y_true[:, i].min(), y_true[:, i].max()], 'r--')
            
            task_name = task_names[i] if task_names else f'任务 {i+1}'
            axes[i].set_title(task_name)
            axes[i].set_xlabel('真实值')
            axes[i].set_ylabel('预测值')
            
            # 计算相关系数
            corr = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
            axes[i].text(0.05, 0.95, f'R: {corr:.3f}', 
                        transform=axes[i].transAxes, fontsize=10)
        
        plt.tight_layout()
        plt.show()

# 3. 训练曲线可视化
def plot_training_curves(train_scores, valid_scores, metric_name='Score'):
    """绘制训练和验证曲线"""
    epochs = range(1, len(train_scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, 'b-', label='训练集', linewidth=2)
    plt.plot(epochs, valid_scores, 'r-', label='验证集', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title('训练和验证曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 4. 化学空间可视化
def visualize_chemical_space(smiles_list, labels=None, method='PCA'):
    """可视化化学空间分布"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # 计算分子描述符
    descriptors = []
    valid_smiles = []
    valid_labels = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol)
            ]
            descriptors.append(desc)
            valid_smiles.append(smiles)
            if labels is not None:
                valid_labels.append(labels[i])
    
    descriptors = np.array(descriptors)
    
    # 降维
    if method == 'PCA':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(descriptors)
        title = 'PCA 化学空间可视化'
    else:
        reducer = TSNE(n_components=2, random_state=42)
        coords = reducer.fit_transform(descriptors)
        title = 't-SNE 化学空间可视化'
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    if valid_labels is not None:
        scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                            c=valid_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
    
    plt.title(title)
    plt.xlabel(f'{method} 维度 1')
    plt.ylabel(f'{method} 维度 2')
    plt.show()

# 5. 交互式可视化
def interactive_molecular_visualization(df, smiles_col='smiles', 
                                      target_col='target', pred_col='prediction'):
    """创建交互式分子可视化"""
    fig = px.scatter(
        df, 
        x=target_col, 
        y=pred_col,
        hover_data=[smiles_col],
        title='交互式预测结果可视化',
        labels={
            target_col: '真实值',
            pred_col: '预测值'
        }
    )
    
    # 添加对角线
    min_val = min(df[target_col].min(), df[pred_col].min())
    max_val = max(df[target_col].max(), df[pred_col].max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='理想预测线',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.show()
```

## 5. 使用方法

### 安装指南

```bash
# 方法1: 使用conda安装（推荐）
conda create -n deepchem python=3.8
conda activate deepchem

# 安装DeepChem和依赖
conda install -c deepchem -c rdkit -c conda-forge deepchem

# 方法2: 使用pip安装
pip install deepchem[tensorflow]  # 使用TensorFlow后端
# 或者
pip install deepchem[torch]       # 使用PyTorch后端

# 安装RDKit（化学信息学库）
conda install -c rdkit rdkit

# 安装其他有用的库
pip install matplotlib seaborn plotly jupyter
```

### 快速上手示例代码

#### 完整的分子毒性预测示例
```python
import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. 数据准备
def create_sample_data():
    """创建示例分子数据"""
    # 一些简单的有机分子SMILES和对应的毒性标签
    data = {
        'smiles': [
            'CCO',          # 乙醇
            'CC(C)O',       # 异丙醇  
            'CCCCO',        # 丁醇
            'CC(=O)O',      # 醋酸
            'c1ccccc1O',    # 苯酚
            'CCCCCl',       # 氯丁烷
            'CC(=O)C',      # 丙酮
            'CCCCCC',       # 己烷
            'c1ccccc1',     # 苯
            'CCO[CH2]',     # 乙醚
        ],
        'toxicity': [0, 0, 0, 0, 1, 1, 0, 0, 1, 0]  # 0=无毒, 1=有毒
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_toxicity.csv', index=False)
    return 'sample_toxicity.csv'

# 2. 完整的训练流程
def molecular_toxicity_prediction_pipeline():
    """完整的分子毒性预测流程"""
    
    print("=== DeepChem 分子毒性预测示例 ===\n")
    
    # 创建示例数据
    csv_file = create_sample_data()
    print("1. 数据准备完成")
    
    # 数据加载和特征化
    print("2. 加载和特征化数据...")
    featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)
    loader = dc.data.CSVLoader(
        tasks=['toxicity'],
        smiles_field='smiles',
        featurizer=featurizer
    )
    dataset = loader.featurize(csv_file)
    print(f"   数据集大小: {len(dataset)}")
    print(f"   特征维度: {dataset.X.shape}")
    
    # 数据分割
    print("3. 分割数据集...")
    splitter = dc.splits.RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2
    )
    print(f"   训练集: {len(train)}, 验证集: {len(valid)}, 测试集: {len(test)}")
    
    # 模型创建和训练
    print("4. 创建和训练模型...")
    model = dc.models.MultitaskClassifier(
        n_tasks=1,
        n_features=1024,
        layer_sizes=[512, 128, 32],
        dropouts=[0.2, 0.2, 0.2],
        learning_rate=0.001
    )
    
    # 训练模型
    model.fit(train, nb_epoch=50)
    print("   模型训练完成")
    
    # 模型评估
    print("5. 模型评估...")
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    
    train_score = model.evaluate(train, [metric])
    valid_score = model.evaluate(valid, [metric])
    test_score = model.evaluate(test, [metric])
    
    print(f"   训练集 ROC-AUC: {train_score['roc_auc_score']:.3f}")
    print(f"   验证集 ROC-AUC: {valid_score['roc_auc_score']:.3f}")
    print(f"   测试集 ROC-AUC: {test_score['roc_auc_score']:.3f}")
    
    # 预测新分子
    print("6. 预测新分子...")
    new_molecules = ['CCCCCO', 'c1ccc(Cl)cc1', 'CCC(=O)O']
    predictions = predict_new_molecules(model, new_molecules, featurizer)
    
    for pred in predictions:
        toxicity = "有毒" if pred['predictions'][0] > 0.5 else "无毒"
        confidence = pred['predictions'][0]
        print(f"   {pred['smiles']} -> {toxicity} (置信度: {confidence:.3f})")
    
    return model, dataset, featurizer

# 运行完整示例
if __name__ == "__main__":
    model, dataset, featurizer = molecular_toxicity_prediction_pipeline()
```

### 小型示例实验

#### 实验1：比较不同特征化方法
```python
def compare_featurizers():
    """比较不同分子特征化方法的效果"""
    
    # 准备数据
    csv_file = create_sample_data()
    
    # 定义不同的特征化方法
    featurizers = {
        'CircularFingerprint': dc.feat.CircularFingerprint(size=1024),
        'MACCSKeysFingerprint': dc.feat.MACCSKeysFingerprint(),
        'RDKitDescriptors': dc.feat.RDKitDescriptors(),
        'CoulombMatrix': dc.feat.CoulombMatrix(max_atoms=50)
    }
    
    results = {}
    
    for name, featurizer in featurizers.items():
        print(f"\n测试特征化方法: {name}")
        
        try:
            # 加载数据
            loader = dc.data.CSVLoader(
                tasks=['toxicity'],
                smiles_field='smiles', 
                featurizer=featurizer
            )
            dataset = loader.featurize(csv_file)
            
            # 分割数据
            splitter = dc.splits.RandomSplitter()
            train, valid, test = splitter.train_valid_test_split(dataset)
            
            # 创建和训练模型
            n_features = dataset.X.shape[1]
            model = dc.models.MultitaskClassifier(
                n_tasks=1,
                n_features=n_features,
                layer_sizes=[min(512, n_features//2), 64],
                learning_rate=0.001
            )
            
            model.fit(train, nb_epoch=30)
            
            # 评估
            metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
            score = model.evaluate(test, [metric])
            
            results[name] = {
                'roc_auc': score['roc_auc_score'],
                'n_features': n_features,
                'dataset_size': len(dataset)
            }
            
            print(f"   特征数量: {n_features}")
            print(f"   ROC-AUC: {score['roc_auc_score']:.3f}")
            
        except Exception as e:
            print(f"   错误: {e}")
            results[name] = {'error': str(e)}
    
    return results

# 运行特征化比较实验
featurizer_results = compare_featurizers()
```

#### 实验2：多任务学习示例
```python
def multitask_learning_example():
    """多任务学习示例：同时预测毒性和溶解度"""
    
    # 创建多任务数据
    data = {
        'smiles': [
            'CCO', 'CC(C)O', 'CCCCO', 'CC(=O)O', 'c1ccccc1O',
            'CCCCCl', 'CC(=O)C', 'CCCCCC', 'c1ccccc1', 'CCO[CH2]'
        ],
        'toxicity': [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        'solubility': [1, 1, 0, 1, 0, 0, 1, 0, 0, 1]  # 1=可溶, 0=难溶
    }
    
    df = pd.DataFrame(data)
    df.to_csv('multitask_data.csv', index=False)
    
    # 加载多任务数据
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(
        tasks=['toxicity', 'solubility'],  # 多个任务
        smiles_field='smiles',
        featurizer=featurizer
    )
    dataset = loader.featurize('multitask_data.csv')
    
    # 分割数据
    splitter = dc.splits.RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(dataset)
    
    # 创建多任务模型
    model = dc.models.MultitaskClassifier(
        n_tasks=2,  # 两个任务
        n_features=1024,
        layer_sizes=[512, 256, 128],
        dropouts=[0.2, 0.2, 0.2],
        learning_rate=0.001
    )
    
    # 训练模型
    model.fit(train, nb_epoch=50)
    
    # 分别评估每个任务
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    test_score = model.evaluate(test, [metric])
    
    print("多任务学习结果:")
    print(f"毒性预测 ROC-AUC: {test_score['roc_auc_score']:.3f}")
    
    # 获取每个任务的单独评估
    predictions = model.predict(test)
    for task_idx, task_name in enumerate(['toxicity', 'solubility']):
        if len(test.y) > 0:
            task_auc = roc_auc_score(test.y[:, task_idx], predictions[:, task_idx])
            print(f"{task_name} ROC-AUC: {task_auc:.3f}")
    
    return model, dataset

# 运行多任务学习实验
multitask_model, multitask_dataset = multitask_learning_example()
```

#### 实验3：模型可解释性分析
```python
def interpretability_analysis():
    """模型可解释性分析示例"""
    
    # 使用之前训练的模型
    csv_file = create_sample_data()
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(
        tasks=['toxicity'],
        smiles_field='smiles',
        featurizer=featurizer
    )
    dataset = loader.featurize(csv_file)
    
    # 训练简单模型
    splitter = dc.splits.RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(dataset)
    
    model = dc.models.MultitaskClassifier(
        n_tasks=1,
        n_features=1024,
        layer_sizes=[256, 64],
        learning_rate=0.001
    )
    model.fit(train, nb_epoch=30)
    
    # 分析特定分子的预测
    test_smiles = ['c1ccccc1O', 'CCO', 'CCCCCl']  # 苯酚, 乙醇, 氯丁烷
    
    print("分子预测分析:")
    for smiles in test_smiles:
        # 特征化单个分子
        features = featurizer.featurize([smiles])
        temp_dataset = dc.data.NumpyDataset(X=features)
        
        # 预测
        pred = model.predict(temp_dataset)[0][0]
        toxicity = "有毒" if pred > 0.5 else "无毒"
        
        # 分子信息
        mol = Chem.MolFromSmiles(smiles)
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        print(f"\n分子: {smiles}")
        print(f"  预测: {toxicity} (概率: {pred:.3f})")
        print(f"  分子量: {mol_weight:.2f}")
        print(f"  脂水分配系数: {logp:.2f}")
        
        # 可视化分子结构
        img = Draw.MolToImage(mol, size=(200, 200))
        # 在实际应用中，这里可以保存或显示图像
        
    return model, dataset

# 运行可解释性分析
interpretability_model, _ = interpretability_analysis()
```

## 6. 实践建议

### 如何使用本项目进行改造或扩展

#### 适配台湾化学品数据库
```python
def adapt_taiwan_chemical_database():
    """适配台湾化学品数据库示例"""
    
    # 模拟台湾环保署化学品资料
    taiwan_chemicals = {
        'smiles': [
            'CCO',                    # 乙醇 - 常见溶剂
            'CC(C)O',                # 异丙醇 - 消毒用品
            'C1=CC=CC=C1',           # 苯 - 工业溶剂
            'CCl4',                  # 四氯化碳 - 受管制物质
            'C1=CC=C(C=C1)O',        # 苯酚 - 化工原料
            'CC(=O)OC1=CC=CC=C1C(=O)O', # 阿司匹林 - 药物
            'CCCCCCCCC(=O)O',        # 壬酸 - 食品添加剂
        ],
        'cas_number': [
            '64-17-5', '67-63-0', '71-43-2', '56-23-5', 
            '108-95-2', '50-78-2', '112-05-0'
        ],
        'toxicity_class': [1, 1, 4, 4, 3, 1, 1],  # 毒性分级 1-4
        'environmental_impact': [1, 1, 3, 4, 3, 1, 2],  # 环境影响 1-4
        'chinese_name': [
            '乙醇', '異丙醇', '苯', '四氯化碳', 
            '苯酚', '阿司匹靈', '壬酸'
        ]
    }
    
    df = pd.DataFrame(taiwan_chemicals)
    
    # 创建多任务学习数据集
    featurizer = dc.feat.CircularFingerprint(size=2048)
    loader = dc.data.CSVLoader(
        tasks=['toxicity_class', 'environmental_impact'],
        smiles_field='smiles',
        featurizer=featurizer
    )
    
    # 保存为临时CSV
    df.to_csv('taiwan_chemicals.csv', index=False)
    dataset = loader.featurize('taiwan_chemicals.csv')
    
    # 转换为分类问题（减1使类别从0开始）
    dataset.y = dataset.y - 1
    
    print("台湾化学品数据集信息:")
    print(f"化学品数量: {len(dataset)}")
    print(f"任务数量: {dataset.y.shape[1]}")
    print(f"特征维度: {dataset.X.shape[1]}")
    
    return dataset, df

# 使用台湾化学品数据训练模型
def train_taiwan_chemical_model():
    """训练台湾化学品分类模型"""
    
    dataset, df = adapt_taiwan_chemical_database()
    
    # 由于数据较少，使用简单的训练验证分割
    splitter = dc.splits.RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset, frac_train=0.7, frac_valid=0.15, frac_test=0.15
    )
    
    # 创建多分类模型
    model = dc.models.MultitaskClassifier(
        n_tasks=2,
        n_features=2048,
        n_classes=4,  # 4个分类等级
        layer_sizes=[512, 128],
        dropouts=[0.3, 0.3],
        learning_rate=0.001
    )
    
    # 训练模型
    model.fit(train, nb_epoch=100)
    
    # 评估模型
    metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    train_score = model.evaluate(train, [metric])
    valid_score = model.evaluate(valid, [metric])
    
    print(f"训练集准确率: {train_score['accuracy_score']:.3f}")
    print(f"验证集准确率: {valid_score['accuracy_score']:.3f}")
    
    return model, dataset, df

taiwan_model, taiwan_dataset, taiwan_df = train_taiwan_chemical_model()
```

#### 扩展到德国REACH法规数据
```python
def extend_to_reach_regulation():
    """扩展到欧盟REACH法规化学品评估"""
    
    # 模拟REACH法规相关的化学品数据
    reach_data = {
        'smiles': [
            'CCCCCCCCCCCCCCCCCCCCCCCC(=O)O',  # 长链脂肪酸
            'C1=CC=C2C(=C1)C=CC=C2',          # 萘 - PAH化合物
            'C1=CC=C(C=C1)C2=CC=CC=C2',       # 联苯
            'ClC1=CC=C(C=C1)Cl',              # 对二氯苯
            'BrC1=CC=C(C=C1)Br',              # 对二溴苯
        ],
        'reach_registration': [1, 1, 1, 0, 0],  # 是否需要REACH注册
        'pbt_assessment': [0, 1, 1, 1, 1],       # PBT物质评估
        'svhc_candidate': [0, 1, 0, 1, 1],       # SVHC候选物质
        'annual_tonnage': [1000, 500, 200, 50, 25]  # 年产量(吨)
    }
    
    df = pd.DataFrame(reach_data)
    
    # 多任务学习：预测REACH相关属性
    featurizer = dc.feat.RDKitDescriptors()
    loader = dc.data.CSVLoader(
        tasks=['reach_registration', 'pbt_assessment', 'svhc_candidate'],
        smiles_field='smiles',
        featurizer=featurizer
    )
    
    df.to_csv('reach_chemicals.csv', index=False)
    dataset = loader.featurize('reach_chemicals.csv')
    
    print("REACH法规数据集信息:")
    print(f"化学品数量: {len(dataset)}")
    print(f"描述符数量: {dataset.X.shape[1]}")
    
    # 训练REACH预测模型
    model = dc.models.MultitaskClassifier(
        n_tasks=3,
        n_features=dataset.X.shape[1],
        layer_sizes=[256, 64],
        learning_rate=0.001
    )
    
    # 由于数据少，使用全部数据训练
    model.fit(dataset, nb_epoch=50)
    
    return model, dataset, df

reach_model, reach_dataset, reach_df = extend_to_reach_regulation()
```

### 可以做的练手项目或可展示的成果

#### 项目1：分子性质预测Web应用
```python
def create_molecular_property_webapp():
    """创建分子性质预测Web应用的后端逻辑"""
    
    class MolecularPropertyPredictor:
        def __init__(self, model_path, featurizer_type='circular'):
            self.model = dc.utils.load_from_disk(model_path)
            
            if featurizer_type == 'circular':
                self.featurizer = dc.feat.CircularFingerprint(size=1024)
            elif featurizer_type == 'rdkit':
                self.featurizer = dc.feat.RDKitDescriptors()
            else:
                self.featurizer = dc.feat.MACCSKeysFingerprint()
        
        def predict_single_molecule(self, smiles):
            """预测单个分子的性质"""
            try:
                # 验证SMILES格式
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return {'error': 'Invalid SMILES format'}
                
                # 特征化
                features = self.featurizer.featurize([smiles])
                temp_dataset = dc.data.NumpyDataset(X=features)
                
                # 预测
                predictions = self.model.predict(temp_dataset)
                
                # 计算分子基本信息
                mol_info = {
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'h_donors': Descriptors.NumHDonors(mol),
                    'h_acceptors': Descriptors.NumHAcceptors(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'tpsa': Descriptors.TPSA(mol)
                }
                
                result = {
                    'smiles': smiles,
                    'predictions': predictions[0].tolist(),
                    'molecular_info': mol_info,
                    'lipinski_rule': self.check_lipinski_rule(mol_info),
                    'molecule_image': self.generate_molecule_image(mol)
                }
                
                return result
                
            except Exception as e:
                return {'error': str(e)}
        
        def check_lipinski_rule(self, mol_info):
            """检查Lipinski规则（药物相似性）"""
            violations = 0
            rules = {}
            
            if mol_info['molecular_weight'] > 500:
                violations += 1
                rules['molecular_weight'] = False
            else:
                rules['molecular_weight'] = True
                
            if mol_info['logp'] > 5:
                violations += 1
                rules['logp'] = False
            else:
                rules['logp'] = True
                
            if mol_info['h_donors'] > 5:
                violations += 1
                rules['h_donors'] = False
            else:
                rules['h_donors'] = True
                
            if mol_info['h_acceptors'] > 10:
                violations += 1
                rules['h_acceptors'] = False
            else:
                rules['h_acceptors'] = True
            
            return {
                'violations': violations,
                'rules': rules,
                'drug_like': violations <= 1
            }
        
        def generate_molecule_image(self, mol):
            """生成分子结构图像（返回base64编码）"""
            import io
            import base64
            
            img = Draw.MolToImage(mol, size=(300, 300))
            
            # 转换为base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        
        def batch_predict(self, smiles_list):
            """批量预测分子性质"""
            results = []
            for smiles in smiles_list:
                result = self.predict_single_molecule(smiles)
                results.append(result)
            return results
    
    return MolecularPropertyPredictor

# 使用示例
# predictor = create_molecular_property_webapp()
# result = predictor.predict_single_molecule('CCO')
```

#### 项目2：分子优化和生成
```python
def molecular_optimization_project():
    """分子优化项目：基于性质目标生成新分子"""
    
    class MolecularOptimizer:
        def __init__(self, property_predictor):
            self.predictor = property_predictor
        
        def generate_similar_molecules(self, seed_smiles, n_variants=10):
            """基于种子分子生成相似分子"""
            from rdkit.Chem import rdMolDescriptors
            
            mol = Chem.MolFromSmiles(seed_smiles)
            if mol is None:
                return []
            
            # 生成分子变体（简化版本）
            variants = []
            
            # 方法1: 添加/删除官能团
            functional_groups = ['C', 'O', 'N', 'Cl', 'F']
            
            for _ in range(n_variants):
                # 这里使用简化的分子修饰方法
                # 在实际应用中，可以使用更复杂的分子生成算法
                variant_smiles = self.modify_molecule(seed_smiles)
                if variant_smiles and variant_smiles != seed_smiles:
                    variants.append(variant_smiles)
            
            return list(set(variants))  # 去重
        
        def modify_molecule(self, smiles):
            """简单的分子修饰方法"""
            import random
            
            # 简化的修饰策略
            modifications = [
                lambda s: s.replace('C', 'CC', 1),     # 添加甲基
                lambda s: s.replace('CC', 'C', 1),     # 删除甲基
                lambda s: s.replace('O', 'N', 1),      # 替换原子
                lambda s: s + 'O' if len(s) < 20 else s,  # 添加羟基
            ]
            
            try:
                mod_func = random.choice(modifications)
                modified = mod_func(smiles)
                
                # 验证修饰后的SMILES
                mol = Chem.MolFromSmiles(modified)
                if mol:
                    return Chem.MolToSmiles(mol)
                else:
                    return None
            except:
                return None
        
        def optimize_for_property(self, seed_smiles, target_property_value, 
                                 max_iterations=50):
            """基于目标性质优化分子"""
            
            best_smiles = seed_smiles
            best_score = 0
        best_params = None
        
        for params in ParameterGrid(param_grid):
            print(f"测试参数: {params}")
            
            try:
                # 创建模型
                model = dc.models.MultitaskClassifier(
                    n_tasks=len(dataset.get_task_names()),
                    n_features=dataset.get_data_shape()[0],
                    layer_sizes=params['layer_sizes'],
                    dropouts=params['dropouts'],
                    learning_rate=params['learning_rate']
                )
                
                # 训练模型
                model.fit(train, nb_epoch=30)
                
                # 评估
                metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
                score = model.evaluate(valid, [metric])['roc_auc_score']
                
                print(f"验证分数: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"参数 {params} 训练失败: {e}")
        
        print(f"\n最佳参数: {best_params}")
        print(f"最佳分数: {best_score:.3f}")
        
        return best_params, best_score
    
    def implement_early_stopping(model, train_dataset, valid_dataset):
        """实现早停机制"""
        best_score = 0
        patience = 10
        patience_counter = 0
        
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        
        for epoch in range(100):
            # 训练一个epoch
            model.fit(train_dataset, nb_epoch=1)
            
            # 验证
            valid_score = model.evaluate(valid_dataset, [metric])['roc_auc_score']
            
            if valid_score > best_score:
                best_score = valid_score
                patience_counter = 0
                # 保存最佳模型
                model.save_checkpoint(max_checkpoints_to_keep=1)
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch + 1}: 验证分数 = {valid_score:.3f}")
            
            if patience_counter >= patience:
                print(f"早停于第 {epoch + 1} 个epoch")
                # 恢复最佳模型
                model.restore()
                break
        
        return model, best_score
    
    def model_ensemble_prediction(models, dataset):
        """模型集成预测"""
        predictions = []
        
        for i, model in enumerate(models):
            pred = model.predict(dataset)
            predictions.append(pred)
            print(f"模型 {i+1} 预测完成")
        
        # 平均集成
        ensemble_pred = np.mean(predictions, axis=0)
        
        # 投票集成（分类任务）
        voting_pred = np.round(ensemble_pred)
        
        return ensemble_pred, voting_pred
    
    return hyperparameter_search, implement_early_stopping, model_ensemble_prediction
```

#### 问题4：数据不平衡处理
```python
def handle_imbalanced_data():
    """处理数据不平衡问题"""
    
    def analyze_class_distribution(dataset):
        """分析类别分布"""
        labels = dataset.y.flatten()
        unique, counts = np.unique(labels, return_counts=True)
        
        print("类别分布:")
        for label, count in zip(unique, counts):
            percentage = count / len(labels) * 100
            print(f"类别 {label}: {count} 样本 ({percentage:.1f}%)")
        
        # 计算不平衡比率
        imbalance_ratio = max(counts) / min(counts)
        print(f"不平衡比率: {imbalance_ratio:.2f}")
        
        return unique, counts, imbalance_ratio
    
    def apply_class_weights(dataset, model_class=dc.models.MultitaskClassifier):
        """应用类别权重"""
        from sklearn.utils.class_weight import compute_class_weight
        
        labels = dataset.y.flatten()
        classes = np.unique(labels)
        
        # 计算类别权重
        class_weights = compute_class_weight(
            'balanced', 
            classes=classes, 
            y=labels
        )
        
        weight_dict = dict(zip(classes, class_weights))
        print(f"类别权重: {weight_dict}")
        
        # 创建带权重的模型（需要自定义实现）
        # DeepChem 的某些模型支持样本权重
        return weight_dict
    
    def oversample_minority_class(dataset):
        """过采样少数类"""
        from sklearn.utils import resample
        
        # 转换为DataFrame便于处理
        df = pd.DataFrame({
            'features': list(dataset.X),
            'labels': dataset.y.flatten()
        })
        
        # 分离不同类别
        class_groups = df.groupby('labels')
        
        # 找到最大类别的样本数
        max_size = class_groups.size().max()
        
        # 过采样每个类别到最大大小
        balanced_dfs = []
        for name, group in class_groups:
            oversampled = resample(
                group, 
                replace=True, 
                n_samples=max_size, 
                random_state=42
            )
            balanced_dfs.append(oversampled)
        
        # 合并平衡后的数据
        balanced_df = pd.concat(balanced_dfs)
        
        # 转换回DeepChem格式
        balanced_X = np.vstack(balanced_df['features'].values)
        balanced_y = balanced_df['labels'].values.reshape(-1, 1)
        
        balanced_dataset = dc.data.NumpyDataset(
            X=balanced_X, 
            y=balanced_y
        )
        
        print(f"原始数据集大小: {len(dataset)}")
        print(f"平衡后数据集大小: {len(balanced_dataset)}")
        
        return balanced_dataset
    
    def stratified_split(dataset):
        """分层抽样分割数据集"""
        # DeepChem 内置的分层分割器
        splitter = dc.splits.StratifiedSplitter()
        
        train, valid, test = splitter.train_valid_test_split(
            dataset, 
            frac_train=0.8, 
            frac_valid=0.1, 
            frac_test=0.1
        )
        
        print("分层分割结果:")
        for name, split_data in [('训练', train), ('验证', valid), ('测试', test)]:
            labels = split_data.y.flatten()
            unique, counts = np.unique(labels, return_counts=True)
            print(f"{name}集: {dict(zip(unique, counts))}")
        
        return train, valid, test
    
    return (analyze_class_distribution, apply_class_weights, 
            oversample_minority_class, stratified_split)
```

### 模型调试或训练优化技巧

#### 1. 学习率调度策略
```python
def advanced_training_strategies():
    """高级训练策略"""
    
    def learning_rate_scheduling():
        """学习率调度示例"""
        
        class LearningRateScheduler:
            def __init__(self, initial_lr=0.001):
                self.initial_lr = initial_lr
                self.current_lr = initial_lr
            
            def step_decay(self, epoch, drop_rate=0.5, epochs_drop=10):
                """阶梯衰减"""
                if epoch > 0 and epoch % epochs_drop == 0:
                    self.current_lr *= drop_rate
                return self.current_lr
            
            def exponential_decay(self, epoch, decay_rate=0.95):
                """指数衰减"""
                self.current_lr = self.initial_lr * (decay_rate ** epoch)
                return self.current_lr
            
            def cosine_annealing(self, epoch, max_epochs=100):
                """余弦退火"""
                import math
                self.current_lr = self.initial_lr * 0.5 * (
                    1 + math.cos(math.pi * epoch / max_epochs)
                )
                return self.current_lr
        
        return LearningRateScheduler
    
    def gradient_clipping_training(model, dataset, clip_value=1.0):
        """梯度裁剪训练"""
        # 在DeepChem中，可以通过自定义训练循环实现梯度裁剪
        # 这里提供概念性示例
        
        print(f"使用梯度裁剪训练，裁剪值: {clip_value}")
        
        # 标准训练，但可以在模型配置中设置梯度裁剪
        model.fit(dataset, nb_epoch=50)
        
        return model
    
    def data_augmentation_for_molecules():
        """分子数据增强技术"""
        
        def canonical_smiles_augmentation(smiles_list):
            """规范SMILES增强"""
            augmented = []
            
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 生成不同的SMILES表示
                    for _ in range(3):
                        aug_smiles = Chem.MolToSmiles(
                            mol, 
                            doRandom=True, 
                            canonical=False
                        )
                        augmented.append(aug_smiles)
            
            return augmented
        
        def conformer_based_augmentation(smiles_list):
            """基于构象的增强"""
            from rdkit.Chem import rdDistGeom
            
            augmented_data = []
            
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 生成3D构象
                    mol_h = Chem.AddHs(mol)
                    
                    # 生成多个构象
                    conf_ids = rdDistGeom.EmbedMultipleConfs(
                        mol_h, 
                        numConfs=5, 
                        randomSeed=42
                    )
                    
                    for conf_id in conf_ids:
                        # 可以提取3D特征或描述符
                        augmented_data.append((smiles, conf_id))
            
            return augmented_data
        
        return canonical_smiles_augmentation, conformer_based_augmentation
    
    return learning_rate_scheduling, gradient_clipping_training, data_augmentation_for_molecules

# 获取高级训练策略
lr_scheduler, gradient_clipping, data_augmentation = advanced_training_strategies()
```

#### 2. 模型诊断和可视化
```python
def model_diagnostics():
    """模型诊断工具"""
    
    def diagnose_overfitting(train_scores, valid_scores):
        """诊断过拟合"""
        
        # 计算训练和验证分数的差异
        score_diff = [abs(t - v) for t, v in zip(train_scores, valid_scores)]
        avg_diff = np.mean(score_diff)
        
        print("过拟合诊断:")
        print(f"平均训练-验证分数差异: {avg_diff:.3f}")
        
        if avg_diff > 0.1:
            print("⚠️ 可能存在过拟合")
            recommendations = [
                "增加Dropout比率",
                "减少模型复杂度",
                "增加训练数据",
                "使用正则化技术",
                "实施早停机制"
            ]
            print("建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("✅ 模型泛化良好")
        
        # 可视化训练曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_scores, 'b-', label='训练分数', linewidth=2)
        plt.plot(valid_scores, 'r-', label='验证分数', linewidth=2)
        plt.fill_between(range(len(score_diff)), 
                        [t - d for t, d in zip(train_scores, score_diff)],
                        [t + d for t, d in zip(train_scores, score_diff)],
                        alpha=0.2, color='gray', label='分数差异区间')
        plt.xlabel('Epoch')
        plt.ylabel('分数')
        plt.title('训练和验证分数曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_prediction_errors(y_true, y_pred, smiles_list=None):
        """分析预测错误"""
        
        errors = np.abs(y_true.flatten() - y_pred.flatten())
        
        # 统计错误分布
        print("预测错误分析:")
        print(f"平均绝对误差: {np.mean(errors):.3f}")
        print(f"标准差: {np.std(errors):.3f}")
        print(f"最大误差: {np.max(errors):.3f}")
        print(f"90百分位误差: {np.percentile(errors, 90):.3f}")
        
        # 找出错误最大的样本
        worst_indices = np.argsort(errors)[-5:]
        
        print("\n错误最大的5个预测:")
        for i, idx in enumerate(worst_indices):
            smiles = smiles_list[idx] if smiles_list else f"样本 {idx}"
            print(f"  {i+1}. {smiles}")
            print(f"     真实值: {y_true.flatten()[idx]:.3f}")
            print(f"     预测值: {y_pred.flatten()[idx]:.3f}")
            print(f"     误差: {errors[idx]:.3f}")
        
        # 误差分布直方图
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('绝对误差')
        plt.ylabel('频数')
        plt.title('预测误差分布')
        plt.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'平均误差: {np.mean(errors):.3f}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.6, c=errors, cmap='Reds')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'k--', label='理想预测')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测vs真实值 (颜色表示误差大小)')
        plt.colorbar(label='绝对误差')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return errors, worst_indices
    
    def feature_importance_analysis(model, dataset, feature_names=None):
        """特征重要性分析（适用于支持的模型）"""
        
        try:
            # 使用置换重要性方法
            from sklearn.inspection import permutation_importance
            
            # 包装DeepChem模型为sklearn兼容格式
            class DCModelWrapper:
                def __init__(self, dc_model):
                    self.model = dc_model
                
                def predict(self, X):
                    temp_dataset = dc.data.NumpyDataset(X=X)
                    return self.model.predict(temp_dataset).flatten()
            
            wrapper = DCModelWrapper(model)
            
            # 计算置换重要性
            result = permutation_importance(
                wrapper, 
                dataset.X, 
                dataset.y.flatten(),
                n_repeats=10, 
                random_state=42
            )
            
            # 创建重要性DataFrame
            if feature_names is None:
                feature_names = [f'特征_{i}' for i in range(dataset.X.shape[1])]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': result.importances_mean,
                'std': result.importances_std
            }).sort_values('importance', ascending=False)
            
            # 可视化前20个最重要的特征
            top_features = importance_df.head(20)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'], 
                    xerr=top_features['std'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('重要性分数')
            plt.title('前20个最重要特征')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return importance_df
            
        except Exception as e:
            print(f"特征重要性分析失败: {e}")
            return None
    
    return diagnose_overfitting, analyze_prediction_errors, feature_importance_analysis

# 获取诊断工具
diagnose_overfitting, analyze_prediction_errors, feature_importance_analysis = model_diagnostics()
```

---

## 总结

DeepChem 为化学和生物学研究提供了强大的深度学习工具包。通过本教学文档，初学者可以：

1. **掌握核心概念**：理解分子表示学习和深度学习在化学中的应用
2. **熟练使用工具**：掌握数据加载、模型训练、预测和评估的完整流程
3. **解决实际问题**：能够处理分子毒性预测、药物发现等实际应用场景
4. **优化模型性能**：了解处理数据不平衡、过拟合等常见问题的方法
5. **扩展应用领域**：具备将DeepChem应用到新领域（如台湾化学品数据库、REACH法规）的能力

### 学习建议
1. **从简单开始**：先用小数据集熟悉工具流程
2. **理解化学背景**：学习基本的化学知识有助于更好地应用工具
3. **实践为主**：通过实际项目加深理解
4. **关注社区**：参与DeepChem社区讨论，获取最新进展
5. **结合其他工具**：学会与RDKit、PyTorch等其他工具配合使用

DeepChem 正在快速发展，新的模型和功能不断加入。建议学习者保持关注官方文档和GitHub仓库的更新，以获得最新的功能和最佳实践。float('inf')
            
            # 预测种子分子的性质
            seed_result = self.predictor.predict_single_molecule(seed_smiles)
            if 'error' in seed_result:
                return {'error': seed_result['error']}
            
            seed_property = seed_result['predictions'][0]
            history = [(seed_smiles, seed_property, 0)]
            
            for iteration in range(max_iterations):
                # 生成变体
                variants = self.generate_similar_molecules(best_smiles, n_variants=5)
                
                for variant in variants:
                    # 预测性质
                    result = self.predictor.predict_single_molecule(variant)
                    if 'error' not in result:
                        predicted_value = result['predictions'][0]
                        score = abs(predicted_value - target_property_value)
                        
                        history.append((variant, predicted_value, iteration + 1))
                        
                        if score < best_score:
                            best_score = score
                            best_smiles = variant
            
            return {
                'optimized_smiles': best_smiles,
                'predicted_property': self.predictor.predict_single_molecule(best_smiles)['predictions'][0],
                'target_property': target_property_value,
                'improvement': abs(seed_property - target_property_value) - best_score,
                'optimization_history': history
            }
    
    return MolecularOptimizer

# 使用示例
# optimizer = molecular_optimization_project()
# result = optimizer.optimize_for_property('CCO', target_property_value=0.8)
```

#### 项目3：化学反应预测系统
```python
def chemical_reaction_prediction():
    """化学反应预测系统"""
    
    class ReactionPredictor:
        def __init__(self):
            # 这里可以加载预训练的反应预测模型
            # 或者使用简化的基于规则的方法
            self.reaction_templates = self.load_reaction_templates()
        
        def load_reaction_templates(self):
            """加载反应模板（简化版本）"""
            # 简化的反应模板
            templates = {
                'esterification': {
                    'reactants': ['carboxylic_acid', 'alcohol'],
                    'products': ['ester', 'water'],
                    'conditions': 'acid_catalyst'
                },
                'hydrolysis': {
                    'reactants': ['ester', 'water'],
                    'products': ['carboxylic_acid', 'alcohol'],
                    'conditions': 'base_or_acid'
                },
                'oxidation': {
                    'reactants': ['alcohol'],
                    'products': ['aldehyde_or_ketone'],
                    'conditions': 'oxidizing_agent'
                }
            }
            return templates
        
        def predict_reaction_products(self, reactants_smiles):
            """预测反应产物"""
            results = []
            
            for smiles in reactants_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 分析分子官能团
                    functional_groups = self.identify_functional_groups(mol)
                    
                    # 基于官能团预测可能的反应
                    possible_reactions = self.match_reactions(functional_groups)
                    
                    results.append({
                        'reactant': smiles,
                        'functional_groups': functional_groups,
                        'possible_reactions': possible_reactions
                    })
            
            return results
        
        def identify_functional_groups(self, mol):
            """识别分子中的官能团"""
            from rdkit.Chem import rdMolDescriptors
            
            functional_groups = []
            
            # 检测常见官能团
            if rdMolDescriptors.NumAliphaticCarbocycles(mol) > 0:
                functional_groups.append('cycloalkyl')
            
            if rdMolDescriptors.NumAromaticCarbocycles(mol) > 0:
                functional_groups.append('aromatic')
            
            # 简化的官能团检测
            smiles = Chem.MolToSmiles(mol)
            
            if 'O' in smiles:
                if 'OH' in smiles:
                    functional_groups.append('alcohol')
                if '(=O)' in smiles:
                    functional_groups.append('carbonyl')
            
            if 'N' in smiles:
                functional_groups.append('nitrogen_containing')
            
            return functional_groups
        
        def match_reactions(self, functional_groups):
            """匹配可能的反应类型"""
            possible_reactions = []
            
            if 'alcohol' in functional_groups:
                possible_reactions.extend(['oxidation', 'esterification'])
            
            if 'carbonyl' in functional_groups:
                possible_reactions.extend(['reduction', 'nucleophilic_addition'])
            
            if 'aromatic' in functional_groups:
                possible_reactions.extend(['electrophilic_substitution'])
            
            return possible_reactions
        
        def estimate_reaction_feasibility(self, reactants, products):
            """估算反应可行性（简化版本）"""
            # 这里可以整合热力学和动力学数据
            # 目前使用简化的评分方法
            
            feasibility_score = 0.5  # 基准分数
            
            # 基于分子复杂度调整
            reactant_complexity = sum(len(Chem.MolFromSmiles(r).GetAtoms()) 
                                    for r in reactants if Chem.MolFromSmiles(r))
            product_complexity = sum(len(Chem.MolFromSmiles(p).GetAtoms()) 
                                   for p in products if Chem.MolFromSmiles(p))
            
            # 简单的启发式规则
            if abs(reactant_complexity - product_complexity) < 5:
                feasibility_score += 0.2
            
            return min(1.0, feasibility_score)
    
    return ReactionPredictor

# 使用示例
# predictor = chemical_reaction_prediction()
# result = predictor.predict_reaction_products(['CCO', 'CC(=O)O'])
```

## 7. 注意事项

### 初学者容易遇到的问题

#### 问题1：RDKit安装和分子解析错误
```python
def handle_rdkit_issues():
    """处理RDKit相关问题的解决方案"""
    
    def safe_mol_from_smiles(smiles):
        """安全地从SMILES创建分子对象"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"警告: 无法解析SMILES: {smiles}")
                return None
            return mol
        except Exception as e:
            print(f"错误: SMILES解析失败 - {smiles}: {e}")
            return None
    
    def validate_dataset_smiles(smiles_list):
        """验证数据集中的SMILES格式"""
        valid_smiles = []
        invalid_smiles = []
        
        for i, smiles in enumerate(smiles_list):
            mol = safe_mol_from_smiles(smiles)
            if mol:
                # 标准化SMILES
                canonical_smiles = Chem.MolToSmiles(mol)
                valid_smiles.append(canonical_smiles)
            else:
                invalid_smiles.append((i, smiles))
        
        if invalid_smiles:
            print(f"发现 {len(invalid_smiles)} 个无效SMILES:")
            for idx, smiles in invalid_smiles[:5]:  # 只显示前5个
                print(f"  索引 {idx}: {smiles}")
        
        return valid_smiles, invalid_smiles
    
    return safe_mol_from_smiles, validate_dataset_smiles

safe_mol_from_smiles, validate_dataset_smiles = handle_rdkit_issues()
```

#### 问题2：内存不足和数据加载问题
```python
def handle_memory_issues():
    """处理内存不足问题的解决方案"""
    
    def create_batched_dataset(csv_file, batch_size=1000):
        """分批处理大型数据集"""
        import pandas as pd
        
        # 读取CSV文件的chunk
        chunks = []
        for chunk in pd.read_csv(csv_file, chunksize=batch_size):
            chunks.append(chunk)
            print(f"处理批次，大小: {len(chunk)}")
        
        return chunks
    
    def memory_efficient_training(model, dataset, batch_size=32):
        """内存高效的训练方法"""
        import gc
        
        # 分批训练
        n_samples = len(dataset)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(10):  # 减少epoch数量
            total_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                # 创建小批次数据集
                batch_X = dataset.X[start_idx:end_idx]
                batch_y = dataset.y[start_idx:end_idx]
                batch_dataset = dc.data.NumpyDataset(X=batch_X, y=batch_y)
                
                # 训练单个批次
                model.fit(batch_dataset, nb_epoch=1)
                
                # 清理内存
                del batch_X, batch_y, batch_dataset
                gc.collect()
            
            print(f"完成第 {epoch + 1} 个epoch")
    
    def reduce_feature_dimensions(dataset, max_features=512):
        """降低特征维度以节省内存"""
        from sklearn.feature_selection import SelectKBest, f_regression
        
        if dataset.X.shape[1] > max_features:
            print(f"降低特征维度从 {dataset.X.shape[1]} 到 {max_features}")
            
            selector = SelectKBest(score_func=f_regression, k=max_features)
            X_reduced = selector.fit_transform(dataset.X, dataset.y.flatten())
            
            reduced_dataset = dc.data.NumpyDataset(
                X=X_reduced, 
                y=dataset.y, 
                ids=dataset.ids
            )
            
            return reduced_dataset, selector
        
        return dataset, None
    
    return create_batched_dataset, memory_efficient_training, reduce_feature_dimensions
```

#### 问题3：模型性能调优
```python
def model_optimization_tips():
    """模型性能调优建议"""
    
    def hyperparameter_search(dataset):
        """超参数搜索"""
        from sklearn.model_selection import ParameterGrid
        
        # 分割数据
        splitter = dc.splits.RandomSplitter()
        train, valid, test = splitter.train_valid_test_split(dataset)
        
        # 定义超参数网格
        param_grid = {
            'layer_sizes': [[512, 128], [256, 64], [128, 32]],
            'dropouts': [[0.2, 0.2], [0.3, 0.3], [0.5, 0.5]],
            'learning_rate': [0.001, 0.01, 0.1]
        }
        
        best_score = 
    