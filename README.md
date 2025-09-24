# GNN分子属性预测

基于图神经网络(GNN)的QM9分子属性预测项目，使用PyTorch Geometric实现分子能量、带隙等属性的回归预测任务。

## 项目特点

- **自动数据处理**: 自动下载QM9数据集并进行预处理
- **灵活的GNN架构**: 支持多层GCN网络和多种池化方式
- **完整训练流程**: 包含训练、验证、测试和可视化
- **GPU加速**: 自动检测并支持GPU训练
- **模型保存**: 自动保存最佳模型和训练结果

## 环境要求

Python 3.8+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

直接运行主程序：

```bash
python gnn_molecular_prediction.py
```

## 程序功能

### 1. 数据加载与预处理
- 自动下载QM9数据集（约134k个小分子）
- 将分子转换为图结构（节点=原子，边=化学键）
- 数据标准化和划分（80%训练，10%验证，10%测试）

### 2. GNN模型架构
- 多层图卷积网络(GCN)
- 批量归一化和Dropout
- 三种池化方式组合（平均、最大、求和池化）
- 全连接输出层

### 3. 训练功能
- Adam优化器
- 学习率自适应调整
- 早停机制防止过拟合
- 训练过程可视化

### 4. 评估指标
- 平均绝对误差(MAE)
- 均方误差(MSE)
- 均方根误差(RMSE)
- 预测值vs真实值对比图

## 可预测的分子属性

程序支持预测QM9数据集中的19种分子属性（修改`main()`函数中的`target_index`即可）：

0. 偶极矩 (dipole_moment)
1. 各向同性极化率 (isotropic_polarizability)
2. HOMO能级 (homo)
3. LUMO能级 (lumo)
4. **HOMO-LUMO能隙 (gap)** - 默认预测目标
5. 电子空间扩展 (electronic_spatial_extent)
6. 零点振动能 (zero_point_vibrational_energy)
7. 0K内能 (internal_energy_0K)
8. 298K内能 (internal_energy_298K)
9. 298K焓 (enthalpy_298K)
10. 298K自由能 (free_energy_298K)
11. 热容 (heat_capacity)
12. 原子化能 (atomization_energy)
13. 原子化焓 (atomization_enthalpy)
14. 原子化自由能 (atomization_free_energy)
15. 转动常数A (rotational_constant_A)
16. 转动常数B (rotational_constant_B)
17. 转动常数C (rotational_constant_C)
18. 振动频率 (vibrational_frequencies)

## 输出文件

运行完成后会生成：
- `best_model.pth`: 最佳模型权重
- `training_history.png`: 训练损失曲线
- `predictions_comparison.png`: 预测值对比图
- `results.npy`: 详细结果数据

## 自定义配置

### 修改预测目标
```python
# 在main()函数中修改target_index
target_index = 4  # 0-18，对应不同的分子属性
```

### 调整模型参数
```python
model = GNNModel(num_features=num_features,
                hidden_dim=128,    # 隐藏层维度
                num_layers=4,      # GNN层数
                num_targets=1,     # 预测目标数
                dropout=0.2)       # Dropout概率
```

### 修改训练参数
```python
num_epochs = 200          # 训练轮数
batch_size = 64          # 批次大小
lr = 0.001               # 学习率
weight_decay = 1e-5      # 权重衰减
```

## 性能参考

在默认设置下（预测HOMO-LUMO能隙）：
- 训练时间：约10-30分钟（取决于硬件）
- 测试MAE：通常在0.1-0.2范围内
- GPU内存需求：约2-4GB

## 故障排除

1. **CUDA内存不足**: 降低`batch_size`或`hidden_dim`
2. **训练过慢**: 检查是否正确使用GPU
3. **依赖安装问题**: 确保PyTorch版本与CUDA版本匹配

## 扩展功能

可以基于此代码扩展：
- 添加其他GNN架构（GAT、GraphSAINT等）
- 实现多任务学习（同时预测多个属性）
- 添加更复杂的分子特征工程
- 集成其他分子数据集