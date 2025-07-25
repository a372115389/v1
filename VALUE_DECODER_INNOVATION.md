# DiSPO with ValueDecoder: Non-linear Task-Conditioned Value Functions

## 创新概述

本实现将原版DiSPO算法的线性价值函数 `V = ψᵀw` 替换为更强大的非线性、任务条件化的价值解码器 `V_θ(ψ, z)`，其中：

- `V_θ`: 神经网络实现的通用函数逼近器
- `ψ`: 后继特征 (successor features)  
- `z`: 低维、可学习的任务编码 (task embedding)

## 核心组件

### 1. ValueDecoder (`models/value_decoder.py`)
- **架构选择**: 采用MLP而非扩散模型，确保训练稳定性
- **任务条件化**: 通过concatenation方式融合后继特征和任务编码
- **残差连接**: 提供更好的梯度流
- **层归一化**: 稳定训练过程

```python
V_θ(ψ, z) = MLP(concat(ψ, z))
```

### 2. TaskEmbedding (`models/value_decoder.py`)
- **少样本适应**: 支持通过梯度式元学习快速适应新任务
- **灵活设计**: 支持单任务和多任务场景
- **低维表示**: 32维编码捕获任务的奖励结构

### 3. 基于梯度的规划 (`utils.py`)
- **动态引导**: `g = ∇_ψ V_θ(ψ, z)` 替代固定的线性引导 `g = w`
- **自适应性**: 引导信号根据当前状态和学习到的价值函数动态调整
- **兼容性**: 支持random shooting和guided diffusion两种规划方式

## 训练策略

### 1. Monte Carlo Returns
- 使用预计算的蒙特卡洛回报作为训练目标
- 比TD学习更稳定，避免bootstrap误差累积
- 在数据集预处理阶段计算，提高训练效率

### 2. 联合训练
- ValueDecoder和TaskEmbedding协同更新
- 通过共享梯度实现元学习式的任务适应
- EMA机制确保推理时的稳定性

## 文件修改总结

1. **新增文件**:
   - `models/value_decoder.py`: 核心ValueDecoder和TaskEmbedding实现

2. **主要修改**:
   - `train.py`: 集成ValueDecoder训练逻辑
   - `utils.py`: 重构planner以使用梯度引导
   - `environments/datasets.py`: 添加Monte Carlo返回计算
   - `configs/dispo_value_decoder.yaml`: 新配置文件

## 使用方法

1. **训练模型**:
```bash
python train.py --config-name=dispo_value_decoder
```

2. **关键配置参数**:
```yaml
model:
  task_embedding_dim: 32        # 任务编码维度
  value_hidden_dims: [512, 256, 128]  # ValueDecoder隐层
  value_dropout: 0.1            # Dropout比率

planning:
  planner: "guided_diffusion"   # 使用梯度引导
  guidance_coef: 1.0            # 引导系数
```

## 理论优势

1. **表达能力**: 非线性价值函数可以建模复杂的奖励结构
2. **泛化性**: 任务编码支持快速适应新任务
3. **规划质量**: 梯度引导提供更智能的后继特征搜索
4. **训练稳定性**: Monte Carlo目标避免TD学习的不稳定性

## 实验建议

1. **消融实验**: 比较线性vs非线性价值函数的性能
2. **任务适应**: 评估少样本任务适应能力
3. **规划质量**: 比较gradient guidance vs linear guidance
4. **计算效率**: 分析额外计算开销vs性能提升权衡

这个实现为DiSPO算法提供了一个更强大、更灵活的价值函数表示，有望在复杂的离线强化学习任务中取得更好的性能。