# 在线世界模型适应：价值条件化的变分推理

## 🎯 核心创新概述

本实现解决了离线强化学习中的一个根本性挑战：**数据集偏斜问题**。我们提出了一种革命性的方法，通过在线世界模型适应(Online World Model Adaptation)，使智能体能够超越离线数据分布的局限，进行创造性规划。

### 理论基础

传统的离线强化学习方法受限于静态的世界模型 `p_φ(ψ|s)`，这个模型本质上只是对历史经验的"复述"。我们的创新将此静态模型转换为一个价值条件化的后验分布：

```
φ'^* = argmax_{φ'} E_{ψ ~ p_{φ'}(ψ|s)}[V_θ(ψ, z^*)] - λ · D_KL(p_{φ'}(ψ|s) || p_φ(ψ|s))
```

这个目标函数实现了两个关键目标的平衡：
1. **价值期望最大化**：推动模型生成高价值的后继特征
2. **KL散度正则化**：确保合理的外推，避免脱离物理可能性

## 🔧 技术实现架构

### 1. 参数化策略 (`train.py`)

我们采用增量参数化方法：`φ' = φ + Δφ`

```python
# 创建与原始参数结构相同的零初始化增量参数
delta_phi_init = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), psi_params)

# 为Δφ创建独立的Adam优化器，使用较小的学习率确保稳定性
delta_phi_optimizer = optax.adam(learning_rate=delta_phi_lr)
```

**设计rationale**: 这种方法确保了参数结构的一致性，同时允许在线微调而不破坏预训练的知识。

### 2. 变分目标函数 (`models/online_adaptation.py`)

#### 挑战1解决方案：可微分价值期望估计

```python
# 使用重参数化技巧实现可微分采样
psi_samples_adapted = psi_sampler(phi_adapted, sample_rng, batch_obs)

# 计算价值期望
values_adapted = value_decoder.model_def.apply(
    value_decoder.ema_params, psi_samples_adapted, task_embed_batch, training=False
).squeeze(-1)
value_expectation = jnp.mean(values_adapted)
```

**技术选择rationale**: 
- 使用VPSDE框架的重参数化特性保持梯度流
- 蒙特卡洛估计提供无偏但可微分的期望估计
- 避免了复杂的score function estimator

#### 挑战2解决方案：KL散度近似

我们采用混合方法来近似KL散度：

```python
# 方法1: 参数空间近似（适用于小Δφ）
param_kl = 0.5 * sum(jnp.sum(delta ** 2) for delta in jax.tree_util.tree_leaves(delta_phi))

# 方法2: 样本空间校正
values_prior = value_decoder.model_def.apply(...)
sample_kl_proxy = jnp.mean((values_adapted - values_prior) ** 2)

# 组合近似
kl_divergence = param_kl + 0.1 * sample_kl_proxy
```

**技术选择rationale**:
1. **参数KL**: 基于Fisher信息矩阵的二阶近似，计算高效
2. **样本KL**: 捕获分布差异，补偿参数近似的不足
3. **组合策略**: 平衡计算效率与准确性

### 3. 在线优化循环 (`utils.py`)

```python
# 短期梯度上升优化Δφ
for step in range(adaptation_steps):
    (loss, info), grads = jax.value_and_grad(objective_fn, argnums=1, has_aux=True)(
        phi_base, current_delta_phi, obs, step_rng
    )
    updates, new_opt_state = delta_phi_optimizer.update(grads, current_opt_state)
    new_delta_phi = optax.apply_updates(current_delta_phi, updates)
```

**实现特点**:
- **实时适应**: 每个决策步骤都进行在线优化
- **梯度稳定性**: 使用Adam优化器的自适应学习率
- **有限步数**: 平衡计算成本与适应质量

## 📊 配置参数说明

```yaml
planning:
  planner: "online_adaptation"  # 启用在线世界模型适应
  delta_phi_lr: 1e-3           # Δφ学习率（建议值：1e-4 to 1e-2）
  adaptation_steps: 10         # 在线优化步数（建议值：5-20）
  kl_coef: 1.0                # KL正则化系数λ（建议值：0.1-2.0）
  n_adaptation_samples: 32     # 蒙特卡洛样本数（建议值：16-64）
```

### 超参数调优指南

- **delta_phi_lr**: 过大导致不稳定，过小导致适应不足
- **adaptation_steps**: 更多步数提高适应质量但增加计算成本
- **kl_coef**: 控制创新性与保守性的平衡
- **n_adaptation_samples**: 影响期望估计的方差

## 🚀 使用方法

```bash
# 使用在线世界模型适应训练
python train.py --config-name=dispo_value_decoder

# 对比实验：使用传统guided diffusion
python train.py --config-name=dispo_value_decoder planning.planner=guided_diffusion
```

## 🔬 核心创新点

### 1. 理论创新
- **从静态到动态**: 将世界模型从固定分布转为价值条件化的动态分布
- **变分推理框架**: 提供了理论严谨的优化目标
- **平衡机制**: KL正则化确保合理外推

### 2. 技术创新
- **增量参数化**: φ' = φ + Δφ 保持预训练知识
- **混合KL近似**: 结合参数空间和样本空间的优势
- **实时适应**: 每步决策时的在线优化

### 3. 实现创新
- **JAX优化**: 充分利用JIT编译和自动微分
- **模块化设计**: 清晰的组件分离便于扩展
- **灵活配置**: 丰富的超参数控制

## 📈 预期优势

1. **突破数据限制**: 能够发现超越离线数据的最优策略
2. **自适应规划**: 根据当前任务动态调整世界模型
3. **理论保证**: 变分推理框架提供理论基础
4. **计算高效**: 短期优化循环保持实时性能

## 🔍 实验建议

### 消融研究
1. **KL系数λ的影响**: 0.1, 0.5, 1.0, 2.0, 5.0
2. **适应步数**: 5, 10, 15, 20步
3. **学习率敏感性**: 1e-4, 5e-4, 1e-3, 5e-3

### 对比实验
1. **vs. 静态规划**: 对比原始guided diffusion
2. **vs. 随机射击**: 评估价值引导的效果
3. **vs. 其他适应方法**: 如果有其他在线适应基线

### 分析指标
- **适应收敛性**: 目标函数的收敛曲线
- **KL散度变化**: 监控模型偏离程度
- **价值提升**: 适应前后的价值估计差异
- **计算开销**: 每步决策的时间成本

这个实现代表了离线强化学习向"创造性规划"智能体的重要进化，为突破数据集偏斜限制提供了一个理论严谨且实用的解决方案。