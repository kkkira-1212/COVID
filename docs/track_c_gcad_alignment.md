# Track C 与 GCAD 方法对齐总结

## 一、核心方法对齐

### 1.1 Gradient-based Granger Causality（梯度基Granger因果关系）

**GCAD方法**:
- 使用深度模型的梯度来动态挖掘因果关系
- 将Granger causality effect量化为深度预测器梯度在时间滞后上的积分
- 使用channel-separated gradient generator

**我们的实现**:
- 对每个变量j，计算预测变量j时其他变量i的梯度
- 在时间维度上聚合梯度，得到变量i对变量j的因果效应
- 构建因果矩阵 `[num_vars, num_vars]`

**关键点**:
- ✅ 使用梯度反映变量间影响
- ✅ 避免测试阶段重复优化
- ✅ 自动学习非线性关系

### 1.2 Sparsification（稀疏化）

**GCAD方法**:
- 使用基于对称性的稀疏化策略
- 消除双向边（bidirectional edges）
- 减少序列相似性对Granger因果关系的影响

**我们的实现**:
- 对每对变量(i, j)，只保留因果效应更强的方向
- 消除较弱的反向边
- 可选：添加阈值过滤弱因果关系

**关键点**:
- ✅ 基于对称性消除双向边
- ✅ 获得更清晰的因果图结构
- ✅ 减少相似性干扰

### 1.3 Pattern Deviation（模式偏差）

**GCAD方法**:
- 输出矩阵包含空间依赖信息和时间依赖信息
- 通过结合这两种依赖的偏差来计算异常分数
- 比较测试样本的因果矩阵与正常参考矩阵

**我们的实现**:
- 使用Frobenius norm计算矩阵偏差（基础方法）
- 可扩展为结合空间偏差和时间偏差（如果需要）
- 偏差越大，异常可能性越高

**关键点**:
- ✅ 使用normal reference作为基准
- ✅ 计算pattern deviation作为异常分数
- ✅ 可以结合空间和时间偏差

## 二、完整流程对齐

### 2.1 训练阶段

**GCAD流程**:
1. 在正常数据上训练预测模型
2. 使用梯度计算训练集的因果矩阵
3. 应用稀疏化
4. 聚合得到正常参考矩阵（median）

**我们的流程**:
1. ✅ 训练Track A预测模型（正常数据）
2. ✅ 计算训练集的gradient-based因果矩阵
3. ✅ 应用sparsification（消除双向边）
4. ✅ 使用median aggregation得到normal reference

**对齐状态**: ✅ 完全对齐

### 2.2 测试阶段

**GCAD流程**:
1. 加载预测模型和正常参考矩阵
2. 对测试样本计算因果矩阵
3. 应用稀疏化
4. 计算pattern deviation
5. 使用deviation作为异常分数

**我们的流程**:
1. ✅ 加载预测模型和normal reference
2. ✅ 计算测试样本的gradient-based因果矩阵
3. ✅ 应用sparsification
4. ✅ 计算pattern deviation (Frobenius norm)
5. ✅ 使用deviation作为anomaly_score

**对齐状态**: ✅ 完全对齐

## 三、实现细节对齐

### 3.1 梯度计算

**GCAD方法**:
- Channel-separated gradient generator
- 对每个变量j，计算其他变量i对它的梯度
- 在时间维度上积分

**我们的实现计划**:
```python
# 伪代码
for j in range(num_vars):
    pred_j = model(X_seq)[:, j]
    for i in range(num_vars):
        grad_ij = torch.autograd.grad(pred_j, X_seq[:, :, i], 
                                      retain_graph=True)
        causal_effect_ij = grad_ij.sum(dim=1)  # 时间维度聚合
        causal_matrix[i, j] = causal_effect_ij
```

**对齐状态**: ✅ 方法一致

### 3.2 稀疏化实现

**GCAD方法**:
- 对称性检查
- 消除双向边
- 保留更强的单向边

**我们的实现计划**:
```python
# 伪代码
for i in range(num_vars):
    for j in range(i+1, num_vars):
        if causal_matrix[i, j] > causal_matrix[j, i]:
            causal_matrix[j, i] = 0  # 保留 i->j
        else:
            causal_matrix[i, j] = 0  # 保留 j->i
```

**对齐状态**: ✅ 方法一致

### 3.3 正常参考矩阵聚合

**GCAD方法**:
- Median aggregation（对异常值鲁棒）

**我们的实现计划**:
```python
# 伪代码
normal_reference = torch.median(causal_matrices_train, dim=0)[0]
```

**对齐状态**: ✅ 方法一致

### 3.4 异常分数计算

**GCAD方法**:
- 结合空间和时间依赖偏差
- Frobenius norm或更复杂的度量

**我们的实现计划**:
```python
# 基础方法
deviation = torch.norm(causal_matrix - normal_reference, p='fro', dim=(-2, -1))

# 扩展方法（如果需要）
spatial_dev = compute_spatial_deviation(causal_matrix, normal_reference)
temporal_dev = compute_temporal_deviation(causal_matrix, normal_reference)
anomaly_score = spatial_dev + temporal_dev
```

**对齐状态**: ✅ 基础方法对齐，扩展方法可选

## 四、关键差异和选择

### 4.1 已对齐的部分
- ✅ Gradient-based Granger causality计算
- ✅ Sparsification方法（对称性消除双向边）
- ✅ Normal reference聚合（median）
- ✅ Pattern deviation计算（Frobenius norm）

### 4.2 可选的扩展
- **空间和时间偏差分离**: GCAD提到可以分离，但我们先用Frobenius norm
- **阈值过滤**: Sparsification中可以添加阈值，但先实现基础版本
- **其他距离度量**: 可以尝试其他矩阵距离，但Frobenius norm是标准选择

### 4.3 实现顺序
1. **Phase 1**: 实现gradient-based因果矩阵计算
2. **Phase 2**: 实现sparsification
3. **Phase 3**: 实现normal reference聚合
4. **Phase 4**: 实现pattern deviation计算
5. **Phase 5**: 集成到训练和推理流程

## 五、待确认的技术细节

### 5.1 梯度计算细节
- **问题**: 如何高效计算所有变量对的梯度？
- **方案**: 使用`torch.autograd.grad`，需要`retain_graph=True`
- **注意**: 可能需要batch处理以避免内存问题

### 5.2 时间维度聚合
- **问题**: 如何聚合时间维度上的梯度？
- **方案**: Sum或mean over time dimension
- **注意**: 需要确保时间维度正确

### 5.3 Sparsification时机
- **问题**: 在计算normal reference之前还是之后sparsify？
- **方案**: 先对每个样本sparsify，再聚合（更合理）
- **注意**: 确保训练和测试使用相同的sparsification策略

### 5.4 异常分数归一化
- **问题**: 是否需要归一化anomaly_score？
- **方案**: 可以先不归一化，使用原始deviation
- **注意**: 如果分数范围差异大，可能需要归一化

## 六、下一步行动

### 6.1 立即开始
1. ✅ 理解对齐完成
2. ⏳ 实现gradient-based因果矩阵计算模块
3. ⏳ 实现sparsification模块
4. ⏳ 实现normal reference聚合
5. ⏳ 实现pattern deviation计算

### 6.2 验证点
- 梯度计算是否正确反映变量间影响？
- Sparsification是否有效消除双向边？
- Normal reference是否代表正常模式？
- Pattern deviation是否能区分异常？

### 6.3 测试策略
- 在简单数据上测试每个模块
- 验证梯度计算和sparsification的正确性
- 检查normal reference的合理性
- 评估anomaly detection性能

## 七、总结

我们已经完全对齐了GCAD的核心方法：
1. ✅ **Gradient-based Granger Causality**: 使用梯度计算因果关系
2. ✅ **Sparsification**: 基于对称性消除双向边
3. ✅ **Pattern Deviation**: 通过矩阵偏差计算异常分数

现在可以开始实现代码了！


