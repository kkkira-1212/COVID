# GCAD论文核心方法理解

## 一、GCAD核心思想

### 1.1 基本假设
- **核心假设**: 当异常发生时，变量间的Granger因果关系模式会发生显著变化
- **方法**: 通过比较测试样本的因果模式与正常参考模式的偏差来检测异常

### 1.2 关键创新点
1. **Gradient-based Granger Causality**: 使用深度模型的梯度来动态挖掘因果关系
2. **Sparsification**: 基于对称性的稀疏化策略，消除双向边
3. **Pattern Deviation**: 结合空间和时间依赖偏差计算异常分数

## 二、Gradient-based Granger Causality 计算

### 2.1 核心思路
根据GCAD论文，Granger causality effect被量化为**深度预测器梯度在时间滞后上的积分**。

### 2.2 实现方法
1. **Channel-separated Gradient Generator**:
   - 对每个变量j，计算预测变量j时，其他变量i的梯度贡献
   - 梯度反映了变量i对变量j预测的影响程度

2. **Gradient计算**:
   ```python
   # 伪代码
   for each variable j:
       # 计算预测变量j时的梯度
       pred_j = model(X_seq)[:, j]  # 预测变量j
       grad_ij = grad(pred_j, X_seq[:, :, i])  # 变量i对变量j的梯度
       # 在时间维度上聚合梯度
       causal_effect_ij = aggregate_over_time(grad_ij)
   ```

3. **时间维度聚合**:
   - 将梯度在时间窗口上积分/求和
   - 得到变量i对变量j的Granger causality effect

### 2.3 输出矩阵
- **形状**: `[num_vars, num_vars]` 或 `[B, num_vars, num_vars]`
- **含义**: `matrix[i, j]` 表示变量i对变量j的Granger causality effect
- **性质**: 包含空间依赖信息（变量间关系）和时间依赖信息（时间滞后效应）

## 三、Sparsification（稀疏化）

### 3.1 目的
- 消除双向边（bidirectional edges）
- 减少序列相似性对Granger因果关系的影响
- 获得更清晰的因果图结构

### 3.2 基于对称性的稀疏化方法
根据论文，使用**对称性**来消除双向边：

```python
# 伪代码
def sparsify(causal_matrix):
    """
    输入: causal_matrix [num_vars, num_vars]
    输出: sparsified_matrix [num_vars, num_vars]
    """
    # 方法1: 保留更强的单向边
    for i in range(num_vars):
        for j in range(i+1, num_vars):
            if causal_matrix[i, j] > causal_matrix[j, i]:
                # i -> j 更强，保留 i -> j，移除 j -> i
                causal_matrix[j, i] = 0
            else:
                # j -> i 更强，保留 j -> i，移除 i -> j
                causal_matrix[i, j] = 0
    
    # 方法2: 或者使用阈值过滤
    # 只保留超过阈值的边
    
    return causal_matrix
```

### 3.3 具体策略
论文提到使用"symmetry-based sparsification method to eliminate bidirectional edges"，可能的实现：
- **策略1**: 对于每对变量(i, j)，只保留因果效应更强的方向
- **策略2**: 设置阈值，只保留超过阈值的因果边
- **策略3**: 结合两者，先阈值过滤，再消除双向边

## 四、Pattern Deviation（模式偏差）

### 4.1 Normal Reference Matrix（正常参考矩阵）
- **来源**: 从训练集（正常数据）的causal matrices聚合得到
- **聚合方法**: Median aggregation（中位数聚合，对异常值鲁棒）
- **形状**: `[num_vars, num_vars]`
- **含义**: 表示正常情况下的Granger因果关系模式

### 4.2 Deviation计算
论文提到输出矩阵包含**空间依赖信息**和**时间依赖信息**，通过结合这两种依赖的偏差来计算异常分数。

```python
# 伪代码
def compute_anomaly_score(causal_matrix, normal_reference):
    """
    输入:
        causal_matrix: [num_vars, num_vars] - 当前样本的因果矩阵
        normal_reference: [num_vars, num_vars] - 正常参考矩阵
    输出:
        anomaly_score: scalar - 异常分数
    """
    # 方法1: Frobenius norm（Frobenius范数）
    deviation = torch.norm(causal_matrix - normal_reference, p='fro')
    
    # 方法2: 结合空间和时间偏差
    # 空间偏差: 变量间关系的偏差
    spatial_dev = compute_spatial_deviation(causal_matrix, normal_reference)
    # 时间偏差: 时间依赖模式的偏差
    temporal_dev = compute_temporal_deviation(causal_matrix, normal_reference)
    anomaly_score = spatial_dev + temporal_dev
    
    return anomaly_score
```

### 4.3 空间和时间偏差
- **空间偏差**: 变量间因果关系结构的偏差
- **时间偏差**: 时间滞后效应模式的偏差
- **结合方式**: 加权求和或直接相加

## 五、完整流程

### 5.1 训练阶段
```
1. 训练预测模型（Track A）
   - 使用正常数据训练
   - 学习正常的预测模式

2. 计算训练集的Causal Matrices
   - 对每个训练样本：
     a. 计算residual_vec
     b. 使用gradient计算Granger causality matrix
     c. 应用sparsification
   - 得到: causal_matrices_train [N_train, num_vars, num_vars]

3. 聚合Normal Reference Matrix
   - 使用median aggregation
   - 得到: normal_reference [num_vars, num_vars]

4. 保存模型和normal_reference
```

### 5.2 测试阶段
```
1. 加载预测模型和normal_reference

2. 对每个测试样本：
   a. 计算pred_all和residual_vec
   b. 使用gradient计算causal_matrix
   c. 应用sparsification
   d. 计算deviation = ||causal_matrix - normal_reference||
   e. deviation作为anomaly_score

3. 使用阈值或排序确定异常
```

## 六、与当前Track C设计的对应关系

### 6.1 Matrix ②的计算
- **当前设计**: 从residual或预测过程中提取
- **GCAD方法**: 使用gradient-based方法，从深度模型的梯度计算
- **更新**: 需要实现gradient-based的Granger causality计算

### 6.2 Sparsification
- **当前设计**: 未明确提到
- **GCAD方法**: 基于对称性的稀疏化，消除双向边
- **更新**: 需要添加sparsification步骤

### 6.3 Anomaly Score计算
- **当前设计**: 使用Frobenius norm
- **GCAD方法**: 结合空间和时间依赖偏差
- **更新**: 可以保留Frobenius norm，或扩展为结合空间和时间偏差

## 七、实现要点

### 7.1 Gradient计算
- 需要使用`torch.autograd.grad`计算梯度
- 对每个变量j，计算其他变量i对它的梯度
- 在时间维度上聚合梯度

### 7.2 Sparsification实现
- 实现对称性检查
- 消除双向边
- 可选：添加阈值过滤

### 7.3 Normal Reference聚合
- 使用median aggregation（对异常值鲁棒）
- 或使用mean aggregation（如果数据干净）

### 7.4 Anomaly Score
- 基础：Frobenius norm
- 扩展：结合空间和时间偏差（如果需要）


