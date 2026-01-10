# Matrix 实现方案讨论

## Matrix ② 计算方式对比：Transfer Entropy vs Gradient-based

### 方案1: Transfer Entropy（CGAD方法）

**原理**:
- 传递熵（Transfer Entropy）衡量变量 X 对变量 Y 的因果影响
- 基于信息论，测量条件信息增益
- 公式：TE_{X→Y} = I(Y_t; X_{t-k:t-1} | Y_{t-k:t-1})

**优点**:
- 理论基础扎实，符合因果关系定义
- 不需要模型梯度，可以独立计算
- 对非线性关系也有效

**缺点**:
- 需要足够的历史数据（每个window需要计算）
- 计算复杂度高：O(num_vars² × window_size²)
- 需要离散化或估计概率分布
- 实现复杂（需要statsmodels或自定义实现）

**实现复杂度**: ⭐⭐⭐⭐ (较高)

### 方案2: Gradient-based 方法

**原理**:
- 利用模型的梯度信息
- 计算 residual 对输入变量的梯度
- 梯度大小反映变量的"影响力"

**基本思路**:
```python
# residual = |X_next - pred|
# 计算每个变量的梯度
for i in range(num_vars):
    grad_i = grad(residual, X_seq[:, :, i], retain_graph=True)
    # grad_i 反映变量i对residual的影响
```

**优点**:
- **实现简单**：PyTorch原生支持autograd
- **计算高效**：O(num_vars × batch_size)，比Transfer Entropy快
- **端到端可微**：可以反向传播（如果需要）
- **适合深度学习框架**：与现有代码集成容易
- **不需要历史数据**：单个样本就能计算

**缺点**:
- 梯度反映的是"敏感性"而非严格意义上的"因果关系"
- 可能受模型训练状态影响
- 对于非线性关系的解释性可能不如Transfer Entropy

**实现复杂度**: ⭐⭐ (较低)

## 我的建议：Gradient-based 更适合

### 理由：

1. **实现简单性**
   - PyTorch的autograd天然支持
   - 代码简洁，易于维护
   - 不需要额外的依赖（如statsmodels）

2. **计算效率**
   - 对于每个样本，只需要一次forward+backward
   - Transfer Entropy需要滑动窗口计算，效率低
   - 特别是batch处理时，gradient-based更高效

3. **与现有框架集成**
   - 我们的模型已经是PyTorch
   - 可以无缝集成到训练/推理流程
   - 不需要额外的前处理步骤

4. **实际效果**
   - 对于异常检测任务，gradient-based可能足够
   - 我们关注的是"偏离正常模式"，而不是严格的因果推断
   - 如果效果不理想，再考虑Transfer Entropy

### Gradient-based 实现思路：

```python
def compute_gradient_based_causal_matrix(X_seq, residual_vec, model):
    """
    从residual计算梯度因果矩阵
    
    Args:
        X_seq: [B, T, num_vars] - 输入序列
        residual_vec: [B, num_vars] - residual向量
        model: 预测模型
    
    Returns:
        causal_matrix: [B, num_vars, num_vars] - 因果矩阵
    """
    B, T, num_vars = X_seq.shape
    causal_matrix = torch.zeros(B, num_vars, num_vars)
    
    # 对每个样本计算
    for b in range(B):
        X_b = X_seq[b:b+1]  # [1, T, num_vars]
        X_b.requires_grad_(True)
        
        # Forward
        pred = model(X_b)  # [1, num_vars]
        # 假设有X_next用于计算residual
        residual_b = residual_vec[b:b+1]  # [1, num_vars]
        
        # 对每个变量j，计算所有变量i的梯度
        for j in range(num_vars):
            # residual_j对X_b的梯度
            grad_j = torch.autograd.grad(
                outputs=residual_b[0, j],
                inputs=X_b,
                retain_graph=True,
                create_graph=False
            )[0]  # [1, T, num_vars]
            
            # 聚合时间维度（平均或sum）
            grad_j_agg = grad_j.mean(dim=1)  # [1, num_vars]
            
            # 取绝对值，表示影响大小
            causal_matrix[b, :, j] = grad_j_agg[0].abs()
    
    return causal_matrix
```

**优化版本（批量计算）**:
```python
def compute_gradient_causal_matrix_batch(X_seq, residual_vec, model):
    """
    批量计算，更高效
    """
    B, T, num_vars = X_seq.shape
    X_seq.requires_grad_(True)
    
    causal_matrix = torch.zeros(B, num_vars, num_vars)
    
    # 计算每个变量j的residual对所有变量的梯度
    for j in range(num_vars):
        grad_j = torch.autograd.grad(
            outputs=residual_vec[:, j].sum(),  # 对所有样本求和
            inputs=X_seq,
            retain_graph=True,
            create_graph=False
        )[0]  # [B, T, num_vars]
        
        # 聚合时间维度
        grad_j_agg = grad_j.mean(dim=1)  # [B, num_vars]
        
        causal_matrix[:, :, j] = grad_j_agg.abs()
    
    return causal_matrix
```

## Matrix ① 实现：可学习参数 A

### 架构设计：

```python
class RegressionHeadWithRelation(nn.Module):
    def __init__(self, d_in, num_vars):
        super().__init__()
        self.num_vars = num_vars
        
        # Matrix ①: 可学习的变量关系矩阵
        self.var_relation_matrix = nn.Parameter(
            torch.randn(num_vars, num_vars) * 0.1  # 初始化
        )
        
        # 基础projection层
        self.base_proj = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_in, num_vars)
        )
        
    def forward(self, z):
        """
        Args:
            z: [B, d_model] - encoder输出
        
        Returns:
            pred: [B, num_vars] - 预测值
        """
        # 基础投影：每个变量初始的表示
        var_base = self.base_proj(z)  # [B, num_vars]
        
        # 使用关系矩阵混合变量信息
        # var_relation_matrix: [num_vars, num_vars]
        # var_base: [B, num_vars]
        # 预测变量j时，使用所有变量的信息，按关系矩阵混合
        pred = torch.matmul(var_base, self.var_relation_matrix.T)  # [B, num_vars]
        # 或者：pred = var_base @ self.var_relation_matrix.T
        
        return pred
```

**另一种实现方式（更灵活）**:
```python
class RegressionHeadWithRelation(nn.Module):
    def __init__(self, d_in, num_vars):
        super().__init__()
        self.num_vars = num_vars
        
        # Matrix ①: 可学习的变量关系矩阵
        self.var_relation_matrix = nn.Parameter(
            torch.randn(num_vars, num_vars) * 0.1
        )
        
        # 每个变量的独立投影
        self.var_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_in // 2),
                nn.ReLU(),
                nn.Linear(d_in // 2, 1)
            ) for _ in range(num_vars)
        ])
        
    def forward(self, z):
        """
        z: [B, d_model]
        """
        # 每个变量独立投影得到基础表示
        var_bases = torch.stack([
            proj(z) for proj in self.var_projs
        ], dim=1)  # [B, num_vars, 1] -> [B, num_vars]
        var_bases = var_bases.squeeze(-1)  # [B, num_vars]
        
        # 使用关系矩阵混合：变量i的信息参与变量j的预测
        # relation_matrix[i, j] 表示变量i对变量j的影响权重
        pred = torch.matmul(var_bases, self.var_relation_matrix)  # [B, num_vars]
        
        return pred
```

### 理解：
- `var_relation_matrix[i, j]` 表示变量 i 的信息对变量 j 预测的贡献
- 预测变量 j 时，使用所有变量的基础表示，按关系矩阵加权组合
- 关系矩阵在训练时通过反向传播更新

## Matrix ③ 聚合方式：中位数

```python
def aggregate_normal_causal_matrix(causal_matrices_train):
    """
    从训练集的Matrix ②聚合得到Matrix ③
    
    Args:
        causal_matrices_train: List of [num_vars, num_vars] or [N, num_vars, num_vars]
    
    Returns:
        normal_causal_matrix: [num_vars, num_vars] - 正常参考矩阵
    """
    if isinstance(causal_matrices_train, list):
        causal_matrices_train = torch.stack(causal_matrices_train)  # [N, num_vars, num_vars]
    
    # 使用中位数聚合（对异常值更鲁棒）
    normal_causal_matrix = torch.median(causal_matrices_train, dim=0)[0]  # [num_vars, num_vars]
    
    return normal_causal_matrix
```

## 总结建议

1. **Matrix ②**: 使用 **Gradient-based 方法**
   - 实现简单，效率高
   - 与PyTorch框架集成好
   - 如果效果不理想，再考虑Transfer Entropy

2. **Matrix ①**: **可学习参数 A [num_vars, num_vars]**
   - 在Prediction Head中实现
   - 预测时使用关系矩阵混合变量信息
   - 通过反向传播更新

3. **Matrix ③**: **中位数聚合**
   - 对异常值更鲁棒
   - 实现简单

