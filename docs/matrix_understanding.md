# 三种矩阵的理解

## 三种矩阵的清晰划分

### Matrix ①: Variable Relation Matrix (在Prediction Head内部)

**位置**: Prediction Head内部
**作用**: 显式建模变量交互，用于预测阶段
**功能**: 
- 变量 i 的信息，如何参与变量 j 的预测
- 让 predictor 在预测阶段就"知道变量之间可以互相影响"
**性质**: 
- 可学习参数（或通过某种机制计算）
- 用于预测过程，不是用于异常检测
- 帮助模型更好地预测（因为考虑了变量间关系）

**示例理解**:
```
预测变量j时，不仅看变量j的历史，还看：
- 变量i对变量j的影响（通过Matrix ①中的权重）
- 变量k对变量j的影响
- 等等...
```

### Matrix ②: Window-level Causal Matrix (每个window一张)

**位置**: 从每个window计算/提取
**作用**: 作为异常的"证据对象"
**功能**: 
- 每个window（一个样本）计算一张因果矩阵
- 反映这个window中变量间的因果关系/交互模式
**计算方式**: 
- 可能从Prediction Head的预测过程/结果中提取
- 或者从residual中计算
- 每个样本一个矩阵 [num_vars, num_vars]

### Matrix ③: Normal Reference Causal Matrix (参考基准)

**位置**: 训练阶段聚合得到
**作用**: 表示"正常结构长什么样"的参考
**功能**: 
- 从训练窗口的Matrix ②聚合出来（平均/中位数等）
- 作为正常的基准因果结构
**使用**: 
- 测试阶段，将Matrix ②与Matrix ③对比
- 偏离程度作为异常分数

## 完整流程梳理

### 训练阶段

```
X_seq [B, T, num_vars]
    ↓
Encoder (Transformer) - 学习时间模式
    ↓
z [B, d_model]
    ↓
Prediction Head (使用Matrix ①建模变量交互)
    ↓
pred_all [B, num_vars]
    ↓
计算residual_vec [B, num_vars]
    ↓
从每个window提取/计算 Matrix ② [B, num_vars, num_vars]
    ↓
聚合训练集的所有Matrix ② → Matrix ③ [num_vars, num_vars] (正常基准)
```

### 测试阶段

```
X_seq [B, T, num_vars]
    ↓
Encoder → z
    ↓
Prediction Head (使用Matrix ①) → pred_all
    ↓
计算residual_vec
    ↓
计算当前window的 Matrix ② [num_vars, num_vars]
    ↓
与Matrix ③对比: deviation = ||Matrix ② - Matrix ③||
    ↓
异常分数 = deviation
    ↓
判断0/1
```

## 关键理解点

1. **Matrix ①（Variable Relation Matrix）**
   - 在Prediction Head内部，用于预测
   - 显式建模"变量i如何影响变量j的预测"
   - 帮助提高预测准确性
   - **不是用于异常检测的**

2. **Matrix ②（Window-level Causal Matrix）**
   - 每个window一张
   - 从预测过程或residual中计算/提取
   - 作为异常检测的"证据对象"

3. **Matrix ③（Normal Reference Causal Matrix）**
   - 训练阶段聚合得到
   - 表示正常的因果结构
   - 用于与Matrix ②对比

## 待确认的问题

1. **Matrix ②的计算方式**：
   - 如何从每个window提取/计算因果矩阵？
   - 是从Prediction Head的预测过程提取？
   - 还是从residual中计算？
   - 使用什么方法（Granger causality、相关性、attention权重等）？

2. **Matrix ①的实现方式**：
   - 在Prediction Head中如何显式建模变量交互？
   - 是作为可学习参数，还是通过某种机制计算？
   - 具体如何体现"变量i的信息参与变量j的预测"？

3. **Matrix ③的聚合方式**：
   - 平均？中位数？其他方式？
   - 是否所有训练样本等权重，还是只考虑正常样本？

