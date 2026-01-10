# Track C 和 Track D 设计文档

## 整体架构流程

```
预测模型 (学习正常模式) → Residual → 两个分支
                                ├─→ Track C: Granger-causality-based structural analysis → 异常检测 (0/1, 可解释性)
                                └─→ Track D: Classification head → 异常类型 (A/B/C/D)
```

## 一、Track A 基础架构回顾

### 当前Track A流程
```
X_seq [B, T, F]
    ↓
TransformerSeqEncoder
    ↓
z [B, d_model]
    ↓
RegressionHead
    ↓
pred [B, num_vars] → pred_target [B, 1]
    ↓
Residual = |true_value - pred_target|  [B, 1]
    ↓
异常检测: residual作为异常分数 (越大越异常)
```

### Residual计算方式
- **residual_vec**: `|X_next - pred_all|`  [B, num_vars] - 所有变量的residual
- **residual**: `residual_vec[:, target_idx]`  [B, 1] - 目标变量的residual

## 二、Track C 设计：Granger-causality-based Structural Analysis

### 目标
- 使用Granger因果关系分析residual的结构
- 进行异常检测（二分类：0/1）
- 提供可解释性（哪些变量之间的因果关系异常）

### 架构设计

```
Residual Vector [B, num_vars]  (从prediction model得到)
    ↓
┌─────────────────────────────────────────────────────────┐
│ Granger Causality Analysis Module                       │
│                                                          │
│  1. 构建因果图 (Causal Graph)                            │
│     - 使用Granger因果关系测试变量间的关系                │
│     - 输出: 因果矩阵/图结构                              │
│                                                          │
│  2. Residual Pattern Analysis                           │
│     - 分析residual向量的模式                             │
│     - 结合因果结构分析异常传播路径                        │
│                                                          │
│  3. Structural Anomaly Score                            │
│     - 基于因果结构的异常评分                             │
│     - 输出: anomaly_score [B, 1]                        │
│                                                          │
│  4. Interpretability Module (可选)                       │
│     - 识别关键的异常因果关系                             │
│     - 输出: 可解释性特征/图结构                          │
└─────────────────────────────────────────────────────────┘
    ↓
Anomaly Detector (Binary Classifier)
    ↓
Anomaly Prediction [B, 1] (0/1)
```

### 实现方案

#### 方案1: 基于统计的Granger Causality（离线分析）
- 在训练/验证集上计算Granger因果关系矩阵
- 使用预计算的因果图分析residual
- 优点：可解释性强，符合传统Granger因果关系定义
- 缺点：需要足够的历史数据，计算复杂度高

#### 方案2: 神经网络学习因果关系（端到端）
- 使用图神经网络（GNN）或注意力机制学习变量间的因果关系
- 从residual中学习因果结构
- 优点：端到端训练，可以处理非线性关系
- 缺点：可解释性相对较弱

#### 推荐方案：混合方案
- 使用预训练的Granger因果关系矩阵作为先验知识
- 使用神经网络学习residual在因果结构上的异常模式
- 结合两者进行异常检测和可解释性分析

### 代码结构

```python
class GrangerCausalityAnalyzer:
    """Granger因果关系分析器"""
    def __init__(self, num_vars, max_lag=3):
        self.num_vars = num_vars
        self.max_lag = max_lag
        
    def compute_granger_matrix(self, data):
        """计算Granger因果关系矩阵"""
        # 使用statsmodels或自定义实现
        # 返回: causal_matrix [num_vars, num_vars]
        pass
    
    def analyze_residual_structure(self, residual_vec, causal_matrix):
        """分析residual的因果结构"""
        # 结合因果矩阵分析residual模式
        pass


class StructuralAnomalyDetector(nn.Module):
    """基于结构分析的异常检测器"""
    def __init__(self, num_vars, d_model=64):
        super().__init__()
        self.causal_encoder = nn.Linear(num_vars, d_model)
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, residual_vec, causal_matrix):
        # 结合residual和因果结构进行异常检测
        pass
```

## 三、Track D 设计：异常类型分类

### 目标
- 基于residual预测异常类型（A/B/C/D）
- 多分类任务

### 架构设计

```
Residual Vector [B, num_vars]
    ↓
┌─────────────────────────────────────────────────────────┐
│ Anomaly Type Classification Head                        │
│                                                          │
│  Input: residual_vec [B, num_vars]                      │
│    ↓                                                     │
│  Feature Extraction                                      │
│    - Linear/MLP提取residual特征                          │
│    ↓                                                     │
│  Classification Head                                     │
│    - Multi-class classifier (4类: A/B/C/D)             │
│    ↓                                                     │
│  Output: anomaly_type_logits [B, 4]                     │
│         anomaly_type_probs [B, 4]                       │
└─────────────────────────────────────────────────────────┘
    ↓
Anomaly Type Prediction [B, 1] (0/1/2/3 → A/B/C/D)
```

### 代码结构

```python
class AnomalyTypeClassifier(nn.Module):
    """异常类型分类器"""
    def __init__(self, num_vars, num_types=4, d_model=64):
        super().__init__()
        self.num_types = num_types  # A, B, C, D
        
        # Feature extraction from residual
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_vars, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_types)
        )
        
    def forward(self, residual_vec):
        """
        Args:
            residual_vec: [B, num_vars] - residual向量
        Returns:
            logits: [B, num_types] - 异常类型logits
        """
        features = self.feature_extractor(residual_vec)
        logits = self.classifier(features)
        return logits
```

### 异常类型定义（需要确认）

需要定义异常类型 A/B/C/D 的含义，可能的定义方式：

1. **基于residual模式**：
   - Type A: 单变量异常（只有目标变量residual大）
   - Type B: 多变量异常（多个变量residual同时大）
   - Type C: 传播异常（某个变量异常导致其他变量异常）
   - Type D: 系统性异常（所有变量residual都大）

2. **基于因果关系**：
   - Type A: 源变量异常（因果链起点异常）
   - Type B: 中间变量异常（因果链中间异常）
   - Type C: 目标变量异常（因果链终点异常）
   - Type D: 多路径异常（多个因果路径同时异常）

3. **基于业务逻辑**（COVID场景）：
   - Type A: 病例异常（NewCases相关变量异常）
   - Type B: 死亡异常（NewDeaths相关变量异常）
   - Type C: 政策异常（Stringency_Index等政策变量异常）
   - Type D: 医疗系统异常（Hosp_Count等医疗变量异常）

**需要用户确认异常类型的定义方式！**

## 四、完整训练流程设计

### Track C 训练流程

```python
# 1. 训练基础预测模型 (Track A)
pred_model = train_track_a(...)  # 复用Track A的训练

# 2. 计算Residual
with torch.no_grad():
    pred_all = pred_model(X)
    residual_vec = |X_next - pred_all|  # [B, num_vars]

# 3. 计算Granger因果关系矩阵（在训练集上）
granger_analyzer = GrangerCausalityAnalyzer(num_vars)
causal_matrix = granger_analyzer.compute_granger_matrix(X_train)

# 4. 训练结构异常检测器
structural_detector = StructuralAnomalyDetector(num_vars)
# 使用residual和异常标签训练
loss = BCE(structural_detector(residual_vec, causal_matrix), y_anomaly)
```

### Track D 训练流程

```python
# 1. 训练基础预测模型 (Track A)
pred_model = train_track_a(...)  # 复用Track A的训练

# 2. 计算Residual
with torch.no_grad():
    pred_all = pred_model(X)
    residual_vec = |X_next - pred_all|  # [B, num_vars]

# 3. 生成异常类型标签（需要实现）
# 方式1: 基于residual模式自动标注
# 方式2: 手动标注
anomaly_type_labels = create_anomaly_type_labels(residual_vec, ...)  # [B, 1] (0/1/2/3)

# 4. 训练异常类型分类器
type_classifier = AnomalyTypeClassifier(num_vars, num_types=4)
# 使用residual和异常类型标签训练
loss = CrossEntropy(type_classifier(residual_vec), anomaly_type_labels)
```

## 五、集成到现有代码结构

### 文件组织

```
model/
├── encoder.py           # Transformer encoder (已有)
├── trainer.py           # Track A/B训练 (已有)
├── track_c.py           # Track C实现 (新建)
│   ├── GrangerCausalityAnalyzer
│   └── StructuralAnomalyDetector
├── track_d.py           # Track D实现 (新建)
│   └── AnomalyTypeClassifier
└── trainer_cd.py        # Track C/D训练函数 (新建)
```

### 训练函数接口设计

```python
def train_track_c(
    bundle_coarse,
    bundle_fine=None,
    pred_model_path,  # Track A训练好的模型路径
    save_path=None,
    ...
):
    """训练Track C模型"""
    pass

def train_track_d(
    bundle_coarse,
    bundle_fine=None,
    pred_model_path,  # Track A训练好的模型路径
    save_path=None,
    num_types=4,  # 异常类型数量
    ...
):
    """训练Track D模型"""
    pass
```

## 六、待确认问题

1. **异常类型定义**：Type A/B/C/D 的具体含义是什么？
2. **标签生成**：如何生成异常类型的标签？自动标注还是手动标注？
3. **Granger因果关系**：
   - 使用统计方法（statsmodels）还是神经网络学习？
   - 是否需要在训练集上预计算因果矩阵？
4. **可解释性**：Track C需要提供哪些可解释性信息？
   - 因果图可视化？
   - 关键异常变量识别？
   - 异常传播路径？
5. **评估指标**：
   - Track C: 使用ROC-AUC、AUPRC等二分类指标
   - Track D: 使用准确率、F1-macro等多分类指标

## 七、下一步行动

1. 确认异常类型定义（A/B/C/D）
2. 实现Granger因果关系分析模块（Track C）
3. 实现异常类型分类器（Track D）
4. 设计标签生成策略
5. 集成到训练流程

