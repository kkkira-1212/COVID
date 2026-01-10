# Track A 和 Track B 实现流程详解

## 一、数据预处理流程（共同部分）

```
原始数据 (Excel/CSV)
    ↓
1. loader() - 加载数据
    ↓
2. engineer_features() - 特征工程
    ↓
3. create_outbreak_labels() - 生成异常标签 (y_next)
    ↓
4. group_by_state() - 按state分组，分别聚合为Day和Week频率
    ├─→ day_data (日数据，每个state一个DataFrame)
    └─→ week_data (周数据，每个state一个DataFrame)
    ↓
5. create_sequences() - 创建滑动窗口序列
    ├─→ seq_day: List[dict] - 每个元素包含：
    │   ├─ X_seq: [window_size, num_features] - 输入序列
    │   ├─ y_next: int - 异常标签 (0/1)
    │   ├─ return_next: float - 回归目标值
    │   └─ state, state_id, dates等元数据
    └─→ seq_week: List[dict] - 同上
    ↓
6. split_sequences() - 划分训练/验证/测试集 (70%/10%/20%)
    ↓
7. sequences_to_bundle() - 转换为bundle格式
    ├─→ bundle_day:
    │   ├─ X_seq: [N, window_day, num_features] - 序列数据
    │   ├─ y_next: [N] - 异常标签
    │   ├─ NewDeaths_ret_next: [N] - 回归目标
    │   ├─ idx_train, idx_val, idx_test - 索引
    │   └─ 其他元数据
    └─→ bundle_week: 同上结构
    ↓
8. standardize() - 标准化 (按训练集统计量)
    ↓
9. add_mapping() - 添加day到week的映射关系 (可选)
```

## 二、Track A 流程 (无监督模式，use_classification=False)

### 模型架构

```
输入: bundle_day, bundle_week
    ↓
┌─────────────────────────────────────────────────────────┐
│ 两个Encoder (共享结构，参数独立)                          │
│                                                          │
│  enc_fine (Day Encoder)           enc_coarse (Week Encoder) │
│  ┌──────────────────┐             ┌──────────────────┐  │
│  │ TransformerSeqEncoder          │ TransformerSeqEncoder │
│  │                                 │                    │  │
│  │ 1. input_proj:                 │ 1. input_proj:     │  │
│  │    [B, T, F] → [B, T, d_model] │    [B, T, F] → ... │  │
│  │                                 │                    │  │
│  │ 2. + pos_encoding (位置编码)    │ 2. + pos_encoding  │  │
│  │                                 │                    │  │
│  │ 3. Transformer Encoder:        │ 3. Transformer Encoder│
│  │    - Multi-head Self-Attention │    - Multi-head Self-Attention│
│  │    - Feed Forward              │    - Feed Forward  │  │
│  │    - Layer Norm                │    - Layer Norm    │  │
│  │    (num_layers层)              │    (num_layers层)  │  │
│  │                                 │                    │  │
│  │ 4. Pooling (last/mean)         │ 4. Pooling         │  │
│  │    [B, T, d_model] → [B, d_model]│  [B, T, d_model] → [B, d_model]│
│  │                                 │                    │  │
│  │ 5. output_proj                 │ 5. output_proj     │  │
│  │                                 │                    │  │
│  └──────────────────┘             └──────────────────┘  │
│         ↓                                    ↓           │
│      z_fine [B, d_model]          z_coarse [B, d_model] │
└─────────────────────────────────────────────────────────┘
         ↓                                    ↓
┌─────────────────────────────────────────────────────────┐
│  RegressionHead (Track A使用)                            │
│                                                          │
│  ┌──────────────────┐             ┌──────────────────┐  │
│  │ regressor_fine   │             │ regressor_coarse │  │
│  │                  │             │                  │  │
│  │ Linear(d_in, d_in)│            │ Linear(d_in, d_in)│  │
│  │ ReLU             │             │ ReLU             │  │
│  │ Dropout(0.2)     │             │ Dropout(0.2)     │  │
│  │ Linear(d_in, num_vars)│        │ Linear(d_in, num_vars)││
│  │                  │             │                  │  │
│  └──────────────────┘             └──────────────────┘  │
│         ↓                                    ↓           │
│  pred_fine_all [B, num_vars]   pred_coarse_all [B, num_vars]│
│         ↓                                    ↓           │
│  pred_fine = pred_fine_all[:, target_idx]   │           │
│                                    pred_coarse = pred_coarse_all[:, target_idx]│
└─────────────────────────────────────────────────────────┘
```

### 训练流程

```python
# Track A 训练循环 (use_classification=False)

for epoch in range(epochs):
    # Forward pass
    z_fine = enc_fine(X_day)      # [B, d_model]
    z_coarse = enc_coarse(X_week)  # [B, d_model]
    
    # Head forward (RegressionHead)
    pred_fine_all, pred_coarse_all = heads(z_fine, z_coarse)
    pred_fine = pred_fine_all[:, target_idx]      # 提取目标变量
    pred_coarse = pred_coarse_all[:, target_idx]
    
    # Loss计算 (仅使用MSE回归损失)
    L_reg_fine = MSE(pred_fine[train_idx], NewDeaths_ret_next_day[train_idx])
    L_reg_coarse = MSE(pred_coarse[train_idx], NewDeaths_ret_next_week[train_idx])
    loss = L_reg_fine + L_reg_coarse
    
    # 可选: 添加L_u损失 (多尺度一致性损失)
    if use_lu:
        u_fine = |NewDeaths_ret_next_day - pred_fine|  # residual
        u_coarse = |NewDeaths_ret_next_week - pred_coarse|
        L_u = SmoothL1Loss(u_fine[mapped], u_coarse[map_to_coarse])
        loss = loss + lambda_u * L_u
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Validation (评估)
    with torch.no_grad():
        # 计算residual作为异常分数
        residual_coarse = |NewDeaths_ret_next_week - pred_coarse|
        # 使用验证集的标签计算ROC-AUC
        roc_auc = roc_auc_score(y_week[val_idx], residual_coarse[val_idx])
```

### Track A 关键点

1. **Head结构**: 使用 `RegressionHead`，只有一个regressor
2. **损失函数**: 仅使用MSE回归损失
3. **训练**: 不直接使用异常标签，只预测下一个时间步的值
4. **异常检测**: 通过预测误差(residual)判断异常，误差大=异常

## 三、Track B 流程 (监督模式，use_classification=True)

### 模型架构

```
输入: bundle_day, bundle_week
    ↓
┌─────────────────────────────────────────────────────────┐
│ 两个Encoder (与Track A相同)                              │
│  enc_fine (Day Encoder)           enc_coarse (Week Encoder) │
│  ... (同Track A) ...             ... (同Track A) ...    │
└─────────────────────────────────────────────────────────┘
         ↓                                    ↓
      z_fine [B, d_model]          z_coarse [B, d_model]
         ↓                                    ↓
┌─────────────────────────────────────────────────────────┐
│  PredictionHeads (Track B使用) ← 关键差异！              │
│                                                          │
│  共享同一个Head，但包含两个分支：                        │
│                                                          │
│  ┌──────────────────────────────────────────────┐      │
│  │ classifier分支 (用于分类损失)                  │      │
│  │                                              │      │
│  │  Linear(d_in, d_in//2)                       │      │
│  │  ReLU                                        │      │
│  │  Dropout(0.2)                                │      │
│  │  Linear(d_in//2, 1)                          │      │
│  │  → logit_fine, logit_coarse [B, 1]          │      │
│  └──────────────────────────────────────────────┘      │
│         ↓                        ↓                      │
│    logit_fine              logit_coarse                 │
│                                                          │
│  ┌──────────────────────────────────────────────┐      │
│  │ regressor分支 (用于回归损失)                   │      │
│  │                                              │      │
│  │  Linear(d_in, d_in)                          │      │
│  │  ReLU                                        │      │
│  │  Linear(d_in, 1)                             │      │
│  │  → pred_fine, pred_coarse [B, 1]            │      │
│  └──────────────────────────────────────────────┘      │
│         ↓                        ↓                      │
│    pred_fine                pred_coarse                 │
│                                                          │
│  注意：两个分支共享同一个输入的embedding (z_fine/z_coarse)│
└─────────────────────────────────────────────────────────┘
```

### 训练流程

```python
# Track B 训练循环 (use_classification=True)

for epoch in range(epochs):
    # Forward pass
    z_fine = enc_fine(X_day)      # [B, d_model]
    z_coarse = enc_coarse(X_week)  # [B, d_model]
    
    # Head forward (PredictionHeads)
    logit_fine, logit_coarse, pred_fine, pred_coarse = heads(z_fine, z_coarse)
    # logit_*: [B, 1] - 分类logits (用于BCE)
    # pred_*: [B, 1] - 回归预测值 (用于MSE)
    
    # 损失计算 (分类损失 + 回归损失)
    
    # 1. 分类损失 (BCE + FocalLoss) - 需要标签！
    L_cls_fine = BCEWithLogitsLoss(logit_fine[train_idx], y_day[train_idx])
    L_cls_coarse = FocalLoss(logit_coarse[train_idx], y_week[train_idx])
    L_cls = 0.6 * L_cls_fine + 0.4 * L_cls_coarse
    
    # 2. 回归损失 (MSE)
    L_reg_fine = MSE(pred_fine[train_idx], NewDeaths_ret_next_day[train_idx])
    L_reg_coarse = MSE(pred_coarse[train_idx], NewDeaths_ret_next_week[train_idx])
    L_reg = L_reg_fine + L_reg_coarse
    
    # 3. 总损失 (加权组合)
    loss = alpha_cls * L_cls + alpha_reg * L_reg
    # 默认: alpha_cls=1.0, alpha_reg=0.1
    
    # 可选: 添加L_u损失 (同Track A)
    if use_lu:
        u_fine = |NewDeaths_ret_next_day - pred_fine|
        u_coarse = |NewDeaths_ret_next_week - pred_coarse|
        L_u = SmoothL1Loss(u_fine[mapped], u_coarse[map_to_coarse])
        loss = loss + lambda_u * L_u
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Validation (评估)
    with torch.no_grad():
        # 使用分类输出
        p_coarse = sigmoid(logit_coarse)  # 异常概率
        f1 = f1_score(y_week[val_idx], (p_coarse[val_idx] >= threshold).astype(int))
```

### Track B 关键点

1. **Head结构**: 使用 `PredictionHeads`，包含两个分支：
   - **classifier**: 输出logits，用于分类损失 (BCE/FocalLoss)
   - **regressor**: 输出预测值，用于回归损失 (MSE)
   
2. **损失函数**: 
   - **分类损失 (L_cls)**: BCE + FocalLoss，需要异常标签
   - **回归损失 (L_reg)**: MSE，预测下一个时间步的值
   - **总损失**: `loss = alpha_cls * L_cls + alpha_reg * L_reg`

3. **训练**: 同时使用异常标签和回归目标，学习区分正常/异常

4. **异常检测**: 直接输出异常概率 (sigmoid(logits))

## 四、关键差异总结

| 特性 | Track A (无监督) | Track B (监督) |
|------|-----------------|----------------|
| **Head类型** | `RegressionHead` | `PredictionHeads` |
| **Head结构** | 只有regressor | classifier + regressor两个分支 |
| **分类损失** | ❌ 无 | ✅ BCE + FocalLoss |
| **回归损失** | ✅ MSE | ✅ MSE |
| **训练是否需要标签** | ❌ 不需要 | ✅ 需要 |
| **异常检测方式** | 预测误差(residual) | 分类概率 |
| **模型名称** | ours_weekonly, ours_multiscale | ours_supervised |

## 五、回答你的问题

**Q: Track B是有俩head吗，一个算MSE一个算BCE的？**

**A:** 不完全准确。Track B使用的是 `PredictionHeads`，它**不是两个独立的head**，而是**一个head包含两个分支**：

1. **classifier分支**: 输出logits → 用于计算BCE/FocalLoss分类损失
2. **regressor分支**: 输出预测值 → 用于计算MSE回归损失

这两个分支共享同一个输入的embedding (z_fine/z_coarse)，但输出不同的值。最终损失是两者的加权组合：
```
loss = alpha_cls * L_cls (BCE) + alpha_reg * L_reg (MSE)
```

所以更准确的说法是：**一个Head，两个输出分支，两个损失函数**。

