# 对话总结：阶段一向量预测实现

## 已完成的工作

### 1. 数据生成修复
- ✅ 修复了`standardize()`函数，现在同时标准化`X_seq`和`X_next`
- ✅ 修复了周频聚合后工程特征丢失问题（`NewCases_MA7`, `NewDeaths_MA7`, `Cases_GrowthRate`, `NewDeaths_return`）
- ✅ 修复了周频聚合后`outbreak_label`丢失问题
- ✅ 删除了`Vax_Dose4`和经纬度常量
- ✅ 处理了Inf/NaN值（特征工程和标准化中）
- ✅ 重新生成了所有数据包（`day_21feat.pt`, `week_21feat.pt`, `day_6feat.pt`, `week_6feat.pt`）

### 2. 阶段一代码实现
- ✅ `RegressionHead`输出维度：`1` → `num_vars`（预测所有变量）
- ✅ `trainer.py`：计算向量residual，提取目标变量residual用于评估
- ✅ `evaluator.py`：修复了`RegressionHead`缺少`num_vars`参数的问题
- ✅ `evaluator.py`：修复了Tensor布尔值判断问题（`or`操作符）
- ✅ `run_baselines.py`：修复了数据加载，添加了`X_next`和`feature_cols`的加载

### 3. 阶段一设计
- **模型预测**：所有变量（向量预测）`[batch, num_vars]`
- **Residual计算**：计算所有变量的residual向量 `residual_vec = |X_next - pred_all|`
- **Residual使用**：只提取目标变量的residual `residual = residual_vec[:, target_idx]`
- **评估**：使用标量residual计算ROC-AUC（只针对目标变量`NewDeaths_return`）

## 当前状态

### 数据质量
- ✅ 无NaN/Inf值
- ✅ 标签分布正常（日频1.5%，周频11.5%）
- ✅ 所有特征都有变化（无常量特征）
- ⚠️ Week数据有极端值：`NewDeaths_ret_next`最大值997（7个样本，0.24%）

### 代码状态
- ✅ 所有相关文件已修改为向量预测
- ✅ 评估逻辑正确（使用目标变量的residual）
- ✅ 数据加载已修复（包含`X_next`和`feature_cols`）

### 6个特征列表
```python
FEATURES_6 = [
    'NewCases', 'NewDeaths',
    'NewCases_MA7', 'NewDeaths_MA7',
    'Cases_GrowthRate', 'NewDeaths_return'
]
```

## 待解决的问题

### 问题1：Multiscale模型效果差（ROC-AUC: 0.4837）
**可能原因**：
1. Week数据样本少（2860 vs Day的20280）
2. Week数据有极端值（`NewDeaths_ret_next`最大值997）
3. 评估时只用week的residual，但week数据质量可能不如day

**建议**：
- 先运行week-only模型验证阶段一
- 如果week-only效果好，说明问题在multiscale的评估逻辑
- 如果week-only效果也差，说明week数据本身有问题

### 问题2：Lambda_u选择
- 默认值：`lambda_u=1.0`
- 测试过的值：`0.3, 0.5, 1.0, 1.5`
- 阶段一建议：使用默认值`1.0`

## 关键代码修改

### 1. `utils/sequences.py`
- `standardize()`：添加了`X_next`的标准化

### 2. `utils/data_utils.py`
- `agg_map`：删除了`Vax_Dose4`
- `loader()`：删除经纬度列
- `group_by_state()`：周频聚合后重新计算工程特征和`outbreak_label`

### 3. `model/encoder.py`
- `RegressionHead`：输出维度改为`num_vars`

### 4. `model/trainer.py`
- Weekly-only和Multi-scale模式：使用向量预测，提取目标变量residual
- 修复了Tensor布尔值判断问题

### 5. `model/evaluator.py`
- 修复了`RegressionHead`缺少`num_vars`参数
- 修复了Tensor布尔值判断问题
- 使用向量residual提取目标变量

### 6. `scripts/run_baselines.py`
- `load_bundle_week()`和`load_bundle_day()`：添加了`X_next`和`feature_cols`的加载

## 运行命令

### 阶段一验证（Week-only）
```bash
python scripts/run_baselines.py \
    --data_path data/processed/week_6feat.pt \
    --models ours_weekonly \
    --epochs 200 \
    --save_dir models/stage1
```

### 阶段一验证（Multiscale）
```bash
python scripts/run_baselines.py \
    --data_path data/processed/week_6feat.pt \
    --data_path_day data/processed/day_6feat.pt \
    --models ours_multiscale \
    --lambda_u 1.0 \
    --epochs 200 \
    --save_dir models/stage1
```

### 21特征数据集
```bash
# Week-only
python scripts/run_baselines.py \
    --data_path data/processed/week_21feat.pt \
    --models ours_weekonly \
    --epochs 200

# Multiscale
python scripts/run_baselines.py \
    --data_path data/processed/week_21feat.pt \
    --data_path_day data/processed/day_21feat.pt \
    --models ours_multiscale \
    --lambda_u 1.0 \
    --epochs 200
```

## 阶段一 vs 阶段二

| 项目 | 阶段一 | 阶段二（未来） |
|------|--------|----------------|
| 模型预测 | 所有变量 | 所有变量 |
| Residual计算 | `residual_vec = \|X_next - pred_all\|` | `residual_vec = \|X_next - pred_all\|` |
| Residual使用 | `residual_vec[:, target_idx]` | `aggregate(residual_vec)` |
| 评估 | 只用目标变量的residual | 用聚合后的residual |

## 下一步行动

1. **立即**：运行week-only模型验证阶段一
2. **如果week-only效果好**：检查multiscale的评估逻辑
3. **如果week-only效果也差**：检查week数据质量和极端值处理
4. **验证通过后**：进入阶段二（多变量residual聚合）

## 关键文件

- `data/processed/*.pt` - 重新生成的数据包（包含`X_next`）
- `model/trainer.py` - 训练逻辑（向量预测）
- `model/evaluator.py` - 评估逻辑（提取目标变量residual）
- `model/encoder.py` - `RegressionHead`输出`num_vars`
- `utils/sequences.py` - 标准化`X_next`
- `utils/data_utils.py` - 特征工程和聚合修复













