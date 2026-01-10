# Track C Implementation To-Do List

## Part 1: Track A Modification (Foundation)

### 1.1 Modify RegressionHead to Include Variable Relation Matrix

**File**: `model/encoder.py`

**Changes**:
- Rename `RegressionHead` to `RegressionHeadWithRelation`
- Add learnable parameter `var_relation_matrix: nn.Parameter` with shape `[num_vars, num_vars]`
- Modify forward to use relation matrix for variable interaction
- Output: `pred_all [B, num_vars]` (all variables, not just target)

**Implementation**:
```python
class RegressionHeadWithRelation(nn.Module):
    def __init__(self, d_in, num_vars):
        self.var_relation_matrix = nn.Parameter(torch.randn(num_vars, num_vars) * 0.1)
        self.base_proj = nn.Sequential(...)
    
    def forward(self, z):
        var_base = self.base_proj(z)
        pred = var_base @ self.var_relation_matrix.T
        return pred
```

### 1.2 Update Trainer to Use All Variables Residual

**File**: `model/trainer.py`

**Changes**:
- Remove `target_idx` extraction
- Use full `residual_vec [B, num_vars]` instead of single variable residual
- Update loss computation to use all variables
- Update evaluation to work with residual vectors

**Key modifications**:
- Line 275-276: Remove `pred_fine = pred_fine_all[:, target_idx]`
- Use `pred_fine_all` and `pred_coarse_all` directly
- Compute `residual_vec = |X_next - pred_all|.abs()` for all variables

### 1.3 Refactor Function Names

**Files**: All model files

**Changes**:
- Remove excessive underscores
- Use concise names
- Examples:
  - `compute_anomaly` → `anomaly`
  - `find_residual_threshold` → `threshold`
  - `evaluate_residual_scores` → `evaluate`
  - `run_inference` → `infer`

## Part 2: Track C Architecture Construction (Based on GCAD)

### 2.1 Create Gradient-based Causal Matrix Computation Module

**New File**: `model/causal.py`

**Components**:
- `GradientCausalMatrix`: Compute gradient-based Granger causality matrix
- Function: `compute(X_seq, model)` → `causal_matrix [B, num_vars, num_vars]`

**Implementation Steps** (Following GCAD):
1. **Gradient Computation**:
   - For each variable j, compute prediction: `pred_j = model(X_seq)[:, j]`
   - For each variable i, compute gradient: `grad_ij = grad(pred_j, X_seq[:, :, i])`
   - Use `torch.autograd.grad` with `retain_graph=True` for multiple gradients
   
2. **Time Dimension Aggregation**:
   - Aggregate gradients across time dimension (integrate over time lag)
   - Sum or mean over time window: `causal_effect_ij = grad_ij.sum(dim=1)` or `mean(dim=1)`
   
3. **Channel-separated Processing**:
   - Process each variable pair (i, j) separately
   - Build causal matrix: `causal_matrix[i, j] = causal_effect_ij`

4. **Output**: Causal matrix `[B, num_vars, num_vars]` for each sample in batch

**Key Points**:
- Use gradient to reflect how variable i affects prediction of variable j
- Gradient-based method avoids repeated optimization during testing
- Captures both spatial (variable-to-variable) and temporal (time lag) dependencies

### 2.2 Create Sparsification Module (GCAD Method)

**File**: `model/causal.py`

**Function**: `sparsify(causal_matrix, method='symmetry')` → `sparsified_matrix [B, num_vars, num_vars]`

**Implementation** (Following GCAD's symmetry-based sparsification):
1. **Eliminate Bidirectional Edges**:
   - For each pair (i, j) where i != j:
     - If `causal_matrix[i, j] > causal_matrix[j, i]`: keep i→j, set j→i = 0
     - Else: keep j→i, set i→j = 0
   - This removes bidirectional edges and reduces impact of sequence similarity

2. **Optional Threshold Filtering**:
   - Apply threshold to remove weak causal relationships
   - `causal_matrix[abs(causal_matrix) < threshold] = 0`

3. **Output**: Sparsified causal matrix with clearer causal structure

**Purpose**:
- Remove bidirectional edges (symmetry-based)
- Reduce impact of sequence similarity on Granger causality
- Obtain sparser, more interpretable causal graph

### 2.3 Create Normal Reference Matrix Aggregator

**File**: `model/causal.py`

**Function**: `aggregate(causal_matrices, method='median')` → `normal_reference [num_vars, num_vars]`

**Implementation**:
- Input: `causal_matrices [N_train, num_vars, num_vars]` from training set
- Method: **Median aggregation** (robust to outliers, as in GCAD)
  - `normal_reference[i, j] = median(causal_matrices[:, i, j])`
- Alternative: Mean aggregation if data is clean
- Output: Single reference matrix representing normal causal patterns

**Key Points**:
- Use median for robustness (GCAD recommendation)
- Aggregate only from training set (normal data)
- This becomes Matrix ③ (Normal Reference Causal Matrix)

### 2.4 Create Pattern Deviation-based Anomaly Detector Module

**New File**: `model/detector.py`

**Components**:
- `CausalAnomalyDetector`: Compare window-level matrix with normal reference using pattern deviation
- Function: `detect(causal_matrix, normal_matrix)` → `anomaly_score [B]`

**Implementation** (Following GCAD's pattern deviation method):
```python
class CausalAnomalyDetector:
    def detect(self, causal, normal):
        """
        Args:
            causal: [B, num_vars, num_vars] - current sample's causal matrix
            normal: [num_vars, num_vars] - normal reference matrix
        Returns:
            anomaly_score: [B] - deviation-based anomaly scores
        """
        # Method 1: Frobenius norm (basic)
        deviation = torch.norm(causal - normal, p='fro', dim=(-2, -1))
        
        # Method 2: Combined spatial and temporal deviation (GCAD approach)
        # Spatial deviation: variable-to-variable relationship deviation
        spatial_dev = self.compute_spatial_deviation(causal, normal)
        # Temporal deviation: time lag pattern deviation (if temporal info is encoded)
        temporal_dev = self.compute_temporal_deviation(causal, normal)
        # Combined score
        anomaly_score = spatial_dev + temporal_dev
        
        return anomaly_score
    
    def compute_spatial_deviation(self, causal, normal):
        """Compute deviation in spatial (variable-to-variable) dependencies"""
        return torch.norm(causal - normal, p='fro', dim=(-2, -1))
    
    def compute_temporal_deviation(self, causal, normal):
        """Compute deviation in temporal (time lag) dependencies"""
        # If temporal info is encoded in matrix structure, compute accordingly
        # For now, can use same Frobenius norm or specific temporal metric
        return torch.norm(causal - normal, p='fro', dim=(-2, -1))
```

**Key Points** (Following GCAD):
- The causal matrix contains both **spatial dependency** (variable relationships) and **temporal dependency** (time lag effects)
- Pattern deviation combines deviations from both types of dependencies
- Frobenius norm is a good baseline; can extend to separate spatial/temporal components if needed

### 2.5 Integrate Track C into Training Pipeline

**File**: `model/trainer.py` or new `model/trainer_c.py`

**Training Flow** (Following GCAD):
1. **Train prediction model (Track A)**:
   - Train on normal data (training set)
   - Learn normal prediction patterns
   - Save model checkpoint

2. **Compute causal matrices for training set**:
   - For each training sample:
     a. Forward pass: `pred_all = model(X_seq)`
     b. Compute gradient-based causal matrix (Matrix ②)
     c. Apply sparsification to remove bidirectional edges
   - Collect all causal matrices: `causal_matrices_train [N_train, num_vars, num_vars]`

3. **Aggregate normal reference matrix**:
   - Use median aggregation: `normal_reference = median(causal_matrices_train, dim=0)`
   - This becomes Matrix ③ (Normal Reference Causal Matrix)

4. **Save components**:
   - Prediction model checkpoint
   - Normal reference matrix (Matrix ③)
   - Optional: Training causal matrices for analysis

**Inference Flow** (Following GCAD):
1. **Load components**:
   - Load prediction model
   - Load normal reference matrix (Matrix ③)

2. **For each test sample**:
   a. Forward pass: `pred_all = model(X_seq)`
   b. Compute gradient-based causal matrix (Matrix ②)
   c. Apply sparsification (same as training)
   d. Compute pattern deviation: `deviation = ||Matrix ② - Matrix ③||`
   e. Use deviation as `anomaly_score`

3. **Anomaly detection**:
   - Use threshold or ranking to determine anomalies
   - Higher deviation → more likely to be anomaly

### 2.6 Create Evaluation Module for Track C

**File**: `model/evaluator_c.py`

**Functions**:
- `evaluate(anomaly_scores, y_true, idx_val, idx_test)`: Evaluate anomaly detection performance
- Return metrics: ROC-AUC, AUPRC, Precision, Recall
- `threshold(anomaly_scores, y_true, method='roc_auc')`: Find optimal threshold for binary classification

**Implementation**:
- Use pattern deviation scores (from detector) as anomaly scores
- Compare with ground truth labels
- Compute standard anomaly detection metrics

## Part 3: SWaT Data Verification

### 3.1 Check SWaT Data Format

**Files to Check**:
- `data/SWaT/processed/swat_hour.pt`
- `data/SWaT/processed/swat_info.json`
- `scripts/process_swat.py`

**Verification Points**:
1. Data structure matches expected format:
   - `X_seq`, `y_next`, `idx_train/val/test` present
   - Feature columns defined
   - Window size appropriate
2. Label format: Binary (0/1) for anomaly detection
3. Sequence creation logic compatible with Track C

### 3.2 Verify Data Compatibility

**Check**:
- Can load SWaT data with existing loader
- Feature dimensions match model input
- Sequence structure supports gradient computation
- No missing values or format issues

### 3.3 Update Data Loading if Needed

**File**: `utils/data_utils.py` or new loader

**If changes needed**:
- Create SWaT-specific loader (if format differs)
- Ensure compatibility with causal matrix computation
- Verify residual computation works correctly

## Part 4: Code Structure Refactoring

### 4.1 Remove All Comments

**Files**: All Python files in `model/`

**Action**: Delete all comments (Chinese, English, docstrings)

### 4.2 Clean Redundant Code

**Files**: All model files

**Actions**:
- Remove unused variables
- Remove duplicate code blocks
- Simplify variable names (remove unnecessary suffixes)
- Remove debug code

### 4.3 Improve Code Architecture

**Structure**:
```
model/
├── encoder.py          (Transformer encoder, unchanged)
├── head.py             (Prediction heads: RegressionHeadWithRelation, PredictionHeads)
├── causal.py           (Causal matrix computation, aggregation)
├── detector.py         (Anomaly detection logic)
├── trainer.py          (Track A training, refactored)
├── trainer_c.py        (Track C training pipeline)
├── evaluator.py        (Evaluation for Track A/B, refactored)
└── evaluator_c.py      (Evaluation for Track C)
```

### 4.4 Function Naming Conventions

**Rules**:
- Maximum one underscore per name
- Use verbs: `compute`, `aggregate`, `detect`, `evaluate`, `train`, `infer`
- Short, descriptive names
- Examples:
  - `compute_causal_matrix` → `causal`
  - `aggregate_normal_reference` → `aggregate`
  - `find_anomaly_threshold` → `threshold`

## Part 5: Track D Module Extension (Brief)

### 5.1 Design Track D Extension

**Concept**: Add anomaly type classification on top of Track C

**Approach**:
- No new model structure
- Add classification loss based on causal matrix
- Multi-task learning: anomaly detection + type classification
- Types: A/B/C/D (4 classes)

### 5.2 Implementation Plan (Future)

**Components**:
- Type classifier head (shared with Track C model)
- Loss: Cross-entropy for types + existing anomaly loss
- Training: Multi-task objective
- Evaluation: Classification metrics + detection metrics

**Note**: Implementation deferred until Track C is complete and validated

## Implementation Order

1. **Phase 1**: Track A Modification
   - Modify RegressionHead (2.1)
   - Update trainer (2.2)
   - Refactor names (2.3)

2. **Phase 2**: Track C Core Components
   - Create causal.py (2.1, 2.2)
   - Create detector.py (2.3)
   - Create evaluator_c.py (2.5)

3. **Phase 3**: Integration
   - Create trainer_c.py (2.4)
   - Integrate with evaluation pipeline

4. **Phase 4**: Data & Cleanup
   - Verify SWaT data (3.1-3.3)
   - Code cleanup (4.1-4.3)

5. **Phase 5**: Track D (Future)
   - Design and implement when Track C is stable

