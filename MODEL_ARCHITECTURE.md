# Model Architecture Documentation

## Overview

The model consists of two main stages:

- **Stage 1**: Multi-scale time series forecasting model that learns to predict future values and produces prediction residuals
- **Stage 2**: Anomaly detection module that leverages causal structure representations and residuals from Stage 1

The entire pipeline is designed to be general-purpose, working with any multivariate time series dataset that can be organized into sequences with labels.

---

## Stage 1: Forecasting Model

### Purpose
Train a multi-scale transformer-based forecasting model to predict future values of multivariate time series. The model learns variable relationships through a shared relation matrix and produces residuals that serve as input for Stage 2.

### Core Components

#### 1. Model Architecture (`model/encoder.py`)

**`TransformerSeqEncoder`** (lines 5-61)
- Encodes time series sequences using Transformer architecture
- Input: `(batch, seq_len, num_vars)` tensor
- Output: `(batch, d_model)` hidden representation
- Components:
  - Input projection: Linear + LayerNorm + GELU + Dropout
  - Positional encoding: Sinusoidal positional embeddings
  - Transformer encoder: Multi-layer self-attention
  - Output projection: Linear + GELU + Dropout
  - Pooling: `last` (final timestep) or `mean` (average pooling)

**`RegressionHeadWithRelation`** (lines 64-87)
- Prediction head that incorporates variable relationships
- Learns a shared relation matrix: `var_relation_matrix` (num_vars × num_vars)
- Separate projection networks for fine and coarse scales
- Forward pass:
  1. Projects hidden representation to variable-specific bases: `base_proj_fine` / `base_proj_coarse`
  2. Multiplies bases with relation matrix: `pred = var_base @ relation_matrix.T`
  3. Returns predictions for both fine and coarse scales

#### 2. Training (`model/trainer.py`)

**`train_ours`** (lines 33-427)
- Main training function for Stage 1 models
- Supports two modes:
  - **Coarse-only**: Single-scale training (only coarse scale)
  - **Multi-scale**: Dual-scale training (fine + coarse scales)
- Training process:
  1. Initializes encoder(s) and regression head
  2. Optimizes MSE loss between predictions and targets
  3. For multi-scale: trains both fine and coarse encoders with shared head
  4. Tracks validation F1 score for model selection
  5. Saves model checkpoint with config and state dicts
- Output: Saved model checkpoint (.pt file) containing:
  - `enc_coarse` / `enc_fine`: Encoder state dicts
  - `head`: Regression head state dict
  - `config`: Hyperparameters (d_model, nhead, num_layers, pooling, coarse_only)

**`FocalLoss`** (lines 9-25)
- Focal loss for classification tasks (optional, currently not used in main pipeline)

**`to_device`** (lines 28-30)
- Utility function to move data dictionary to specified device

#### 3. Evaluation (`model/evaluator.py`)

**`infer`** (lines 44-183)
- Inference function for Stage 1 models
- Loads trained model and computes residuals on input data
- Returns:
  - `residual`: Prediction residuals (absolute difference between true and predicted values)
  - `y_true`: Ground truth labels
  - `idx_val`, `idx_test`: Validation and test indices
- Supports post-processing for multi-scale models (fuses fine and coarse residuals)

**`evaluate`** (lines 29-41)
- Evaluates residuals using standard metrics
- Finds optimal threshold on validation set
- Computes metrics on test set:
  - Threshold, F1, Precision, Recall, AUPRC, ROC-AUC

**`residualthreshold`** (lines 21-26)
- Finds optimal threshold by maximizing F1 score on validation set

**`threshold`** (lines 15-18)
- Generic threshold finder for probability scores

**`anomaly`** (lines 9-12)
- Computes anomaly scores using z-score normalization

---

## Stage 2: Anomaly Detection

### Purpose
Detect anomalies by analyzing both spatial (causal structure) and temporal (residual magnitude) deviations from normal patterns.

### Core Components

#### 1. Causal Analysis (`model/causal.py`)

**`compute_gradient_causal_matrix`** (lines 3-38)
- Computes causal matrices using gradient-based method (alternative approach, not used in main pipeline)
- Uses gradients of predictions w.r.t. inputs to estimate causal effects

**`sparsify`** (lines 41-67)
- Sparsifies causal matrices by removing weak connections
- Removes diagonal (self-connections)
- Removes weaker directional edges
- Applies percentile-based thresholding

**`aggregate_normal_reference`** (lines 70-76)
- Aggregates causal matrices from normal training samples
- Methods: median or mean aggregation

**`compute_spatial_deviation`** (lines 79-82)
- Computes Frobenius norm of difference between test causal matrix and normal reference
- Measures structural changes in variable relationships

**`compute_normal_temporal_stats`** (lines 85-90)
- Computes median and MAD (Median Absolute Deviation) of residuals from normal training samples

**`compute_temporal_deviation`** (lines 93-97)
- Computes MAD-based deviation score for residuals
- Normalizes by training normal statistics

**`normalize_scores`** (lines 100-103)
- Min-max normalization of scores

**`fuse_scores`** (lines 106-111)
- Combines spatial and temporal deviation scores with weighted average

#### 2. Stage 2 Validation (`scripts/validate_stage2_minimal.py`)

**`extract_causal_matrix_and_residual`** (lines 15-98)
- Extracts per-window causal matrix representations and residuals from Stage 1 model
- Process:
  1. Loads Stage 1 model checkpoint
  2. For each window:
     - Encodes sequence to hidden representation
     - Projects to variable bases using `base_proj_coarse`
     - Computes per-window causal matrix: `var_base.unsqueeze(-1) * relation_matrix.T`
     - Computes residual from prediction error
  3. Returns causal matrices, residuals, labels, and indices

**`compute_anomaly_scores`** (lines 147-210)
- Computes anomaly scores for all windows
- Steps:
  1. Builds reference causal matrix from normal training samples (median aggregation)
  2. For each window:
     - Computes spatial deviation (Frobenius norm from reference)
     - Computes temporal deviation (MAD-based residual deviation)
     - Combines into anomaly score: `spatial_weight * spatial_dev + temporal_weight * temporal_dev`
- Returns scores, spatial deviations, temporal deviations, and reference statistics

**`evaluate_stage2`** (lines 213-295)
- Evaluates Stage 2 anomaly detection performance
- Uses 99th percentile of training normal scores as threshold
- Computes metrics: AUPRC, ROC-AUC, effect size (attack median / normal median)
- Returns comprehensive evaluation results

**Main function** (lines 298-502)
- Command-line interface for Stage 2 validation
- Supports both SWAT and PSM datasets (auto-detection)
- Configurable parameters:
  - Model path, data directory, dataset type
  - Sparsify threshold, spatial/temporal weights
  - Batch size, device
- Outputs detailed evaluation report

---

## Data Processing Utilities

### 1. Data Loading (`utils/data_utils.py`)

**`loader`** (lines 26-45)
- General-purpose data loader
- Reads Excel/CSV files, handles date columns, removes location columns
- Returns cleaned DataFrame

**`engineer_features`** (lines 47-106)
- Feature engineering functions
- Computes moving averages, growth rates, return rates
- Dataset-specific features (can be customized per dataset)

**`group_by_state`** (lines 66-106)
- Groups data by entity (e.g., state, sensor) and frequency
- Aggregates features according to aggregation map
- Returns dictionary: `{entity: DataFrame}`

**`agg_map`** (lines 3-24)
- Aggregation map for different feature types
- Defines how to aggregate features when resampling (sum, mean, last)

### 2. Sequence Creation (`utils/sequences.py`)

**`create_sequences`** (lines 5-52)
- Creates sliding window sequences from time series data
- Input: Dictionary of entity dataframes, feature columns, window size
- Output: List of sequence dictionaries with:
  - `X_seq`: Input sequence (window_size × num_features)
  - `X_next`: Target values
  - `y_next`: Labels
  - `state_id`, `target_time_id`: Metadata

**`split_sequences`** (lines 54-70)
- Splits sequences into train/val/test sets
- Maintains temporal order (no shuffling)
- Returns indices for each split

**`sequences_to_bundle`** (lines 72-111)
- Converts sequence list to PyTorch bundle format
- Stacks sequences into tensors
- Creates train/val/test indices
- Returns dictionary with:
  - `X_seq`: (N, window_size, num_features)
  - `X_next`: (N, num_features)
  - `y_next`: (N,)
  - `idx_train`, `idx_val`, `idx_test`: Indices

**`standardize`** (lines 113-141)
- Standardizes features using mean and std from training set
- Stores statistics for inference-time normalization

### 3. Label Creation (`utils/labels.py`)

**`create_outbreak_labels`** (lines 4-61)
- Creates anomaly labels based on multiple criteria
- Combines return-based, level-based, and trend-based detection
- General-purpose function that can be adapted to different labeling schemes

### 4. Multi-Scale Mapping (`utils/mapping.py`)

**`compute_fine_coarse_mapping`** (lines 6-42)
- Computes mapping from fine-scale to coarse-scale sequences
- Matches sequences based on entity and temporal proximity
- Returns mapping tensor: `fine_to_coarse_index`

**`add_mapping`** (lines 45-48)
- Adds fine-to-coarse mapping to fine-scale bundle
- Required for multi-scale training

### 5. Utility Functions (`utils/utils.py`)

- KL divergence computation
- Random seed setting
- Sliding window utilities
- Causal structure evaluation metrics
- POT (Peaks Over Threshold) method for anomaly detection
- Top-K evaluation utilities

---

## Pipeline Structure

### Data Flow

```
Raw Data (CSV/Excel)
    ↓
[utils/data_utils.py: loader, engineer_features, group_by_state]
    ↓
Entity-grouped DataFrames
    ↓
[utils/sequences.py: create_sequences]
    ↓
Sequence List
    ↓
[utils/sequences.py: split_sequences, sequences_to_bundle]
    ↓
Data Bundle (PyTorch format)
    ↓
[model/trainer.py: train_ours]  ← Stage 1 Training
    ↓
Trained Model (.pt file)
    ↓
[model/evaluator.py: infer]  ← Stage 1 Inference
    ↓
Residuals + Ground Truth
    ↓
[scripts/validate_stage2_minimal.py: extract_causal_matrix_and_residual]  ← Stage 2
    ↓
Causal Matrices + Residuals
    ↓
[scripts/validate_stage2_minimal.py: compute_anomaly_scores, evaluate_stage2]
    ↓
Anomaly Scores + Evaluation Metrics
```

### Multi-Scale Pipeline

For multi-scale models:

```
Fine-scale Data → Fine Bundle
Coarse-scale Data → Coarse Bundle
    ↓
[utils/mapping.py: add_mapping]
    ↓
Coarse Bundle (with mapping)
Fine Bundle (with fine_to_coarse_index)
    ↓
[model/trainer.py: train_ours (coarse_only=False)]
    ↓
Trained Multi-Scale Model
    ↓
[model/evaluator.py: infer] (uses both bundles)
    ↓
Coarse Residuals (used for Stage 2)
```

---

## General vs Dataset-Specific Code

### General (Reusable) Modules

All code in the following directories is **general-purpose**:

- **`model/`**: Core model architecture and training/evaluation functions
- **`utils/`**: General data processing utilities
- **`scripts/validate_stage2_minimal.py`**: General Stage 2 validation (supports multiple datasets)

### Dataset-Specific Scripts

The following scripts are dataset-specific and should be refactored or replaced with general versions:

- **`scripts/train_psm_stage1.py`**: PSM-specific training script
  - Should be replaced with general `scripts/train_stage1.py` that accepts dataset type as parameter

- **`scripts/train_swat_stage1.py`**: SWAT-specific training script
  - Should be merged into general training script

- **`scripts/process_psm.py`**: PSM-specific data processing
  - Should be replaced with general `scripts/process_data.py`

- **`scripts/process_swat.py`**: SWAT-specific data processing
  - Should be merged into general processing script

- **`pipeline/pipeline.py`**: COVID-specific pipeline (contains hardcoded feature names)
  - Contains `DEFAULT_FEATURES` list specific to COVID dataset
  - Should be made configurable

### Notes

- Dataset-specific scripts should be consolidated into general scripts with command-line arguments for dataset configuration
- Feature engineering should be made configurable through parameter files or function arguments
- All model checkpoints in `models/` are preserved and not deleted

---

## File Structure Summary

```
model/
├── encoder.py              # Core model architecture (TransformerSeqEncoder, RegressionHeadWithRelation)
├── trainer.py              # Stage 1 training (train_ours, FocalLoss, to_device)
├── evaluator.py            # Stage 1 evaluation (infer, evaluate, threshold functions)
├── causal.py               # Causal analysis utilities (spatial deviation computation)
├── data_pipeline.py        # Dataset construction (build_dataset)
├── patchtst.py             # PatchTST baseline model
├── dlinear.py              # DLinear baseline model
├── lstm.py                 # LSTM baseline model
├── AERCA.py                # AERCA baseline model
├── trainer_c.py            # Alternative training function
├── trainer_unified.py      # Unified training function
└── evaluator_c.py          # Alternative evaluation function

utils/
├── data_utils.py           # Data loading and feature engineering (loader, engineer_features, group_by_state)
├── sequences.py            # Sequence creation and bundling (create_sequences, split_sequences, sequences_to_bundle)
├── labels.py               # Label creation (create_outbreak_labels)
├── mapping.py              # Multi-scale mapping (compute_fine_coarse_mapping, add_mapping)
└── utils.py                # General utilities (KL divergence, POT, evaluation metrics)

scripts/
├── validate_stage2_minimal.py  # Stage 2 validation (GENERAL - supports multiple datasets)
├── train_psm_stage1.py         # PSM-specific Stage 1 training (TO BE REFACTORED)
├── train_swat_stage1.py        # SWAT-specific Stage 1 training (TO BE REFACTORED)
├── process_psm.py              # PSM-specific data processing (TO BE REFACTORED)
├── process_swat.py             # SWAT-specific data processing (TO BE REFACTORED)
└── [other analysis scripts]    # Various analysis and baseline scripts

pipeline/
└── pipeline.py             # COVID-specific pipeline (contains hardcoded features - TO BE MADE GENERAL)
```

---

## Key Design Principles

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Generality**: Core modules work with any multivariate time series data
3. **Multi-scale Support**: Built-in support for fine and coarse scale modeling
4. **Separation of Concerns**: Stage 1 (forecasting) and Stage 2 (anomaly detection) are clearly separated
5. **Configurability**: Model hyperparameters are stored in checkpoints for reproducibility
6. **Extensibility**: Easy to add new baseline models or evaluation metrics

---

## Usage Example

### Stage 1 Training

```python
from model.trainer import train_ours

# Load data bundles
bundle_coarse = torch.load('data/processed/coarse.pt')
bundle_fine = torch.load('data/processed/fine.pt')  # Optional

# Train model
model_path = train_ours(
    bundle_coarse=bundle_coarse,
    bundle_fine=bundle_fine,  # None for coarse-only
    save_path='models/stage1_model.pt',
    coarse_only=False,  # True for single-scale
    epochs=50,
    d_model=64,
    nhead=4,
    num_layers=2,
    device='cuda'
)
```

### Stage 1 Evaluation

```python
from model.evaluator import infer, evaluate

# Inference
result = infer(
    model_path='models/stage1_model.pt',
    bundle_coarse=bundle_coarse,
    bundle_fine=bundle_fine,
    device='cuda'
)

# Evaluation
metrics = evaluate(
    result['residual'],
    result['y_true'],
    result['idx_val'],
    result['idx_test']
)
```

### Stage 2 Validation

```bash
python scripts/validate_stage2_minimal.py \
    --model_path models/stage1_model.pt \
    --data_dir data/processed \
    --dataset auto \
    --spatial_weight 0.5 \
    --temporal_weight 0.5 \
    --batch_size 32 \
    --device cuda
```

---

## Future Refactoring Recommendations

1. **Consolidate training scripts**: Merge `train_psm_stage1.py` and `train_swat_stage1.py` into a single general `train_stage1.py`
2. **Consolidate processing scripts**: Merge `process_psm.py` and `process_swat.py` into a single general `process_data.py`
3. **Make pipeline.py general**: Remove hardcoded feature lists, make it configurable
4. **Configuration files**: Use YAML/JSON config files for dataset-specific settings
5. **Unified CLI**: Create a main CLI tool that handles all stages of the pipeline

