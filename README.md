# Multi-Scale Time Series Anomaly Detection

Multi-scale time series anomaly detection framework supporting COVID-19 outbreak prediction and PSM anomaly detection.

## Features

- **Multi-scale architecture**: Leverages fine and coarse temporal scales (e.g., daily/weekly for COVID-19, fine/coarse aggregation for PSM)
- **L_u constraint loss**: Aligns residuals between fine and coarse scale predictions
- **Comprehensive baselines**: PatchTST, DLinear, LSTM, AERCA
- **Ablation studies**: Systematic evaluation of design components
- **PSM dataset support**: Fine/coarse scale processing with explicit mapping (coarse_t ↔ fine_[t*k : (t+1)*k])

## Project Structure

```
├── model/              # Core model and causal scoring logic
├── utils/              # Shared utilities (I/O, causal eval, sequence tools)
├── experiments/        # Dataset-specific utilities and side branches
├── data/processing/    # Dataset processing scripts
├── scripts/            # Main training and evaluation entrypoints
└── notebooks/          # Analysis notebooks with results
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data:
   - Place raw data files under `data/`
   - Run processing scripts in `data/processing/` to generate `.pt` bundles

3. Run experiments:

### Stage-1 training:
```bash
python scripts/train_stage1.py \
  --data_dir data/SWaT/processed \
  --coarse_file swat_coarse.pt \
  --fine_file swat_fine.pt \
  --model_save_path models/swat_stage1.pt
```

### Causal stability evaluation:
```bash
python scripts/verify_causal_stability.py \
  --model_path models/swat_stage1.pt \
  --data_dir data/SWaT/processed \
  --data_file data/SWaT/processed/swat_fine.pt \
  --scale fine \
  --global_threshold_percentile 10
```

### Fusion sweep:
```bash
python scripts/sweep_causal_fusion.py \
  --model_path models/swat_stage1.pt \
  --data_dir data/SWaT/processed \
  --data_file data/SWaT/processed/swat_fine.pt \
  --scale fine \
  --global_threshold_percentile 10 \
  --fusion_normalize --struct_metric relative \
  --fusion_mode relu_gate \
  --sweep_relu \
  --relu_tau_percentile 90 \
  --lambda_start 0 --lambda_end 1 --lambda_step 0.1 \
  --save_dir analysis
```

## Results

Experimental results are documented in `notebooks/`, including:
- Performance metrics (F1, AUPRC, ROC-AUC)

## Data & Models

- **Data files** are not included due to size constraints. Please prepare data separately:
  - COVID-19: Place Excel files in `data/` directory
  - PSM: Place CSV files (`train.csv`, `test.csv`, `test_label.csv`) in `data/PSM/PSM/`
- **Trained models** are not included. Models can be trained using the provided scripts.
- **Processed data** (`.pt` files) are excluded from git but can be generated using processing scripts.

## PSM Dataset Details

- **Fine scale**: Original sequences (1-step, e.g., per minute)
- **Coarse scale**: Aggregated sequences (k fine steps → 1 coarse step, default k=5)
- **Mapping**: `coarse_t ↔ fine_[t*k : (t+1)*k]` (explicit index mapping)
- Processing script: `data/processing/process_psm.py`

## Citation

If you use this code, please cite our work.

