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
├── model/              # Model implementations (Transformer encoder, prediction heads)
├── utils/              # Utility functions (data loading, sequence creation, mapping)
├── pipeline/           # Data pipeline
├── scripts/            # Training and evaluation scripts
│   ├── run_baselines.py    # Run baseline models
│   └── run_ablation.sh     # Run ablation studies
└── notebooks/          # Analysis notebooks with results
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data:
   - Place data files in `data/` directory
   - Processed data files should be in `data/processed/`

3. Run experiments:

### COVID-19 Dataset:
```bash
# Ablation study on 6-feature dataset
bash scripts/run_ablation.sh 6feat all

# Run baseline models
python scripts/run_baselines.py --data_path data/processed/week_21feat.pt --models baselines
```

### PSM Dataset:
```bash
# Process PSM data into fine/coarse scales
python scripts/process_psm.py --data_dir data/PSM/PSM --output_dir data/PSM/processed --k 5

# Train coarse-only baseline (sanity check)
python scripts/train_psm_stage1.py --data_dir data/PSM/processed --epochs 50 --batch_size 64

# Train multi-scale model
python scripts/train_psm_multiscale.py --data_dir data/PSM/processed --epochs 50 --batch_size 64 --use_lu --lambda_u 1.0
```

## Results

Experimental results are documented in `notebooks/results.ipynb`, including:
- Ablation studies on multi-scale design and L_u constraint
- Baseline model comparisons
- Performance metrics (F1, AUPRC, ROC-AUC) on 6-feature and 21-feature datasets

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
- Processing script: `scripts/process_psm.py`

## Citation

If you use this code, please cite our work.

