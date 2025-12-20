# COVID-19 Anomaly Detection

Multi-scale time series anomaly detection for COVID-19 outbreak prediction using daily and weekly temporal patterns.

## Features

- **Multi-scale architecture**: Leverages both daily and weekly time series data
- **L_u constraint loss**: Aligns residuals between daily and weekly predictions
- **Comprehensive baselines**: PatchTST, DLinear, LSTM, AERCA
- **Ablation studies**: Systematic evaluation of design components

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
```bash
# Ablation study on 6-feature dataset
bash scripts/run_ablation.sh 6feat all

# Run baseline models
python scripts/run_baselines.py --data_path data/processed/week_21feat.pt --models baselines
```

## Results

Experimental results are documented in `notebooks/results.ipynb`, including:
- Ablation studies on multi-scale design and L_u constraint
- Baseline model comparisons
- Performance metrics (F1, AUPRC, ROC-AUC) on 6-feature and 21-feature datasets

## Data & Models

- **Data files** are not included due to size constraints (>400MB). Please prepare data separately.
- **Trained models** are not included. Models can be trained using the provided scripts.

## Citation

If you use this code, please cite our work.

