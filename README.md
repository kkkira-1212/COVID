# COVID-19 Anomaly Detection

Multi-scale time series anomaly detection for COVID-19 outbreak prediction.

## Project Structure

```
├── model/              # Model implementations
├── utils/              # Utility functions
├── pipeline/           # Data pipeline
├── scripts/            # Training and evaluation scripts
├── notebooks/          # Analysis notebooks
└── docs/               # Documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data:
   - Place data files in `data/` directory
   - Processed data files should be in `data/processed/`

## Usage

### Run ablation study:
```bash
bash scripts/run_ablation.sh [6feat|21feat] [all|weekonly|multiscale|lambda]
```

### Run baseline models:
```bash
python scripts/run_baselines.py --data_path data/processed/week_21feat.pt --models [model_name]
```

## Data

Data files are not included in this repository due to size constraints. Please download them separately and place in the `data/` directory.

## Models

Trained models are not included in this repository. Models can be trained using the provided scripts.

