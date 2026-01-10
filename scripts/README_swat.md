# SWaT Data Processing

## What it does

`process_swat.py` converts SWaT industrial control system dataset into model-ready format:

1. **Loads data**: Reads SWaT CSV files (normal.csv, attack.csv, or merged.csv)
2. **Selects target feature**: Chooses FIT101 (flow sensor) as regression target
3. **Feature selection**: Selects top 30 features by information content (variance, std, CV)
4. **Time aggregation**: 
   - Minute-level (fine-grained): 1-minute windows
   - Hour-level (coarse-grained): 1-hour windows
5. **Creates sequences**: Sliding windows with specified sizes
6. **Splits data**: Anomaly detection split strategy:
   - **Training set**: Only normal data (for learning normal patterns)
   - **Validation set**: Normal + some attack data (for hyperparameter tuning)
   - **Test set**: Normal + attack data (for evaluation)
7. **Standardizes**: Normalizes features using training set statistics
8. **Saves bundles**: PyTorch tensors ready for model training

## Data Split Strategy

The script uses **anomaly detection split strategy**:
- **Training set contains ONLY normal data** - This is correct for anomaly detection tasks where we learn normal patterns
- **Validation/Test sets contain both normal and attack data** - For tuning and evaluation

This ensures the model learns from normal data only, which is the standard approach for anomaly detection.

## Output files

- `swat_minute.pt`: Fine-grained bundle (minute-level sequences)
- `swat_hour.pt`: Coarse-grained bundle (hour-level sequences)
- `swat_info.json`: Dataset metadata

## Usage

```bash
python scripts/process_swat.py \
    --data_dir data/SWaT \
    --output_dir data/SWaT/processed \
    --window_minute 60 \
    --window_hour 24 \
    --train_ratio 0.7 \
    --val_ratio 0.1
```

## Load processed data

```python
import torch

bundle_minute = torch.load('data/SWaT/processed/swat_minute.pt', weights_only=False)
bundle_hour = torch.load('data/SWaT/processed/swat_hour.pt', weights_only=False)

# Use in model training
from model.trainer import train_ours
train_ours(
    bundle_coarse=bundle_hour,
    bundle_fine=bundle_minute,
    save_path='models/swat_model.pt'
)
```

