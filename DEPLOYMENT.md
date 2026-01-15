# Deployment Guide

## For Laboratory Server

### 1. Clone Repository

```bash
git clone <repository-url>
cd COVID
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

#### PSM Dataset:
- Place PSM CSV files in `data/PSM/PSM/`:
  - `train.csv`
  - `test.csv`
  - `test_label.csv`

- Process data:
```bash
python scripts/process_psm.py \
  --data_dir data/PSM/PSM \
  --output_dir data/PSM/processed \
  --window_fine 60 \
  --window_coarse 12 \
  --k 5 \
  --max_features 25
```

### 4. Run Training

#### Quick Validation (with subset):
```bash
# Coarse-only baseline (sanity check)
python scripts/train_psm_stage1.py \
  --data_dir data/PSM/processed \
  --model_save_path models/psm_stage1_coarse_only.pt \
  --epochs 50 \
  --batch_size 64 \
  --device cuda \
  --max_samples 5000

# Multi-scale training
python scripts/train_psm_multiscale.py \
  --data_dir data/PSM/processed \
  --model_save_path models/psm_stage1_multiscale.pt \
  --epochs 50 \
  --batch_size 64 \
  --device cuda \
  --max_samples 5000 \
  --use_lu \
  --lambda_u 1.0
```

#### Full Dataset Training:
```bash
# Remove --max_samples flag for full dataset
python scripts/train_psm_multiscale.py \
  --data_dir data/PSM/processed \
  --model_save_path models/psm_stage1_multiscale_full.pt \
  --epochs 50 \
  --batch_size 64 \
  --device cuda \
  --use_lu \
  --lambda_u 1.0
```

### 5. Check GPU Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Notes

- Data files (`.pt`) are excluded from git due to size - generate them locally
- Model files are excluded from git - train them on the server
- Use `--device cuda` for GPU training (much faster than CPU)
- Use `--max_samples` for quick validation with subset
- Remove `--max_samples` for full dataset training





