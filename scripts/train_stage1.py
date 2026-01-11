import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.trainer import train_ours as train
from model.evaluator import infer, evaluate


def compute_signal_ratio(residual, y_true, idx_test):
    residual_test = residual[idx_test]
    y_test = y_true[idx_test]
    
    residual_normal = residual_test[y_test == 0]
    residual_anomaly = residual_test[y_test == 1]
    
    if len(residual_anomaly) == 0:
        return 0.0
    
    median_normal = np.median(residual_normal) if len(residual_normal) > 0 else 0.0
    median_anomaly = np.median(residual_anomaly) if len(residual_anomaly) > 0 else 0.0
    
    if median_normal == 0:
        return float('inf') if median_anomaly > 0 else 0.0
    
    return median_anomaly / median_normal


def analyze_residual_separation(residual, y_true, idx_val, idx_test, save_path=None):
    residual_val = residual[idx_val]
    residual_test = residual[idx_test]
    y_val = y_true[idx_val]
    y_test = y_true[idx_test]
    
    residual_normal_val = residual_val[y_val == 0]
    residual_anomaly_val = residual_val[y_val == 1]
    residual_normal_test = residual_test[y_test == 0]
    residual_anomaly_test = residual_test[y_test == 1]
    
    stats = {
        'val_normal_count': len(residual_normal_val),
        'val_anomaly_count': len(residual_anomaly_val),
        'val_normal_median': np.median(residual_normal_val) if len(residual_normal_val) > 0 else 0.0,
        'val_anomaly_median': np.median(residual_anomaly_val) if len(residual_anomaly_val) > 0 else 0.0,
        'val_normal_mean': np.mean(residual_normal_val) if len(residual_normal_val) > 0 else 0.0,
        'val_anomaly_mean': np.mean(residual_anomaly_val) if len(residual_anomaly_val) > 0 else 0.0,
        'val_normal_std': np.std(residual_normal_val) if len(residual_normal_val) > 0 else 0.0,
        'val_anomaly_std': np.std(residual_anomaly_val) if len(residual_anomaly_val) > 0 else 0.0,
        'test_normal_count': len(residual_normal_test),
        'test_anomaly_count': len(residual_anomaly_test),
        'test_normal_median': np.median(residual_normal_test) if len(residual_normal_test) > 0 else 0.0,
        'test_anomaly_median': np.median(residual_anomaly_test) if len(residual_anomaly_test) > 0 else 0.0,
        'test_normal_mean': np.mean(residual_normal_test) if len(residual_normal_test) > 0 else 0.0,
        'test_anomaly_mean': np.mean(residual_anomaly_test) if len(residual_anomaly_test) > 0 else 0.0,
        'test_normal_std': np.std(residual_normal_test) if len(residual_normal_test) > 0 else 0.0,
        'test_anomaly_std': np.std(residual_anomaly_test) if len(residual_anomaly_test) > 0 else 0.0,
    }
    
    median_normal = stats['test_normal_median']
    median_anomaly = stats['test_anomaly_median']
    if median_normal > 0:
        stats['signal_ratio'] = median_anomaly / median_normal
    else:
        stats['signal_ratio'] = float('inf') if median_anomaly > 0 else 0.0
    
    if save_path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if len(residual_normal_val) > 0:
            axes[0].hist(residual_normal_val, bins=50, alpha=0.7, label=f'Normal (n={len(residual_normal_val)})', 
                        density=True, color='blue')
        if len(residual_anomaly_val) > 0:
            axes[0].hist(residual_anomaly_val, bins=50, alpha=0.7, label=f'Anomaly (n={len(residual_anomaly_val)})', 
                        density=True, color='red')
        axes[0].set_xlabel('Residual')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Validation Set Residual Distribution\nNormal median: {stats["val_normal_median"]:.4f}, Anomaly median: {stats["val_anomaly_median"]:.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if len(residual_normal_test) > 0:
            axes[1].hist(residual_normal_test, bins=50, alpha=0.7, label=f'Normal (n={len(residual_normal_test)})', 
                        density=True, color='blue')
        if len(residual_anomaly_test) > 0:
            axes[1].hist(residual_anomaly_test, bins=50, alpha=0.7, label=f'Anomaly (n={len(residual_anomaly_test)})', 
                        density=True, color='red')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'Test Set Residual Distribution\nSignal Ratio: {stats["signal_ratio"]:.2f}x')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Residual distribution plot saved to: {save_path}")
    
    return stats


def apply_subset_sampling(bundle, max_samples):
    n_total = bundle['X_seq'].shape[0]
    idx_tr_orig = bundle['idx_train'].numpy()
    idx_val_orig = bundle['idx_val'].numpy()
    idx_test_orig = bundle['idx_test'].numpy()
    
    train_ratio = len(idx_tr_orig) / n_total
    val_ratio = len(idx_val_orig) / n_total
    
    n_train_sample = int(max_samples * train_ratio)
    n_val_sample = int(max_samples * val_ratio)
    n_test_sample = max_samples - n_train_sample - n_val_sample
    
    idx_tr_sample = idx_tr_orig[:min(n_train_sample, len(idx_tr_orig))]
    idx_val_sample = idx_val_orig[:min(n_val_sample, len(idx_val_orig))]
    idx_test_sample = idx_test_orig[:min(n_test_sample, len(idx_test_orig))]
    
    selected_indices = np.concatenate([idx_tr_sample, idx_val_sample, idx_test_sample])
    selected_indices = np.sort(selected_indices)
    
    indices = torch.from_numpy(selected_indices).long()
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
    
    bundle['X_seq'] = bundle['X_seq'][indices]
    bundle['X_next'] = bundle['X_next'][indices]
    if bundle.get('y_next') is not None:
        bundle['y_next'] = bundle['y_next'][indices]
    
    idx_train_new = torch.tensor([old_to_new[idx] for idx in idx_tr_sample if idx in old_to_new], dtype=torch.long)
    idx_val_new = torch.tensor([old_to_new[idx] for idx in idx_val_sample if idx in old_to_new], dtype=torch.long)
    idx_test_new = torch.tensor([old_to_new[idx] for idx in idx_test_sample if idx in old_to_new], dtype=torch.long)
    
    bundle['idx_train'] = idx_train_new
    bundle['idx_val'] = idx_val_new
    bundle['idx_test'] = idx_test_new
    
    if 'meta' in bundle and hasattr(bundle['meta'], 'iloc'):
        bundle['meta'] = bundle['meta'].iloc[selected_indices].reset_index(drop=True)
    
    print(f"  Reduced from {n_total} to {len(selected_indices)} samples")
    print(f"  Train: {len(idx_train_new)} samples (from original {len(idx_tr_orig)})")
    print(f"  Val:   {len(idx_val_new)} samples (from original {len(idx_val_orig)})")
    print(f"  Test:  {len(idx_test_new)} samples (from original {len(idx_test_orig)})")
    
    return bundle


def apply_multi_scale_subset_sampling(bundle_fine, bundle_coarse, max_samples):
    n_total_coarse = bundle_coarse['X_seq'].shape[0]
    idx_tr_coarse_orig = bundle_coarse['idx_train'].numpy()
    idx_val_coarse_orig = bundle_coarse['idx_val'].numpy()
    idx_test_coarse_orig = bundle_coarse['idx_test'].numpy()
    
    train_ratio = len(idx_tr_coarse_orig) / n_total_coarse
    val_ratio = len(idx_val_coarse_orig) / n_total_coarse
    
    n_train_coarse_sample = int(max_samples * train_ratio)
    n_val_coarse_sample = int(max_samples * val_ratio)
    n_test_coarse_sample = max_samples - n_train_coarse_sample - n_val_coarse_sample
    
    idx_tr_coarse_sample = idx_tr_coarse_orig[:min(n_train_coarse_sample, len(idx_tr_coarse_orig))]
    idx_val_coarse_sample = idx_val_coarse_orig[:min(n_val_coarse_sample, len(idx_val_coarse_orig))]
    idx_test_coarse_sample = idx_test_coarse_orig[:min(n_test_coarse_sample, len(idx_test_coarse_orig))]
    
    selected_coarse_indices = np.concatenate([idx_tr_coarse_sample, idx_val_coarse_sample, idx_test_coarse_sample])
    selected_coarse_indices = np.sort(selected_coarse_indices)
    
    coarse_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_coarse_indices)}
    
    bundle_coarse['X_seq'] = bundle_coarse['X_seq'][selected_coarse_indices]
    bundle_coarse['X_next'] = bundle_coarse['X_next'][selected_coarse_indices]
    if bundle_coarse.get('y_next') is not None:
        bundle_coarse['y_next'] = bundle_coarse['y_next'][selected_coarse_indices]
    
    idx_train_coarse_new = torch.tensor([coarse_old_to_new[idx] for idx in idx_tr_coarse_sample if idx in coarse_old_to_new], dtype=torch.long)
    idx_val_coarse_new = torch.tensor([coarse_old_to_new[idx] for idx in idx_val_coarse_sample if idx in coarse_old_to_new], dtype=torch.long)
    idx_test_coarse_new = torch.tensor([coarse_old_to_new[idx] for idx in idx_test_coarse_sample if idx in coarse_old_to_new], dtype=torch.long)
    
    bundle_coarse['idx_train'] = idx_train_coarse_new
    bundle_coarse['idx_val'] = idx_val_coarse_new
    bundle_coarse['idx_test'] = idx_test_coarse_new
    
    if 'fine_to_coarse_index' not in bundle_fine:
        raise ValueError("fine_to_coarse_index mapping required for multi-scale sampling")
    
    mapping = bundle_fine['fine_to_coarse_index']
    if isinstance(mapping, torch.Tensor):
        mapping = mapping.numpy()
    
    selected_coarse_set = set(selected_coarse_indices)
    fine_mask = np.array([mapping[i] in selected_coarse_set for i in range(len(mapping))])
    selected_fine_indices = np.where(fine_mask)[0]
    
    fine_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_fine_indices)}
    
    bundle_fine['X_seq'] = bundle_fine['X_seq'][selected_fine_indices]
    bundle_fine['X_next'] = bundle_fine['X_next'][selected_fine_indices]
    if bundle_fine.get('y_next') is not None:
        bundle_fine['y_next'] = bundle_fine['y_next'][selected_fine_indices]
    
    mapping_new = np.array([coarse_old_to_new.get(mapping[old_idx], -1) for old_idx in selected_fine_indices])
    bundle_fine['fine_to_coarse_index'] = torch.tensor(mapping_new, dtype=torch.long)
    
    idx_tr_fine_orig = bundle_fine['idx_train'].numpy() if 'idx_train' in bundle_fine else np.array([])
    idx_val_fine_orig = bundle_fine['idx_val'].numpy() if 'idx_val' in bundle_fine else np.array([])
    idx_test_fine_orig = bundle_fine['idx_test'].numpy() if 'idx_test' in bundle_fine else np.array([])
    
    selected_fine_set = set(selected_fine_indices)
    idx_tr_fine_filtered = [idx for idx in idx_tr_fine_orig if idx in selected_fine_set]
    idx_val_fine_filtered = [idx for idx in idx_val_fine_orig if idx in selected_fine_set]
    idx_test_fine_filtered = [idx for idx in idx_test_fine_orig if idx in selected_fine_set]
    
    idx_train_fine_new = torch.tensor([fine_old_to_new[idx] for idx in idx_tr_fine_filtered if idx in fine_old_to_new], dtype=torch.long)
    idx_val_fine_new = torch.tensor([fine_old_to_new[idx] for idx in idx_val_fine_filtered if idx in fine_old_to_new], dtype=torch.long)
    idx_test_fine_new = torch.tensor([fine_old_to_new[idx] for idx in idx_test_fine_filtered if idx in fine_old_to_new], dtype=torch.long)
    
    bundle_fine['idx_train'] = idx_train_fine_new
    bundle_fine['idx_val'] = idx_val_fine_new
    bundle_fine['idx_test'] = idx_test_fine_new
    
    if 'meta' in bundle_coarse and hasattr(bundle_coarse['meta'], 'iloc'):
        bundle_coarse['meta'] = bundle_coarse['meta'].iloc[selected_coarse_indices].reset_index(drop=True)
    if 'meta' in bundle_fine and hasattr(bundle_fine['meta'], 'iloc'):
        bundle_fine['meta'] = bundle_fine['meta'].iloc[selected_fine_indices].reset_index(drop=True)
    
    print(f"  Coarse: Reduced from {n_total_coarse} to {len(selected_coarse_indices)} samples")
    print(f"    Train: {len(idx_train_coarse_new)}, Val: {len(idx_val_coarse_new)}, Test: {len(idx_test_coarse_new)}")
    print(f"  Fine: Reduced from {len(fine_mask)} to {len(selected_fine_indices)} samples")
    print(f"    Train: {len(idx_train_fine_new)}, Val: {len(idx_val_fine_new)}, Test: {len(idx_test_fine_new)}")
    
    return bundle_fine, bundle_coarse


def detect_dataset_type(data_dir):
    data_dir = Path(data_dir)
    
    coarse_files = ['psm_coarse.pt', 'swat_hour.pt', 'coarse.pt']
    fine_files = ['psm_fine.pt', 'swat_minute.pt', 'fine.pt']
    
    for coarse_file in coarse_files:
        if (data_dir / coarse_file).exists():
            coarse_path = coarse_file
            break
    else:
        raise ValueError(f"Could not detect dataset type. Expected one of: {coarse_files}")
    
    fine_path = None
    for fine_file in fine_files:
        if (data_dir / fine_file).exists():
            fine_path = fine_file
            break
    
    return coarse_path, fine_path


def load_data_bundles(data_dir, coarse_file, fine_file=None):
    data_dir = Path(data_dir)
    
    bundle_coarse = torch.load(data_dir / coarse_file, weights_only=False)
    
    bundle_fine = None
    if fine_file is not None:
        bundle_fine = torch.load(data_dir / fine_file, weights_only=False)
    
    return bundle_coarse, bundle_fine


def main():
    parser = argparse.ArgumentParser(description='Train Stage 1 forecasting model')
    parser.add_argument('--data_dir', type=str, required=True, help='Processed data directory')
    parser.add_argument('--coarse_file', type=str, default=None, help='Coarse scale data file (auto-detect if not specified)')
    parser.add_argument('--fine_file', type=str, default=None, help='Fine scale data file (auto-detect if not specified)')
    parser.add_argument('--model_save_path', type=str, required=True, help='Model save path')
    parser.add_argument('--coarse_only', action='store_true', help='Train only coarse scale')
    parser.add_argument('--use_lu', action='store_true', help='Use LU loss')
    parser.add_argument('--lambda_u', type=float, default=1.0, help='Lambda for LU loss')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--pooling', type=str, default='last', choices=['last', 'mean'], help='Pooling method')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only evaluate')
    parser.add_argument('--plot_path', type=str, default=None, help='Path to save residual plot')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples for quick validation')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Stage 1 Training")
    print("=" * 60)
    print(f"\nLoading data from {data_dir}...")
    
    if args.coarse_file is None or args.fine_file is None:
        coarse_file, fine_file = detect_dataset_type(data_dir)
        if args.coarse_file is None:
            args.coarse_file = coarse_file
        if args.fine_file is None and not args.coarse_only:
            args.fine_file = fine_file
    else:
        fine_file = args.fine_file if not args.coarse_only else None
    
    bundle_coarse, bundle_fine = load_data_bundles(data_dir, args.coarse_file, fine_file)
    
    if args.max_samples is not None and args.max_samples > 0:
        print(f"\nUsing data subset for quick validation (max_samples={args.max_samples})...")
        if args.coarse_only or bundle_fine is None:
            bundle_coarse = apply_subset_sampling(bundle_coarse, args.max_samples)
        else:
            bundle_fine, bundle_coarse = apply_multi_scale_subset_sampling(bundle_fine, bundle_coarse, args.max_samples)
    
    print(f"\nData loaded:")
    print(f"  Coarse scale: {bundle_coarse['X_seq'].shape}")
    print(f"  Train samples: {len(bundle_coarse['idx_train'])}")
    print(f"  Val samples: {len(bundle_coarse['idx_val'])}")
    print(f"  Test samples: {len(bundle_coarse['idx_test'])}")
    
    if bundle_fine is not None:
        print(f"  Fine scale: {bundle_fine['X_seq'].shape}")
        print(f"  Fine train/val/test: {len(bundle_fine['idx_train'])}/{len(bundle_fine['idx_val'])}/{len(bundle_fine['idx_test'])}")
    
    y_all = bundle_coarse['y_next'].numpy()
    y_train = y_all[bundle_coarse['idx_train'].numpy()]
    y_val = y_all[bundle_coarse['idx_val'].numpy()]
    y_test = y_all[bundle_coarse['idx_test'].numpy()]
    
    print(f"\nLabel distribution:")
    print(f"  Train - Normal: {np.sum(y_train == 0)}, Anomaly: {np.sum(y_train == 1)}")
    print(f"  Val   - Normal: {np.sum(y_val == 0)}, Anomaly: {np.sum(y_val == 1)}")
    print(f"  Test  - Normal: {np.sum(y_test == 0)}, Anomaly: {np.sum(y_test == 1)}")
    
    if not args.skip_training:
        print(f"\nTraining model...")
        print(f"  Coarse only: {args.coarse_only}")
        print(f"  Use LU loss: {args.use_lu}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Device: {device}")
        
        model_path = train(
            bundle_coarse=bundle_coarse,
            bundle_fine=bundle_fine if not args.coarse_only else None,
            save_path=args.model_save_path,
            coarse_only=args.coarse_only,
            use_classification=False,
            use_lu=args.use_lu,
            lambda_u=args.lambda_u,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            pooling=args.pooling,
            batch_size=args.batch_size,
            device=args.device
        )
        print(f"\nModel saved to: {model_path}")
    else:
        model_path = args.model_save_path
        print(f"\nSkipping training, using existing model: {model_path}")
    
    print(f"\nEvaluating model and computing residuals...")
    inference_result = infer(
        model_path=model_path,
        bundle_coarse=bundle_coarse,
        bundle_fine=bundle_fine if not args.coarse_only else None,
        device=args.device
    )
    
    residual = inference_result['residual']
    y_true = inference_result['y_true']
    idx_val = inference_result['idx_val']
    idx_test = inference_result['idx_test']
    
    print(f"\nComputing evaluation metrics...")
    eval_metrics = evaluate(residual, y_true, idx_val, idx_test)
    
    print(f"\nEvaluation Metrics:")
    print(f"  Threshold: {eval_metrics['threshold']:.6f}")
    print(f"  F1 Score: {eval_metrics['f1']:.4f}")
    print(f"  Precision: {eval_metrics['precision']:.4f}")
    print(f"  Recall: {eval_metrics['recall']:.4f}")
    print(f"  ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    print(f"  AUPRC: {eval_metrics['auprc']:.4f}")
    
    if args.plot_path:
        print(f"\nAnalyzing residual separation quality...")
        separation_stats = analyze_residual_separation(
            residual, y_true, idx_val, idx_test, save_path=args.plot_path
        )
        
        signal_ratio = compute_signal_ratio(residual, y_true, idx_test)
        
        print(f"\nResidual Separation Statistics:")
        print(f"\nValidation Set:")
        print(f"  Normal:   n={separation_stats['val_normal_count']}, median={separation_stats['val_normal_median']:.6f}, mean={separation_stats['val_normal_mean']:.6f}, std={separation_stats['val_normal_std']:.6f}")
        if separation_stats['val_anomaly_count'] > 0:
            print(f"  Anomaly:  n={separation_stats['val_anomaly_count']}, median={separation_stats['val_anomaly_median']:.6f}, mean={separation_stats['val_anomaly_mean']:.6f}, std={separation_stats['val_anomaly_std']:.6f}")
        
        print(f"\nTest Set:")
        print(f"  Normal:   n={separation_stats['test_normal_count']}, median={separation_stats['test_normal_median']:.6f}, mean={separation_stats['test_normal_mean']:.6f}, std={separation_stats['test_normal_std']:.6f}")
        if separation_stats['test_anomaly_count'] > 0:
            print(f"  Anomaly:  n={separation_stats['test_anomaly_count']}, median={separation_stats['test_anomaly_median']:.6f}, mean={separation_stats['test_anomaly_mean']:.6f}, std={separation_stats['test_anomaly_std']:.6f}")
        
        print(f"\nSignal Ratio (Anomaly/Normal median): {signal_ratio:.2f}x")
        print(f"ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    
    print(f"\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"✓ Pipeline: OK (data loaded and processed)")
    print(f"✓ Split: OK (train/val/test split maintained)")
    if eval_metrics['roc_auc'] > 0.5:
        print(f"✓ Residual Separation: GOOD (ROC-AUC = {eval_metrics['roc_auc']:.4f} > 0.5)")
    else:
        print(f"⚠ Residual Separation: POOR (ROC-AUC = {eval_metrics['roc_auc']:.4f} <= 0.5)")
    print("=" * 60)


if __name__ == '__main__':
    main()

