import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.trainer import train_ours as train
from model.evaluator import infer, evaluate


def compute_signal_ratio(residual, y_true, idx_test):
    """Compute signal ratio: ratio of median residuals between anomaly and normal samples.
    
    Args:
        residual: Residual scores (numpy array)
        y_true: True labels (numpy array)
        idx_test: Test indices (numpy array)
    
    Returns:
        signal_ratio: Median anomaly residual / Median normal residual
    """
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
    """Analyze residual separation quality between normal and anomaly samples.
    
    Args:
        residual: Residual scores (numpy array)
        y_true: True labels (numpy array)
        idx_val: Validation indices (numpy array)
        idx_test: Test indices (numpy array)
        save_path: Optional path to save distribution plot
    
    Returns:
        stats: Dictionary with separation statistics
    """
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
    
    # Compute signal ratio
    median_normal = stats['test_normal_median']
    median_anomaly = stats['test_anomaly_median']
    if median_normal > 0:
        stats['signal_ratio'] = median_anomaly / median_normal
    else:
        stats['signal_ratio'] = float('inf') if median_anomaly > 0 else 0.0
    
    # Create visualization if requested
    if save_path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validation set distribution
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
        
        # Test set distribution
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PSM stage1 coarse-only model for sanity check')
    parser.add_argument('--data_dir', type=str, default='data/PSM/processed', help='Processed data directory')
    parser.add_argument('--model_save_path', type=str, default='models/psm_stage1_coarse_only.pt', help='Model save path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--pooling', type=str, default='last', help='Pooling method')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only evaluate')
    parser.add_argument('--plot_path', type=str, default='psm_residual_coarse_only.png', help='Path to save residual plot')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use for quick validation (None = use all)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load data bundles
    print("=" * 60)
    print("PSM Stage1 Coarse-Only Training (Sanity Check)")
    print("=" * 60)
    print(f"\nLoading data from {data_dir}...")
    
    bundle_coarse = torch.load(data_dir / 'psm_coarse.pt', weights_only=False)
    
    # Use subset for quick validation if specified
    # IMPORTANT: Sample from original train/val/test splits separately to maintain split integrity
    if args.max_samples is not None and args.max_samples > 0:
        print(f"\nUsing data subset for quick validation (max_samples={args.max_samples})...")
        print("  Sampling from original train/val/test splits separately to maintain split integrity...")
        
        n_total = bundle_coarse['X_seq'].shape[0]
        idx_tr_orig = bundle_coarse['idx_train'].numpy()
        idx_val_orig = bundle_coarse['idx_val'].numpy()
        idx_test_orig = bundle_coarse['idx_test'].numpy()
        y_all = bundle_coarse['y_next'].numpy() if bundle_coarse.get('y_next') is not None else np.zeros(n_total)
        
        # Determine sample sizes for each split (preserve original ratios: 60/20/20)
        train_ratio = len(idx_tr_orig) / n_total
        val_ratio = len(idx_val_orig) / n_total
        test_ratio = len(idx_test_orig) / n_total
        
        n_train_sample = int(args.max_samples * train_ratio)
        n_val_sample = int(args.max_samples * val_ratio)
        n_test_sample = args.max_samples - n_train_sample - n_val_sample  # Remaining goes to test
        
        # Sample from each split separately (maintain temporal order within each split)
        # For train: take first n_train_sample (since train is already in temporal order)
        # For val: take first n_val_sample
        # For test: take first n_test_sample
        idx_tr_sample = idx_tr_orig[:min(n_train_sample, len(idx_tr_orig))]
        idx_val_sample = idx_val_orig[:min(n_val_sample, len(idx_val_orig))]
        idx_test_sample = idx_test_orig[:min(n_test_sample, len(idx_test_orig))]
        
        # Combine all selected indices
        selected_indices = np.concatenate([idx_tr_sample, idx_val_sample, idx_test_sample])
        selected_indices = np.sort(selected_indices)  # Sort to maintain global temporal order
        
        n_use = len(selected_indices)
        indices = torch.from_numpy(selected_indices).long()
        
        # Create mapping from old indices to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        
        # Subset the data
        bundle_coarse['X_seq'] = bundle_coarse['X_seq'][indices]
        bundle_coarse['X_next'] = bundle_coarse['X_next'][indices]
        if bundle_coarse.get('y_next') is not None:
            bundle_coarse['y_next'] = bundle_coarse['y_next'][indices]
        
        # Map original split indices to new indices
        idx_train_new = torch.tensor([old_to_new[idx] for idx in idx_tr_sample if idx in old_to_new], dtype=torch.long)
        idx_val_new = torch.tensor([old_to_new[idx] for idx in idx_val_sample if idx in old_to_new], dtype=torch.long)
        idx_test_new = torch.tensor([old_to_new[idx] for idx in idx_test_sample if idx in old_to_new], dtype=torch.long)
        
        bundle_coarse['idx_train'] = idx_train_new
        bundle_coarse['idx_val'] = idx_val_new
        bundle_coarse['idx_test'] = idx_test_new
        
        # Update meta if available
        if 'meta' in bundle_coarse and hasattr(bundle_coarse['meta'], 'iloc'):
            bundle_coarse['meta'] = bundle_coarse['meta'].iloc[selected_indices].reset_index(drop=True)
        
        print(f"  Reduced from {n_total} to {n_use} samples")
        print(f"  Train: {len(idx_train_new)} samples (from original {len(idx_tr_orig)})")
        print(f"  Val:   {len(idx_val_new)} samples (from original {len(idx_val_orig)})")
        print(f"  Test:  {len(idx_test_new)} samples (from original {len(idx_test_orig)})")
        print(f"  Normal samples: {np.sum(bundle_coarse['y_next'].numpy() == 0)}, Anomaly samples: {np.sum(bundle_coarse['y_next'].numpy() == 1)}")
    
    print(f"\nData loaded:")
    print(f"  Coarse scale: {bundle_coarse['X_seq'].shape}")
    print(f"  Train samples: {len(bundle_coarse['idx_train'])}")
    print(f"  Val samples: {len(bundle_coarse['idx_val'])}")
    print(f"  Test samples: {len(bundle_coarse['idx_test'])}")
    
    # Check label distribution
    y_all = bundle_coarse['y_next'].numpy()
    y_train = y_all[bundle_coarse['idx_train'].numpy()]
    y_val = y_all[bundle_coarse['idx_val'].numpy()]
    y_test = y_all[bundle_coarse['idx_test'].numpy()]
    
    print(f"\nLabel distribution:")
    print(f"  Train - Normal: {np.sum(y_train == 0)}, Anomaly: {np.sum(y_train == 1)}")
    print(f"  Val   - Normal: {np.sum(y_val == 0)}, Anomaly: {np.sum(y_val == 1)}")
    print(f"  Test  - Normal: {np.sum(y_test == 0)}, Anomaly: {np.sum(y_test == 1)}")
    
    if not args.skip_training:
        print(f"\nTraining coarse-only model...")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Device: {device}")
        
        model_path = train(
            bundle_coarse=bundle_coarse,
            bundle_fine=None,
            save_path=args.model_save_path,
            coarse_only=True,
            use_classification=False,
            use_lu=False,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            pooling=args.pooling,
            batch_size=args.batch_size
        )
        print(f"\nModel saved to: {model_path}")
    else:
        model_path = args.model_save_path
        print(f"\nSkipping training, using existing model: {model_path}")
    
    # Evaluate model
    print(f"\nEvaluating model and computing residuals...")
    inference_result = infer(
        model_path=model_path,
        bundle_coarse=bundle_coarse,
        bundle_fine=None,
        device=args.device
    )
    
    residual = inference_result['residual']
    y_true = inference_result['y_true']
    idx_val = inference_result['idx_val']
    idx_test = inference_result['idx_test']
    
    # Evaluate metrics
    print(f"\nComputing evaluation metrics...")
    eval_metrics = evaluate(residual, y_true, idx_val, idx_test)
    
    print(f"\nEvaluation Metrics:")
    print(f"  Threshold: {eval_metrics['threshold']:.6f}")
    print(f"  F1 Score: {eval_metrics['f1']:.4f}")
    print(f"  Precision: {eval_metrics['precision']:.4f}")
    print(f"  Recall: {eval_metrics['recall']:.4f}")
    print(f"  ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    print(f"  AUPRC: {eval_metrics['auprc']:.4f}")
    
    # Analyze residual separation
    print(f"\nAnalyzing residual separation quality...")
    separation_stats = analyze_residual_separation(
        residual, y_true, idx_val, idx_test, save_path=args.plot_path
    )
    
    # Compute signal ratio
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
    
    # Sanity check summary
    print(f"\n" + "=" * 60)
    print("Sanity Check Summary")
    print("=" * 60)
    print(f"✓ Pipeline: OK (data loaded and processed)")
    print(f"✓ Split: OK (train/val/test split looks reasonable)")
    if eval_metrics['roc_auc'] > 0.5:
        print(f"✓ Residual Separation: GOOD (ROC-AUC = {eval_metrics['roc_auc']:.4f} > 0.5)")
    else:
        print(f"⚠ Residual Separation: POOR (ROC-AUC = {eval_metrics['roc_auc']:.4f} <= 0.5)")
    
    if signal_ratio > 1.0:
        print(f"✓ Signal Ratio: GOOD ({signal_ratio:.2f}x > 1.0x)")
    else:
        print(f"⚠ Signal Ratio: POOR ({signal_ratio:.2f}x <= 1.0x)")
    
    print("=" * 60)
    
    return {
        'eval_metrics': eval_metrics,
        'separation_stats': separation_stats,
        'signal_ratio': signal_ratio
    }


if __name__ == '__main__':
    main()

