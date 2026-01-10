import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.trainer import train_ours as train
from model.evaluator import infer, evaluate


def compute_signal_ratio(residual, y_true, idx_test):
    """Compute signal ratio: ratio of median residuals between anomaly and normal samples."""
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


def apply_same_subset_sampling(bundle_fine, bundle_coarse, max_samples):
    """Apply the same subset sampling to both fine and coarse bundles, maintaining split integrity.
    
    This ensures both bundles use the same data subset while preserving their mapping relationship.
    """
    print(f"\nApplying same subset sampling to fine and coarse bundles (max_samples={max_samples})...")
    print("  Sampling from original train/val/test splits separately to maintain split integrity...")
    
    n_total_coarse = bundle_coarse['X_seq'].shape[0]
    n_total_fine = bundle_fine['X_seq'].shape[0]
    
    # Get original splits
    idx_tr_coarse_orig = bundle_coarse['idx_train'].numpy()
    idx_val_coarse_orig = bundle_coarse['idx_val'].numpy()
    idx_test_coarse_orig = bundle_coarse['idx_test'].numpy()
    
    # Determine sample sizes for coarse split (preserve original ratios: 60/20/20)
    train_ratio = len(idx_tr_coarse_orig) / n_total_coarse
    val_ratio = len(idx_val_coarse_orig) / n_total_coarse
    test_ratio = len(idx_test_coarse_orig) / n_total_coarse
    
    n_train_coarse_sample = int(max_samples * train_ratio)
    n_val_coarse_sample = int(max_samples * val_ratio)
    n_test_coarse_sample = max_samples - n_train_coarse_sample - n_val_coarse_sample
    
    # Sample from coarse splits (take first N samples from each split)
    idx_tr_coarse_sample = idx_tr_coarse_orig[:min(n_train_coarse_sample, len(idx_tr_coarse_orig))]
    idx_val_coarse_sample = idx_val_coarse_orig[:min(n_val_coarse_sample, len(idx_val_coarse_orig))]
    idx_test_coarse_sample = idx_test_coarse_orig[:min(n_test_coarse_sample, len(idx_test_coarse_orig))]
    
    # Combine coarse selected indices
    selected_coarse_indices = np.concatenate([idx_tr_coarse_sample, idx_val_coarse_sample, idx_test_coarse_sample])
    selected_coarse_indices = np.sort(selected_coarse_indices)
    
    # Create mapping from old coarse indices to new coarse indices
    coarse_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_coarse_indices)}
    
    # Subset coarse bundle
    bundle_coarse['X_seq'] = bundle_coarse['X_seq'][selected_coarse_indices]
    bundle_coarse['X_next'] = bundle_coarse['X_next'][selected_coarse_indices]
    if bundle_coarse.get('y_next') is not None:
        bundle_coarse['y_next'] = bundle_coarse['y_next'][selected_coarse_indices]
    
    # Map coarse split indices to new indices
    idx_train_coarse_new = torch.tensor([coarse_old_to_new[idx] for idx in idx_tr_coarse_sample if idx in coarse_old_to_new], dtype=torch.long)
    idx_val_coarse_new = torch.tensor([coarse_old_to_new[idx] for idx in idx_val_coarse_sample if idx in coarse_old_to_new], dtype=torch.long)
    idx_test_coarse_new = torch.tensor([coarse_old_to_new[idx] for idx in idx_test_coarse_sample if idx in coarse_old_to_new], dtype=torch.long)
    
    bundle_coarse['idx_train'] = idx_train_coarse_new
    bundle_coarse['idx_val'] = idx_val_coarse_new
    bundle_coarse['idx_test'] = idx_test_coarse_new
    
    # Now handle fine bundle: use mapping to find corresponding fine samples
    if 'fine_to_coarse_index' not in bundle_fine:
        raise ValueError("fine_to_coarse_index mapping required for multi-scale sampling")
    
    mapping = bundle_fine['fine_to_coarse_index']
    if isinstance(mapping, torch.Tensor):
        mapping = mapping.numpy()
    
    # Find fine indices that map to selected coarse indices
    selected_coarse_set = set(selected_coarse_indices)
    fine_mask = np.array([mapping[i] in selected_coarse_set for i in range(len(mapping))])
    selected_fine_indices = np.where(fine_mask)[0]
    
    # Create mapping from old fine indices to new fine indices
    fine_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_fine_indices)}
    
    # Subset fine bundle
    bundle_fine['X_seq'] = bundle_fine['X_seq'][selected_fine_indices]
    bundle_fine['X_next'] = bundle_fine['X_next'][selected_fine_indices]
    if bundle_fine.get('y_next') is not None:
        bundle_fine['y_next'] = bundle_fine['y_next'][selected_fine_indices]
    
    # Remap fine_to_coarse_index to new coarse indices
    mapping_new = np.array([coarse_old_to_new.get(mapping[old_idx], -1) for old_idx in selected_fine_indices])
    bundle_fine['fine_to_coarse_index'] = torch.tensor(mapping_new, dtype=torch.long)
    
    # Get fine split indices based on temporal order (fine should maintain temporal split)
    # Use original fine splits, but filter to only selected indices
    idx_tr_fine_orig = bundle_fine['idx_train'].numpy() if 'idx_train' in bundle_fine else np.array([])
    idx_val_fine_orig = bundle_fine['idx_val'].numpy() if 'idx_val' in bundle_fine else np.array([])
    idx_test_fine_orig = bundle_fine['idx_test'].numpy() if 'idx_test' in bundle_fine else np.array([])
    
    # Filter to only selected fine indices and map to new indices
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
    
    # Update meta if available
    if 'meta' in bundle_coarse and hasattr(bundle_coarse['meta'], 'iloc'):
        bundle_coarse['meta'] = bundle_coarse['meta'].iloc[selected_coarse_indices].reset_index(drop=True)
    if 'meta' in bundle_fine and hasattr(bundle_fine['meta'], 'iloc'):
        bundle_fine['meta'] = bundle_fine['meta'].iloc[selected_fine_indices].reset_index(drop=True)
    
    n_use_coarse = len(selected_coarse_indices)
    n_use_fine = len(selected_fine_indices)
    
    print(f"  Coarse: Reduced from {n_total_coarse} to {n_use_coarse} samples")
    print(f"    Train: {len(idx_train_coarse_new)}, Val: {len(idx_val_coarse_new)}, Test: {len(idx_test_coarse_new)}")
    print(f"  Fine: Reduced from {n_total_fine} to {n_use_fine} samples")
    print(f"    Train: {len(idx_train_fine_new)}, Val: {len(idx_val_fine_new)}, Test: {len(idx_test_fine_new)}")
    print(f"  Mapping: {np.sum(mapping_new >= 0)}/{len(mapping_new)} fine sequences mapped to coarse")
    
    return bundle_fine, bundle_coarse


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PSM multi-scale model')
    parser.add_argument('--data_dir', type=str, default='data/PSM/processed', help='Processed data directory')
    parser.add_argument('--model_save_path', type=str, default='models/psm_stage1_multiscale.pt', help='Model save path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--pooling', type=str, default='last', help='Pooling method')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only evaluate')
    parser.add_argument('--max_samples', type=int, default=5000, help='Maximum number of samples for quick validation')
    parser.add_argument('--use_lu', action='store_true', help='Use LU loss')
    parser.add_argument('--lambda_u', type=float, default=1.0, help='Lambda for LU loss')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("PSM Multi-Scale Training")
    print("=" * 60)
    print(f"\nLoading data from {data_dir}...")
    
    bundle_fine = torch.load(data_dir / 'psm_fine.pt', weights_only=False)
    bundle_coarse = torch.load(data_dir / 'psm_coarse.pt', weights_only=False)
    
    # Apply same subset sampling if specified
    if args.max_samples is not None and args.max_samples > 0:
        bundle_fine, bundle_coarse = apply_same_subset_sampling(
            bundle_fine, bundle_coarse, args.max_samples
        )
    
    print(f"\nData loaded:")
    print(f"  Fine scale: {bundle_fine['X_seq'].shape}")
    print(f"  Coarse scale: {bundle_coarse['X_seq'].shape}")
    print(f"  Fine train/val/test: {len(bundle_fine['idx_train'])}/{len(bundle_fine['idx_val'])}/{len(bundle_fine['idx_test'])}")
    print(f"  Coarse train/val/test: {len(bundle_coarse['idx_train'])}/{len(bundle_coarse['idx_val'])}/{len(bundle_coarse['idx_test'])}")
    
    # Check mapping
    mapping = bundle_fine.get('fine_to_coarse_index', None)
    if mapping is not None:
        if isinstance(mapping, torch.Tensor):
            mapping = mapping.numpy()
        n_valid = np.sum(mapping >= 0)
        print(f"  Mapping: {n_valid}/{len(mapping)} fine sequences mapped to coarse")
    
    # Check label distribution
    y_fine = bundle_fine['y_next'].numpy() if bundle_fine.get('y_next') is not None else None
    y_coarse = bundle_coarse['y_next'].numpy() if bundle_coarse.get('y_next') is not None else None
    
    if y_fine is not None:
        y_fine_train = y_fine[bundle_fine['idx_train'].numpy()]
        y_fine_val = y_fine[bundle_fine['idx_val'].numpy()]
        y_fine_test = y_fine[bundle_fine['idx_test'].numpy()]
        print(f"\nFine label distribution:")
        print(f"  Train - Normal: {np.sum(y_fine_train == 0)}, Anomaly: {np.sum(y_fine_train == 1)}")
        print(f"  Val   - Normal: {np.sum(y_fine_val == 0)}, Anomaly: {np.sum(y_fine_val == 1)}")
        print(f"  Test  - Normal: {np.sum(y_fine_test == 0)}, Anomaly: {np.sum(y_fine_test == 1)}")
    
    if not args.skip_training:
        print(f"\nTraining multi-scale model...")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Use LU loss: {args.use_lu}")
        print(f"  Lambda U: {args.lambda_u}")
        print(f"  Device: {device}")
        
        model_path = train(
            bundle_coarse=bundle_coarse,
            bundle_fine=bundle_fine,
            save_path=args.model_save_path,
            coarse_only=False,
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
    
    # Evaluate model
    print(f"\nEvaluating model and computing residuals...")
    inference_result = infer(
        model_path=model_path,
        bundle_coarse=bundle_coarse,
        bundle_fine=bundle_fine,
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
    
    # Compute signal ratio
    signal_ratio = compute_signal_ratio(residual, y_true, idx_test)
    print(f"\nSignal Ratio (Anomaly/Normal median): {signal_ratio:.2f}x")
    print(f"ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    
    print("\n" + "=" * 60)
    print("Multi-Scale Training Summary")
    print("=" * 60)
    print(f"✓ Pipeline: OK (data loaded and processed)")
    print(f"✓ Split: OK (train/val/test split maintained)")
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
        'signal_ratio': signal_ratio
    }


if __name__ == '__main__':
    main()


