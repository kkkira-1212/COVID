import torch
import numpy as np
from pathlib import Path
import sys
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.evaluator import infer, evaluate
from scripts.validate_stage2_minimal import (
    extract_causal_matrix_and_residual,
    compute_anomaly_scores
)


def detect_dataset_type(data_dir):
    """Auto-detect dataset type from data directory."""
    data_dir = Path(data_dir)
    
    coarse_files = ['psm_coarse.pt', 'swat_hour.pt', 'smap_coarse.pt', 'coarse.pt']
    fine_files = ['psm_fine.pt', 'swat_minute.pt', 'smap_fine.pt', 'fine.pt']
    
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


def compare_models(
    model_paths,
    data_dir,
    coarse_file,
    fine_file=None,
    device='cuda',
    topk_percents=[5, 10],
    use_stage2=False,
    spatial_weight=0.5,
    temporal_weight=0.5
):
    data_dir = Path(data_dir)
    
    bundle_coarse = torch.load(data_dir / coarse_file, weights_only=False)
    bundle_fine = None
    if fine_file is not None:
        bundle_fine = torch.load(data_dir / fine_file, weights_only=False)
    
    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(f"\nData: {coarse_file}")
    if fine_file:
        print(f"Fine scale: {fine_file}")
    print(f"Coarse scale shape: {bundle_coarse['X_seq'].shape}")
    if bundle_fine:
        print(f"Fine scale shape: {bundle_fine['X_seq'].shape}")
    
    # Note: Test set info will be determined from each model's inference result
    # since coarse-only, fine-only, and multi-scale models may use different data sources
    print(f"\nNote: Test set statistics will be shown per model (may differ for coarse-only vs fine-only vs multi-scale)")
    
    results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")
        
        print(f"\nLoading model and computing residuals...")
        inference_result = infer(
            model_path=model_path,
            bundle_coarse=bundle_coarse,
            bundle_fine=bundle_fine,
            device=device
        )
        
        residual = inference_result['residual']
        y_true = inference_result['y_true']
        idx_val = inference_result['idx_val']
        idx_test = inference_result['idx_test']
        
        residual_test = residual[idx_test]
        y_test_actual = y_true[idx_test]
        
        residual_normal = residual_test[y_test_actual == 0]
        residual_anomaly = residual_test[y_test_actual == 1]
        
        print(f"\nResidual statistics on test set:")
        print(f"  Total samples: {len(residual_test)}")
        print(f"  Normal samples: {len(residual_normal)}")
        print(f"  Anomaly samples: {len(residual_anomaly)}")
        print(f"  Actual anomaly ratio: {len(residual_anomaly)/len(residual_test)*100:.2f}%")
        print(f"\n  Overall:")
        print(f"    Min: {residual_test.min():.6f}")
        print(f"    Max: {residual_test.max():.6f}")
        print(f"    Mean: {residual_test.mean():.6f}")
        print(f"    Median: {np.median(residual_test):.6f}")
        print(f"    95th percentile: {np.percentile(residual_test, 95):.6f}")
        print(f"    99th percentile: {np.percentile(residual_test, 99):.6f}")
        if len(residual_normal) > 0:
            print(f"\n  Normal samples:")
            print(f"    Mean: {residual_normal.mean():.6f}")
            print(f"    Median: {np.median(residual_normal):.6f}")
            print(f"    95th percentile: {np.percentile(residual_normal, 95):.6f}")
        if len(residual_anomaly) > 0:
            print(f"\n  Anomaly samples:")
            print(f"    Mean: {residual_anomaly.mean():.6f}")
            print(f"    Median: {np.median(residual_anomaly):.6f}")
            print(f"    95th percentile: {np.percentile(residual_anomaly, 95):.6f}")
            if len(residual_normal) > 0:
                signal_ratio = np.median(residual_anomaly) / np.median(residual_normal) if np.median(residual_normal) > 0 else float('inf')
                print(f"    Signal ratio (anomaly/normal median): {signal_ratio:.2f}x")
        
        model_results = {}
        
        for topk in topk_percents:
            print(f"\n{'─'*80}")
            print(f"Using Top {topk}% Residual as Anomaly Threshold")
            print(f"{'─'*80}")
            
            eval_metrics = evaluate(
                residual, y_true, idx_val, idx_test,
                use_topk=True, topk_percent=topk
            )
            
            th = eval_metrics['threshold']
            pred = (residual_test >= th).astype(int)
            
            print(f"  Threshold: {th:.6f} (top {topk}% = {np.sum(pred == 1)} samples)")
            print(f"  Predicted anomalies: {np.sum(pred == 1)}")
            print(f"  Actual anomalies: {np.sum(y_test_actual == 1)}")
            print(f"  True Positives: {np.sum((pred == 1) & (y_test_actual == 1))}")
            print(f"  False Positives: {np.sum((pred == 1) & (y_test_actual == 0))}")
            print(f"  False Negatives: {np.sum((pred == 0) & (y_test_actual == 1))}")
            print(f"  True Negatives: {np.sum((pred == 0) & (y_test_actual == 0))}")
            print(f"\n  Metrics:")
            print(f"    Precision: {eval_metrics['precision']:.4f}")
            print(f"    Recall: {eval_metrics['recall']:.4f}")
            print(f"    F1 Score: {eval_metrics['f1']:.4f}")
            print(f"    ROC-AUC: {eval_metrics['roc_auc']:.4f}")
            print(f"    AUPRC: {eval_metrics['auprc']:.4f}")
            
            model_results[f'top{topk}%'] = eval_metrics
        
        results[model_name] = model_results
    
    print(f"\n{'='*80}")
    print("Comparison Summary")
    print(f"{'='*80}")
    
    # Create a comprehensive comparison table
    for topk in topk_percents:
        print(f"\n{'─'*80}")
        print(f"Top {topk}% Threshold Comparison")
        print(f"{'─'*80}")
        print(f"{'Model':<45} {'F1':<8} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10} {'AUPRC':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
        print(f"{'-'*80}")
        
        for model_name, model_results in results.items():
            metrics = model_results[f'top{topk}%']
            # Calculate TP, FP, FN from predicted and actual anomalies
            pred_anomalies = metrics.get('predicted_anomalies', 0)
            actual_anomalies = metrics.get('actual_anomalies', 0)
            tp = int(metrics.get('precision', 0) * pred_anomalies) if metrics.get('precision', 0) > 0 else 0
            fp = pred_anomalies - tp
            fn = actual_anomalies - tp
            
            print(f"{model_name:<45} {metrics['f1']:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['roc_auc']:<10.4f} {metrics['auprc']:<10.4f} {tp:<6} {fp:<6} {fn:<6}")
    
    # Summary statistics table
    print(f"\n{'='*80}")
    print("Best Performance Summary (across all top-k%)")
    print(f"{'='*80}")
    print(f"{'Model':<45} {'Best F1':<10} {'Best ROC-AUC':<12} {'Best AUPRC':<12} {'Best Top-k%':<12}")
    print(f"{'-'*80}")
    
    for model_name, model_results in results.items():
        best_f1 = max([m['f1'] for m in model_results.values()])
        best_roc = max([m['roc_auc'] for m in model_results.values()])
        best_auprc = max([m['auprc'] for m in model_results.values()])
        
        # Find which topk gave best F1
        best_f1_topk = None
        for topk_key, metrics in model_results.items():
            if metrics['f1'] == best_f1:
                best_f1_topk = topk_key
                break
        
        print(f"{model_name:<45} {best_f1:<10.4f} {best_roc:<12.4f} {best_auprc:<12.4f} {best_f1_topk:<12}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare multiple trained models')
    parser.add_argument('--data_dir', type=str, required=True, help='Processed data directory')
    parser.add_argument('--coarse_file', type=str, default=None, help='Coarse scale data file (auto-detect if not specified)')
    parser.add_argument('--fine_file', type=str, default=None, help='Fine scale data file (auto-detect if not specified)')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Paths to model files to compare')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--topk_percents', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30], help='Top-k percentages to use (default: 5 10 15 20 25 30)')
    parser.add_argument('--use_stage2', action='store_true', help='Use Stage 2 scores (spatial + temporal deviation) instead of Stage 1 residual only')
    parser.add_argument('--spatial_weight', type=float, default=0.5, help='Weight for spatial deviation in Stage 2 (default: 0.5)')
    parser.add_argument('--temporal_weight', type=float, default=0.5, help='Weight for temporal deviation in Stage 2 (default: 0.5)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Auto-detect data files if not specified
    if args.coarse_file is None or args.fine_file is None:
        coarse_file, fine_file = detect_dataset_type(data_dir)
        if args.coarse_file is None:
            args.coarse_file = coarse_file
        if args.fine_file is None:
            args.fine_file = fine_file
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    results = compare_models(
        model_paths=args.model_paths,
        data_dir=args.data_dir,
        coarse_file=args.coarse_file,
        fine_file=args.fine_file,
        device=args.device,
        topk_percents=args.topk_percents
    )
    
    print(f"\n{'='*80}")
    print("Comparison completed!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

