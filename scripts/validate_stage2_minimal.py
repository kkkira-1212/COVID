import torch
import numpy as np
import argparse
from pathlib import Path
import sys
from sklearn.metrics import average_precision_score, roc_auc_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.encoder import TransformerSeqEncoder, RegressionHeadWithRelation
from model.trainer import to_device as todevice


def extract_causal_matrix_and_residual(model_path, bundle_coarse, device='cuda', batch_size=32):
    """
    从模型提取每个 window 的 causal matrix representation 和 residual
    返回:
    - causal_matrices: (N, num_vars, num_vars) - 每个 window 的 causal matrix
    - residuals: (N,) - 每个 window 的 residual
    - y_true: (N,) - 标签
    - idx_train, idx_val, idx_test - 索引
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    data_coarse = todevice(bundle_coarse, device)
    
    coarse_only = config.get('coarse_only', False)
    num_vars = data_coarse['X_seq'].shape[2]
    
    # For multi-scale models, we only use coarse (hour) scale for Stage 2 validation
    # We still need to load the model correctly even if it's multi-scale
    
    # Load model
    enc_coarse = TransformerSeqEncoder(
        input_dim=data_coarse['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_coarse['X_seq'].shape[1] + 5
    ).to(device)
    enc_coarse.pooling = config['pooling']
    
    head = RegressionHeadWithRelation(config['d_model'], num_vars).to(device)
    enc_coarse.load_state_dict(checkpoint['enc_coarse'])
    head.load_state_dict(checkpoint['head'])
    
    # For multi-scale models, we only use coarse scale encoder and head
    enc_coarse.eval()
    head.eval()
    
    # Extract fixed relation matrix (reference causal structure)
    relation_matrix_ref = head.var_relation_matrix.detach().cpu().numpy()
    
    # Extract per-window causal representations and residuals
    N = data_coarse['X_seq'].shape[0]
    causal_matrices = []
    residuals = []
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch_idx = torch.arange(i, end_idx, device=device)
            
            X_batch = data_coarse['X_seq'][batch_idx]
            X_next_batch = data_coarse['X_next'][batch_idx]
            
            # Get hidden representation
            z_batch = enc_coarse(X_batch)  # (batch, d_model)
            
            # Get var_base (variable-specific base prediction)
            var_base = head.base_proj_coarse(z_batch)  # (batch, num_vars)
            
            # Generate per-window causal matrix: 
            # We weight the relation matrix by var_base to get per-window causal patterns
            # causal_mat[i] = var_base[i].unsqueeze(-1) * relation_matrix.T
            # This gives (batch, num_vars, num_vars) where each matrix is weighted by var_base
            var_base_expanded = var_base.unsqueeze(-1)  # (batch, num_vars, 1)
            # Get relation matrix from model (torch tensor on device)
            relation_matrix_torch = head.var_relation_matrix.T.unsqueeze(0)  # (1, num_vars, num_vars)
            causal_mat = var_base_expanded * relation_matrix_torch  # (batch, num_vars, num_vars)
            causal_matrices.append(causal_mat.cpu().numpy())
            
            # Get prediction and residual
            _, pred_batch = head(z_batch, z_batch)
            residual_batch = (X_next_batch - pred_batch).abs().mean(dim=1)  # (batch,)
            residuals.append(residual_batch.cpu().numpy())
    
    causal_matrices = np.concatenate(causal_matrices, axis=0)  # (N, num_vars, num_vars)
    residuals = np.concatenate(residuals, axis=0)  # (N,)
    
    y_true = data_coarse['y_next'].cpu().numpy()
    idx_train = data_coarse['idx_train'].cpu().numpy()
    idx_val = data_coarse['idx_val'].cpu().numpy()
    idx_test = data_coarse['idx_test'].cpu().numpy()
    
    return causal_matrices, residuals, y_true, idx_train, idx_val, idx_test, relation_matrix_ref


def sparsify_causal_matrix(causal_mat, threshold_percentile=90):
    """
    Sparsify causal matrix by keeping only top connections
    Args:
        causal_mat: (num_vars, num_vars)
        threshold_percentile: percentile to use as threshold
    Returns:
        sparse_causal_mat: (num_vars, num_vars)
    """
    abs_mat = np.abs(causal_mat)
    threshold = np.percentile(abs_mat, threshold_percentile)
    sparse_mat = np.where(abs_mat >= threshold, causal_mat, 0)
    return sparse_mat


def compute_spatial_deviation(causal_mat, reference_mat):
    """
    Compute spatial deviation using Frobenius norm
    Args:
        causal_mat: (num_vars, num_vars) - per-window causal matrix
        reference_mat: (num_vars, num_vars) - reference causal matrix (median)
    Returns:
        spatial_dev: scalar
    """
    diff = causal_mat - reference_mat
    fro_norm = np.linalg.norm(diff, ord='fro')
    return fro_norm


def compute_temporal_deviation(residual, train_normal_residuals):
    """
    Compute temporal deviation using MAD-based score
    Args:
        residual: scalar - current window residual
        train_normal_residuals: array - training normal residuals for statistics
    Returns:
        temporal_dev: scalar - MAD-based deviation score
    """
    median_residual = np.median(train_normal_residuals)
    mad = np.median(np.abs(train_normal_residuals - median_residual))
    mad = max(mad, 1e-8)  # avoid division by zero
    
    temporal_dev = np.abs(residual - median_residual) / mad
    return temporal_dev


def compute_anomaly_scores(
    causal_matrices, residuals, y_true,
    idx_train, idx_val, idx_test,
    relation_matrix_ref,
    sparsify_threshold=90,
    spatial_weight=0.5,
    temporal_weight=0.5
):
    """
    Compute anomaly scores for all windows
    """
    # Step 1: Build reference from training normal windows
    train_normal_mask = (y_true[idx_train] == 0)
    train_normal_indices = idx_train[train_normal_mask]
    
    if len(train_normal_indices) == 0:
        raise ValueError("No normal samples in training set!")
    
    # Get causal matrices for training normal windows
    train_normal_causal = causal_matrices[train_normal_indices]  # (n_normal, num_vars, num_vars)
    
    # Compute median reference causal matrix
    reference_causal = np.median(train_normal_causal, axis=0)  # (num_vars, num_vars)
    
    # Get training normal residuals for MAD computation
    train_normal_residuals = residuals[train_normal_indices]
    
    print(f"Reference computed from {len(train_normal_indices)} normal training windows")
    print(f"Reference causal matrix shape: {reference_causal.shape}")
    print(f"Training normal residual stats: median={np.median(train_normal_residuals):.6f}, MAD={np.median(np.abs(train_normal_residuals - np.median(train_normal_residuals))):.6f}")
    
    # Step 2: Compute scores for all windows
    all_scores = []
    spatial_devs = []
    temporal_devs = []
    
    N = len(causal_matrices)
    for i in range(N):
        # Spatial deviation
        causal_mat = causal_matrices[i]
        sparse_causal = sparsify_causal_matrix(causal_mat, sparsify_threshold)
        spatial_dev = compute_spatial_deviation(sparse_causal, reference_causal)
        spatial_devs.append(spatial_dev)
        
        # Temporal deviation
        residual = residuals[i]
        temporal_dev = compute_temporal_deviation(residual, train_normal_residuals)
        temporal_devs.append(temporal_dev)
        
        # Combined anomaly score
        anomaly_score = spatial_weight * spatial_dev + temporal_weight * temporal_dev
        all_scores.append(anomaly_score)
    
    scores = np.array(all_scores)
    spatial_devs = np.array(spatial_devs)
    temporal_devs = np.array(temporal_devs)
    
    return {
        'scores': scores,
        'spatial_devs': spatial_devs,
        'temporal_devs': temporal_devs,
        'reference_causal': reference_causal,
        'train_normal_residuals': train_normal_residuals
    }


def evaluate_stage2(scores, y_true, idx_train, idx_val, idx_test, train_normal_residuals, 
                    spatial_devs=None, temporal_devs=None):
    """
    Evaluate Stage 2 using stable metrics:
    - Threshold from 99% percentile of training normal scores
    - AUPRC and ROC-AUC on test set
    - Effect size (attack median / normal median)
    - Spatial and temporal deviation breakdown
    """
    # Get training normal scores for threshold
    train_normal_mask = (y_true[idx_train] == 0)
    train_normal_indices = idx_train[train_normal_mask]
    train_normal_scores = scores[train_normal_indices]
    
    # Threshold: 99% percentile of training normal scores
    threshold = np.percentile(train_normal_scores, 99)
    
    # Test set evaluation
    test_scores = scores[idx_test]
    test_y = y_true[idx_test]
    test_normal_scores = test_scores[test_y == 0]
    test_attack_scores = test_scores[test_y == 1]
    
    # Metrics
    auprc = average_precision_score(test_y, test_scores)
    roc_auc = roc_auc_score(test_y, test_scores)
    
    # Effect size
    normal_median = np.median(test_normal_scores) if len(test_normal_scores) > 0 else 0
    attack_median = np.median(test_attack_scores) if len(test_attack_scores) > 0 else 0
    effect_size = attack_median / normal_median if normal_median > 0 else 0
    
    # Spatial and temporal deviation breakdown on test set
    test_spatial_normal_median = None
    test_spatial_attack_median = None
    test_temporal_normal_median = None
    test_temporal_attack_median = None
    
    if spatial_devs is not None and temporal_devs is not None:
        test_spatial_devs = spatial_devs[idx_test]
        test_temporal_devs = temporal_devs[idx_test]
        
        test_spatial_normal = test_spatial_devs[test_y == 0]
        test_spatial_attack = test_spatial_devs[test_y == 1]
        test_temporal_normal = test_temporal_devs[test_y == 0]
        test_temporal_attack = test_temporal_devs[test_y == 1]
        
        test_spatial_normal_median = np.median(test_spatial_normal) if len(test_spatial_normal) > 0 else 0
        test_spatial_attack_median = np.median(test_spatial_attack) if len(test_spatial_attack) > 0 else 0
        test_temporal_normal_median = np.median(test_temporal_normal) if len(test_temporal_normal) > 0 else 0
        test_temporal_attack_median = np.median(test_temporal_attack) if len(test_temporal_attack) > 0 else 0
    
    # Val set (for reference)
    val_scores = scores[idx_val]
    val_y = y_true[idx_val]
    val_normal_scores = val_scores[val_y == 0]
    val_attack_scores = val_scores[val_y == 1]
    
    val_normal_median = np.median(val_normal_scores) if len(val_normal_scores) > 0 else 0
    val_attack_median = np.median(val_attack_scores) if len(val_attack_scores) > 0 else 0
    val_effect_size = val_attack_median / val_normal_median if val_normal_median > 0 else 0
    
    results = {
        'threshold': threshold,
        'test_auprc': auprc,
        'test_roc_auc': roc_auc,
        'test_normal_median': normal_median,
        'test_attack_median': attack_median,
        'test_effect_size': effect_size,
        'val_normal_median': val_normal_median,
        'val_attack_median': val_attack_median,
        'val_effect_size': val_effect_size,
        'train_normal_q99': np.percentile(train_normal_scores, 99),
        'train_normal_median': np.median(train_normal_scores),
        'test_normal_scores': test_normal_scores,
        'test_attack_scores': test_attack_scores,
        'test_spatial_normal_median': test_spatial_normal_median,
        'test_spatial_attack_median': test_spatial_attack_median,
        'test_temporal_normal_median': test_temporal_normal_median,
        'test_temporal_attack_median': test_temporal_attack_median,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Stage 2 minimal validation for SWaT/PSM')
    parser.add_argument('--model_path', type=str, 
                       default='models/swat_stage1_multi_scale.pt',
                       help='Path to Stage 1 trained model')
    parser.add_argument('--data_dir', type=str, 
                       default='data/SWaT/processed',
                       help='Path to processed data directory (SWaT or PSM)')
    parser.add_argument('--dataset', type=str, default='auto',
                       choices=['auto', 'swat', 'psm'],
                       help='Dataset type: auto (detect from data_dir), swat, or psm')
    parser.add_argument('--sparsify_threshold', type=int, default=90,
                       help='Percentile threshold for sparsifying causal matrix')
    parser.add_argument('--spatial_weight', type=float, default=0.5,
                       help='Weight for spatial deviation in anomaly score')
    parser.add_argument('--temporal_weight', type=float, default=0.5,
                       help='Weight for temporal deviation in anomaly score')
    parser.add_argument('--spatial_only', action='store_true',
                       help='Use only spatial deviation, ignore temporal (overrides temporal_weight)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Auto-detect dataset type from data_dir if not specified
    data_dir = Path(args.data_dir)
    if args.dataset == 'auto':
        if 'swat' in str(data_dir).lower() or (data_dir / 'swat_hour.pt').exists():
            dataset_type = 'swat'
        elif 'psm' in str(data_dir).lower() or (data_dir / 'psm_coarse.pt').exists():
            dataset_type = 'psm'
        else:
            # Try to detect by checking which files exist
            if (data_dir / 'swat_hour.pt').exists():
                dataset_type = 'swat'
            elif (data_dir / 'psm_coarse.pt').exists():
                dataset_type = 'psm'
            else:
                raise ValueError(f"Cannot auto-detect dataset type from {data_dir}. Please specify --dataset explicitly.")
    else:
        dataset_type = args.dataset
    
    print("=" * 60)
    print("Stage 2 Minimal Validation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_type.upper()}")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_dir}")
    print(f"  Sparsify threshold: {args.sparsify_threshold}%")
    if args.spatial_only:
        print(f"  Mode: SPATIAL ONLY (temporal deviation disabled)")
        print(f"  Spatial weight: 1.0")
        print(f"  Temporal weight: 0.0")
    else:
        print(f"  Mode: COMBINED (spatial + temporal)")
        print(f"  Spatial weight: {args.spatial_weight}")
        print(f"  Temporal weight: {args.temporal_weight}")
    
    # Load data based on dataset type
    if dataset_type == 'swat':
        bundle_coarse = torch.load(data_dir / 'swat_hour.pt', weights_only=False)
        scale_name = 'Hour'
    elif dataset_type == 'psm':
        bundle_coarse = torch.load(data_dir / 'psm_coarse.pt', weights_only=False)
        scale_name = 'Coarse'
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"\nData loaded ({dataset_type.upper()}):")
    print(f"  {scale_name} scale shape: {bundle_coarse['X_seq'].shape}")
    print(f"  Train/Val/Test sizes: {len(bundle_coarse['idx_train'])}/{len(bundle_coarse['idx_val'])}/{len(bundle_coarse['idx_test'])}")
    
    # Extract causal matrices and residuals
    print(f"\nExtracting causal matrices and residuals...")
    causal_matrices, residuals, y_true, idx_train, idx_val, idx_test, relation_matrix_ref = \
        extract_causal_matrix_and_residual(
            args.model_path, bundle_coarse, device=args.device, batch_size=args.batch_size
        )
    
    print(f"  Causal matrices shape: {causal_matrices.shape}")
    print(f"  Residuals shape: {residuals.shape}")
    print(f"  Reference relation matrix shape: {relation_matrix_ref.shape}")
    
    # Compute anomaly scores
    print(f"\nComputing anomaly scores...")
    if args.spatial_only:
        spatial_weight = 1.0
        temporal_weight = 0.0
        print(f"  Using SPATIAL DEVIATION ONLY (temporal ignored)")
    else:
        spatial_weight = args.spatial_weight
        temporal_weight = args.temporal_weight
        print(f"  Using COMBINED MODE: spatial_weight={spatial_weight}, temporal_weight={temporal_weight}")
    
    score_results = compute_anomaly_scores(
        causal_matrices, residuals, y_true,
        idx_train, idx_val, idx_test,
        relation_matrix_ref,
        sparsify_threshold=args.sparsify_threshold,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight
    )
    
    scores = score_results['scores']
    
    print(f"\nScore statistics (all windows):")
    print(f"  Mean: {np.mean(scores):.6f}")
    print(f"  Std: {np.std(scores):.6f}")
    print(f"  Median: {np.median(scores):.6f}")
    print(f"  Q90: {np.percentile(scores, 90):.6f}")
    print(f"  Q95: {np.percentile(scores, 95):.6f}")
    print(f"  Q99: {np.percentile(scores, 99):.6f}")
    
    print(f"\nSpatial deviation statistics:")
    print(f"  Mean: {np.mean(score_results['spatial_devs']):.6f}")
    print(f"  Median: {np.median(score_results['spatial_devs']):.6f}")
    
    print(f"\nTemporal deviation statistics:")
    print(f"  Mean: {np.mean(score_results['temporal_devs']):.6f}")
    print(f"  Median: {np.median(score_results['temporal_devs']):.6f}")
    
    # Evaluate
    print(f"\nEvaluating Stage 2...")
    eval_results = evaluate_stage2(
        scores, y_true, idx_train, idx_val, idx_test,
        score_results['train_normal_residuals'],
        spatial_devs=score_results['spatial_devs'],
        temporal_devs=score_results['temporal_devs']
    )
    
    print(f"\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nThreshold (99% percentile of train normal): {eval_results['threshold']:.6f}")
    
    mode_label = "Spatial Only" if args.spatial_only else "Combined (Spatial + Temporal)"
    print(f"\nTest Set Metrics ({mode_label}):")
    print(f"  AUPRC: {eval_results['test_auprc']:.4f} {'✓' if eval_results['test_auprc'] > 0.5 else '✗'}")
    print(f"  ROC-AUC: {eval_results['test_roc_auc']:.4f} {'✓' if eval_results['test_roc_auc'] > 0.5 else '✗'}")
    print(f"  Normal score median: {eval_results['test_normal_median']:.6f}")
    print(f"  Attack score median: {eval_results['test_attack_median']:.6f}")
    print(f"  Effect size (attack/normal): {eval_results['test_effect_size']:.2f}x {'✓' if eval_results['test_effect_size'] > 1.2 else '✗'}")
    
    if eval_results['test_spatial_normal_median'] is not None:
        print(f"\nTest Set Deviation Breakdown:")
        print(f"  Spatial deviation median:")
        print(f"    Normal: {eval_results['test_spatial_normal_median']:.6f}")
        print(f"    Attack: {eval_results['test_spatial_attack_median']:.6f}")
        print(f"  Temporal deviation median:")
        print(f"    Normal: {eval_results['test_temporal_normal_median']:.6f}")
        print(f"    Attack: {eval_results['test_temporal_attack_median']:.6f}")
    
    print(f"\nValidation Set (reference):")
    print(f"  Normal score median: {eval_results['val_normal_median']:.6f}")
    print(f"  Attack score median: {eval_results['val_attack_median']:.6f}")
    print(f"  Effect size (attack/normal): {eval_results['val_effect_size']:.2f}x")
    
    print(f"\nTraining Normal Statistics:")
    print(f"  Median: {eval_results['train_normal_median']:.6f}")
    print(f"  Q99: {eval_results['train_normal_q99']:.6f}")
    
    # Decision
    print(f"\n" + "=" * 60)
    print("DECISION CRITERIA")
    print("=" * 60)
    
    passed_auprc = eval_results['test_auprc'] > 0.5
    passed_roc = eval_results['test_roc_auc'] > 0.5
    passed_effect = eval_results['test_effect_size'] > 1.2
    
    print(f"  AUPRC > 0.5: {'PASS' if passed_auprc else 'FAIL'} ({eval_results['test_auprc']:.4f})")
    print(f"  ROC-AUC > 0.5: {'PASS' if passed_roc else 'FAIL'} ({eval_results['test_roc_auc']:.4f})")
    print(f"  Effect size > 1.2x: {'PASS' if passed_effect else 'FAIL'} ({eval_results['test_effect_size']:.2f}x)")
    
    # Decision for spatial-only mode
    use_spatial_only = args.spatial_only
    if use_spatial_only:
        print(f"\n" + "=" * 60)
        print("SPATIAL-ONLY VALIDATION RESULT")
        print("=" * 60)
        if passed_auprc and passed_roc and passed_effect:
            print(f"✓ Spatial deviation (structural shift) is DETECTABLE!")
            print(f"  Even if residual magnitude fails, structural shift can still detect anomalies.")
        else:
            print(f"✗ Spatial deviation alone is NOT sufficient for detection.")
            if not passed_auprc:
                print(f"  Issue: AUPRC too low ({eval_results['test_auprc']:.4f})")
            if not passed_roc:
                print(f"  Issue: ROC-AUC too low ({eval_results['test_roc_auc']:.4f})")
            if not passed_effect:
                print(f"  Issue: Effect size too small ({eval_results['test_effect_size']:.2f}x)")
    else:
        if passed_auprc and passed_roc and passed_effect:
            print(f"\n✓ Stage 2 validation PASSED - Worth continuing!")
        else:
            print(f"\n✗ Stage 2 validation FAILED - Consider fixing Stage 1")
    
    print("=" * 60)


if __name__ == '__main__':
    main()

