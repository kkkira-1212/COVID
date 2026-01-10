import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .trainer import todevice, getregressiontarget
from .encoder import TransformerSeqEncoder, RegressionHeadWithRelation
from .causal import (
    compute_gradient_causal_matrix,
    sparsify,
    compute_spatial_deviation,
    compute_temporal_deviation,
    fuse_scores
)


def infer_c(
    model_path,
    bundle_coarse,
    bundle_fine=None,
    device='cuda',
    batch_size=32
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    data_coarse = todevice(bundle_coarse, device)
    coarse_only = config.get('coarse_only', False)
    num_vars = data_coarse['X_seq'].shape[2]
    threshold_percentile = config.get('threshold_percentile', 75.0)
    weight_spatial = config.get('weight_spatial', 0.5)
    
    if coarse_only:
        enc_coarse = TransformerSeqEncoder(
            input_dim=data_coarse['X_seq'].shape[2],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_seq_len=data_coarse['X_seq'].shape[1] + 5
        ).to(device)
        enc_coarse.pooling = config['pooling']
        enc_coarse.load_state_dict(checkpoint['enc_coarse'])
        
        head = RegressionHeadWithRelation(config['d_model'], num_vars).to(device)
        head.load_state_dict(checkpoint['head'])
        
        normal_reference_coarse = checkpoint['normal_reference_coarse'].to(device)
        normal_median_coarse = checkpoint['normal_median_coarse'].to(device)
        normal_mad_coarse = checkpoint['normal_mad_coarse'].to(device)
        
        enc_coarse.eval()
        head.eval()
        
        X_all = data_coarse['X_seq']
        anomaly_scores = []
        
        with torch.no_grad():
            z_coarse = enc_coarse(X_all)
            _, pred_coarse_all = head(z_coarse, z_coarse)
            
            if 'X_next' in data_coarse:
                residual_vec_coarse = (data_coarse['X_next'] - pred_coarse_all).abs()
            else:
                target = getregressiontarget(data_coarse).unsqueeze(1).expand_as(pred_coarse_all)
                residual_vec_coarse = (target - pred_coarse_all).abs()
        
        for i in range(0, len(X_all), batch_size):
            batch_X = X_all[i:i+batch_size]
            causal_batch = compute_gradient_causal_matrix(batch_X, enc_coarse, head, use_fine=False)
            causal_batch = sparsify(causal_batch, threshold_percentile=threshold_percentile)
            
            spatial_scores = compute_spatial_deviation(causal_batch, normal_reference_coarse)
            temporal_scores = compute_temporal_deviation(
                residual_vec_coarse[i:i+batch_size],
                normal_median_coarse,
                normal_mad_coarse
            )
            
            fused_scores = fuse_scores(spatial_scores, temporal_scores, weight_spatial)
            anomaly_scores.append(fused_scores.cpu())
        
        anomaly_scores = torch.cat(anomaly_scores, dim=0).numpy()
        
        return {
            'anomaly_scores': anomaly_scores,
            'y_true': data_coarse['y_next'].cpu().numpy(),
            'idx_val': data_coarse['idx_val'].cpu().numpy(),
            'idx_test': data_coarse['idx_test'].cpu().numpy(),
        }
    
    if bundle_fine is None:
        raise ValueError("bundle_fine required for multi-scale models")
    
    data_fine = todevice(bundle_fine, device)
    
    enc_fine = TransformerSeqEncoder(
        input_dim=data_fine['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_fine['X_seq'].shape[1] + 5
    ).to(device)
    enc_fine.pooling = config['pooling']
    enc_fine.load_state_dict(checkpoint['enc_fine'])
    
    enc_coarse = TransformerSeqEncoder(
        input_dim=data_coarse['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_coarse['X_seq'].shape[1] + 5
    ).to(device)
    enc_coarse.pooling = config['pooling']
    enc_coarse.load_state_dict(checkpoint['enc_coarse'])
    
    head = RegressionHeadWithRelation(config['d_model'], num_vars).to(device)
    head.load_state_dict(checkpoint['head'])
    
    normal_reference_fine = checkpoint['normal_reference_fine'].to(device)
    normal_reference_coarse = checkpoint['normal_reference_coarse'].to(device)
    normal_median_fine = checkpoint['normal_median_fine'].to(device)
    normal_mad_fine = checkpoint['normal_mad_fine'].to(device)
    normal_median_coarse = checkpoint['normal_median_coarse'].to(device)
    normal_mad_coarse = checkpoint['normal_mad_coarse'].to(device)
    
    enc_fine.eval()
    enc_coarse.eval()
    head.eval()
    
    X_all_fine = data_fine['X_seq']
    X_all_coarse = data_coarse['X_seq']
    
    anomaly_scores_fine = []
    anomaly_scores_coarse = []
    
    with torch.no_grad():
        z_fine = enc_fine(X_all_fine)
        z_coarse = enc_coarse(X_all_coarse)
        pred_fine_all, pred_coarse_all = head(z_fine, z_coarse)
        
        if 'X_next' in data_fine:
            residual_vec_fine = (data_fine['X_next'] - pred_fine_all).abs()
        else:
            target_fine = getregressiontarget(data_fine).unsqueeze(1).expand_as(pred_fine_all)
            residual_vec_fine = (target_fine - pred_fine_all).abs()
        
        if 'X_next' in data_coarse:
            residual_vec_coarse = (data_coarse['X_next'] - pred_coarse_all).abs()
        else:
            target_coarse = getregressiontarget(data_coarse).unsqueeze(1).expand_as(pred_coarse_all)
            residual_vec_coarse = (target_coarse - pred_coarse_all).abs()
    
    for i in range(0, len(X_all_fine), batch_size):
        batch_X = X_all_fine[i:i+batch_size]
        causal_batch = compute_gradient_causal_matrix(batch_X, enc_fine, head, use_fine=True)
        causal_batch = sparsify(causal_batch, threshold_percentile=threshold_percentile)
        
        spatial_scores = compute_spatial_deviation(causal_batch, normal_reference_fine)
        temporal_scores = compute_temporal_deviation(
            residual_vec_fine[i:i+batch_size],
            normal_median_fine,
            normal_mad_fine
        )
        
        fused_scores = fuse_scores(spatial_scores, temporal_scores, weight_spatial)
        anomaly_scores_fine.append(fused_scores.cpu())
    
    for i in range(0, len(X_all_coarse), batch_size):
        batch_X = X_all_coarse[i:i+batch_size]
        causal_batch = compute_gradient_causal_matrix(batch_X, enc_coarse, head, use_fine=False)
        causal_batch = sparsify(causal_batch, threshold_percentile=threshold_percentile)
        
        spatial_scores = compute_spatial_deviation(causal_batch, normal_reference_coarse)
        temporal_scores = compute_temporal_deviation(
            residual_vec_coarse[i:i+batch_size],
            normal_median_coarse,
            normal_mad_coarse
        )
        
        fused_scores = fuse_scores(spatial_scores, temporal_scores, weight_spatial)
        anomaly_scores_coarse.append(fused_scores.cpu())
    
    anomaly_scores_fine = torch.cat(anomaly_scores_fine, dim=0).numpy()
    anomaly_scores_coarse = torch.cat(anomaly_scores_coarse, dim=0).numpy()
    
    map_fc = data_fine.get('fine_to_coarse_index', None)
    if map_fc is None:
        map_fc = data_fine.get('day_to_week_index', None)
    
    if map_fc is not None:
        if isinstance(map_fc, torch.Tensor):
            map_fc = map_fc.cpu().numpy()
        else:
            map_fc = np.array(map_fc)
        
        anomaly_scores_coarse_aligned = np.zeros_like(anomaly_scores_fine)
        for i, coarse_idx in enumerate(map_fc):
            if coarse_idx >= 0 and coarse_idx < len(anomaly_scores_coarse):
                anomaly_scores_coarse_aligned[i] = anomaly_scores_coarse[coarse_idx]
        
        anomaly_scores = (anomaly_scores_fine + anomaly_scores_coarse_aligned) / 2.0
    else:
        anomaly_scores = anomaly_scores_coarse
    
    return {
        'anomaly_scores': anomaly_scores,
        'y_true': data_coarse['y_next'].cpu().numpy(),
        'idx_val': data_coarse['idx_val'].cpu().numpy(),
        'idx_test': data_coarse['idx_test'].cpu().numpy(),
    }


def evaluate_c(anomaly_scores, y_true, idx_val, idx_test):
    scores_val = anomaly_scores[idx_val]
    scores_test = anomaly_scores[idx_test]
    y_val = y_true[idx_val]
    y_test = y_true[idx_test]
    
    if len(np.unique(y_test)) < 2:
        return {
            "roc_auc": 0.0,
            "auprc": 0.0,
        }
    
    roc_auc = roc_auc_score(y_test, scores_test)
    auprc = average_precision_score(y_test, scores_test)
    
    return {
        "roc_auc": roc_auc,
        "auprc": auprc,
    }

