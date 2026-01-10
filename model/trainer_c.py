import torch
from .trainer import todevice, getregressiontarget
from .encoder import TransformerSeqEncoder, RegressionHeadWithRelation
from .causal import (
    compute_gradient_causal_matrix,
    sparsify,
    aggregate_normal_reference,
    compute_normal_temporal_stats
)


def train_track_c(
    bundle_coarse,
    bundle_fine=None,
    pred_model_path=None,
    save_path=None,
    coarse_only=False,
    threshold_percentile=75.0,
    weight_spatial=0.5,
    batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pred_model_path is None:
        raise ValueError("pred_model_path required for Track C training")
    
    checkpoint = torch.load(pred_model_path, map_location=device)
    config = checkpoint['config']
    
    data_coarse = todevice(bundle_coarse, device)
    coarse_only = coarse_only or config.get('coarse_only', False)
    num_vars = data_coarse['X_seq'].shape[2]
    
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
        
        enc_coarse.eval()
        head.eval()
        
        idx_train = data_coarse['idx_train']
        X_train = data_coarse['X_seq'][idx_train]
        
        causal_matrices_coarse = []
        residual_vecs_coarse = []
        
        with torch.no_grad():
            z_coarse = enc_coarse(X_train)
            _, pred_coarse_all = head(z_coarse, z_coarse)
            
            if 'X_next' in data_coarse:
                residual_vec_coarse = (data_coarse['X_next'][idx_train] - pred_coarse_all).abs()
            else:
                target = getregressiontarget(data_coarse)[idx_train].unsqueeze(1).expand_as(pred_coarse_all)
                residual_vec_coarse = (target - pred_coarse_all).abs()
            
            residual_vecs_coarse.append(residual_vec_coarse)
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            causal_batch = compute_gradient_causal_matrix(batch_X, enc_coarse, head, use_fine=False)
            causal_batch = sparsify(causal_batch, threshold_percentile=threshold_percentile)
            causal_matrices_coarse.append(causal_batch.cpu())
        
        causal_matrices_coarse = torch.cat(causal_matrices_coarse, dim=0)
        residual_vecs_coarse = torch.cat(residual_vecs_coarse, dim=0)
        
        normal_reference_coarse = aggregate_normal_reference(causal_matrices_coarse)
        normal_median_coarse, normal_mad_coarse = compute_normal_temporal_stats(residual_vecs_coarse)
        
        torch.save({
            "enc_coarse": enc_coarse.state_dict(),
            "head": head.state_dict(),
            "normal_reference_coarse": normal_reference_coarse,
            "normal_median_coarse": normal_median_coarse,
            "normal_mad_coarse": normal_mad_coarse,
            "config": {
                **config,
                "coarse_only": True,
                "threshold_percentile": threshold_percentile,
                "weight_spatial": weight_spatial
            }
        }, save_path)
        
        return save_path
    
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
    
    enc_fine.eval()
    enc_coarse.eval()
    head.eval()
    
    idx_train_fine = data_fine['idx_train']
    idx_train_coarse = data_coarse['idx_train']
    X_train_fine = data_fine['X_seq'][idx_train_fine]
    X_train_coarse = data_coarse['X_seq'][idx_train_coarse]
    
    causal_matrices_fine = []
    causal_matrices_coarse = []
    residual_vecs_fine = []
    residual_vecs_coarse = []
    
    with torch.no_grad():
        z_fine = enc_fine(X_train_fine)
        z_coarse = enc_coarse(X_train_coarse)
        pred_fine_all, pred_coarse_all = head(z_fine, z_coarse)
        
        if 'X_next' in data_fine:
            residual_vec_fine = (data_fine['X_next'][idx_train_fine] - pred_fine_all).abs()
        else:
            target_fine = getregressiontarget(data_fine)[idx_train_fine].unsqueeze(1).expand_as(pred_fine_all)
            residual_vec_fine = (target_fine - pred_fine_all).abs()
        
        if 'X_next' in data_coarse:
            residual_vec_coarse = (data_coarse['X_next'][idx_train_coarse] - pred_coarse_all).abs()
        else:
            target_coarse = getregressiontarget(data_coarse)[idx_train_coarse].unsqueeze(1).expand_as(pred_coarse_all)
            residual_vec_coarse = (target_coarse - pred_coarse_all).abs()
        
        residual_vecs_fine.append(residual_vec_fine)
        residual_vecs_coarse.append(residual_vec_coarse)
    
    for i in range(0, len(X_train_fine), batch_size):
        batch_X = X_train_fine[i:i+batch_size]
        causal_batch = compute_gradient_causal_matrix(batch_X, enc_fine, head, use_fine=True)
        causal_batch = sparsify(causal_batch, threshold_percentile=threshold_percentile)
        causal_matrices_fine.append(causal_batch.cpu())
    
    for i in range(0, len(X_train_coarse), batch_size):
        batch_X = X_train_coarse[i:i+batch_size]
        causal_batch = compute_gradient_causal_matrix(batch_X, enc_coarse, head, use_fine=False)
        causal_batch = sparsify(causal_batch, threshold_percentile=threshold_percentile)
        causal_matrices_coarse.append(causal_batch.cpu())
    
    causal_matrices_fine = torch.cat(causal_matrices_fine, dim=0)
    causal_matrices_coarse = torch.cat(causal_matrices_coarse, dim=0)
    residual_vecs_fine = torch.cat(residual_vecs_fine, dim=0)
    residual_vecs_coarse = torch.cat(residual_vecs_coarse, dim=0)
    
    normal_reference_fine = aggregate_normal_reference(causal_matrices_fine)
    normal_reference_coarse = aggregate_normal_reference(causal_matrices_coarse)
    normal_median_fine, normal_mad_fine = compute_normal_temporal_stats(residual_vecs_fine)
    normal_median_coarse, normal_mad_coarse = compute_normal_temporal_stats(residual_vecs_coarse)
    
    torch.save({
        "enc_fine": enc_fine.state_dict(),
        "enc_coarse": enc_coarse.state_dict(),
        "head": head.state_dict(),
        "normal_reference_fine": normal_reference_fine,
        "normal_reference_coarse": normal_reference_coarse,
        "normal_median_fine": normal_median_fine,
        "normal_mad_fine": normal_mad_fine,
        "normal_median_coarse": normal_median_coarse,
        "normal_mad_coarse": normal_mad_coarse,
        "config": {
            **config,
            "coarse_only": False,
            "threshold_percentile": threshold_percentile,
            "weight_spatial": weight_spatial
        }
    }, save_path)
    
    return save_path

