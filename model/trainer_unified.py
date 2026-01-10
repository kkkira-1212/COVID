import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from .trainer import todevice, getregressiontarget
from .encoder import TransformerSeqEncoder, RegressionHeadWithRelation
from .causal import (
    compute_gradient_causal_matrix,
    sparsify,
    aggregate_normal_reference,
    compute_normal_temporal_stats,
    compute_spatial_deviation,
    compute_temporal_deviation,
    fuse_scores
)


def train_unified(
    bundle_coarse,
    bundle_fine=None,
    save_path=None,
    coarse_only=False,
    use_lu=False,
    lambda_u=1.0,
    epochs=200,
    lr=3e-4,
    weight_decay=1e-4,
    patience_limit=30,
    d_model=64,
    nhead=4,
    num_layers=2,
    pooling="last",
    threshold_percentile=75.0,
    weight_spatial=0.5,
    batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_coarse = todevice(bundle_coarse, device)
    
    bundle_week = bundle_coarse
    bundle_day = bundle_fine
    weekly_only = coarse_only
    data_week = data_coarse
    
    if weekly_only:
        r_coarse = getregressiontarget(data_week)
        num_vars = data_week['X_seq'].shape[2]
        
        data = {
            'Xc': data_week['X_seq'],
            'Xc_next': data_week.get('X_next', None),
            'rc': r_coarse,
            'yc': data_week['y_next'],
            'idxc_tr': data_week['idx_train'],
            'idxc_val': data_week['idx_val'],
            'idxc_te': data_week['idx_test'],
        }
        
        enc_coarse = TransformerSeqEncoder(
            input_dim=data['Xc'].shape[2], d_model=d_model, nhead=nhead,
            num_layers=num_layers, max_seq_len=data['Xc'].shape[1] + 5
        ).to(device)
        enc_coarse.pooling = pooling
        
        head = RegressionHeadWithRelation(d_model, num_vars).to(device)
        head_coarse = head
        
        opt = torch.optim.AdamW(
            list(enc_coarse.parameters()) + list(head.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        
        best_roc_auc = 0.0
        patience = 0
        
        for ep in range(epochs):
            enc_coarse.train()
            head.train()
            opt.zero_grad()
            
            z_coarse = enc_coarse(data['Xc'])
            _, pred_coarse_all = head_coarse(z_coarse, z_coarse)
            
            if data['Xc_next'] is not None:
                loss = F.mse_loss(pred_coarse_all[data['idxc_tr']], data['Xc_next'][data['idxc_tr']])
            else:
                target = data['rc'][data['idxc_tr']].unsqueeze(1).expand(-1, num_vars)
                loss = F.mse_loss(pred_coarse_all[data['idxc_tr']], target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            
            enc_coarse.eval()
            head.eval()
            with torch.no_grad():
                z_coarse = enc_coarse(data['Xc'])
                _, pred_coarse_all = head_coarse(z_coarse, z_coarse)
                
                if data['Xc_next'] is not None:
                    residual_vec = (data['Xc_next'] - pred_coarse_all).abs()
                else:
                    target = data['rc'].unsqueeze(1).expand_as(pred_coarse_all)
                    residual_vec = (target - pred_coarse_all).abs()
                
                residual = residual_vec.mean(dim=1)
                res_val = residual[data['idxc_val']].cpu().numpy()
                y_val = data['yc'][data['idxc_val']].cpu().numpy()
                if len(np.unique(y_val)) >= 2:
                    roc_auc = roc_auc_score(y_val, res_val)
                else:
                    roc_auc = 0.0
            
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                patience = 0
                torch.save({
                    "epoch": ep,
                    "enc_coarse": enc_coarse.state_dict(),
                    "head": head.state_dict(),
                    "config": {
                        "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                        "pooling": pooling, "coarse_only": True,
                    }
                }, save_path)
            else:
                patience += 1
                if patience >= patience_limit:
                    break
        
        checkpoint = torch.load(save_path, map_location=device)
        enc_coarse.load_state_dict(checkpoint['enc_coarse'])
        head.load_state_dict(checkpoint['head'])
        enc_coarse.eval()
        head.eval()
        
        idx_train = data['idxc_tr']
        X_train = data['Xc'][idx_train]
        
        causal_matrices_coarse = []
        residual_vecs_coarse = []
        
        with torch.no_grad():
            z_coarse = enc_coarse(X_train)
            _, pred_coarse_all = head(z_coarse, z_coarse)
            
            if data['Xc_next'] is not None:
                residual_vec_coarse = (data['Xc_next'][idx_train] - pred_coarse_all).abs()
            else:
                target = data['rc'][idx_train].unsqueeze(1).expand_as(pred_coarse_all)
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
                "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                "pooling": pooling, "coarse_only": True,
                "threshold_percentile": threshold_percentile,
                "weight_spatial": weight_spatial
            }
        }, save_path)
        
        return save_path
    
    if bundle_fine is None:
        raise ValueError("bundle_fine required for multi-scale models")
    
    data_fine = todevice(bundle_fine, device)
    r_fine = getregressiontarget(data_fine)
    r_coarse = getregressiontarget(data_week)
    num_vars = data_week['X_seq'].shape[2]
    
    map_fc = data_fine.get('fine_to_coarse_index', None)
    if map_fc is None:
        map_fc = data_fine.get('day_to_week_index', None)
    
    data = {
        'Xf': data_fine['X_seq'],
        'Xf_next': data_fine.get('X_next', None),
        'rf': r_fine,
        'idxf_tr': data_fine['idx_train'],
        'idxf_val': data_fine['idx_val'],
        'idxf_te': data_fine['idx_test'],
        'map_fc': map_fc,
        'Xc': data_week['X_seq'],
        'Xc_next': data_week.get('X_next', None),
        'rc': r_coarse,
        'yc': data_week['y_next'],
        'idxc_tr': data_week['idx_train'],
        'idxc_val': data_week['idx_val'],
        'idxc_te': data_week['idx_test'],
    }
    
    enc_fine = TransformerSeqEncoder(
        input_dim=data['Xf'].shape[2], d_model=d_model, nhead=nhead,
        num_layers=num_layers, max_seq_len=data['Xf'].shape[1] + 5
    ).to(device)
    enc_fine.pooling = pooling
    
    enc_coarse = TransformerSeqEncoder(
        input_dim=data['Xc'].shape[2], d_model=d_model, nhead=nhead,
        num_layers=num_layers, max_seq_len=data['Xc'].shape[1] + 5
    ).to(device)
    enc_coarse.pooling = pooling
    
    heads = RegressionHeadWithRelation(d_model, num_vars).to(device)
    
    opt = torch.optim.AdamW(
        list(enc_fine.parameters()) + list(enc_coarse.parameters()) + list(heads.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    
    best_roc_auc = 0.0
    patience = 0
    
    for ep in range(epochs):
        enc_fine.train()
        enc_coarse.train()
        heads.train()
        opt.zero_grad()
        
        z_fine = enc_fine(data['Xf'])
        z_coarse = enc_coarse(data['Xc'])
        
        pred_fine_all, pred_coarse_all = heads(z_fine, z_coarse)
        
        if data['Xf_next'] is not None:
            mse_fine = F.mse_loss(pred_fine_all[data['idxf_tr']], data['Xf_next'][data['idxf_tr']])
        else:
            target_fine = data['rf'][data['idxf_tr']].unsqueeze(1).expand(-1, num_vars)
            mse_fine = F.mse_loss(pred_fine_all[data['idxf_tr']], target_fine)
        
        if data['Xc_next'] is not None:
            mse_coarse = F.mse_loss(pred_coarse_all[data['idxc_tr']], data['Xc_next'][data['idxc_tr']])
        else:
            target_coarse = data['rc'][data['idxc_tr']].unsqueeze(1).expand(-1, num_vars)
            mse_coarse = F.mse_loss(pred_coarse_all[data['idxc_tr']], target_coarse)
        
        loss = mse_fine + mse_coarse
        
        if use_lu and data['map_fc'] is not None:
            if data['Xf_next'] is not None:
                u_fine = (data['Xf_next'] - pred_fine_all).abs().mean(dim=1)
            else:
                target_fine = data['rf'].unsqueeze(1).expand_as(pred_fine_all)
                u_fine = (target_fine - pred_fine_all).abs().mean(dim=1)
            
            if data['Xc_next'] is not None:
                u_coarse = (data['Xc_next'] - pred_coarse_all).abs().mean(dim=1)
            else:
                target_coarse = data['rc'].unsqueeze(1).expand_as(pred_coarse_all)
                u_coarse = (target_coarse - pred_coarse_all).abs().mean(dim=1)
            
            map_tr = data['map_fc'][data['idxf_tr']]
            mask = (map_tr >= 0)
            if mask.sum() > 0:
                Lu = F.smooth_l1_loss(u_fine[data['idxf_tr']][mask], u_coarse[map_tr[mask]])
                loss = loss + lambda_u * Lu
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        opt.step()
        
        enc_fine.eval()
        enc_coarse.eval()
        heads.eval()
        with torch.no_grad():
            z_fine = enc_fine(data['Xf'])
            z_coarse = enc_coarse(data['Xc'])
            pred_fine_all, pred_coarse_all = heads(z_fine, z_coarse)
            
            if data['Xc_next'] is not None:
                residual_vec_coarse = (data['Xc_next'] - pred_coarse_all).abs()
            else:
                target_coarse = data['rc'].unsqueeze(1).expand_as(pred_coarse_all)
                residual_vec_coarse = (target_coarse - pred_coarse_all).abs()
            
            residual = residual_vec_coarse.mean(dim=1)
            res_val = residual[data['idxc_val']].cpu().numpy()
            y_val = data['yc'][data['idxc_val']].cpu().numpy()
            if len(np.unique(y_val)) >= 2:
                roc_auc = roc_auc_score(y_val, res_val)
            else:
                roc_auc = 0.0
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            patience = 0
            save_dict = {
                "epoch": ep,
                "enc_fine": enc_fine.state_dict(),
                "enc_coarse": enc_coarse.state_dict(),
                "head": heads.state_dict(),
                "config": {
                    "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                    "pooling": pooling, "coarse_only": False,
                    "use_lu": use_lu
                }
            }
            if use_lu:
                save_dict["config"]["lambda_u"] = lambda_u
            torch.save(save_dict, save_path)
        else:
            patience += 1
            if patience >= patience_limit:
                break
    
    checkpoint = torch.load(save_path, map_location=device)
    enc_fine.load_state_dict(checkpoint['enc_fine'])
    enc_coarse.load_state_dict(checkpoint['enc_coarse'])
    head.load_state_dict(checkpoint['head'])
    enc_fine.eval()
    enc_coarse.eval()
    head.eval()
    
    idx_train_fine = data['idxf_tr']
    idx_train_coarse = data['idxc_tr']
    X_train_fine = data['Xf'][idx_train_fine]
    X_train_coarse = data['Xc'][idx_train_coarse]
    
    causal_matrices_fine = []
    causal_matrices_coarse = []
    residual_vecs_fine = []
    residual_vecs_coarse = []
    
    with torch.no_grad():
        z_fine = enc_fine(X_train_fine)
        z_coarse = enc_coarse(X_train_coarse)
        pred_fine_all, pred_coarse_all = head(z_fine, z_coarse)
        
        if data['Xf_next'] is not None:
            residual_vec_fine = (data['Xf_next'][idx_train_fine] - pred_fine_all).abs()
        else:
            target_fine = data['rf'][idx_train_fine].unsqueeze(1).expand_as(pred_fine_all)
            residual_vec_fine = (target_fine - pred_fine_all).abs()
        
        if data['Xc_next'] is not None:
            residual_vec_coarse = (data['Xc_next'][idx_train_coarse] - pred_coarse_all).abs()
        else:
            target_coarse = data['rc'][idx_train_coarse].unsqueeze(1).expand_as(pred_coarse_all)
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
            "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
            "pooling": pooling, "coarse_only": False,
            "use_lu": use_lu, "lambda_u": lambda_u if use_lu else None,
            "threshold_percentile": threshold_percentile,
            "weight_spatial": weight_spatial
        }
    }, save_path)
    
    return save_path

