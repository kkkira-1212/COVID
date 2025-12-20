import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from .encoder import TransformerSeqEncoder, PredictionHeads, RegressionHead
from .trainer import to_device


# def normalize_residual_np(residual, idx_train, eps=1e-8):
#     residual_train = residual[idx_train]
#     residual_mean = residual_train.mean()
#     residual_std = residual_train.std()
#     if residual_std < eps:
#         return residual
#     normalized_residual = (residual - residual_mean) / (residual_std + eps)
#     return normalized_residual


def compute_anomaly(residual, idx_val):
    mu = residual[idx_val].mean()
    sigma = residual[idx_val].std()
    return np.abs(residual - mu) / (sigma + 1e-8)


def find_threshold(probs, y, idx_val):
    ths = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y[idx_val], (probs[idx_val] >= t).astype(int), zero_division=0) for t in ths]
    return ths[np.argmax(f1s)]


def tune_strength(p_week, residual_week, y_week, idx_val):
    anomaly = compute_anomaly(residual_week, idx_val)
    strengths = [0.3, 0.5, 0.7, 0.9]
    best_s, best_f1 = 0.5, 0
    for s in strengths:
        p = np.clip(p_week + s * anomaly, 0, 1)
        ths = np.linspace(0.01, 0.99, 99)
        f1 = max(
            f1_score(y_week[idx_val], (p[idx_val] >= t).astype(int), zero_division=0)
            for t in ths
        )
        if f1 > best_f1:
            best_f1, best_s = f1, s
    return best_s


def find_residual_threshold(residual, y_true, idx_val):
    res = residual[idx_val]
    y = y_true[idx_val]
    ths = np.linspace(np.percentile(res, 10), np.percentile(res, 95), 30)
    f1s = [f1_score(y, (res >= t).astype(int), zero_division=0) for t in ths]
    return ths[np.argmax(f1s)] if len(ths) else 0.0


def evaluate_residual_scores(residual, y_true, idx_val, idx_test):
    th = find_residual_threshold(residual, y_true, idx_val)
    scores_te = residual[idx_test]
    y_te = y_true[idx_test]
    pred = (scores_te >= th).astype(int)
    return {
        "threshold": th,
        "recall": np.sum((pred == 1) & (y_te == 1)) / max(np.sum(y_te == 1), 1),
        "precision": np.sum((pred == 1) & (y_te == 1)) / max(np.sum(pred == 1), 1),
        "auprc": average_precision_score(y_te, scores_te),
        "roc_auc": roc_auc_score(y_te, scores_te),
        "f1": f1_score(y_te, pred, zero_division=0),
    }


def evaluate_supervised(probs, y_true, idx_val, idx_test):
    th = find_threshold(probs, y_true, idx_val)
    pred_test = (probs[idx_test] >= th).astype(int)
    y_test = y_true[idx_test]
    return {
        "threshold": th,
        "recall": np.sum((pred_test == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1),
        "precision": np.sum((pred_test == 1) & (y_test == 1)) / max(np.sum(pred_test == 1), 1),
        "auprc": average_precision_score(y_test, probs[idx_test]),
        "roc_auc": roc_auc_score(y_test, probs[idx_test]),
        "f1": f1_score(y_test, pred_test, zero_division=0),
    }


def evaluate_2b(probs, residuals, strength, y_true, idx_val, idx_test):
    anomaly = compute_anomaly(residuals, idx_val)
    probs_enhanced = np.clip(probs + strength * anomaly, 0, 1)
    th = find_threshold(probs_enhanced, y_true, idx_val)
    pred_test = (probs_enhanced[idx_test] >= th).astype(int)
    y_test = y_true[idx_test]
    return {
        "threshold": th,
        "recall": np.sum((pred_test == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1),
        "precision": np.sum((pred_test == 1) & (y_test == 1)) / max(np.sum(pred_test == 1), 1),
        "auprc": average_precision_score(y_test, probs_enhanced[idx_test]),
        "roc_auc": roc_auc_score(y_test, probs_enhanced[idx_test]),
        "f1": f1_score(y_test, pred_test, zero_division=0),
    }


def run_inference(
    model_path,
    bundle_week,
    bundle_day=None,
    device='cuda',
    use_postprocessing=False
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    data_week = to_device(bundle_week, device)
    
    weekly_only = config.get('weekly_only', False)
    use_classification = config.get('use_classification', False)
    
    if weekly_only:
        enc_w = TransformerSeqEncoder(
            input_dim=data_week['X_seq'].shape[2],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_seq_len=data_week['X_seq'].shape[1] + 5
        ).to(device)
        enc_w.pooling = config['pooling']
        head = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['d_model'], 1)
        ).to(device)
        enc_w.load_state_dict(checkpoint['enc_week'])
        head.load_state_dict(checkpoint['head'])
        enc_w.eval()
        head.eval()
        
        with torch.no_grad():
            zw = enc_w(data_week['X_seq'])
            pred_w = head(zw).squeeze(1)
        
        residual = (data_week['NewDeaths_ret_next'] - pred_w).abs().cpu().numpy()
        # idx_train = data_week['idx_train'].cpu().numpy()
        # residual = normalize_residual_np(residual, idx_train)
        
        return {
            'residual': residual,
            'y_true': data_week['y_next'].cpu().numpy(),
            'idx_val': data_week['idx_val'].cpu().numpy(),
            'idx_test': data_week['idx_test'].cpu().numpy(),
        }
    
    if bundle_day is None:
        raise ValueError("bundle_day required for multi-scale models")
    
    data_day = to_device(bundle_day, device)
    
    enc_d = TransformerSeqEncoder(
        input_dim=data_day['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_day['X_seq'].shape[1] + 5
    ).to(device)
    enc_d.pooling = config['pooling']
    
    enc_w = TransformerSeqEncoder(
        input_dim=data_week['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_week['X_seq'].shape[1] + 5
    ).to(device)
    enc_w.pooling = config['pooling']
    
    if use_classification:
        heads = PredictionHeads(config['d_model']).to(device)
        heads.load_state_dict(checkpoint['heads'])
    else:
        heads = RegressionHead(config['d_model']).to(device)
        heads.load_state_dict(checkpoint['head'])
    
    enc_d.load_state_dict(checkpoint['enc_day'])
    enc_w.load_state_dict(checkpoint['enc_week'])
    enc_d.eval()
    enc_w.eval()
    heads.eval()
    
    with torch.no_grad():
        zd = enc_d(data_day['X_seq'])
        zw = enc_w(data_week['X_seq'])
        
        if use_classification:
            logit_d, logit_w, pred_d, pred_w = heads(zd, zw)
            p_day = torch.sigmoid(logit_d).cpu().numpy()
            p_week = torch.sigmoid(logit_w).cpu().numpy()
            residual_week = (data_week['NewDeaths_ret_next'] - pred_w).abs().cpu().numpy()
            residual_day = (data_day['NewDeaths_ret_next'] - pred_d).abs().cpu().numpy()
            
            if use_postprocessing:
                strength = tune_strength(
                    p_week, residual_week,
                    data_week['y_next'].cpu().numpy(),
                    data_week['idx_val'].cpu().numpy()
                )
                anomaly_week = compute_anomaly(residual_week, data_week['idx_val'].cpu().numpy())
                p_enhanced = np.clip(p_week + strength * anomaly_week, 0, 1)
                residual = 1.0 - p_enhanced
            else:
                residual = 1.0 - p_week
            
            return {
                'p_day': p_day,
                'p_week': p_week,
                'res_day': residual_day,
                'res_week': residual_week,
                'y_day': data_day['y_next'].cpu().numpy(),
                'y_week': data_week['y_next'].cpu().numpy(),
                'idx_val_day': data_day['idx_val'].cpu().numpy(),
                'idx_val_week': data_week['idx_val'].cpu().numpy(),
                'idx_test_day': data_day['idx_test'].cpu().numpy(),
                'idx_test_week': data_week['idx_test'].cpu().numpy(),
            }
        else:
            pred_d, pred_w = heads(zd, zw)
            residual_week = (data_week['NewDeaths_ret_next'] - pred_w).abs().cpu().numpy()
            residual_day = (data_day['NewDeaths_ret_next'] - pred_d).abs().cpu().numpy()
            # idx_train_week = data_week['idx_train'].cpu().numpy()
            # idx_train_day = data_day['idx_train'].cpu().numpy()
            # residual_week = normalize_residual_np(residual_week, idx_train_week)
            # residual_day = normalize_residual_np(residual_day, idx_train_day)
            
            if use_postprocessing:
                map_dw = data_day.get('day_to_week_index', None)
                if map_dw is None:
                    residual = residual_week
                else:
                    if isinstance(map_dw, torch.Tensor):
                        map_dw = map_dw.cpu().numpy()
                    else:
                        map_dw = np.array(map_dw)
                    idx_val_week = data_week['idx_val'].cpu().numpy()
                    idx_val_day = data_day['idx_val'].cpu().numpy()
                    y_val = data_week['y_next'][data_week['idx_val']].cpu().numpy()
                    
                    res_val_week = residual_week[idx_val_week]
                    res_val_day_aligned = np.zeros_like(res_val_week)
                    
                    for i, w_idx in enumerate(idx_val_week):
                        day_mask = (map_dw == w_idx) & np.isin(np.arange(len(map_dw)), idx_val_day)
                        day_indices = np.where(day_mask)[0]
                        if len(day_indices) > 0:
                            res_val_day_aligned[i] = residual_day[day_indices].mean()
                        else:
                            res_val_day_aligned[i] = res_val_week[i]
                    
                    alphas = np.linspace(0.0, 1.0, 11)
                    best_alpha, best_f1 = 0.0, 0.0
                    for alpha in alphas:
                        fused_res = (1 - alpha) * res_val_week + alpha * res_val_day_aligned
                        ths = np.linspace(np.percentile(fused_res, 10), np.percentile(fused_res, 95), 25)
                        f1s = [f1_score(y_val, (fused_res >= t).astype(int), zero_division=0) for t in ths]
                        max_f1 = max(f1s) if len(f1s) else 0.0
                        if max_f1 > best_f1:
                            best_f1, best_alpha = max_f1, alpha
                    
                    res_day_aligned_all = np.zeros_like(residual_week)
                    for w_idx in range(len(residual_week)):
                        day_indices = np.where(map_dw == w_idx)[0]
                        if len(day_indices) > 0:
                            res_day_aligned_all[w_idx] = residual_day[day_indices].mean()
                        else:
                            res_day_aligned_all[w_idx] = residual_week[w_idx]
                    
                    residual = (1 - best_alpha) * residual_week + best_alpha * res_day_aligned_all
            else:
                residual = residual_week
            
            return {
                'residual': residual,
                'y_true': data_week['y_next'].cpu().numpy(),
                'idx_val': data_week['idx_val'].cpu().numpy(),
                'idx_test': data_week['idx_test'].cpu().numpy(),
            }
