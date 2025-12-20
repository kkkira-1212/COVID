import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from .encoder import TransformerSeqEncoder, PredictionHeads, RegressionHead


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=torch.tensor(self.pos_weight, device=logits.device)
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        w = (1 - pt)**self.gamma
        a = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (a * w * bce).mean()


def to_device(data_dict, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data_dict.items()}


# def normalize_residual(residual, idx_train, eps=1e-8):
#     residual_train = residual[idx_train]
#     residual_mean = residual_train.mean()
#     residual_std = residual_train.std()
#     if residual_std < eps:
#         return residual
#     normalized_residual = (residual - residual_mean) / (residual_std + eps)
#     return normalized_residual


def train_ours(
    bundle_week,
    bundle_day=None,
    save_path=None,
    weekly_only=False,
    use_classification=False,
    use_lu=False,
    alpha_cls=1.0,
    alpha_reg=0.1,
    lambda_u=1.0,
    epochs=200,
    lr=3e-4,
    weight_decay=1e-4,
    patience_limit=30,
    d_model=64,
    nhead=4,
    num_layers=2,
    pooling="last"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_week = to_device(bundle_week, device)
    
    if weekly_only:
        data = {
            'Xw': data_week['X_seq'],
            'rw': data_week['NewDeaths_ret_next'],
            'yw': data_week['y_next'],
            'idxw_tr': data_week['idx_train'],
            'idxw_val': data_week['idx_val'],
            'idxw_te': data_week['idx_test'],
        }
        
        enc_w = TransformerSeqEncoder(
            input_dim=data['Xw'].shape[2], d_model=d_model, nhead=nhead,
            num_layers=num_layers, max_seq_len=data['Xw'].shape[1] + 5
        ).to(device)
        enc_w.pooling = pooling
        
        head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        ).to(device)
        
        opt = torch.optim.AdamW(
            list(enc_w.parameters()) + list(head.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        
        best_f1 = 0.0
        patience = 0
        
        for ep in range(epochs):
            enc_w.train()
            head.train()
            opt.zero_grad()
            
            zw = enc_w(data['Xw'])
            pred_w = head(zw).squeeze(1)
            loss = F.mse_loss(pred_w[data['idxw_tr']], data['rw'][data['idxw_tr']])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            
            enc_w.eval()
            head.eval()
            with torch.no_grad():
                residual_w = (data['rw'] - pred_w).abs()
                # residual_w = normalize_residual(residual_w, data['idxw_tr'])
                res_val = residual_w[data['idxw_val']].cpu().numpy()
                y_val = data['yw'][data['idxw_val']].cpu().numpy()
                ths = np.linspace(np.percentile(res_val, 10), np.percentile(res_val, 95), 25)
                f1s = [f1_score(y_val, (res_val >= t).astype(int), zero_division=0) for t in ths]
                f1 = max(f1s) if len(f1s) else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                patience = 0
                torch.save({
                    "epoch": ep,
                    "enc_week": enc_w.state_dict(),
                    "head": head.state_dict(),
                    "config": {
                        "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                        "pooling": pooling, "weekly_only": True, "use_classification": False
                    }
                }, save_path)
            else:
                patience += 1
                if patience >= patience_limit:
                    break
        
        return save_path
    
    if bundle_day is None:
        raise ValueError("bundle_day required for multi-scale models")
    
    data_day = to_device(bundle_day, device)
    data = {
        'Xd': data_day['X_seq'],
        'rd': data_day['NewDeaths_ret_next'],
        'idxd_tr': data_day['idx_train'],
        'idxd_val': data_day['idx_val'],
        'idxd_te': data_day['idx_test'],
        'map_dw': data_day.get('day_to_week_index', None),
        'Xw': data_week['X_seq'],
        'rw': data_week['NewDeaths_ret_next'],
        'yw': data_week['y_next'],
        'idxw_tr': data_week['idx_train'],
        'idxw_val': data_week['idx_val'],
        'idxw_te': data_week['idx_test'],
    }
    
    if use_classification:
        data['yd'] = data_day['y_next']
    
    enc_d = TransformerSeqEncoder(
        input_dim=data['Xd'].shape[2], d_model=d_model, nhead=nhead,
        num_layers=num_layers, max_seq_len=data['Xd'].shape[1] + 5
    ).to(device)
    enc_d.pooling = pooling
    
    enc_w = TransformerSeqEncoder(
        input_dim=data['Xw'].shape[2], d_model=d_model, nhead=nhead,
        num_layers=num_layers, max_seq_len=data['Xw'].shape[1] + 5
    ).to(device)
    enc_w.pooling = pooling
    
    if use_classification:
        heads = PredictionHeads(d_model).to(device)
        pos_w_d = torch.tensor(
            len(data['idxd_tr']) - data['yd'][data['idxd_tr']].sum().item(), device=device
        ).sqrt()
        pos_w_w = torch.tensor(
            len(data['idxw_tr']) - data['yw'][data['idxw_tr']].sum().item(), device=device
        ).sqrt()
        loss_d_cls = nn.BCEWithLogitsLoss(pos_weight=pos_w_d)
        loss_w_cls = FocalLoss(pos_weight=pos_w_w.item())
    else:
        heads = RegressionHead(d_model).to(device)
    
    opt = torch.optim.AdamW(
        list(enc_d.parameters()) + list(enc_w.parameters()) + list(heads.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    
    best_f1 = 0.0
    patience = 0
    
    for ep in range(epochs):
        enc_d.train()
        enc_w.train()
        heads.train()
        opt.zero_grad()
        
        zd = enc_d(data['Xd'])
        zw = enc_w(data['Xw'])
        
        if use_classification:
            logd, logw, pred_d, pred_w = heads(zd, zw)
            Ld_cls = loss_d_cls(logd[data['idxd_tr']], data['yd'][data['idxd_tr']])
            Lw_cls = loss_w_cls(logw[data['idxw_tr']], data['yw'][data['idxw_tr']])
            L_cls = 0.6 * Ld_cls + 0.4 * Lw_cls
            mse_d = F.mse_loss(pred_d[data['idxd_tr']], data['rd'][data['idxd_tr']])
            mse_w = F.mse_loss(pred_w[data['idxw_tr']], data['rw'][data['idxw_tr']])
            L_reg = mse_d + mse_w
            loss = alpha_cls * L_cls + alpha_reg * L_reg
        else:
            pred_d, pred_w = heads(zd, zw)
            mse_d = F.mse_loss(pred_d[data['idxd_tr']], data['rd'][data['idxd_tr']])
            mse_w = F.mse_loss(pred_w[data['idxw_tr']], data['rw'][data['idxw_tr']])
            loss = mse_d + mse_w
        
        if use_lu and data['map_dw'] is not None:
            ud = (data['rd'] - pred_d).abs()
            uw = (data['rw'] - pred_w).abs()
            # ud = normalize_residual(ud, data['idxd_tr'])
            # uw = normalize_residual(uw, data['idxw_tr'])
            map_tr = data['map_dw'][data['idxd_tr']]
            mask = (map_tr >= 0)
            if mask.sum() > 0:
                Lu = F.smooth_l1_loss(ud[data['idxd_tr']][mask], uw[map_tr[mask]])
                loss = loss + lambda_u * Lu
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        opt.step()
        
        enc_d.eval()
        enc_w.eval()
        heads.eval()
        with torch.no_grad():
            if use_classification:
                logd, logw, _, _ = heads(zd, zw)
                pd = torch.sigmoid(logd[data['idxd_val']]).cpu().numpy()
                pw = torch.sigmoid(logw[data['idxw_val']]).cpu().numpy()
                yd_val = data['yd'][data['idxd_val']].cpu().numpy()
                yw_val = data['yw'][data['idxw_val']].cpu().numpy()
                ths = np.linspace(0.05, 0.95, 30)
                f1_d = max([f1_score(yd_val, (pd >= t).astype(int), zero_division=0) for t in ths])
                f1_w = max([f1_score(yw_val, (pw >= t).astype(int), zero_division=0) for t in ths])
                f1 = (f1_d + f1_w) / 2
            else:
                pred_d, pred_w = heads(zd, zw)
                residual_w = (data['rw'] - pred_w).abs()
                # residual_w = normalize_residual(residual_w, data['idxw_tr'])
                res_val = residual_w[data['idxw_val']].cpu().numpy()
                y_val = data['yw'][data['idxw_val']].cpu().numpy()
                ths = np.linspace(np.percentile(res_val, 10), np.percentile(res_val, 95), 25)
                f1s = [f1_score(y_val, (res_val >= t).astype(int), zero_division=0) for t in ths]
                f1 = max(f1s) if len(f1s) else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            save_dict = {
                "epoch": ep,
                "enc_day": enc_d.state_dict(),
                "enc_week": enc_w.state_dict(),
                "config": {
                    "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                    "pooling": pooling, "weekly_only": False,
                    "use_classification": use_classification, "use_lu": use_lu
                }
            }
            if use_classification:
                save_dict["heads"] = heads.state_dict()
                save_dict["config"]["alpha_cls"] = alpha_cls
                save_dict["config"]["alpha_reg"] = alpha_reg
            else:
                save_dict["head"] = heads.state_dict()
            if use_lu:
                save_dict["config"]["lambda_u"] = lambda_u
            torch.save(save_dict, save_path)
        else:
            patience += 1
            if patience >= patience_limit:
                break
    
    return save_path
