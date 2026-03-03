import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from .encoder import TransformerSeqEncoder, RegressionHeadWithRelation
from experiments.mapping import build_batch_mapping
from utils.training import (
    to_device,
    todevice,
    getregressiontarget,
    _count_coarse_groups,
    _train_coarse_group,
    _log_multiscale_epoch,
    _validate_multiscale,
)


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


def _train_single_scale(
    X_seq,
    X_next,
    y,
    idx_tr,
    idx_val,
    idx_test,
    d_model,
    nhead,
    num_layers,
    pooling,
    lr,
    weight_decay,
    epochs,
    batch_size,
    device,
    save_path,
    config_overrides,
    enc_state_key,
    resume_from=None,
    clip_scope="all",
):
    enc = TransformerSeqEncoder(
        input_dim=X_seq.shape[2],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=X_seq.shape[1] + 5,
    ).to(device)
    enc.pooling = pooling

    num_vars = X_seq.shape[2]
    head = RegressionHeadWithRelation(d_model, num_vars).to(device)

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_batches = []
    for i in range(0, len(idx_tr), batch_size):
        end_idx = min(i + batch_size, len(idx_tr))
        train_batches.append(idx_tr[i:end_idx])

    start_epoch = 0
    best_auc = None
    best_epoch = 0
    if resume_from is not None:
        print(f"\nResuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        enc.load_state_dict(checkpoint[enc_state_key])
        head.load_state_dict(checkpoint["head"])
        if "optimizer" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_auc = checkpoint.get("best_auc", None)
        best_epoch = checkpoint.get("best_epoch", 0)
        print(f"  Resumed from epoch {start_epoch}, best AUC: {best_auc}, best epoch: {best_epoch+1}")

    for ep in range(start_epoch, epochs):
        enc.train()
        head.train()
        opt.zero_grad()

        n_batches = len(train_batches)
        total_loss_scalar = 0.0

        for batch_idx_idx, batch_idx in enumerate(train_batches):
            X_batch = X_seq[batch_idx].to(device)
            X_next_batch = X_next[batch_idx].to(device)

            z_batch = enc(X_batch)
            _, pred_batch = head(z_batch, z_batch)
            pred_batch = pred_batch.mean(dim=1)
            target_batch = X_next_batch.mean(dim=1)

            batch_loss = F.mse_loss(pred_batch, target_batch)
            total_loss_scalar += batch_loss.item()

            if n_batches > 0:
                scaled_loss = batch_loss / n_batches
                is_last_batch = (batch_idx_idx == len(train_batches) - 1)
                scaled_loss.backward(retain_graph=not is_last_batch)
                del scaled_loss

            del X_batch, X_next_batch, z_batch, pred_batch, target_batch, batch_loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if n_batches > 0:
            loss = torch.tensor(total_loss_scalar / n_batches, device=device)
            if clip_scope == "head":
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        enc.eval()
        head.eval()
        with torch.no_grad():
            pred_val_list = []
            for i in range(0, len(idx_val), batch_size):
                end_idx = min(i + batch_size, len(idx_val))
                batch_idx = idx_val[i:end_idx]
                X_batch = X_seq[batch_idx].to(device)
                X_next_batch = X_next[batch_idx].to(device)
                z_batch = enc(X_batch)
                _, pred_batch = head(z_batch, z_batch)
                pred_val_list.append(pred_batch.mean(dim=1).cpu())

            if len(pred_val_list) > 0:
                pred_val = torch.cat(pred_val_list, dim=0)
                X_next_val = X_next[idx_val].mean(dim=1)
                residual = (X_next_val - pred_val).abs()
                res_val = residual.numpy()

                if y is not None:
                    y_val = y[idx_val].numpy() if isinstance(y, torch.Tensor) else y[idx_val]
                    unique_classes = np.unique(y_val)
                    if len(unique_classes) < 2:
                        auc = None
                    else:
                        try:
                            auc = roc_auc_score(y_val, res_val)
                        except (ValueError, Exception):
                            auc = None
                else:
                    auc = None
            else:
                auc = None

            if auc is not None and (best_auc is None or auc > best_auc):
                best_auc = auc
                best_epoch = ep

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            auc_str = f"{auc:.4f}" if auc is not None else "N/A (single class)"
            print(f"Epoch {ep+1}/{epochs} - Loss: {loss.item():.6f}, Val AUC-ROC: {auc_str}", flush=True)

        if auc is not None and (best_auc is None or auc > best_auc):
            best_auc = auc
            best_epoch = ep
            torch.save({
                "epoch": ep,
                enc_state_key: enc.state_dict(),
                "head": head.state_dict(),
                "optimizer": opt.state_dict(),
                "config": {
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "pooling": pooling,
                    "num_vars": num_vars,
                    **config_overrides,
                },
                "best_auc": best_auc,
                "best_epoch": best_epoch,
            }, save_path)

        if ep == epochs - 1:
            if best_auc is None:
                best_epoch = ep
            torch.save({
                "epoch": ep,
                enc_state_key: enc.state_dict(),
                "head": head.state_dict(),
                "optimizer": opt.state_dict(),
                "config": {
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "pooling": pooling,
                    "num_vars": num_vars,
                    **config_overrides,
                },
                "best_auc": best_auc,
                "best_epoch": best_epoch,
            }, save_path)
            best_auc_str = f"{best_auc:.4f}" if best_auc is not None else "N/A (single class in validation)"
            print(
                f"\nTraining completed. Best AUC-ROC: {best_auc_str} at epoch {best_epoch+1}. Final model saved at epoch {ep+1}.",
                flush=True,
            )

    return save_path


def _compute_lu_loss(z_fine_subset, z_coarse_subset, lu_predictor, lu_detach_coarse):
    if lu_predictor is None:
        return None
    if z_fine_subset.size(0) < 2:
        return None
    p_fine = lu_predictor(z_fine_subset)
    p_fine = F.normalize(p_fine, dim=-1)
    z_coarse_align = z_coarse_subset.detach() if lu_detach_coarse else z_coarse_subset
    z_coarse_norm = F.normalize(z_coarse_align, dim=-1)
    return F.mse_loss(p_fine, z_coarse_norm)


def _save_multiscale_checkpoint(
    save_path,
    enc_fine,
    enc_coarse,
    head,
    opt,
    d_model,
    nhead,
    num_layers,
    pooling,
    use_classification,
    use_lu,
    lambda_u,
    lu_detach_coarse,
    lu_predictor,
    best_auc,
    best_epoch,
    ep,
):
    save_dict = {
        "epoch": ep,
        "enc_fine": enc_fine.state_dict(),
        "enc_coarse": enc_coarse.state_dict(),
        "head": head.state_dict(),
        "optimizer": opt.state_dict(),
        "config": {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "pooling": pooling,
            "coarse_only": False,
            "use_classification": use_classification,
            "use_lu": use_lu,
        },
        "best_auc": best_auc,
        "best_epoch": best_epoch,
    }
    if use_lu:
        save_dict["config"]["lambda_u"] = lambda_u
        save_dict["config"]["lu_detach_coarse"] = lu_detach_coarse
        save_dict["config"]["lu_predictor"] = True
        save_dict["lu_predictor"] = lu_predictor.state_dict()
    torch.save(save_dict, save_path)


def train_ours(
    bundle_coarse,
    bundle_fine=None,
    save_path=None,
    coarse_only=False,
    fine_only=False,
    use_classification=False,
    use_lu=True,
    lu_detach_coarse=False,
    alpha_cls=1.0,
    alpha_reg=0.1,
    lambda_u=1.0,
    epochs=200,
    lr=3e-4,
    weight_decay=1e-4,
    d_model=64,
    nhead=4,
    num_layers=2,
    pooling="last",
    batch_size=32,
    device='cuda',
    resume_from=None
):
    if isinstance(device, str):
        if device == 'cpu':
            device = torch.device('cpu')
        elif device == 'cuda':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = str(save_path)
    
    if fine_only:
        if bundle_fine is None:
            raise ValueError("bundle_fine required for fine_only training")
        
        return _train_single_scale(
            X_seq=bundle_fine['X_seq'],
            X_next=bundle_fine['X_next'],
            y=bundle_fine.get('y_next', None),
            idx_tr=bundle_fine['idx_train'],
            idx_val=bundle_fine['idx_val'],
            idx_test=bundle_fine['idx_test'],
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            pooling=pooling,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            save_path=save_path,
            config_overrides={
                "coarse_only": False,
                "fine_only": True,
                "use_classification": False,
            },
            enc_state_key="enc_fine",
            resume_from=None,
            clip_scope="all",
        )
    
    if coarse_only:
        return _train_single_scale(
            X_seq=bundle_coarse['X_seq'],
            X_next=bundle_coarse['X_next'],
            y=bundle_coarse.get('y_next', None),
            idx_tr=bundle_coarse['idx_train'],
            idx_val=bundle_coarse['idx_val'],
            idx_test=bundle_coarse['idx_test'],
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            pooling=pooling,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            save_path=save_path,
            config_overrides={
                "coarse_only": True,
                "use_classification": False,
            },
            enc_state_key="enc_coarse",
            resume_from=resume_from,
            clip_scope="head",
        )
    
    if bundle_fine is None:
        raise ValueError("bundle_fine required for multi-scale models")
    
    data_fine = bundle_fine
    
    if 'fine_to_coarse_index' not in data_fine:
        raise ValueError("fine_to_coarse_index mapping required for multi-scale training")
    
    mapping = data_fine['fine_to_coarse_index']
    if (mapping < 0).all():
        raise ValueError("All mapping indices are invalid. Data must have valid fine_to_coarse_index mapping.")
    
    data = {
        'X_fine': data_fine['X_seq'],
        'X_next_fine': data_fine['X_next'],
        'y_fine': data_fine.get('y_next', None),
        'idx_fine_tr': data_fine['idx_train'],
        'idx_fine_val': data_fine['idx_val'],
        'idx_fine_test': data_fine['idx_test'],
        'mapping': mapping,
        'X_coarse': bundle_coarse['X_seq'],
        'X_next_coarse': bundle_coarse['X_next'],
        'y_coarse': bundle_coarse.get('y_next', None),
        'idx_coarse_tr': bundle_coarse['idx_train'],
        'idx_coarse_val': bundle_coarse['idx_val'],
        'idx_coarse_test': bundle_coarse['idx_test'],
    }
    
    if use_classification:
        if data['y_fine'] is None or data['y_coarse'] is None:
            raise ValueError("y_next required for classification mode")
    
    enc_fine = TransformerSeqEncoder(
        input_dim=data['X_fine'].shape[2], d_model=d_model, nhead=nhead,
        num_layers=num_layers, max_seq_len=data['X_fine'].shape[1] + 5
    ).to(device)
    enc_fine.pooling = pooling
    
    enc_coarse = TransformerSeqEncoder(
        input_dim=data['X_coarse'].shape[2], d_model=d_model, nhead=nhead,
        num_layers=num_layers, max_seq_len=data['X_coarse'].shape[1] + 5
    ).to(device)
    enc_coarse.pooling = pooling
    
    num_vars = data['X_fine'].shape[2]
    
    if use_classification:
        raise NotImplementedError("Classification not yet supported in multi-scale mode")
    else:
        head = RegressionHeadWithRelation(d_model, num_vars).to(device)
    
    lu_predictor = None
    if use_lu:
        lu_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        ).to(device)
    
    opt = torch.optim.AdamW(
        list(enc_fine.parameters()) + list(enc_coarse.parameters()) + list(head.parameters()) +
        (list(lu_predictor.parameters()) if lu_predictor is not None else []),
        lr=lr, weight_decay=weight_decay
    )
    
    
    idx_fine_tr = data['idx_fine_tr']
    idx_coarse_tr = data['idx_coarse_tr']
    n_train_fine = len(idx_fine_tr)
    
    fine_batches = []
    for i in range(0, n_train_fine, batch_size):
        end_idx = min(i + batch_size, n_train_fine)
        fine_batches.append(idx_fine_tr[i:end_idx])
    
    start_epoch = 0
    best_auc = None
    best_epoch = 0
    if resume_from is not None:
        print(f"\nResuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        enc_fine.load_state_dict(checkpoint['enc_fine'])
        enc_coarse.load_state_dict(checkpoint['enc_coarse'])
        head.load_state_dict(checkpoint['head'])
        if lu_predictor is not None and 'lu_predictor' in checkpoint:
            lu_predictor.load_state_dict(checkpoint['lu_predictor'])
        if 'optimizer' in checkpoint:
            opt.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_auc = checkpoint.get('best_auc', None)
        best_epoch = checkpoint.get('best_epoch', 0)
        print(f"  Resumed from epoch {start_epoch}, best AUC: {best_auc}, best epoch: {best_epoch+1}")
    
    for ep in range(start_epoch, epochs):
        enc_fine.train()
        enc_coarse.train()
        head.train()
        total_loss_scalar = 0.0
        n_batches = 0
        stats = {
            "mse_fine_sum": 0.0,
            "mse_coarse_sum": 0.0,
            "lu_sum": 0.0,
            "lu_count": 0,
        }
        
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"Epoch {ep+1}/{epochs} - Training...", flush=True)
        
        opt.zero_grad()
        
        total_loss_count = _count_coarse_groups(
            fine_batches,
            data['mapping'],
            build_batch_mapping,
        )
        
        for batch_idx_fine in fine_batches:
            X_fine_batch = data['X_fine'][batch_idx_fine].to(device)
            X_next_fine_batch = data['X_next_fine'][batch_idx_fine].to(device)
            
            mapping_batch, valid_mask, coarse_indices, unique_coarse_indices, fine_to_coarse_map = build_batch_mapping(
                batch_idx_fine, data['mapping']
            )

            if valid_mask.sum() == 0:
                continue

            z_fine_batch = enc_fine(X_fine_batch)
            X_coarse_batch = data['X_coarse'][unique_coarse_indices].to(device)
            X_next_coarse_batch = data['X_next_coarse'][unique_coarse_indices].to(device)
            z_coarse_batch = enc_coarse(X_coarse_batch)
            
            coarse_idx_list = list(fine_to_coarse_map.keys())
            for idx, (coarse_idx, fine_indices_list) in enumerate(fine_to_coarse_map.items()):
                loss_inc, batch_inc = _train_coarse_group(
                    fine_indices_list,
                    batch_idx_fine,
                    unique_coarse_indices,
                    coarse_idx,
                    z_fine_batch,
                    X_next_fine_batch,
                    z_coarse_batch,
                    X_next_coarse_batch,
                    head,
                    _compute_lu_loss,
                    lu_predictor,
                    lu_detach_coarse,
                    lambda_u,
                    total_loss_count,
                    idx,
                    len(coarse_idx_list),
                    stats,
                )
                total_loss_scalar += loss_inc
                n_batches += batch_inc
                
                if device.type == 'cuda' and n_batches % 5 == 0:
                    torch.cuda.empty_cache()
            
            del z_fine_batch, z_coarse_batch, X_coarse_batch, X_next_coarse_batch, X_fine_batch, X_next_fine_batch
        
        if n_batches > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            loss_scalar = total_loss_scalar / n_batches
        else:
            loss_scalar = 0.0
        
        _log_multiscale_epoch(stats, n_batches, loss_scalar, use_lu, lambda_u, ep, epochs)
        
        enc_fine.eval()
        enc_coarse.eval()
        head.eval()
        auc = _validate_multiscale(
            enc_coarse, head, data, batch_size, device, roc_auc_score, np
        )
        
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            auc_str = f"{auc:.4f}" if auc is not None else "N/A (single class)"
            print(f"  Validation AUC-ROC: {auc_str}", flush=True)
        
        if auc is not None and (best_auc is None or auc > best_auc):
            best_auc = auc
            best_epoch = ep
            _save_multiscale_checkpoint(
                save_path,
                enc_fine,
                enc_coarse,
                head,
                opt,
                d_model,
                nhead,
                num_layers,
                pooling,
                use_classification,
                use_lu,
                lambda_u,
                lu_detach_coarse,
                lu_predictor,
                best_auc,
                best_epoch,
                ep,
            )
            if (ep + 1) % 10 == 0 or ep == 0:
                best_auc_str = f"{best_auc:.4f}" if best_auc is not None else "N/A"
                print(f"  -> Best model saved (AUC-ROC={best_auc_str} at epoch {ep+1})", flush=True)
        
        if ep == epochs - 1:
            if best_auc is None:
                best_epoch = ep
            _save_multiscale_checkpoint(
                save_path,
                enc_fine,
                enc_coarse,
                head,
                opt,
                d_model,
                nhead,
                num_layers,
                pooling,
                use_classification,
                use_lu,
                lambda_u,
                lu_detach_coarse,
                lu_predictor,
                best_auc,
                best_epoch,
                ep,
            )
            best_auc_str = f"{best_auc:.4f}" if best_auc is not None else "N/A (single class in validation)"
            print(f"\nTraining completed. Best model (AUC-ROC={best_auc_str}) saved from epoch {best_epoch+1}.", flush=True)
    
    return save_path


