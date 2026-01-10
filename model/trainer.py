import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from .encoder import TransformerSeqEncoder, RegressionHeadWithRelation


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


def train_ours(
    bundle_coarse,
    bundle_fine=None,
    save_path=None,
    coarse_only=False,
    use_classification=False,
    use_lu=False,
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
    device='cuda'
):
    # Handle device parameter: convert string to device or use default
    if isinstance(device, str):
        if device == 'cpu':
            device = torch.device('cpu')
        elif device == 'cuda':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if coarse_only:
        # Keep data on CPU, move to GPU only when needed for batch processing
        data = {
            'X_coarse': bundle_coarse['X_seq'],  # Keep on CPU
            'X_next_coarse': bundle_coarse['X_next'],  # Keep on CPU
            'y_coarse': bundle_coarse.get('y_next', None),  # Keep on CPU
            'idx_tr': bundle_coarse['idx_train'],
            'idx_val': bundle_coarse['idx_val'],
            'idx_test': bundle_coarse['idx_test'],
        }
        
        enc_coarse = TransformerSeqEncoder(
            input_dim=data['X_coarse'].shape[2], d_model=d_model, nhead=nhead,
            num_layers=num_layers, max_seq_len=data['X_coarse'].shape[1] + 5
        ).to(device)
        enc_coarse.pooling = pooling
        
        num_vars = data['X_coarse'].shape[2]
        head = RegressionHeadWithRelation(d_model, num_vars).to(device)
        
        opt = torch.optim.AdamW(
            list(enc_coarse.parameters()) + list(head.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        
        idx_tr = data['idx_tr']
        n_train = len(idx_tr)
        train_batches = []
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            train_batches.append(idx_tr[i:end_idx])
        
        # Track best validation F1 for model saving
        best_f1 = 0.0
        best_epoch = 0
        
        for ep in range(epochs):
            enc_coarse.train()
            head.train()
            opt.zero_grad()
            
            total_loss = 0.0
            n_batches = 0
            
            # Process training data in batches
            for batch_idx in train_batches:
                X_coarse_batch = data['X_coarse'][batch_idx].to(device)
                X_next_coarse_batch = data['X_next_coarse'][batch_idx].to(device)
                
                z_coarse_batch = enc_coarse(X_coarse_batch)
                _, pred_coarse_batch = head(z_coarse_batch, z_coarse_batch)
                pred_coarse_batch = pred_coarse_batch.mean(dim=1)
                target_batch = X_next_coarse_batch.mean(dim=1)
                
                batch_loss = F.mse_loss(pred_coarse_batch, target_batch)
                total_loss = total_loss + batch_loss
                n_batches += 1
                
                # Clear GPU cache periodically
                if device.type == 'cuda' and n_batches % 10 == 0:
                    torch.cuda.empty_cache()
            
            if n_batches > 0:
                loss = total_loss / n_batches
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            enc_coarse.eval()
            head.eval()
            with torch.no_grad():
                # Compute predictions for validation set in batches
                idx_val = data['idx_val']
                pred_val_list = []
                for i in range(0, len(idx_val), batch_size):
                    end_idx = min(i + batch_size, len(idx_val))
                    batch_idx = idx_val[i:end_idx]
                    X_coarse_batch = data['X_coarse'][batch_idx].to(device)
                    X_next_coarse_batch = data['X_next_coarse'][batch_idx].to(device)
                    z_coarse_batch = enc_coarse(X_coarse_batch)
                    _, pred_coarse_batch = head(z_coarse_batch, z_coarse_batch)
                    pred_val_list.append(pred_coarse_batch.mean(dim=1).cpu())
                
                if len(pred_val_list) > 0:
                    pred_coarse_val = torch.cat(pred_val_list, dim=0)
                    X_next_coarse_val = data['X_next_coarse'][idx_val].mean(dim=1)
                    residual_coarse = (X_next_coarse_val - pred_coarse_val).abs()
                    res_val = residual_coarse.numpy()
                    
                    if data['y_coarse'] is not None:
                        y_val = data['y_coarse'][idx_val].numpy() if isinstance(data['y_coarse'], torch.Tensor) else data['y_coarse'][idx_val]
                        ths = np.linspace(np.percentile(res_val, 10), np.percentile(res_val, 95), 25)
                        f1s = [f1_score(y_val, (res_val >= t).astype(int), zero_division=0) for t in ths]
                        f1 = max(f1s) if len(f1s) else 0.0
                    else:
                        f1 = 0.0
                else:
                    f1 = 0.0
                
                # Track best F1
                if f1 > best_f1:
                    best_f1 = f1
                    best_epoch = ep
            
            if (ep + 1) % 10 == 0 or ep == epochs - 1:
                print(f"Epoch {ep+1}/{epochs} - Loss: {loss.item():.6f}, Val F1: {f1:.4f}", flush=True)
            
            # Save best model and final model
            if f1 > best_f1 or ep == epochs - 1:
                torch.save({
                    "epoch": ep,
                    "enc_coarse": enc_coarse.state_dict(),
                    "head": head.state_dict(),
                    "config": {
                        "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                        "pooling": pooling, "coarse_only": True, "use_classification": False,
                        "num_vars": num_vars
                    },
                    "best_f1": best_f1,
                    "best_epoch": best_epoch
                }, save_path)
                if ep == epochs - 1:
                    print(f"\nTraining completed. Best F1: {best_f1:.4f} at epoch {best_epoch+1}. Final model saved at epoch {ep+1}.", flush=True)
        
        return save_path
    
    if bundle_fine is None:
        raise ValueError("bundle_fine required for multi-scale models")
    
    # For multi-scale, keep data on CPU, move to GPU only when needed
    data_fine = bundle_fine
    
    if 'fine_to_coarse_index' not in data_fine:
        raise ValueError("fine_to_coarse_index mapping required for multi-scale training")
    
    mapping = data_fine['fine_to_coarse_index']
    if (mapping < 0).all():
        raise ValueError("All mapping indices are invalid. Data must have valid fine_to_coarse_index mapping.")
    
    data = {
        'X_fine': data_fine['X_seq'],  # Keep on CPU
        'X_next_fine': data_fine['X_next'],  # Keep on CPU
        'y_fine': data_fine.get('y_next', None),  # Keep on CPU
        'idx_fine_tr': data_fine['idx_train'],
        'idx_fine_val': data_fine['idx_val'],
        'idx_fine_test': data_fine['idx_test'],
        'mapping': mapping,
        'X_coarse': bundle_coarse['X_seq'],  # Keep on CPU
        'X_next_coarse': bundle_coarse['X_next'],  # Keep on CPU
        'y_coarse': bundle_coarse.get('y_next', None),  # Keep on CPU
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
    
    opt = torch.optim.AdamW(
        list(enc_fine.parameters()) + list(enc_coarse.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    
    
    idx_fine_tr = data['idx_fine_tr']
    idx_coarse_tr = data['idx_coarse_tr']
    n_train_fine = len(idx_fine_tr)
    
    fine_batches = []
    for i in range(0, n_train_fine, batch_size):
        end_idx = min(i + batch_size, n_train_fine)
        fine_batches.append(idx_fine_tr[i:end_idx])
    
    # Track best validation F1 for model saving
    best_f1 = 0.0
    best_epoch = 0
    
    for ep in range(epochs):
        enc_fine.train()
        enc_coarse.train()
        head.train()
        opt.zero_grad()
        
        total_loss = 0.0
        n_batches = 0
        
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"Epoch {ep+1}/{epochs} - Training...", flush=True)
        
        for batch_idx_fine in fine_batches:
            X_fine_batch = data['X_fine'][batch_idx_fine]
            X_next_fine_batch = data['X_next_fine'][batch_idx_fine]
            mapping_batch = data['mapping'][batch_idx_fine]
            valid_mask = (mapping_batch >= 0)
            
            if valid_mask.sum() == 0:
                continue
            
            z_fine_batch = enc_fine(X_fine_batch)
            
            coarse_indices = mapping_batch[valid_mask]
            unique_coarse_indices = coarse_indices.unique()
            X_coarse_batch = data['X_coarse'][unique_coarse_indices]
            X_next_coarse_batch = data['X_next_coarse'][unique_coarse_indices]
            z_coarse_batch = enc_coarse(X_coarse_batch)
            
            fine_to_coarse_map = {}
            for i, coarse_idx in enumerate(coarse_indices):
                fine_idx = batch_idx_fine[valid_mask][i]
                if coarse_idx.item() not in fine_to_coarse_map:
                    fine_to_coarse_map[coarse_idx.item()] = []
                fine_to_coarse_map[coarse_idx.item()].append(fine_idx)
            
            for coarse_idx, fine_indices_list in fine_to_coarse_map.items():
                coarse_pos = (unique_coarse_indices == coarse_idx).nonzero(as_tuple=True)[0]
                if len(coarse_pos) == 0:
                    continue
                coarse_pos = coarse_pos[0]
                
                fine_positions_in_batch = []
                for fi in fine_indices_list:
                    pos = torch.where(batch_idx_fine == fi)[0]
                    if len(pos) > 0:
                        fine_positions_in_batch.append(pos[0].item())
                
                if len(fine_positions_in_batch) == 0:
                    continue
                
                fine_positions_tensor = torch.tensor(fine_positions_in_batch, device=z_fine_batch.device)
                z_fine_subset = z_fine_batch[fine_positions_tensor]
                X_next_fine_subset = X_next_fine_batch[fine_positions_tensor]
                z_coarse_subset = z_coarse_batch[coarse_pos:coarse_pos+1].expand(len(fine_positions_in_batch), -1)
                X_next_coarse_subset = X_next_coarse_batch[coarse_pos:coarse_pos+1]
                
                pred_fine_batch, pred_coarse_batch = head(z_fine_subset, z_coarse_subset)
                
                mse_fine = F.mse_loss(pred_fine_batch, X_next_fine_subset)
                mse_coarse = F.mse_loss(pred_coarse_batch.mean(dim=0, keepdim=True), X_next_coarse_subset)
                batch_loss = mse_fine + mse_coarse
                
                if use_lu:
                    u_fine = (X_next_fine_subset - pred_fine_batch).abs().mean(dim=1).mean()
                    u_coarse = (X_next_coarse_subset - pred_coarse_batch.mean(dim=0, keepdim=True)).abs().mean()
                    Lu = F.smooth_l1_loss(u_fine.unsqueeze(0), u_coarse.unsqueeze(0))
                    batch_loss = batch_loss + lambda_u * Lu
                
                total_loss = total_loss + batch_loss
                n_batches += 1
        
        if n_batches > 0:
            loss = total_loss / n_batches
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  Loss: {loss.item():.6f}, Batches: {n_batches}", flush=True)
        
        enc_fine.eval()
        enc_coarse.eval()
        head.eval()
        with torch.no_grad():
            idx_coarse_val = data['idx_coarse_val']
            
            pred_coarse_val_list = []
            
            for i in range(0, len(idx_coarse_val), batch_size):
                end_idx = min(i + batch_size, len(idx_coarse_val))
                batch_idx = idx_coarse_val[i:end_idx]
                X_coarse_batch = data['X_coarse'][batch_idx]
                z_coarse_batch = enc_coarse(X_coarse_batch)
                _, pred_coarse_batch = head(z_coarse_batch, z_coarse_batch)
                pred_coarse_val_list.append(pred_coarse_batch.cpu())
            
            if len(pred_coarse_val_list) > 0:
                pred_coarse_val = torch.cat(pred_coarse_val_list, dim=0)
                pred_coarse_val_mean = pred_coarse_val.mean(dim=1).to(device)
                residual_coarse = (data['X_next_coarse'][idx_coarse_val].mean(dim=1) - pred_coarse_val_mean).abs()
                res_val = residual_coarse.cpu().numpy()
                if data['y_coarse'] is not None:
                    y_val = data['y_coarse'][idx_coarse_val].cpu().numpy()
                    ths = np.linspace(np.percentile(res_val, 10), np.percentile(res_val, 95), 25)
                    f1s = [f1_score(y_val, (res_val >= t).astype(int), zero_division=0) for t in ths]
                    f1 = max(f1s) if len(f1s) else 0.0
                else:
                    f1 = 0.0
            else:
                f1 = 0.0
        
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  Validation F1: {f1:.4f}", flush=True)
        
        # Save best model based on validation F1
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = ep
            save_dict = {
                "epoch": ep,
                "enc_fine": enc_fine.state_dict(),
                "enc_coarse": enc_coarse.state_dict(),
                "head": head.state_dict(),
                "config": {
                    "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                    "pooling": pooling, "coarse_only": False,
                    "use_classification": use_classification, "use_lu": use_lu
                },
                "best_f1": best_f1,
                "best_epoch": best_epoch
            }
            if use_lu:
                save_dict["config"]["lambda_u"] = lambda_u
            torch.save(save_dict, save_path)
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  -> Best model saved (F1={best_f1:.4f} at epoch {ep+1})", flush=True)
        
        # Also save at regular intervals and final epoch
        if ep == epochs - 1 or (ep + 1) % 10 == 0:
            if ep != best_epoch:  # Don't save again if already saved as best
                save_dict = {
                    "epoch": ep,
                    "enc_fine": enc_fine.state_dict(),
                    "enc_coarse": enc_coarse.state_dict(),
                    "head": head.state_dict(),
                    "config": {
                        "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                        "pooling": pooling, "coarse_only": False,
                        "use_classification": use_classification, "use_lu": use_lu
                    },
                    "best_f1": best_f1,
                    "best_epoch": best_epoch
                }
                if use_lu:
                    save_dict["config"]["lambda_u"] = lambda_u
                torch.save(save_dict, save_path)
            if ep == epochs - 1:
                print(f"\nTraining completed. Best model (F1={best_f1:.4f}) saved from epoch {best_epoch+1}.", flush=True)
    
    return save_path
