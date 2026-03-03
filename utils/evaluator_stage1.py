import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from model.encoder import TransformerSeqEncoder, RegressionHeadWithRelation


def residualthreshold(residual, y_true, idx_val):
    res = residual[idx_val]
    y = y_true[idx_val]
    ths = np.linspace(np.percentile(res, 10), np.percentile(res, 95), 30)
    f1s = [f1_score(y, (res >= t).astype(int), zero_division=0) for t in ths]
    return ths[np.argmax(f1s)] if len(ths) else 0.0


def evaluate(residual, y_true, idx_val, idx_test, use_topk=False, topk_percent=5):
    if use_topk:
        scores_te = residual[idx_test]
        y_te = y_true[idx_test]
        th = np.percentile(scores_te, 100 - topk_percent)
        pred = (scores_te >= th).astype(int)
        return {
            "threshold": th,
            "threshold_type": f"top{topk_percent}%",
            "recall": np.sum((pred == 1) & (y_te == 1)) / max(np.sum(y_te == 1), 1),
            "precision": np.sum((pred == 1) & (y_te == 1)) / max(np.sum(pred == 1), 1),
            "auprc": average_precision_score(y_te, scores_te),
            "roc_auc": roc_auc_score(y_te, scores_te),
            "f1": f1_score(y_te, pred, zero_division=0),
            "topk_percent": topk_percent,
            "predicted_anomalies": np.sum(pred == 1),
            "actual_anomalies": np.sum(y_te == 1),
        }
    else:
        th = residualthreshold(residual, y_true, idx_val)
        scores_te = residual[idx_test]
        y_te = y_true[idx_test]
        pred = (scores_te >= th).astype(int)
        return {
            "threshold": th,
            "threshold_type": "validation_optimized",
            "recall": np.sum((pred == 1) & (y_te == 1)) / max(np.sum(y_te == 1), 1),
            "precision": np.sum((pred == 1) & (y_te == 1)) / max(np.sum(pred == 1), 1),
            "auprc": average_precision_score(y_te, scores_te),
            "roc_auc": roc_auc_score(y_te, scores_te),
            "f1": f1_score(y_te, pred, zero_division=0),
        }


def infer(
    model_path,
    bundle_coarse,
    bundle_fine=None,
    device='cuda',
    use_postprocessing=False
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    data_coarse = bundle_coarse

    coarse_only = config.get('coarse_only', False)
    fine_only = config.get('fine_only', False)

    if fine_only:
        if bundle_fine is None:
            raise ValueError("bundle_fine required for fine_only models")

        data_fine = bundle_fine
        num_vars = data_fine['X_seq'].shape[2]

        enc_fine = TransformerSeqEncoder(
            input_dim=data_fine['X_seq'].shape[2],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_seq_len=data_fine['X_seq'].shape[1] + 5
        ).to(device)
        enc_fine.pooling = config['pooling']

        head = RegressionHeadWithRelation(config['d_model'], num_vars).to(device)
        enc_fine.load_state_dict(checkpoint['enc_fine'])
        head.load_state_dict(checkpoint['head'])
        enc_fine.eval()
        head.eval()

        with torch.no_grad():
            batch_size = 512
            n_samples_fine = data_fine['X_seq'].shape[0]

            z_fine_list = []
            for i in range(0, n_samples_fine, batch_size):
                end_idx = min(i + batch_size, n_samples_fine)
                X_fine_batch = data_fine['X_seq'][i:end_idx].to(device)
                z_fine_batch = enc_fine(X_fine_batch)
                z_fine_list.append(z_fine_batch.cpu())
                del X_fine_batch, z_fine_batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            z_fine = torch.cat(z_fine_list, dim=0).to(device)
            del z_fine_list
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            pred_fine_all, _ = head(z_fine, z_fine)
            del z_fine
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            residual_vec_fine = (data_fine['X_next'].to(device) - pred_fine_all).abs()
            del pred_fine_all

            residual = residual_vec_fine.mean(dim=1).cpu().numpy()
            del residual_vec_fine
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return {
            'residual': residual,
            'y_true': data_fine['y_next'].cpu().numpy(),
            'idx_val': data_fine['idx_val'].cpu().numpy(),
            'idx_test': data_fine['idx_test'].cpu().numpy(),
        }

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

        head = RegressionHeadWithRelation(config['d_model'], num_vars).to(device)
        enc_coarse.load_state_dict(checkpoint['enc_coarse'])
        head.load_state_dict(checkpoint['head'])
        enc_coarse.eval()
        head.eval()

        with torch.no_grad():
            batch_size = 512
            n_samples_coarse = data_coarse['X_seq'].shape[0]

            z_coarse_list = []
            for i in range(0, n_samples_coarse, batch_size):
                end_idx = min(i + batch_size, n_samples_coarse)
                X_seq_batch = data_coarse['X_seq'][i:end_idx].to(device)
                z_coarse_batch = enc_coarse(X_seq_batch)
                z_coarse_list.append(z_coarse_batch.cpu())
                del X_seq_batch, z_coarse_batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            z_coarse = torch.cat(z_coarse_list, dim=0).to(device)
            del z_coarse_list
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            pred_coarse_list = []
            for i in range(0, n_samples_coarse, batch_size):
                end_idx = min(i + batch_size, n_samples_coarse)
                z_coarse_batch = z_coarse[i:end_idx]
                _, pred_coarse_batch = head(z_coarse_batch, z_coarse_batch)
                pred_coarse_list.append(pred_coarse_batch.cpu())
                del z_coarse_batch, pred_coarse_batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            pred_coarse_all = torch.cat(pred_coarse_list, dim=0).to(device)
            del z_coarse, pred_coarse_list
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            residual_list = []
            for i in range(0, n_samples_coarse, batch_size):
                end_idx = min(i + batch_size, n_samples_coarse)
                X_next_batch = data_coarse['X_next'][i:end_idx].to(device)
                pred_batch = pred_coarse_all[i:end_idx]
                residual_vec_batch = (X_next_batch - pred_batch).abs()
                residual_batch = residual_vec_batch.mean(dim=1).cpu().numpy()
                residual_list.append(residual_batch)
                del X_next_batch, pred_batch, residual_vec_batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            residual = np.concatenate(residual_list, axis=0)
            del pred_coarse_all, residual_list
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return {
            'residual': residual,
            'y_true': data_coarse['y_next'].cpu().numpy(),
            'idx_val': data_coarse['idx_val'].cpu().numpy(),
            'idx_test': data_coarse['idx_test'].cpu().numpy(),
        }

    if bundle_fine is None:
        raise ValueError("bundle_fine required for multi-scale models")

    data_coarse = bundle_coarse
    data_fine = bundle_fine

    enc_fine = TransformerSeqEncoder(
        input_dim=data_fine['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_fine['X_seq'].shape[1] + 5
    ).to(device)
    enc_fine.pooling = config['pooling']

    enc_coarse = TransformerSeqEncoder(
        input_dim=data_coarse['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_coarse['X_seq'].shape[1] + 5
    ).to(device)
    enc_coarse.pooling = config['pooling']

    head = RegressionHeadWithRelation(config['d_model'], num_vars).to(device)
    head.load_state_dict(checkpoint['head'])

    enc_fine.load_state_dict(checkpoint['enc_fine'])
    enc_coarse.load_state_dict(checkpoint['enc_coarse'])
    enc_fine.eval()
    enc_coarse.eval()
    head.eval()

    with torch.no_grad():
        batch_size = 512
        n_samples_fine = data_fine['X_seq'].shape[0]
        n_samples_coarse = data_coarse['X_seq'].shape[0]

        z_fine_list = []
        for i in range(0, n_samples_fine, batch_size):
            end_idx = min(i + batch_size, n_samples_fine)
            X_fine_batch = data_fine['X_seq'][i:end_idx].to(device)
            z_fine_batch = enc_fine(X_fine_batch)
            z_fine_list.append(z_fine_batch.cpu())
            del X_fine_batch, z_fine_batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        z_coarse_list = []
        for i in range(0, n_samples_coarse, batch_size):
            end_idx = min(i + batch_size, n_samples_coarse)
            X_coarse_batch = data_coarse['X_seq'][i:end_idx].to(device)
            z_coarse_batch = enc_coarse(X_coarse_batch)
            z_coarse_list.append(z_coarse_batch.cpu())
            del X_coarse_batch, z_coarse_batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        z_fine = torch.cat(z_fine_list, dim=0).to(device)
        z_coarse = torch.cat(z_coarse_list, dim=0).to(device)
        del z_fine_list, z_coarse_list
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        pred_fine_all, pred_coarse_all = head(z_fine, z_coarse)
        del z_fine, z_coarse
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        X_next_coarse = data_coarse['X_next'].to(device)
        X_next_fine = data_fine['X_next'].to(device)
        residual_vec_coarse = (X_next_coarse - pred_coarse_all).abs()
        residual_vec_fine = (X_next_fine - pred_fine_all).abs()
        del pred_fine_all, pred_coarse_all, X_next_coarse, X_next_fine
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        residual_coarse = residual_vec_coarse.mean(dim=1).cpu().numpy()
        residual_fine = residual_vec_fine.mean(dim=1).cpu().numpy()
        del residual_vec_coarse, residual_vec_fine
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if use_postprocessing:
            mapping = data_fine.get('fine_to_coarse_index', None)
            if mapping is None:
                residual = residual_coarse
            else:
                if isinstance(mapping, torch.Tensor):
                    mapping = mapping.cpu().numpy()
                else:
                    mapping = np.array(mapping)
                idx_val_coarse = data_coarse['idx_val'].cpu().numpy()
                idx_val_fine = data_fine['idx_val'].cpu().numpy()
                y_val = data_coarse['y_next'][data_coarse['idx_val']].cpu().numpy()

                res_val_coarse = residual_coarse[idx_val_coarse]
                res_val_fine_aligned = np.zeros_like(res_val_coarse)

                for i, c_idx in enumerate(idx_val_coarse):
                    fine_mask = (mapping == c_idx) & np.isin(np.arange(len(mapping)), idx_val_fine)
                    fine_indices = np.where(fine_mask)[0]
                    if len(fine_indices) > 0:
                        res_val_fine_aligned[i] = residual_fine[fine_indices].mean()
                    else:
                        res_val_fine_aligned[i] = res_val_coarse[i]

                alphas = np.linspace(0.0, 1.0, 11)
                best_alpha, best_f1 = 0.0, 0.0
                for alpha in alphas:
                    fused_res = (1 - alpha) * res_val_coarse + alpha * res_val_fine_aligned
                    ths = np.linspace(np.percentile(fused_res, 10), np.percentile(fused_res, 95), 25)
                    f1s = [f1_score(y_val, (fused_res >= t).astype(int), zero_division=0) for t in ths]
                    max_f1 = max(f1s) if len(f1s) else 0.0
                    if max_f1 > best_f1:
                        best_f1, best_alpha = max_f1, alpha

                res_fine_aligned_all = np.zeros_like(residual_coarse)
                for c_idx in range(len(residual_coarse)):
                    fine_indices = np.where(mapping == c_idx)[0]
                    if len(fine_indices) > 0:
                        res_fine_aligned_all[c_idx] = residual_fine[fine_indices].mean()
                    else:
                        res_fine_aligned_all[c_idx] = residual_coarse[c_idx]

                residual = (1 - best_alpha) * residual_coarse + best_alpha * res_fine_aligned_all
        else:
            residual = residual_coarse

        return {
            'residual': residual,
            'y_true': data_coarse['y_next'].cpu().numpy(),
            'idx_val': data_coarse['idx_val'].cpu().numpy(),
            'idx_test': data_coarse['idx_test'].cpu().numpy(),
        }
