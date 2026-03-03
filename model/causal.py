import math
import numpy as np
import torch

def grad_causal_matrix(X_seq, encoder, head, use_fine=True):
    B, T, num_vars = X_seq.shape
    device = X_seq.device
    
    X_seq = X_seq.clone().requires_grad_(True)
    
    encoder.eval()
    head.eval()
    
    with torch.enable_grad():
        z = encoder(X_seq)
        if use_fine:
            pred_all, _ = head(z, z)
        else:
            _, pred_all = head(z, z)
    
    causal_matrix = torch.zeros(B, num_vars, num_vars, device=device)
    
    for j in range(num_vars):
        pred_j = pred_all[:, j]
        
        grad_outputs = torch.ones_like(pred_j)
        grads = torch.autograd.grad(
            outputs=pred_j,
            inputs=X_seq,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False
        )[0]
        
        for i in range(num_vars):
            grad_ij = grads[:, :, i]
            causal_effect_ij = grad_ij.abs().sum(dim=1)
            causal_matrix[:, i, j] = causal_effect_ij
    
    return causal_matrix


def sparsify(causal_matrix, threshold=None, threshold_percentile=75.0):
    B, num_vars, _ = causal_matrix.shape
    sparsified = causal_matrix.clone()
    
    for b in range(B):
        mat = sparsified[b]
        
        for i in range(num_vars):
            mat[i, i] = 0.0
        
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                if mat[i, j] > mat[j, i]:
                    mat[j, i] = 0.0
                else:
                    mat[i, j] = 0.0
        
        if threshold is None:
            if threshold_percentile > 0:
                values = mat[mat > 0]
                if len(values) > 0:
                    th = torch.quantile(values, threshold_percentile / 100.0)
                    mat[mat < th] = 0.0
        else:
            mat[mat < threshold] = 0.0
    
    return sparsified


def aggregate_normal_reference(causal_matrices, method='median'):
    if method == 'median':
        return torch.median(causal_matrices, dim=0)[0]
    elif method == 'mean':
        return torch.mean(causal_matrices, dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def build_reference_mask(reference_mat, percentile=0.0, abs_threshold=0.0):
    if torch.is_tensor(reference_mat):
        abs_ref = reference_mat.abs().clone()
        abs_ref.fill_diagonal_(0.0)
        values = abs_ref[abs_ref > 0]
        if values.numel() == 0:
            return torch.zeros_like(reference_mat), 0.0

        if abs_threshold and abs_threshold > 0:
            threshold = abs_threshold
        elif percentile and percentile > 0:
            threshold = torch.quantile(values, percentile / 100.0).item()
        else:
            return torch.ones_like(reference_mat), 0.0

        mask = (abs_ref >= threshold).to(reference_mat.dtype)
        return mask, float(threshold)

    abs_ref = np.abs(reference_mat)
    np.fill_diagonal(abs_ref, 0.0)
    values = abs_ref[abs_ref > 0]
    if values.size == 0:
        return np.zeros_like(reference_mat, dtype=float), 0.0

    if abs_threshold and abs_threshold > 0:
        threshold = abs_threshold
    elif percentile and percentile > 0:
        threshold = float(np.percentile(values, percentile))
    else:
        return np.ones_like(reference_mat, dtype=float), 0.0

    mask = (abs_ref >= threshold).astype(float)
    return mask, float(threshold)


def compute_spatial_deviation(causal_mats, reference_mat, mask=None):
    if torch.is_tensor(causal_mats):
        diff = causal_mats - reference_mat.unsqueeze(0)
        if mask is not None:
            diff = diff * mask.unsqueeze(0)
        return torch.linalg.norm(diff, ord="fro", dim=(1, 2))

    diff = causal_mats - reference_mat[None, :, :]
    if mask is not None:
        diff = diff * mask[None, :, :]
    return np.linalg.norm(diff, ord="fro", axis=(1, 2))


def compute_normal_temporal_stats(residual_vec_train):
    normal_median = torch.median(residual_vec_train, dim=0).values
    abs_deviations = torch.abs(residual_vec_train - normal_median)
    normal_mad = torch.median(abs_deviations, dim=0).values
    normal_mad = torch.clamp(normal_mad, min=1e-8)
    return normal_median, normal_mad


def compute_temporal_stats(residual_train):
    if torch.is_tensor(residual_train):
        return compute_normal_temporal_stats(residual_train)

    normal_median = np.median(residual_train, axis=0)
    normal_mad = np.median(np.abs(residual_train - normal_median), axis=0)
    normal_mad = np.maximum(normal_mad, 1e-8)
    return normal_median, normal_mad


def compute_temporal_scores(residuals, normal_median, normal_mad, topk_percent):
    if torch.is_tensor(residuals):
        return _temporal_topk_score(
            residuals,
            normal_median,
            normal_mad,
            topk_percent=topk_percent,
            topk_min=1,
        )

    z = np.abs(residuals - normal_median) / normal_mad
    k = max(1, int(np.ceil(z.shape[1] * topk_percent)))
    return np.mean(np.partition(z, -k, axis=1)[:, -k:], axis=1)


def _temporal_topk_score(
    residual_vec,
    normal_residual_median,
    normal_residual_mad,
    topk_percent=0.1,
    topk_min=1,
):
    z = torch.abs(residual_vec - normal_residual_median) / normal_residual_mad
    if topk_percent is None or topk_percent <= 0:
        return z.mean(dim=1)

    num_vars = z.shape[1]
    k = max(topk_min, int(math.ceil(num_vars * topk_percent)))
    k = min(k, num_vars)
    topk_vals, _ = torch.topk(z, k, dim=1)
    return topk_vals.mean(dim=1)


def normalize_scores(scores):
    if torch.is_tensor(scores):
        if scores.max() == scores.min():
            return torch.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    scores = np.asarray(scores, dtype=float)
    min_v = np.min(scores)
    max_v = np.max(scores)
    if max_v == min_v:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v + 1e-8)


class TemporalScorer:
    def __init__(self, normal_median, normal_mad, topk_percent=0.1, topk_min=1):
        self.normal_median = normal_median
        self.normal_mad = normal_mad
        self.topk_percent = topk_percent
        self.topk_min = topk_min

    @classmethod
    def fit(cls, residual_vec_train, topk_percent=0.1, topk_min=1):
        normal_median, normal_mad = compute_normal_temporal_stats(residual_vec_train)
        return cls(normal_median, normal_mad, topk_percent=topk_percent, topk_min=topk_min)

    def score(self, residual_vec):
        return _temporal_topk_score(
            residual_vec,
            self.normal_median,
            self.normal_mad,
            topk_percent=self.topk_percent,
            topk_min=self.topk_min,
        )


def fuse_scores(
    spatial_scores,
    temporal_scores,
    weight_spatial=0.5,
    max_spatial_weight=0.3,
    gate_tau=0.0,
):
    spatial_norm = normalize_scores(spatial_scores)
    temporal_norm = normalize_scores(temporal_scores)

    spatial_weight = min(float(weight_spatial), float(max_spatial_weight))
    gated = torch.relu(spatial_norm - float(gate_tau))
    fused = temporal_norm + spatial_weight * gated
    return fused

