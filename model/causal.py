import torch

def compute_gradient_causal_matrix(X_seq, encoder, head, use_fine=True):
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


def compute_spatial_deviation(causal_matrix, normal_reference):
    diff = causal_matrix - normal_reference.unsqueeze(0)
    deviations = torch.norm(diff, p='fro', dim=(-2, -1))
    return deviations


def compute_normal_temporal_stats(residual_vec_train):
    residual_mean_train = residual_vec_train.mean(dim=1)
    normal_median = torch.median(residual_mean_train)
    abs_deviations = torch.abs(residual_mean_train - normal_median)
    normal_mad = torch.median(abs_deviations)
    return normal_median, normal_mad


def compute_temporal_deviation(residual_vec, normal_residual_median, normal_residual_mad):
    residual_mean = residual_vec.mean(dim=1)
    mu_diff = torch.abs(residual_mean - normal_residual_median)
    deviations = mu_diff / (normal_residual_mad + 1e-8)
    return deviations


def normalize_scores(scores):
    if scores.max() == scores.min():
        return torch.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def fuse_scores(spatial_scores, temporal_scores, weight_spatial=0.5):
    spatial_norm = normalize_scores(spatial_scores)
    temporal_norm = normalize_scores(temporal_scores)
    
    fused = weight_spatial * spatial_norm + (1 - weight_spatial) * temporal_norm
    return fused

