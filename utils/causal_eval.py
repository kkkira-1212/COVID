import numpy as np
import torch

from model.causal import grad_causal_matrix, sparsify


def extract_causal_matrices(
    enc,
    head,
    X_seq,
    indices,
    device,
    batch_size=64,
    scale="coarse",
    sparsify_percentile=0.0,
):
    causal_matrices = []
    use_fine = scale == "fine"
    with torch.enable_grad():
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = torch.as_tensor(indices[start:end], device=X_seq.device)
            X_batch = X_seq[batch_idx].to(device)
            causal_batch = grad_causal_matrix(X_batch, enc, head, use_fine=use_fine)
            if sparsify_percentile and sparsify_percentile > 0:
                causal_batch = sparsify(causal_batch, threshold_percentile=sparsify_percentile)
            causal_matrices.append(causal_batch.cpu().numpy())
    return np.concatenate(causal_matrices, axis=0)


def extract_residual_vectors(
    enc,
    head,
    X_seq,
    X_next,
    indices,
    device,
    batch_size=64,
    scale="coarse",
):
    residual_vecs = []
    use_fine = scale == "fine"
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = torch.as_tensor(indices[start:end], device=X_seq.device)
            X_batch = X_seq[batch_idx].to(device)
            X_next_batch = X_next[batch_idx].to(device)
            z_batch = enc(X_batch)
            if use_fine:
                pred_batch, _ = head(z_batch, z_batch)
            else:
                _, pred_batch = head(z_batch, z_batch)
            residual_batch = (X_next_batch - pred_batch).abs()
            residual_vecs.append(residual_batch.cpu().numpy())
    return np.concatenate(residual_vecs, axis=0)


def compute_relative_change_scores(causal_mats, reference_mat, mask=None, diag_only=False, eps=1e-8):
    ref = reference_mat.copy()
    if mask is not None:
        ref = ref * mask
    abs_ref = np.abs(ref) + eps
    if diag_only:
        ref_diag = np.diag(abs_ref)
        curr_diag = np.diagonal(causal_mats, axis1=1, axis2=2)
        scores = np.mean(np.abs(curr_diag - np.diag(ref)) / (ref_diag + eps), axis=1)
        return scores
    diff = np.abs(causal_mats - reference_mat[None, :, :])
    if mask is not None:
        diff = diff * mask[None, :, :]
    scores = np.mean(diff / abs_ref[None, :, :], axis=(1, 2))
    return scores


def apply_global_threshold(causal_mats, threshold_percentile, enforce_direction=True):
    num_vars = causal_mats.shape[1]
    abs_mean = np.mean(np.abs(causal_mats), axis=0)
    abs_mean_no_diag = abs_mean.copy()
    np.fill_diagonal(abs_mean_no_diag, 0.0)

    direction_mask = np.ones((num_vars, num_vars), dtype=bool)
    if enforce_direction:
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                if abs_mean_no_diag[i, j] >= abs_mean_no_diag[j, i]:
                    direction_mask[j, i] = False
                else:
                    direction_mask[i, j] = False

    abs_mean_dir = abs_mean_no_diag * direction_mask
    values = abs_mean_dir[abs_mean_dir > 0]
    if len(values) == 0:
        return causal_mats, abs_mean, 0.0, direction_mask

    threshold = np.percentile(values, threshold_percentile)
    keep_mask = abs_mean_dir >= threshold
    masked = causal_mats * keep_mask[None, :, :]
    return masked, abs_mean, threshold, keep_mask
