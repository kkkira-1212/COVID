import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_f1(pred, labels):
    pred = np.asarray(pred, dtype=int)
    labels = np.asarray(labels, dtype=int)
    tp = np.sum((pred == 1) & (labels == 1))
    fp = np.sum((pred == 1) & (labels == 0))
    fn = np.sum((pred == 0) & (labels == 1))
    if tp == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_auc(scores, labels):
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def compute_auprc(scores, labels):
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def fuse_residual_base(struct_scores, temp_scores, weight_spatial, max_spatial_weight=0.3):
    spatial_weight = min(float(weight_spatial), float(max_spatial_weight))
    return temp_scores + spatial_weight * struct_scores


def fuse_relu_gate(struct_scores, temp_scores, tau, lam):
    return temp_scores + lam * np.maximum(struct_scores - tau, 0.0)
