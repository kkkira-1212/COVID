import argparse
from pathlib import Path
import sys

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.causal import (
    build_reference_mask,
    compute_spatial_deviation,
    compute_temporal_stats,
    compute_temporal_scores,
    normalize_scores,
)
from utils.causal_eval import (
    extract_causal_matrices,
    extract_residual_vectors,
    compute_relative_change_scores,
    apply_global_threshold,
)
from utils.causal_io import resolve_bundle_path, load_model, select_normal_indices
from utils.causal_sweep import (
    compute_f1,
    compute_auc,
    compute_auprc,
    fuse_residual_base,
    fuse_relu_gate,
)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep fusion settings for causal stability scores"
    )
    data_args = parser.add_argument_group("data")
    data_args.add_argument("--model_path", type=str, required=True)
    data_args.add_argument("--data_dir", type=str, required=True)
    data_args.add_argument("--dataset", type=str, default="auto", choices=["auto", "psm", "swat", "smap"])
    data_args.add_argument("--scale", type=str, default="coarse", choices=["coarse", "fine"])
    data_args.add_argument("--data_file", type=str, default=None)
    data_args.add_argument("--device", type=str, default="cuda")
    data_args.add_argument("--batch_size", type=int, default=64)
    data_args.add_argument("--max_windows", type=int, default=None)
    data_args.add_argument("--sample_method", type=str, default="random", choices=["random", "first"])
    data_args.add_argument("--seed", type=int, default=42)

    causal_args = parser.add_argument_group("causal")
    causal_args.add_argument("--sparsify_percentile", type=float, default=0.0)
    causal_args.add_argument("--global_threshold_percentile", type=float, default=0.0)
    causal_args.add_argument("--ref_mask_percentile", type=float, default=90.0)
    causal_args.add_argument("--ref_mask_abs", type=float, default=0.0)
    causal_args.add_argument("--struct_metric", type=str, default="fro", choices=["fro", "relative"])
    causal_args.add_argument("--temporal_topk_percent", type=float, default=0.1)

    fusion_args = parser.add_argument_group("fusion")
    fusion_args.add_argument("--fusion_normalize", action="store_true")
    fusion_args.add_argument("--baseline_alpha_a", type=float, default=None)
    fusion_args.add_argument("--threshold_percentile", type=float, default=99.0)
    fusion_args.add_argument("--fusion_mode", type=str, default="residual_base", choices=["residual_base", "relu_gate"])
    fusion_args.add_argument("--max_spatial_weight", type=float, default=0.3)
    fusion_args.add_argument("--relu_tau_percentile", type=float, default=90.0)
    fusion_args.add_argument("--relu_lambda", type=float, default=0.3)

    sweep_args = parser.add_argument_group("sweep")
    sweep_args.add_argument("--sweep_relu", action="store_true")
    sweep_args.add_argument("--tau_start", type=float, default=90.0)
    sweep_args.add_argument("--tau_end", type=float, default=90.0)
    sweep_args.add_argument("--tau_step", type=float, default=1.0)
    sweep_args.add_argument("--lambda_start", type=float, default=0.0)
    sweep_args.add_argument("--lambda_end", type=float, default=1.0)
    sweep_args.add_argument("--lambda_step", type=float, default=0.1)

    save_args = parser.add_argument_group("save")
    save_args.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle_path, dataset_name = resolve_bundle_path(
        args.data_dir, args.dataset, args.scale, args.data_file
    )

    print("=" * 60)
    print("Causal Fusion Sweep")
    print("=" * 60)
    print(f"Dataset: {dataset_name.upper()}  Scale: {args.scale}")
    print(f"Model: {args.model_path}")
    print(f"Bundle: {bundle_path}")
    print(f"Device: {device}")

    bundle = torch.load(bundle_path, weights_only=False)
    X_seq = bundle["X_seq"]
    y_true = bundle["y_next"].cpu().numpy() if isinstance(bundle["y_next"], torch.Tensor) else bundle["y_next"]
    idx_train = bundle["idx_train"].cpu().numpy() if isinstance(bundle["idx_train"], torch.Tensor) else bundle["idx_train"]
    idx_test = bundle["idx_test"].cpu().numpy() if isinstance(bundle["idx_test"], torch.Tensor) else bundle["idx_test"]

    normal_indices = select_normal_indices(
        y_true,
        idx_train,
        max_windows=args.max_windows,
        seed=args.seed,
        sample_method=args.sample_method,
    )

    print(f"Normal windows selected: {len(normal_indices)} / {len(idx_train)} train windows")

    num_vars = X_seq.shape[2]
    seq_len = X_seq.shape[1]
    enc, head = load_model(args.model_path, num_vars, seq_len, device)

    print("Extracting causal matrices for normal windows...")
    causal_mats = extract_causal_matrices(
        enc,
        head,
        X_seq,
        normal_indices,
        device,
        batch_size=args.batch_size,
        scale=args.scale,
        sparsify_percentile=args.sparsify_percentile,
    )

    global_threshold = None
    global_mask = None
    global_abs_mean = None
    if args.global_threshold_percentile and args.global_threshold_percentile > 0:
        if args.sparsify_percentile and args.sparsify_percentile > 0:
            print(
                "Warning: both per-window sparsify and global threshold are enabled. "
                "Global threshold will be applied on top of sparsify."
            )
        causal_mats, global_abs_mean, global_threshold, global_mask = apply_global_threshold(
            causal_mats, args.global_threshold_percentile, enforce_direction=True
        )
        nonzero_ratio = float(np.mean(global_mask))
        print(
            f"\nGlobal threshold applied (abs_mean percentile={args.global_threshold_percentile}): "
            f"threshold={global_threshold:.6f}, nonzero_ratio={nonzero_ratio:.4f}"
        )

    if "X_next" not in bundle:
        raise ValueError("F1 comparison requires X_next in the bundle.")

    train_normal_mask = y_true[idx_train] == 0
    train_normal_indices = idx_train[train_normal_mask]
    if len(train_normal_indices) == 0:
        raise ValueError("Train normal indices required for baseline F1.")

    test_indices = idx_test
    test_labels = y_true[test_indices]

    reference_mat = np.median(causal_mats, axis=0)
    ref_mask = None
    if args.ref_mask_abs > 0 or args.ref_mask_percentile > 0:
        ref_mask, _ = build_reference_mask(
            reference_mat,
            percentile=args.ref_mask_percentile,
            abs_threshold=args.ref_mask_abs,
        )

    train_causal = extract_causal_matrices(
        enc,
        head,
        X_seq,
        train_normal_indices,
        device,
        batch_size=args.batch_size,
        scale=args.scale,
        sparsify_percentile=args.sparsify_percentile,
    )
    test_causal = extract_causal_matrices(
        enc,
        head,
        X_seq,
        test_indices,
        device,
        batch_size=args.batch_size,
        scale=args.scale,
        sparsify_percentile=args.sparsify_percentile,
    )

    if args.struct_metric == "relative":
        struct_train = compute_relative_change_scores(
            train_causal, reference_mat, mask=ref_mask
        )
        struct_test = compute_relative_change_scores(
            test_causal, reference_mat, mask=ref_mask
        )
    else:
        struct_train = compute_spatial_deviation(
            train_causal, reference_mat, mask=ref_mask
        )
        struct_test = compute_spatial_deviation(
            test_causal, reference_mat, mask=ref_mask
        )

    residual_train = extract_residual_vectors(
        enc,
        head,
        X_seq,
        bundle["X_next"],
        train_normal_indices,
        device,
        batch_size=args.batch_size,
        scale=args.scale,
    )
    normal_median, normal_mad = compute_temporal_stats(residual_train)

    residual_test = extract_residual_vectors(
        enc,
        head,
        X_seq,
        bundle["X_next"],
        test_indices,
        device,
        batch_size=args.batch_size,
        scale=args.scale,
    )
    temp_train_residual = compute_temporal_scores(
        residual_train,
        normal_median,
        normal_mad,
        args.temporal_topk_percent,
    )
    temp_test_residual = compute_temporal_scores(
        residual_test,
        normal_median,
        normal_mad,
        args.temporal_topk_percent,
    )
    temp_train_diag = compute_relative_change_scores(
        train_causal, reference_mat, mask=ref_mask, diag_only=True
    )
    temp_test_diag = compute_relative_change_scores(
        test_causal, reference_mat, mask=ref_mask, diag_only=True
    )

    if args.fusion_normalize:
        struct_all = normalize_scores(np.concatenate([struct_train, struct_test]))
        struct_train = struct_all[: len(struct_train)]
        struct_test = struct_all[len(struct_train):]
        temp_all = normalize_scores(np.concatenate([temp_train_residual, temp_test_residual]))
        temp_train_residual = temp_all[: len(temp_train_residual)]
        temp_test_residual = temp_all[len(temp_train_residual):]
        if temp_train_diag is not None and temp_test_diag is not None:
            temp_all = normalize_scores(np.concatenate([temp_train_diag, temp_test_diag]))
            temp_train_diag = temp_all[: len(temp_train_diag)]
            temp_test_diag = temp_all[len(temp_train_diag):]

    def eval_baseline(label, alpha, temp_train, temp_test):
        if args.fusion_mode == "relu_gate":
            tau = float(np.percentile(struct_train, args.relu_tau_percentile))
            fused_train = fuse_relu_gate(struct_train, temp_train, tau, args.relu_lambda)
            fused_test = fuse_relu_gate(struct_test, temp_test, tau, args.relu_lambda)
        else:
            fused_train = fuse_residual_base(struct_train, temp_train, alpha, args.max_spatial_weight)
            fused_test = fuse_residual_base(struct_test, temp_test, alpha, args.max_spatial_weight)
        th = np.percentile(fused_train, args.threshold_percentile)
        pred_test = (fused_test >= th).astype(int)
        f1 = compute_f1(pred_test, test_labels)
        auc = compute_auc(fused_test, test_labels)
        auprc = compute_auprc(fused_test, test_labels)
        print(
            f"  [{label}] alpha={alpha:.2f} threshold(q{args.threshold_percentile:.0f})={th:.6f} "
            f"F1={f1:.4f} ROC-AUC={auc:.4f} AUPRC={auprc:.4f}"
        )

    alpha_a = 0.0 if args.baseline_alpha_a is None else args.baseline_alpha_a
    print("\nBaseline F1 comparison (test set):")
    eval_baseline("Baseline A (Residual)", alpha_a, temp_train_residual, temp_test_residual)

    if args.sweep_relu:
        print("\nReLU-gate sweep (test set):")
        lam_values = np.arange(args.lambda_start, args.lambda_end + 1e-8, args.lambda_step)
        tau_values = np.arange(args.tau_start, args.tau_end + 1e-8, args.tau_step)
        for tau_pct in tau_values:
            tau_val = float(np.percentile(struct_train, tau_pct))
            for lam in lam_values:
                fused_train = fuse_relu_gate(struct_train, temp_train_residual, tau_val, lam)
                fused_test = fuse_relu_gate(struct_test, temp_test_residual, tau_val, lam)
                th = np.percentile(fused_train, args.threshold_percentile)
                pred_test = (fused_test >= th).astype(int)
                f1 = compute_f1(pred_test, test_labels)
                auc = compute_auc(fused_test, test_labels)
                auprc = compute_auprc(fused_test, test_labels)
                print(
                    f"  tau=q{tau_pct:.0f} lam={lam:.2f} "
                    f"F1={f1:.4f} ROC-AUC={auc:.4f} AUPRC={auprc:.4f}"
                )

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{dataset_name}_{args.scale}_causal_stability.npz"
        save_payload = {
            "normal_indices": normal_indices,
            "global_abs_mean": global_abs_mean,
            "global_threshold": global_threshold,
            "global_mask": global_mask,
        }
        np.savez(out_path, **save_payload)
        print(f"\nSaved stats to: {out_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
