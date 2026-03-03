import argparse
from pathlib import Path
import sys

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.causal_eval import extract_causal_matrices, apply_global_threshold
from utils.causal_io import resolve_bundle_path, load_model, select_normal_indices


def main():
    parser = argparse.ArgumentParser(
        description="Verify causal graph stability on normal windows (stage2 graph validation)"
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

    save_args = parser.add_argument_group("save")
    save_args.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle_path, dataset_name = resolve_bundle_path(
        args.data_dir, args.dataset, args.scale, args.data_file
    )

    print("=" * 60)
    print("Causal Graph Stability Check (Normal Windows)")
    print("=" * 60)
    print(f"Dataset: {dataset_name.upper()}  Scale: {args.scale}")
    print(f"Model: {args.model_path}")
    print(f"Bundle: {bundle_path}")
    print(f"Device: {device}")

    bundle = torch.load(bundle_path, weights_only=False)
    X_seq = bundle["X_seq"]
    y_true = bundle["y_next"].cpu().numpy() if isinstance(bundle["y_next"], torch.Tensor) else bundle["y_next"]
    idx_train = bundle["idx_train"].cpu().numpy() if isinstance(bundle["idx_train"], torch.Tensor) else bundle["idx_train"]
    idx_val = bundle["idx_val"].cpu().numpy() if isinstance(bundle["idx_val"], torch.Tensor) else bundle["idx_val"]
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

