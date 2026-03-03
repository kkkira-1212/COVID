import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.sequence_common import (
    create_sequences_with_mapping,
    create_coarse_sequences_from_fine,
    create_fine_to_coarse_mapping,
    split_sequences_preserve_test,
    sequences_to_bundle,
    standardize,
)


def load_psm_data(data_dir, split="train"):
    data_dir = Path(data_dir)

    if split == "train":
        df = pd.read_csv(data_dir / "train.csv")
        df["outbreak_label"] = 0
    elif split == "test":
        df = pd.read_csv(data_dir / "test.csv")
        df_label = pd.read_csv(data_dir / "test_label.csv")
        df = df.merge(df_label, on="timestamp_(min)", how="left")
        df["outbreak_label"] = df["label"].fillna(0).astype(int)
        df = df.drop(columns=["label"])
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split}")

    base_date = pd.Timestamp("2020-01-01 00:00:00")
    df["Date"] = base_date + pd.to_timedelta(df["timestamp_(min)"], unit="m")
    df["State"] = "PSM"

    feature_cols = [
        col
        for col in df.columns
        if col not in ["timestamp_(min)", "Date", "State", "outbreak_label"]
    ]

    for col in feature_cols:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill().fillna(0)

    df = df.sort_values("timestamp_(min)").reset_index(drop=True)

    return df, feature_cols


def process_psm_data(
    data_dir="data/PSM/PSM",
    output_dir="data/PSM/processed",
    window_fine=60,
    window_coarse=12,
    k=5,
    train_ratio=0.8,
    val_ratio=0.2,
    agg_func="mean",
    feature_cols=None,
    max_features=None,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PSM data...")
    df_train, feature_cols_all = load_psm_data(data_dir, split="train")
    df_test, _ = load_psm_data(data_dir, split="test")

    print(f"Train samples: {len(df_train):,}")
    print(f"Test samples: {len(df_test):,}")
    print(
        f"Test anomalies: {df_test['outbreak_label'].sum():,} "
        f"({df_test['outbreak_label'].mean()*100:.2f}%)"
    )

    if feature_cols is None:
        if max_features is not None and max_features < len(feature_cols_all):
            use_features = feature_cols_all[:max_features]
        else:
            use_features = feature_cols_all
    else:
        use_features = [f for f in feature_cols if f in feature_cols_all]
        if len(use_features) == 0:
            raise ValueError(
                f"None of the specified features found. Available: {feature_cols_all[:10]}"
            )

    print(f"Using {len(use_features)} features")

    target_feature = use_features[0]
    df_train["target_return"] = df_train[target_feature].pct_change().fillna(0)
    df_test["target_return"] = df_test[target_feature].pct_change().fillna(0)

    fine_data_train = {"PSM": df_train.copy()}
    fine_data_test = {"PSM": df_test.copy()}

    print("Creating fine sequences...")
    fine_sequences_train, fine_states_train = create_sequences_with_mapping(
        fine_data_train,
        use_features,
        window_fine,
        stride=1,
        time_col="timestamp_(min)",
        date_col="Date",
        label_col="outbreak_label",
        return_col="target_return",
        base_date="1970-01-01",
        time_unit="m",
    )
    fine_sequences_test, fine_states_test = create_sequences_with_mapping(
        fine_data_test,
        use_features,
        window_fine,
        stride=1,
        time_col="timestamp_(min)",
        date_col="Date",
        label_col="outbreak_label",
        return_col="target_return",
        base_date="1970-01-01",
        time_unit="m",
    )

    print(f"Created {len(fine_sequences_train):,} train fine sequences")
    print(f"Created {len(fine_sequences_test):,} test fine sequences")
    print(f"Total fine sequences: {len(fine_sequences_train) + len(fine_sequences_test):,}")

    print(f"Creating coarse sequences (k={k})...")
    coarse_sequences_train = create_coarse_sequences_from_fine(
        fine_sequences_train, k, agg_func=agg_func
    )
    coarse_sequences_test = create_coarse_sequences_from_fine(
        fine_sequences_test, k, agg_func=agg_func
    )

    print(f"Created {len(coarse_sequences_train):,} train coarse sequences")
    print(f"Created {len(coarse_sequences_test):,} test coarse sequences")
    print(f"Total coarse sequences: {len(coarse_sequences_train) + len(coarse_sequences_test):,}")

    print("Splitting sequences (preserving test set)...")
    (
        idx_tr_fine,
        idx_v_fine,
        idx_te_fine,
        idx_tr_coarse,
        idx_v_coarse,
        idx_te_coarse,
        fine_sequences_all,
        coarse_sequences_all,
    ) = split_sequences_preserve_test(
        fine_sequences_train,
        fine_sequences_test,
        coarse_sequences_train,
        coarse_sequences_test,
        train_ratio,
        val_ratio,
    )

    print(
        f"Fine scale - Train: {len(idx_tr_fine):,}, "
        f"Val: {len(idx_v_fine):,}, Test: {len(idx_te_fine):,}"
    )
    print(
        f"Coarse scale - Train: {len(idx_tr_coarse):,}, "
        f"Val: {len(idx_v_coarse):,}, Test: {len(idx_te_coarse):,}"
    )

    print("Creating bundles...")
    bundle_fine = sequences_to_bundle(
        fine_sequences_all, idx_tr_fine, idx_v_fine, idx_te_fine,
        fine_states_train, use_features, window_fine
    )
    bundle_coarse = sequences_to_bundle(
        coarse_sequences_all, idx_tr_coarse, idx_v_coarse, idx_te_coarse,
        fine_states_train, use_features, window_fine
    )

    print("Standardizing...")
    bundle_fine = standardize(bundle_fine)
    bundle_coarse = standardize(bundle_coarse)

    print("Creating fine-to-coarse mapping...")
    mapping = create_fine_to_coarse_mapping(fine_sequences_all, coarse_sequences_all, k)
    bundle_fine["fine_to_coarse_index"] = mapping

    n_valid = (mapping >= 0).sum().item()
    print(f"Mapping created: {n_valid}/{len(mapping)} fine sequences mapped to coarse sequences")

    y_true_fine = bundle_fine["y_next"]
    y_true_coarse = bundle_coarse["y_next"]

    print("\nAnomaly ratios:")
    print(
        f"  Fine - Train: {y_true_fine[idx_tr_fine].sum().item() / len(idx_tr_fine):.4f} "
        f"({y_true_fine[idx_tr_fine].sum().item()}/{len(idx_tr_fine)})"
    )
    print(
        f"  Fine - Val: {y_true_fine[idx_v_fine].sum().item() / len(idx_v_fine):.4f} "
        f"({y_true_fine[idx_v_fine].sum().item()}/{len(idx_v_fine)})"
    )
    print(
        f"  Fine - Test: {y_true_fine[idx_te_fine].sum().item() / len(idx_te_fine):.4f} "
        f"({y_true_fine[idx_te_fine].sum().item()}/{len(idx_te_fine)})"
    )
    print(
        f"  Coarse - Test: {y_true_coarse[idx_te_coarse].sum().item() / len(idx_te_coarse):.4f} "
        f"({y_true_coarse[idx_te_coarse].sum().item()}/{len(idx_te_coarse)})"
    )

    print(f"\nSaving to {output_dir}...")
    torch.save(bundle_fine, output_dir / "psm_fine.pt")
    torch.save(bundle_coarse, output_dir / "psm_coarse.pt")

    info = {
        "dataset": "PSM",
        "target_feature": target_feature,
        "fine_shape": list(bundle_fine["X_seq"].shape),
        "coarse_shape": list(bundle_coarse["X_seq"].shape),
        "features": use_features,
        "n_features": len(use_features),
        "window_fine": window_fine,
        "window_coarse": window_coarse,
        "k": k,
        "agg_func": agg_func,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "n_fine_sequences": len(fine_sequences_all),
        "n_coarse_sequences": len(coarse_sequences_all),
        "n_train_fine": len(idx_tr_fine),
        "n_val_fine": len(idx_v_fine),
        "n_test_fine": len(idx_te_fine),
        "n_train_coarse": len(idx_tr_coarse),
        "n_val_coarse": len(idx_v_coarse),
        "n_test_coarse": len(idx_te_coarse),
        "mapping_valid_ratio": float(n_valid / len(mapping)),
        "test_anomaly_ratio_fine": float(y_true_fine[idx_te_fine].mean().item()),
        "test_anomaly_ratio_coarse": float(y_true_coarse[idx_te_coarse].mean().item()),
    }

    with open(output_dir / "psm_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Processing complete!")
    return bundle_fine, bundle_coarse, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process PSM data into fine/coarse scales")
    parser.add_argument("--data_dir", type=str, default="data/PSM/PSM")
    parser.add_argument("--output_dir", type=str, default="data/PSM/processed")
    parser.add_argument("--window_fine", type=int, default=60)
    parser.add_argument("--window_coarse", type=int, default=12)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--agg_func", type=str, default="mean", choices=["mean", "sum", "max", "min"])
    parser.add_argument("--max_features", type=int, default=None)

    args = parser.parse_args()

    bundle_fine, bundle_coarse, info = process_psm_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_fine=args.window_fine,
        window_coarse=args.window_coarse,
        k=args.k,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        agg_func=args.agg_func,
        max_features=args.max_features,
    )

    print("\nProcessing Summary:")
    print(f"  Fine sequences: {info['n_fine_sequences']:,}")
    print(f"  Coarse sequences: {info['n_coarse_sequences']:,}")
    print(f"  Aggregation factor k: {info['k']}")
    print(f"  Mapping valid ratio: {info['mapping_valid_ratio']:.2%}")
