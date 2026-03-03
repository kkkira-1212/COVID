import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.sequence_common import (
    create_sequences_with_mapping,
    create_coarse_sequences_and_mapping,
    split_train_val_test,
    sequences_to_bundle,
    standardize,
    apply_multi_scale_subset_sampling,
    apply_test_subset_sampling,
)


def _read_smd_matrix(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        data = np.array(data, dtype=np.float32)
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    data = np.genfromtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if np.isnan(data).mean() > 0.5 or data.shape[1] == 1:
        data_ws = np.genfromtxt(path, delimiter=None)
        if data_ws.ndim == 1:
            data_ws = data_ws.reshape(-1, 1)
        if data_ws.shape[1] > 1:
            data = data_ws

    data = np.array(data, dtype=np.float32)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def _list_machine_files(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = []
    for p in data_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".txt", ".csv", ".npy"}:
            files.append(p.name)

    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    return sorted(files)


def _load_flat_smd(data_dir):
    data_dir = Path(data_dir)
    train_path = data_dir / "SMD_train.npy"
    test_path = data_dir / "SMD_test.npy"
    label_path = data_dir / "SMD_test_label.npy"

    if not (train_path.exists() and test_path.exists() and label_path.exists()):
        return None

    train_data = _read_smd_matrix(train_path)
    test_data = _read_smd_matrix(test_path)
    test_labels = _read_smd_matrix(label_path).astype(int).reshape(-1)

    if len(test_labels) != test_data.shape[0]:
        min_len = min(len(test_labels), test_data.shape[0])
        test_data = test_data[:min_len]
        test_labels = test_labels[:min_len]

    feature_cols = [f"feature_{i}" for i in range(train_data.shape[1])]

    def build_df(data, labels, name):
        n_samples = data.shape[0]
        df = pd.DataFrame(data, columns=feature_cols)
        df["outbreak_label"] = labels
        df["timestamp"] = np.arange(n_samples)
        base_date = pd.Timestamp("2020-01-01 00:00:00")
        df["Date"] = base_date + pd.to_timedelta(df["timestamp"], unit="m")
        df["State"] = name
        target_feature = feature_cols[0]
        df["target_return"] = df[target_feature].pct_change().fillna(0)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].ffill().fillna(0)
        return df

    df_train = build_df(train_data, np.zeros(train_data.shape[0], dtype=int), "SMD")
    df_test = build_df(test_data, test_labels, "SMD")
    return df_train, df_test, feature_cols


def load_smd_split(data_dir, machine_file, split="train"):
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    data_path = split_dir / machine_file
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {split} file: {data_path}")

    data = _read_smd_matrix(data_path)
    n_samples, n_features = data.shape

    if split == "train":
        labels = np.zeros(n_samples, dtype=int)
    elif split == "test":
        label_path = data_dir / "test_label" / machine_file
        if not label_path.exists():
            raise FileNotFoundError(f"Missing test label file: {label_path}")
        labels = _read_smd_matrix(label_path).astype(int).reshape(-1)
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split}")

    if len(labels) != n_samples:
        min_len = min(len(labels), n_samples)
        data = data[:min_len]
        labels = labels[:min_len]
        n_samples = min_len

    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_cols)
    df["outbreak_label"] = labels

    df["timestamp"] = np.arange(n_samples)
    base_date = pd.Timestamp("2020-01-01 00:00:00")
    df["Date"] = base_date + pd.to_timedelta(df["timestamp"], unit="m")
    df["State"] = Path(machine_file).stem

    target_feature = feature_cols[0]
    df["target_return"] = df[target_feature].pct_change().fillna(0)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill().fillna(0)

    return df, feature_cols


def process_smd_data(
    data_dir="data/SMD",
    output_dir="data/SMD/processed",
    window_fine=100,
    window_coarse=20,
    k=5,
    train_ratio=0.8,
    val_ratio=0.2,
    agg_func="mean",
    max_features=None,
    max_samples=None,
    max_test_samples=None,
    stratified_test=True,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flat_loaded = _load_flat_smd(data_dir)
    if flat_loaded is not None:
        df_train, df_test, feature_cols = flat_loaded
        train_data = {"SMD": df_train}
        test_data = {"SMD": df_test}
        min_features = len(feature_cols)
        if max_features is not None:
            min_features = min(min_features, max_features)
        use_features = [f"feature_{i}" for i in range(min_features)]
        print(f"Using {len(use_features)} features (flat SMD)")
        states_count = 1
    else:
        train_files = _list_machine_files(data_dir / "train")
        train_data = {}
        test_data = {}
        feature_counts = []

        for machine_file in train_files:
            df_train, feature_cols = load_smd_split(data_dir, machine_file, split="train")
            df_test, _ = load_smd_split(data_dir, machine_file, split="test")
            train_data[Path(machine_file).stem] = df_train
            test_data[Path(machine_file).stem] = df_test
            feature_counts.append(len(feature_cols))

        min_features = min(feature_counts) if feature_counts else 0
        if min_features == 0:
            raise ValueError("No features found in SMD data")

        if max_features is not None:
            min_features = min(min_features, max_features)

        use_features = [f"feature_{i}" for i in range(min_features)]
        print(f"Using {len(use_features)} features across {len(train_files)} machines")
        states_count = len(train_files)

    print("Creating fine sequences...")
    fine_sequences_train, states_order = create_sequences_with_mapping(
        train_data, use_features, window_fine, stride=1, time_col="timestamp", date_col="Date"
    )
    fine_sequences_test, _ = create_sequences_with_mapping(
        test_data, use_features, window_fine, stride=1, time_col="timestamp", date_col="Date"
    )

    print(f"Created {len(fine_sequences_train):,} train fine sequences")
    print(f"Created {len(fine_sequences_test):,} test fine sequences")

    print(f"Creating coarse sequences (k={k})...")
    coarse_sequences_train, mapping_train = create_coarse_sequences_and_mapping(
        fine_sequences_train, states_order, k, agg_func=agg_func
    )
    coarse_sequences_test, mapping_test = create_coarse_sequences_and_mapping(
        fine_sequences_test, states_order, k, agg_func=agg_func
    )

    fine_sequences_all = fine_sequences_train + fine_sequences_test
    coarse_sequences_all = coarse_sequences_train + coarse_sequences_test

    idx_tr_fine, idx_v_fine, idx_te_fine = split_train_val_test(
        fine_sequences_train, fine_sequences_test, train_ratio, val_ratio
    )
    idx_tr_coarse, idx_v_coarse, idx_te_coarse = split_train_val_test(
        coarse_sequences_train, coarse_sequences_test, train_ratio, val_ratio
    )

    mapping = torch.full((len(fine_sequences_all),), -1, dtype=torch.long)
    if len(fine_sequences_train) > 0:
        mapping[:len(fine_sequences_train)] = mapping_train
    if len(fine_sequences_test) > 0:
        offset = len(coarse_sequences_train)
        test_mapping = mapping_test.clone()
        valid_mask = test_mapping >= 0
        test_mapping[valid_mask] += offset
        mapping[len(fine_sequences_train):] = test_mapping

    print("Creating bundles...")
    bundle_fine = sequences_to_bundle(
        fine_sequences_all, idx_tr_fine, idx_v_fine, idx_te_fine,
        states_order, use_features, window_fine
    )
    bundle_coarse = sequences_to_bundle(
        coarse_sequences_all, idx_tr_coarse, idx_v_coarse, idx_te_coarse,
        states_order, use_features, window_fine
    )

    print("Standardizing...")
    bundle_fine = standardize(bundle_fine)
    bundle_coarse = standardize(bundle_coarse)
    bundle_fine["fine_to_coarse_index"] = mapping

    if max_samples is not None and max_samples > 0:
        print(f"\nApplying subset sampling (max_samples={max_samples})...")
        bundle_fine, bundle_coarse = apply_multi_scale_subset_sampling(
            bundle_fine, bundle_coarse, max_samples, stratified_test=stratified_test
        )

    if max_test_samples is not None and max_test_samples > 0:
        print(
            f"\nApplying test-only subset (max_test_samples={max_test_samples}, "
            f"stratified={stratified_test})..."
        )
        bundle_fine, bundle_coarse = apply_test_subset_sampling(
            bundle_fine, bundle_coarse, max_test_samples, stratified=stratified_test
        )

    print(f"Saving to {output_dir}...")
    torch.save(bundle_fine, output_dir / "smd_fine.pt")
    torch.save(bundle_coarse, output_dir / "smd_coarse.pt")

    y_true_fine = bundle_fine["y_next"]
    y_true_coarse = bundle_coarse["y_next"]
    idx_tr_fine = bundle_fine["idx_train"]
    idx_v_fine = bundle_fine["idx_val"]
    idx_te_fine = bundle_fine["idx_test"]
    idx_tr_coarse = bundle_coarse["idx_train"]
    idx_v_coarse = bundle_coarse["idx_val"]
    idx_te_coarse = bundle_coarse["idx_test"]

    info = {
        "dataset": "SMD",
        "n_machines": states_count,
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
        "mapping_valid_ratio": float((mapping >= 0).sum().item() / len(mapping)) if len(mapping) > 0 else 0.0,
        "test_anomaly_ratio_fine": float(y_true_fine[idx_te_fine].mean().item()) if len(idx_te_fine) > 0 else 0.0,
        "test_anomaly_ratio_coarse": float(y_true_coarse[idx_te_coarse].mean().item()) if len(idx_te_coarse) > 0 else 0.0,
        "split_strategy": "normal_train_attack_test",
        "max_samples": max_samples,
        "max_test_samples": max_test_samples,
        "stratified_test": stratified_test,
    }

    with open(output_dir / "smd_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Processing complete!")
    return bundle_fine, bundle_coarse, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process SMD data into fine/coarse scales")
    parser.add_argument("--data_dir", type=str, default="data/SMD")
    parser.add_argument("--output_dir", type=str, default="data/SMD/processed")
    parser.add_argument("--window_fine", type=int, default=100)
    parser.add_argument("--window_coarse", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--agg_func", type=str, default="mean", choices=["mean", "sum", "max", "min"])
    parser.add_argument("--max_features", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--no_stratified_test", action="store_true")

    args = parser.parse_args()

    bundle_fine, bundle_coarse, info = process_smd_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_fine=args.window_fine,
        window_coarse=args.window_coarse,
        k=args.k,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        agg_func=args.agg_func,
        max_features=args.max_features,
        max_samples=args.max_samples,
        max_test_samples=args.max_test_samples,
        stratified_test=not args.no_stratified_test,
    )

    print("\nProcessing Summary:")
    print(f"  Fine sequences: {info['n_fine_sequences']:,}")
    print(f"  Coarse sequences: {info['n_coarse_sequences']:,}")
    print(f"  Mapping valid ratio: {info['mapping_valid_ratio']:.2%}")
