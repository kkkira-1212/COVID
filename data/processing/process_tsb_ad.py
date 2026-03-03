import sys
import re
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
)


def _parse_train_len(filename):
    match = re.search(r"_tr_(\d+)_1st_(\d+)", filename)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _dataset_name_from_file(path):
    parts = Path(path).name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"


def _list_tsb_files(data_dir, include=None, exclude=None, max_files=None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    if include:
        include_l = [s.lower() for s in include]
        files = [p for p in files if _dataset_name_from_file(p).lower() in include_l]
    if exclude:
        exclude_l = [s.lower() for s in exclude]
        files = [p for p in files if _dataset_name_from_file(p).lower() not in exclude_l]

    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise FileNotFoundError("No CSV files found after filtering.")
    return files


def _load_tsb_file(path):
    df = pd.read_csv(path)
    label_col = "Label" if "Label" in df.columns else "label" if "label" in df.columns else None
    if label_col is None:
        raise ValueError(f"Missing Label column in {path}")

    labels = df[label_col].fillna(0).astype(int).values
    features = df.drop(columns=[label_col])

    for col in features.columns:
        if features[col].dtype == "object":
            features[col] = pd.to_numeric(features[col], errors="coerce")

    values = np.array(features.values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    feature_cols = [f"feature_{i}" for i in range(values.shape[1])]
    df_feat = pd.DataFrame(values, columns=feature_cols)
    df_feat["outbreak_label"] = labels
    df_feat["timestamp"] = np.arange(len(df_feat))
    base_date = pd.Timestamp("2020-01-01 00:00:00")
    df_feat["Date"] = base_date + pd.to_timedelta(df_feat["timestamp"], unit="m")
    df_feat["State"] = Path(path).stem

    if feature_cols:
        df_feat["target_return"] = df_feat[feature_cols[0]].pct_change().fillna(0)
    else:
        df_feat["target_return"] = 0.0

    num_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat[num_cols] = df_feat[num_cols].ffill().fillna(0)
    return df_feat, feature_cols


def _split_train_test(df, train_len):
    if train_len is None or train_len <= 0:
        train_len = len(df)
    df_train = df.iloc[:train_len].copy()
    df_test = df.iloc[train_len:].copy()
    return df_train, df_test


def process_tsb_ad_data(
    data_dir="data/TSB-AD-M",
    output_dir="data/TSB-AD-M/processed",
    include=None,
    exclude=None,
    max_files=None,
    window_fine=100,
    window_coarse=20,
    k=5,
    stride=1,
    train_ratio=0.8,
    val_ratio=0.2,
    agg_func="mean",
    max_features=None,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _list_tsb_files(data_dir, include=include, exclude=exclude, max_files=max_files)

    train_data = {}
    test_data = {}
    feature_counts = []
    dataset_names = []

    for path in files:
        df, feature_cols = _load_tsb_file(path)
        train_len, first_anom = _parse_train_len(path.name)
        df_train, df_test = _split_train_test(df, train_len)

        if df_train["outbreak_label"].sum() > 0:
            print(f"Warning: train split in {path.name} contains anomalies.")
        if first_anom is not None and len(df_test) > 0:
            if df_test["outbreak_label"].iloc[:max(0, first_anom - (train_len or 0))].sum() > 0:
                print(f"Warning: anomalies appear earlier than 1st_ in {path.name}.")

        state_name = Path(path).stem
        train_data[state_name] = df_train
        test_data[state_name] = df_test
        feature_counts.append(len(feature_cols))
        dataset_names.append(_dataset_name_from_file(path))

    min_features = min(feature_counts) if feature_counts else 0
    if min_features == 0:
        raise ValueError("No features found in TSB-AD-M data")
    if max_features is not None:
        min_features = min(min_features, max_features)

    use_features = [f"feature_{i}" for i in range(min_features)]
    print(f"Using {len(use_features)} features across {len(files)} files")

    print("Creating fine sequences...")
    fine_sequences_train, states_order = create_sequences_with_mapping(
        train_data, use_features, window_fine, stride=stride, time_col="timestamp", date_col="Date"
    )
    fine_sequences_test, _ = create_sequences_with_mapping(
        test_data, use_features, window_fine, stride=stride, time_col="timestamp", date_col="Date"
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

    if include and len(include) == 1:
        base_name = include[0]
    elif include and len(include) > 1:
        base_name = "tsb_ad_mix"
    else:
        base_name = "tsb_ad_all"

    print(f"Saving to {output_dir}...")
    torch.save(bundle_fine, output_dir / f"{base_name}_fine.pt")
    torch.save(bundle_coarse, output_dir / f"{base_name}_coarse.pt")

    y_true_fine = bundle_fine["y_next"]
    y_true_coarse = bundle_coarse["y_next"]
    idx_tr_fine = bundle_fine["idx_train"]
    idx_v_fine = bundle_fine["idx_val"]
    idx_te_fine = bundle_fine["idx_test"]
    idx_tr_coarse = bundle_coarse["idx_train"]
    idx_v_coarse = bundle_coarse["idx_val"]
    idx_te_coarse = bundle_coarse["idx_test"]

    info = {
        "dataset": "TSB-AD-M",
        "files": [p.name for p in files],
        "datasets": sorted(set(dataset_names)),
        "n_files": len(files),
        "features": use_features,
        "n_features": len(use_features),
        "window_fine": window_fine,
        "window_coarse": window_coarse,
        "k": k,
        "stride": stride,
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
    }

    with open(output_dir / f"{base_name}_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Processing complete!")
    return bundle_fine, bundle_coarse, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process TSB-AD-M data into fine/coarse scales")
    parser.add_argument("--data_dir", type=str, default="data/TSB-AD-M")
    parser.add_argument("--output_dir", type=str, default="data/TSB-AD-M/processed")
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--window_fine", type=int, default=100)
    parser.add_argument("--window_coarse", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--agg_func", type=str, default="mean", choices=["mean", "sum", "max", "min"])
    parser.add_argument("--max_features", type=int, default=None)

    args = parser.parse_args()

    bundle_fine, bundle_coarse, info = process_tsb_ad_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        include=args.include,
        exclude=args.exclude,
        max_files=args.max_files,
        window_fine=args.window_fine,
        window_coarse=args.window_coarse,
        k=args.k,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        agg_func=args.agg_func,
        max_features=args.max_features,
    )

    print("\nProcessing Summary:")
    print(f"  Fine sequences: {info['n_fine_sequences']:,}")
    print(f"  Coarse sequences: {info['n_coarse_sequences']:,}")
    print(f"  Mapping valid ratio: {info['mapping_valid_ratio']:.2%}")
