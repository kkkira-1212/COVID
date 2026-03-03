import numpy as np
import pandas as pd
import torch

def create_sequences_with_mapping(
    state_data,
    feature_cols,
    window_size,
    stride=1,
    time_col="timestamp_(min)",
    date_col="Date",
    label_col="outbreak_label",
    return_col="target_return",
    base_date="1970-01-01",
    time_unit="m",
):
    sequences = []
    states_order = sorted(state_data.keys())
    state_to_id = {s: i for i, s in enumerate(states_order)}
    base_date = pd.Timestamp(base_date)

    for state in states_order:
        df_state = state_data[state]
        state_id = state_to_id[state]

        X = df_state[feature_cols].values
        y = df_state[label_col].values if label_col in df_state.columns else np.zeros(len(df_state))

        if time_col in df_state.columns:
            timestamps = df_state[time_col].values
        else:
            timestamps = np.arange(len(df_state))

        if date_col in df_state.columns:
            dates = pd.to_datetime(df_state[date_col].values)
        else:
            dates = [base_date + pd.to_timedelta(float(ts), unit=time_unit) for ts in timestamps]

        base_timestamp = timestamps[0] if len(timestamps) > 0 else 0
        T = len(df_state)

        r = df_state[return_col].values if return_col in df_state.columns else np.zeros(len(df_state))

        for t in range(window_size, T, stride):
            X_win = X[t - window_size:t]
            timestamps_win = timestamps[t - window_size:t]

            time_ids = (timestamps_win - base_timestamp).astype(int)
            target_time_id = int(timestamps[t] - base_timestamp)

            sequences.append({
                "state": state,
                "state_id": state_id,
                "X_seq": X_win,
                "X_next": X[t],
                "seq_time_ids": time_ids.astype(int),
                "y_next": int(y[t]),
                "return_next": float(r[t]),
                "target_time_id": int(target_time_id),
                "target_timestamp": float(timestamps[t]),
                "target_date": dates[t],
            })

    return sequences, states_order


def create_sequences_with_dates(
    state_data,
    feature_cols,
    window_size,
    stride=1,
    date_col="Date",
    label_col="outbreak_label",
    return_col="target_return",
    time_unit="m",
):
    unit_seconds = {
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }[time_unit]

    sequences = []
    states_order = sorted(state_data.keys())
    state_to_id = {s: i for i, s in enumerate(states_order)}

    for state in states_order:
        df_state = state_data[state]
        state_id = state_to_id[state]

        X = df_state[feature_cols].values
        y = df_state[label_col].values if label_col in df_state.columns else np.zeros(len(df_state))
        dates = pd.to_datetime(df_state[date_col].values)
        r = df_state[return_col].values if return_col in df_state.columns else np.zeros(len(df_state))

        base_date = dates[0]
        T = len(df_state)

        for t in range(window_size, T, stride):
            X_win = X[t - window_size:t]
            dates_win = dates[t - window_size:t]

            time_ids = ((dates_win - base_date).total_seconds() / unit_seconds).astype(int)
            target_time_id = int((dates[t] - base_date).total_seconds() / unit_seconds)

            sequences.append({
                "state": state,
                "state_id": state_id,
                "X_seq": X_win,
                "X_next": X[t],
                "seq_time_ids": time_ids,
                "y_next": int(y[t]),
                "return_next": float(r[t]),
                "target_time_id": target_time_id,
                "target_date": dates[t],
            })

    return sequences, states_order


def create_coarse_sequences_from_fine(fine_sequences, k, agg_func="mean"):
    if len(fine_sequences) == 0:
        return []

    n_fine = len(fine_sequences)
    coarse_sequences = []

    for coarse_idx in range(0, n_fine, k):
        fine_group = fine_sequences[coarse_idx:coarse_idx + k]
        if len(fine_group) == 0:
            continue

        last_seq = fine_group[-1]
        X_next_group = np.array([s["X_next"] for s in fine_group])

        if agg_func == "mean":
            X_next_agg = X_next_group.mean(axis=0)
        elif agg_func == "sum":
            X_next_agg = X_next_group.sum(axis=0)
        elif agg_func == "max":
            X_next_agg = X_next_group.max(axis=0)
        elif agg_func == "min":
            X_next_agg = X_next_group.min(axis=0)
        else:
            X_next_agg = X_next_group.mean(axis=0)

        X_seq_agg = last_seq["X_seq"].copy()
        y_agg = max([s["y_next"] for s in fine_group])
        r_agg = np.mean([s["return_next"] for s in fine_group])
        target_timestamp = fine_group[-1]["target_timestamp"]
        target_date = fine_group[-1]["target_date"]
        seq_time_ids = last_seq["seq_time_ids"].copy()
        target_time_id = last_seq["target_time_id"]

        coarse_sequences.append({
            "state": last_seq["state"],
            "state_id": last_seq["state_id"],
            "X_seq": X_seq_agg,
            "X_next": X_next_agg,
            "seq_time_ids": seq_time_ids,
            "y_next": int(y_agg),
            "return_next": float(r_agg),
            "target_time_id": int(target_time_id),
            "target_timestamp": float(target_timestamp),
            "target_date": target_date,
        })

    return coarse_sequences


def create_coarse_sequences_and_mapping(fine_sequences, states_order, k, agg_func="mean"):
    fine_indices_by_state = {state_id: [] for state_id in range(len(states_order))}
    for idx, seq in enumerate(fine_sequences):
        fine_indices_by_state[seq["state_id"]].append(idx)

    coarse_sequences = []
    mapping = torch.full((len(fine_sequences),), -1, dtype=torch.long)

    for state_id in range(len(states_order)):
        fine_indices = fine_indices_by_state.get(state_id, [])
        if not fine_indices:
            continue

        fine_seqs_state = [fine_sequences[i] for i in fine_indices]
        coarse_start = len(coarse_sequences)
        coarse_state = create_coarse_sequences_from_fine(fine_seqs_state, k, agg_func=agg_func)
        coarse_sequences.extend(coarse_state)

        n_coarse_state = len(coarse_state)
        if n_coarse_state == 0:
            continue

        for local_idx, fine_idx in enumerate(fine_indices):
            coarse_local = min(local_idx // k, n_coarse_state - 1)
            mapping[fine_idx] = coarse_start + coarse_local

    return coarse_sequences, mapping


def create_fine_to_coarse_mapping(fine_sequences, coarse_sequences, k):
    N_fine = len(fine_sequences)
    N_coarse = len(coarse_sequences)
    mapping = torch.full((N_fine,), -1, dtype=torch.long)

    for fine_idx in range(N_fine):
        coarse_idx = fine_idx // k
        if coarse_idx < N_coarse:
            mapping[fine_idx] = coarse_idx
        else:
            if N_coarse > 0:
                mapping[fine_idx] = N_coarse - 1

    return mapping


def split_sequences_preserve_test(
    fine_sequences_train,
    fine_sequences_test,
    coarse_sequences_train,
    coarse_sequences_test,
    train_ratio=0.8,
    val_ratio=0.2,
):
    fine_sequences_all = fine_sequences_train + fine_sequences_test
    coarse_sequences_all = coarse_sequences_train + coarse_sequences_test

    n_train_fine = len(fine_sequences_train)
    n_train_coarse = len(coarse_sequences_train)

    n_train_split = int(n_train_fine * train_ratio)
    n_val_split = int(n_train_fine * val_ratio)

    idx_tr_fine = torch.arange(0, n_train_split, dtype=torch.long)
    idx_v_fine = torch.arange(n_train_split, n_train_split + n_val_split, dtype=torch.long)
    idx_te_fine = torch.arange(n_train_fine, len(fine_sequences_all), dtype=torch.long)

    n_train_coarse_split = int(n_train_coarse * train_ratio)
    n_val_coarse_split = int(n_train_coarse * val_ratio)

    idx_tr_coarse = torch.arange(0, n_train_coarse_split, dtype=torch.long)
    idx_v_coarse = torch.arange(n_train_coarse_split, n_train_coarse_split + n_val_coarse_split, dtype=torch.long)
    idx_te_coarse = torch.arange(n_train_coarse, len(coarse_sequences_all), dtype=torch.long)

    return (
        idx_tr_fine,
        idx_v_fine,
        idx_te_fine,
        idx_tr_coarse,
        idx_v_coarse,
        idx_te_coarse,
        fine_sequences_all,
        coarse_sequences_all,
    )


def split_train_val_test(train_seqs, test_seqs, train_ratio=0.8, val_ratio=0.2):
    train_count = len(train_seqs)
    train_split = int(train_count * train_ratio)
    val_split = int(train_count * val_ratio)
    idx_train = torch.arange(0, train_split, dtype=torch.long)
    idx_val = torch.arange(train_split, train_split + val_split, dtype=torch.long)
    idx_test = torch.arange(train_count, train_count + len(test_seqs), dtype=torch.long)
    return idx_train, idx_val, idx_test


def split_sequences_anomaly_detection(seqs, train_ratio=0.6, val_ratio=0.2):
    dates = pd.to_datetime([s["target_date"] for s in seqs])
    order = dates.argsort()

    attack_indices = [i for i, s in enumerate(seqs) if s["y_next"] == 1]
    normal_indices = [i for i, s in enumerate(seqs) if s["y_next"] == 0]

    normal_order = [i for i in order if i in normal_indices]
    attack_order = [i for i in order if i in attack_indices]

    n_normal = len(normal_order)
    n_attack = len(attack_order)
    n_train_normal = int(n_normal * train_ratio)
    n_val_normal = int(n_normal * val_ratio)
    n_train_attack = int(n_attack * train_ratio)
    n_val_attack = int(n_attack * val_ratio)

    idx_train = normal_order[:n_train_normal] + attack_order[:n_train_attack]
    idx_val = normal_order[n_train_normal:n_train_normal + n_val_normal] + attack_order[n_train_attack:n_train_attack + n_val_attack]
    idx_test = normal_order[n_train_normal + n_val_normal:] + attack_order[n_train_attack + n_val_attack:]

    idx_train_sorted = sorted(idx_train, key=lambda x: dates[x])
    idx_val_sorted = sorted(idx_val, key=lambda x: dates[x])
    idx_test_sorted = sorted(idx_test, key=lambda x: dates[x])

    return (
        torch.tensor(idx_train_sorted, dtype=torch.long),
        torch.tensor(idx_val_sorted, dtype=torch.long),
        torch.tensor(idx_test_sorted, dtype=torch.long),
    )


def sequences_to_bundle(seqs, idx_tr, idx_v, idx_te, states, feats, win, return_key="return_next"):
    N = len(seqs)
    D = len(feats)
    X = np.zeros((N, win, D), np.float32)
    Xn = np.zeros((N, D), np.float32)
    sid = np.zeros(N, np.int64)
    tids = np.zeros((N, win), np.int64)
    tidt = np.zeros(N, np.int64)
    y = np.zeros(N, np.float32)
    r = np.zeros(N, np.float32)
    meta = []
    for i, s in enumerate(seqs):
        X[i] = s["X_seq"]
        Xn[i] = s["X_next"]
        sid[i] = s["state_id"]
        tids[i] = s["seq_time_ids"]
        tidt[i] = s["target_time_id"]
        y[i] = s["y_next"]
        r[i] = s["return_next"]
        meta.append({
            "index": i,
            "state": s["state"],
            "target_date": str(s["target_date"]),
            "y_next": s["y_next"],
        })
    return {
        "X_seq": torch.tensor(X, dtype=torch.float32),
        "X_next": torch.tensor(Xn, dtype=torch.float32),
        "state_ids": torch.tensor(sid),
        "seq_time_ids": torch.tensor(tids),
        "target_time_ids": torch.tensor(tidt),
        "y_next": torch.tensor(y),
        return_key: torch.tensor(r, dtype=torch.float32),
        "idx_train": idx_tr,
        "idx_val": idx_v,
        "idx_test": idx_te,
        "meta": pd.DataFrame(meta),
        "states_order": states,
        "feature_cols": feats,
        "window_size": win,
    }


def standardize(bundle):
    X = bundle["X_seq"].numpy()
    idx = bundle["idx_train"].numpy()
    N, T, D = X.shape

    if len(idx) == 0:
        raise ValueError("Training indices are empty. Cannot compute standardization statistics.")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xtr = X[idx].reshape(-1, D)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-8] = 1.0

    X = (X - mean.reshape(1, 1, D)) / std.reshape(1, 1, D)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    bundle["X_seq"] = torch.tensor(X, dtype=torch.float32)

    if "X_next" in bundle:
        Xn = bundle["X_next"].numpy()
        if Xn.shape != (N, D):
            raise ValueError(f"X_next shape mismatch: expected ({N}, {D}), got {Xn.shape}")
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        Xn = (Xn - mean.reshape(1, D)) / std.reshape(1, D)
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        bundle["X_next"] = torch.tensor(Xn, dtype=torch.float32)

    bundle["feature_means"] = mean
    bundle["feature_stds"] = std
    return bundle


def apply_multi_scale_subset_sampling(bundle_fine, bundle_coarse, max_samples, stratified_test=True):
    n_total_coarse = bundle_coarse["X_seq"].shape[0]
    idx_tr_coarse_orig = bundle_coarse["idx_train"].numpy()
    idx_val_coarse_orig = bundle_coarse["idx_val"].numpy()
    idx_test_coarse_orig = bundle_coarse["idx_test"].numpy()

    train_ratio = len(idx_tr_coarse_orig) / n_total_coarse if n_total_coarse > 0 else 0.0
    val_ratio = len(idx_val_coarse_orig) / n_total_coarse if n_total_coarse > 0 else 0.0

    n_train_coarse_sample = int(max_samples * train_ratio)
    n_val_coarse_sample = int(max_samples * val_ratio)
    n_test_coarse_sample = max_samples - n_train_coarse_sample - n_val_coarse_sample

    idx_tr_coarse_sample = idx_tr_coarse_orig[:min(n_train_coarse_sample, len(idx_tr_coarse_orig))]
    idx_val_coarse_sample = idx_val_coarse_orig[:min(n_val_coarse_sample, len(idx_val_coarse_orig))]
    if stratified_test and bundle_coarse.get("y_next") is not None and n_test_coarse_sample > 0:
        y_test_coarse = bundle_coarse["y_next"][idx_test_coarse_orig].numpy().astype(int)
        anomaly_idx = idx_test_coarse_orig[y_test_coarse == 1]
        normal_idx = idx_test_coarse_orig[y_test_coarse == 0]

        ratio = y_test_coarse.mean() if len(y_test_coarse) > 0 else 0.0
        n_anom = int(round(n_test_coarse_sample * ratio))
        n_norm = n_test_coarse_sample - n_anom

        anomaly_sel = anomaly_idx[:min(n_anom, len(anomaly_idx))]
        normal_sel = normal_idx[:min(n_norm, len(normal_idx))]
        idx_test_coarse_sample = np.concatenate([anomaly_sel, normal_sel])
        idx_test_coarse_sample = np.sort(idx_test_coarse_sample)
    else:
        idx_test_coarse_sample = idx_test_coarse_orig[:min(n_test_coarse_sample, len(idx_test_coarse_orig))]

    selected_coarse_indices = np.concatenate([idx_tr_coarse_sample, idx_val_coarse_sample, idx_test_coarse_sample])
    selected_coarse_indices = np.sort(selected_coarse_indices)

    coarse_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_coarse_indices)}

    bundle_coarse["X_seq"] = bundle_coarse["X_seq"][selected_coarse_indices]
    bundle_coarse["X_next"] = bundle_coarse["X_next"][selected_coarse_indices]
    if bundle_coarse.get("y_next") is not None:
        bundle_coarse["y_next"] = bundle_coarse["y_next"][selected_coarse_indices]

    idx_train_coarse_new = torch.tensor(
        [coarse_old_to_new[idx] for idx in idx_tr_coarse_sample if idx in coarse_old_to_new],
        dtype=torch.long,
    )
    idx_val_coarse_new = torch.tensor(
        [coarse_old_to_new[idx] for idx in idx_val_coarse_sample if idx in coarse_old_to_new],
        dtype=torch.long,
    )
    idx_test_coarse_new = torch.tensor(
        [coarse_old_to_new[idx] for idx in idx_test_coarse_sample if idx in coarse_old_to_new],
        dtype=torch.long,
    )

    bundle_coarse["idx_train"] = idx_train_coarse_new
    bundle_coarse["idx_val"] = idx_val_coarse_new
    bundle_coarse["idx_test"] = idx_test_coarse_new

    if "fine_to_coarse_index" not in bundle_fine:
        raise ValueError("fine_to_coarse_index mapping required for multi-scale sampling")

    mapping = bundle_fine["fine_to_coarse_index"]
    if isinstance(mapping, torch.Tensor):
        mapping = mapping.numpy()

    selected_coarse_set = set(selected_coarse_indices)
    fine_mask = np.array([mapping[i] in selected_coarse_set for i in range(len(mapping))])
    selected_fine_indices = np.where(fine_mask)[0]

    fine_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_fine_indices)}

    bundle_fine["X_seq"] = bundle_fine["X_seq"][selected_fine_indices]
    bundle_fine["X_next"] = bundle_fine["X_next"][selected_fine_indices]
    if bundle_fine.get("y_next") is not None:
        bundle_fine["y_next"] = bundle_fine["y_next"][selected_fine_indices]

    mapping_new = np.array([coarse_old_to_new.get(mapping[old_idx], -1) for old_idx in selected_fine_indices])
    bundle_fine["fine_to_coarse_index"] = torch.tensor(mapping_new, dtype=torch.long)

    idx_tr_fine_orig = bundle_fine["idx_train"].numpy() if "idx_train" in bundle_fine else np.array([])
    idx_val_fine_orig = bundle_fine["idx_val"].numpy() if "idx_val" in bundle_fine else np.array([])
    idx_test_fine_orig = bundle_fine["idx_test"].numpy() if "idx_test" in bundle_fine else np.array([])

    selected_fine_set = set(selected_fine_indices)
    idx_tr_fine_filtered = [idx for idx in idx_tr_fine_orig if idx in selected_fine_set]
    idx_val_fine_filtered = [idx for idx in idx_val_fine_orig if idx in selected_fine_set]
    idx_test_fine_filtered = [idx for idx in idx_test_fine_orig if idx in selected_fine_set]

    idx_train_fine_new = torch.tensor(
        [fine_old_to_new[idx] for idx in idx_tr_fine_filtered if idx in fine_old_to_new],
        dtype=torch.long,
    )
    idx_val_fine_new = torch.tensor(
        [fine_old_to_new[idx] for idx in idx_val_fine_filtered if idx in fine_old_to_new],
        dtype=torch.long,
    )
    idx_test_fine_new = torch.tensor(
        [fine_old_to_new[idx] for idx in idx_test_fine_filtered if idx in fine_old_to_new],
        dtype=torch.long,
    )

    bundle_fine["idx_train"] = idx_train_fine_new
    bundle_fine["idx_val"] = idx_val_fine_new
    bundle_fine["idx_test"] = idx_test_fine_new

    if "meta" in bundle_coarse and hasattr(bundle_coarse["meta"], "iloc"):
        bundle_coarse["meta"] = bundle_coarse["meta"].iloc[selected_coarse_indices].reset_index(drop=True)
    if "meta" in bundle_fine and hasattr(bundle_fine["meta"], "iloc"):
        bundle_fine["meta"] = bundle_fine["meta"].iloc[selected_fine_indices].reset_index(drop=True)

    print(f"  Coarse: Reduced from {n_total_coarse} to {len(selected_coarse_indices)} samples")
    print(f"    Train: {len(idx_train_coarse_new)}, Val: {len(idx_val_coarse_new)}, Test: {len(idx_test_coarse_new)}")
    print(f"  Fine: Reduced to {len(selected_fine_indices)} samples")

    return bundle_fine, bundle_coarse


def apply_test_subset_sampling(bundle_fine, bundle_coarse, max_test_samples, stratified=True):
    idx_test_coarse_orig = bundle_coarse["idx_test"].numpy()
    if len(idx_test_coarse_orig) <= max_test_samples:
        print("  Test subset: skipped (already within limit)")
        return bundle_fine, bundle_coarse

    if stratified and bundle_coarse.get("y_next") is not None:
        y_test_coarse = bundle_coarse["y_next"][idx_test_coarse_orig].numpy().astype(int)
        anomaly_idx = idx_test_coarse_orig[y_test_coarse == 1]
        normal_idx = idx_test_coarse_orig[y_test_coarse == 0]

        ratio = y_test_coarse.mean() if len(y_test_coarse) > 0 else 0.0
        n_anom = int(round(max_test_samples * ratio))
        n_norm = max_test_samples - n_anom

        anomaly_sel = anomaly_idx[:min(n_anom, len(anomaly_idx))]
        normal_sel = normal_idx[:min(n_norm, len(normal_idx))]
        selected_test = np.concatenate([anomaly_sel, normal_sel])
        selected_test = np.sort(selected_test)
    else:
        selected_test = np.sort(idx_test_coarse_orig[:max_test_samples])

    selected_coarse_indices = np.sort(
        np.concatenate([
            bundle_coarse["idx_train"].numpy(),
            bundle_coarse["idx_val"].numpy(),
            selected_test,
        ])
    )

    coarse_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_coarse_indices)}

    bundle_coarse["X_seq"] = bundle_coarse["X_seq"][selected_coarse_indices]
    bundle_coarse["X_next"] = bundle_coarse["X_next"][selected_coarse_indices]
    if bundle_coarse.get("y_next") is not None:
        bundle_coarse["y_next"] = bundle_coarse["y_next"][selected_coarse_indices]

    idx_train_coarse_new = torch.tensor(
        [coarse_old_to_new[idx] for idx in bundle_coarse["idx_train"].numpy() if idx in coarse_old_to_new],
        dtype=torch.long,
    )
    idx_val_coarse_new = torch.tensor(
        [coarse_old_to_new[idx] for idx in bundle_coarse["idx_val"].numpy() if idx in coarse_old_to_new],
        dtype=torch.long,
    )
    idx_test_coarse_new = torch.tensor(
        [coarse_old_to_new[idx] for idx in selected_test if idx in coarse_old_to_new],
        dtype=torch.long,
    )

    bundle_coarse["idx_train"] = idx_train_coarse_new
    bundle_coarse["idx_val"] = idx_val_coarse_new
    bundle_coarse["idx_test"] = idx_test_coarse_new

    mapping = bundle_fine["fine_to_coarse_index"]
    if isinstance(mapping, torch.Tensor):
        mapping = mapping.numpy()

    selected_coarse_set = set(selected_coarse_indices)
    fine_mask = np.array([mapping[i] in selected_coarse_set for i in range(len(mapping))])
    selected_fine_indices = np.where(fine_mask)[0]

    fine_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_fine_indices)}

    bundle_fine["X_seq"] = bundle_fine["X_seq"][selected_fine_indices]
    bundle_fine["X_next"] = bundle_fine["X_next"][selected_fine_indices]
    if bundle_fine.get("y_next") is not None:
        bundle_fine["y_next"] = bundle_fine["y_next"][selected_fine_indices]

    mapping_new = np.array([coarse_old_to_new.get(mapping[old_idx], -1) for old_idx in selected_fine_indices])
    bundle_fine["fine_to_coarse_index"] = torch.tensor(mapping_new, dtype=torch.long)

    idx_tr_fine_orig = bundle_fine["idx_train"].numpy() if "idx_train" in bundle_fine else np.array([])
    idx_val_fine_orig = bundle_fine["idx_val"].numpy() if "idx_val" in bundle_fine else np.array([])
    idx_test_fine_orig = bundle_fine["idx_test"].numpy() if "idx_test" in bundle_fine else np.array([])

    selected_fine_set = set(selected_fine_indices)
    idx_tr_fine_filtered = [idx for idx in idx_tr_fine_orig if idx in selected_fine_set]
    idx_val_fine_filtered = [idx for idx in idx_val_fine_orig if idx in selected_fine_set]
    idx_test_fine_filtered = [idx for idx in idx_test_fine_orig if idx in selected_fine_set]

    idx_train_fine_new = torch.tensor(
        [fine_old_to_new[idx] for idx in idx_tr_fine_filtered if idx in fine_old_to_new],
        dtype=torch.long,
    )
    idx_val_fine_new = torch.tensor(
        [fine_old_to_new[idx] for idx in idx_val_fine_filtered if idx in fine_old_to_new],
        dtype=torch.long,
    )
    idx_test_fine_new = torch.tensor(
        [fine_old_to_new[idx] for idx in idx_test_fine_filtered if idx in fine_old_to_new],
        dtype=torch.long,
    )

    bundle_fine["idx_train"] = idx_train_fine_new
    bundle_fine["idx_val"] = idx_val_fine_new
    bundle_fine["idx_test"] = idx_test_fine_new

    if "meta" in bundle_coarse and hasattr(bundle_coarse["meta"], "iloc"):
        bundle_coarse["meta"] = bundle_coarse["meta"].iloc[selected_coarse_indices].reset_index(drop=True)
    if "meta" in bundle_fine and hasattr(bundle_fine["meta"], "iloc"):
        bundle_fine["meta"] = bundle_fine["meta"].iloc[selected_fine_indices].reset_index(drop=True)

    print(f"  Test subset (coarse): {len(idx_test_coarse_orig)} -> {len(idx_test_coarse_new)}")
    print(f"  Fine kept: {len(selected_fine_indices)}")

    return bundle_fine, bundle_coarse
