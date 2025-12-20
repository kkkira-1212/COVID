import numpy as np
import pandas as pd
import torch

def create_sequences(state_data, feature_cols, window_size=14, stride=1, freq='D'):
    sequences = []
    states_order = sorted(state_data.keys())
    state_to_id = {s: i for i, s in enumerate(states_order)}

    date_col = "Date" if freq == "D" else "WeekStart"

    for state in states_order:
        df = state_data[state]
        state_id = state_to_id[state]

        X = df[feature_cols].values 
        y = df["outbreak_label"].values 
        dates = pd.to_datetime(df[date_col].values)

        r = df["NewDeaths_return"].values if "NewDeaths_return" in df else np.zeros(len(df))

        base_date = dates[0]
        T = len(df)

        for t in range(window_size, T, stride):
            X_win = X[t-window_size:t]
            dates_win = dates[t-window_size:t]

            time_ids = (dates_win - base_date).days
            target_time_id = (dates[t] - base_date).days

            sequences.append({
                "state": state,
                "state_id": state_id,
                "X_seq": X_win,
                "seq_time_ids": time_ids.astype(int),
                "y_next": int(y[t]),
                "return_next": float(r[t]),
                "target_time_id": int(target_time_id),
                "target_date": dates[t],
            })

    return sequences, states_order


def split_sequences(seqs, train_ratio=0.7, val_ratio=0.1):
    dates = pd.to_datetime([s['target_date'] for s in seqs])
    order = dates.argsort()

    N = len(seqs)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    idx_train = order[:n_train]
    idx_val = order[n_train:n_train+n_val]
    idx_test = order[n_train+n_val:]

    return (
        torch.tensor(idx_train, dtype=torch.long),
        torch.tensor(idx_val, dtype=torch.long),
        torch.tensor(idx_test, dtype=torch.long)
    )

def sequences_to_bundle(seqs, idx_tr, idx_v, idx_te, states, feats, win):
    N = len(seqs); D = len(feats)
    X = np.zeros((N, win, D), np.float32)
    sid = np.zeros(N, np.int64)
    tids = np.zeros((N, win), np.int64)
    tidt = np.zeros(N, np.int64)
    y = np.zeros(N, np.float32)
    r = np.zeros(N, np.float32)
    meta = []
    for i, s in enumerate(seqs):
        X[i] = s['X_seq']
        sid[i] = s['state_id']
        tids[i] = s['seq_time_ids']
        tidt[i] = s['target_time_id']
        y[i] = s['y_next']
        r[i] = s['return_next']
        meta.append({
            'index': i,
            'state': s['state'],
            'target_date': str(s['target_date']),
            'y_next': s['y_next'],
        })
    return {
        'X_seq': torch.tensor(X, dtype=torch.float32),
        'state_ids': torch.tensor(sid),
        'seq_time_ids': torch.tensor(tids),
        'target_time_ids': torch.tensor(tidt),
        'y_next': torch.tensor(y),
        'NewDeaths_ret_next': torch.tensor(r, dtype=torch.float32),
        'idx_train': idx_tr,
        'idx_val': idx_v,
        'idx_test': idx_te,
        'meta': pd.DataFrame(meta),
        'states_order': states,
        'feature_cols': feats,
        'window_size': win,
    }

def standardize(bundle):
    X = bundle['X_seq'].numpy()
    idx = bundle['idx_train'].numpy()
    N, T, D = X.shape
    Xtr = X[idx].reshape(-1, D)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-8] = 1.0
    X = (X - mean.reshape(1,1,D)) / std.reshape(1,1,D)
    bundle['X_seq'] = torch.tensor(X, dtype=torch.float32)
    bundle['feature_means'] = mean
    bundle['feature_stds'] = std
    return bundle