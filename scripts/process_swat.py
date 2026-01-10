import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.sequences import create_sequences, split_sequences, sequences_to_bundle, standardize
from utils.mapping import add_mapping


def load_swat_data(data_dir, use_merged=True):
    data_dir = Path(data_dir)
    
    if use_merged and (data_dir / 'merged.csv').exists():
        df = pd.read_csv(data_dir / 'merged.csv')
    else:
        df_normal = pd.read_csv(data_dir / 'normal.csv')
        df_attack = pd.read_csv(data_dir / 'attack.csv')
        df = pd.concat([df_normal, df_attack], ignore_index=True)
        df = df.sort_values(' Timestamp').reset_index(drop=True)
    
    df.columns = df.columns.str.strip()
    
    timestamp_col = 'Timestamp' if 'Timestamp' in df.columns else ' Timestamp'
    df[timestamp_col] = df[timestamp_col].astype(str).str.strip()
    df['Timestamp'] = pd.to_datetime(df[timestamp_col], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
    if df['Timestamp'].isna().any():
        df['Timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    
    df['outbreak_label'] = (df['Normal/Attack'] == 'Attack').astype(int)
    df['State'] = 'SWaT'
    df['Date'] = df['Timestamp']
    df = df.drop(columns=['Normal/Attack', timestamp_col])
    
    feature_cols = [col for col in df.columns 
                    if col not in ['Timestamp', 'State', 'Date', 'outbreak_label']]
    
    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill().fillna(0)
    df = df.sort_values(['State', 'Date']).reset_index(drop=True)
    
    return df, feature_cols


def select_target_feature(df, feature_cols):
    priority_features = [col for col in feature_cols if 'LIT' in col or 'FIT' in col]
    
    if len(priority_features) > 0:
        target_feature = priority_features[0]
    else:
        target_feature = feature_cols[0]
    
    return target_feature


def select_features_by_information(df, feature_cols, target_feature, max_features=30):
    feature_scores = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        try:
            values = df[col].values
            std = np.std(values)
            var = np.var(values)
            range_val = np.max(values) - np.min(values)
            mean_val = np.abs(np.mean(values))
            cv = std / (mean_val + 1e-8)
            
            score = var + range_val * 0.1 + cv * 0.1
            feature_scores[col] = score
        except:
            feature_scores[col] = 0.0
    
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    selected = []
    target_included = False
    
    for feat, score in sorted_features:
        if len(selected) >= max_features:
            break
        if feat == target_feature:
            target_included = True
        selected.append(feat)
    
    if not target_included and target_feature in feature_cols:
        if len(selected) >= max_features:
            selected[-1] = target_feature
        else:
            selected.append(target_feature)
    
    if target_feature not in selected:
        selected.append(target_feature)
    
    return selected


def process_swat_data(
    data_dir='data/SWaT',
    output_dir='data/SWaT/processed',
    window_minute=60,
    window_hour=12,
    train_ratio=0.6,
    val_ratio=0.2,
    use_merged=True,
    train_normal_only=True
):
    data_dir = Path(data_dir)
    df, feature_cols = load_swat_data(data_dir, use_merged=use_merged)
    
    target_feature = select_target_feature(df, feature_cols)
    
    df['target_return'] = df.groupby('State')[target_feature].transform(
        lambda x: x.pct_change().fillna(0)
    )
    
    states = sorted(df['State'].unique())
    
    agg_map = {col: 'mean' for col in feature_cols}
    agg_map['target_return'] = 'mean'
    agg_map['outbreak_label'] = 'max'
    
    minute_data = {}
    hour_data = {}
    
    for s in states:
        state_df = df[df['State'] == s].copy()
        state_df = state_df.set_index('Date')
        
        d_minute = state_df.resample('1min').agg(agg_map).reset_index()
        minute_data[s] = d_minute
        
        d_hour = state_df.resample('1h').agg(agg_map).reset_index()
        d_hour = d_hour.rename(columns={'Date': 'HourStart'})
        hour_data[s] = d_hour
    
    use_features = select_features_by_information(df, feature_cols, target_feature, max_features=30)
    
    def create_sequences_custom(state_data, feature_cols, window_size, date_col='Date'):
        sequences = []
        states_order = sorted(state_data.keys())
        state_to_id = {s: i for i, s in enumerate(states_order)}
        
        for state in states_order:
            df_state = state_data[state]
            state_id = state_to_id[state]
            
            X = df_state[feature_cols].values
            y = df_state["outbreak_label"].values
            dates = pd.to_datetime(df_state[date_col].values)
            r = df_state["target_return"].values if "target_return" in df_state.columns else np.zeros(len(df_state))
            
            base_date = dates[0]
            T = len(df_state)
            
            for t in range(window_size, T):
                X_win = X[t-window_size:t]
                dates_win = dates[t-window_size:t]
                
                if date_col == 'Date':
                    time_ids = ((dates_win - base_date).total_seconds() / 60).astype(int)
                    target_time_id = int((dates[t] - base_date).total_seconds() / 60)
                else:
                    time_ids = ((dates_win - base_date).total_seconds() / 3600).astype(int)
                    target_time_id = int((dates[t] - base_date).total_seconds() / 3600)
                
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
    
    seq_minute, states_minute = create_sequences_custom(
        minute_data, use_features, window_minute, date_col='Date'
    )
    seq_hour, states_hour = create_sequences_custom(
        hour_data, use_features, window_hour, date_col='HourStart'
    )
    
    def split_sequences_anomaly_detection(seqs, train_ratio=0.6, val_ratio=0.2):
        dates = pd.to_datetime([s['target_date'] for s in seqs])
        order = dates.argsort()
        
        attack_indices = [i for i, s in enumerate(seqs) if s['y_next'] == 1]
        normal_indices = [i for i, s in enumerate(seqs) if s['y_next'] == 0]
        
        normal_order = [i for i in order if i in normal_indices]
        attack_order = [i for i in order if i in attack_indices]
        
        n_normal = len(normal_order)
        n_attack = len(attack_order)
        n_train_normal = int(n_normal * train_ratio)
        n_val_normal = int(n_normal * val_ratio)
        
        n_train_attack = int(n_attack * train_ratio)
        n_val_attack = int(n_attack * val_ratio)
        
        idx_train = normal_order[:n_train_normal] + attack_order[:n_train_attack]
        idx_val = normal_order[n_train_normal:n_train_normal+n_val_normal] + attack_order[n_train_attack:n_train_attack+n_val_attack]
        idx_test = normal_order[n_train_normal+n_val_normal:] + attack_order[n_train_attack+n_val_attack:]
        
        idx_train_sorted = sorted(idx_train, key=lambda x: dates[x])
        idx_val_sorted = sorted(idx_val, key=lambda x: dates[x])
        idx_test_sorted = sorted(idx_test, key=lambda x: dates[x])
        
        return (
            torch.tensor(idx_train_sorted, dtype=torch.long),
            torch.tensor(idx_val_sorted, dtype=torch.long),
            torch.tensor(idx_test_sorted, dtype=torch.long)
        )
    
    idx_tr_m, idx_v_m, idx_te_m = split_sequences_anomaly_detection(seq_minute, train_ratio, val_ratio)
    idx_tr_h, idx_v_h, idx_te_h = split_sequences_anomaly_detection(seq_hour, train_ratio, val_ratio)
    
    train_attacks_m = sum([seq_minute[i]['y_next'] for i in idx_tr_m.numpy()])
    val_attacks_m = sum([seq_minute[i]['y_next'] for i in idx_v_m.numpy()])
    test_attacks_m = sum([seq_minute[i]['y_next'] for i in idx_te_m.numpy()])
    
    bundle_minute = sequences_to_bundle(
        seq_minute, idx_tr_m, idx_v_m, idx_te_m,
        states_minute, use_features, window_minute
    )
    bundle_hour = sequences_to_bundle(
        seq_hour, idx_tr_h, idx_v_h, idx_te_h,
        states_hour, use_features, window_hour
    )
    
    bundle_minute = standardize(bundle_minute)
    bundle_hour = standardize(bundle_hour)
    
    if len(seq_hour) > 0:
        bundle_minute = add_mapping(bundle_minute, bundle_hour)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(bundle_minute, output_dir / 'swat_minute.pt')
    if len(seq_hour) > 0:
        torch.save(bundle_hour, output_dir / 'swat_hour.pt')
    
    info = {
        'dataset': 'SWaT',
        'target_feature': target_feature,
        'minute_shape': list(bundle_minute['X_seq'].shape),
        'hour_shape': list(bundle_hour['X_seq'].shape) if len(seq_hour) > 0 else None,
        'features': use_features,
        'window_minute': window_minute,
        'window_hour': window_hour,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'n_train_minute': len(idx_tr_m),
        'n_val_minute': len(idx_v_m),
        'n_test_minute': len(idx_te_m),
        'n_train_hour': len(idx_tr_h),
        'n_val_hour': len(idx_v_h),
        'n_test_hour': len(idx_te_h),
        'train_attacks_minute': int(train_attacks_m),
        'val_attacks_minute': int(val_attacks_m),
        'test_attacks_minute': int(test_attacks_m),
        'split_strategy': 'anomaly_detection'
    }
    
    with open(output_dir / 'swat_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    return bundle_minute, bundle_hour, info


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/SWaT')
    parser.add_argument('--output_dir', type=str, default='data/SWaT/processed')
    parser.add_argument('--window_minute', type=int, default=60)
    parser.add_argument('--window_hour', type=int, default=24)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    
    args = parser.parse_args()
    
    bundle_minute, bundle_hour, info = process_swat_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_minute=args.window_minute,
        window_hour=args.window_hour,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
