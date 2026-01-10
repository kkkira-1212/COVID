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


def load_psm_data(data_dir, split='train'):
    """Load PSM data from CSV files.
    
    Args:
        data_dir: Path to PSM data directory (should contain train.csv, test.csv, test_label.csv)
        split: 'train' or 'test'
    
    Returns:
        df: DataFrame with features and timestamp
        feature_cols: List of feature column names
    """
    data_dir = Path(data_dir)
    
    if split == 'train':
        df = pd.read_csv(data_dir / 'train.csv')
        df['outbreak_label'] = 0  # PSM train data is all normal
    elif split == 'test':
        df = pd.read_csv(data_dir / 'test.csv')
        df_label = pd.read_csv(data_dir / 'test_label.csv')
        # Merge labels based on timestamp
        df = df.merge(df_label, on='timestamp_(min)', how='left')
        df['outbreak_label'] = df['label'].fillna(0).astype(int)
        df = df.drop(columns=['label'])
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split}")
    
    # Create Date from timestamp (assuming timestamp is minutes from start)
    # Use a base date and add minutes
    base_date = pd.Timestamp('2020-01-01 00:00:00')
    df['Date'] = base_date + pd.to_timedelta(df['timestamp_(min)'], unit='m')
    # Use timestamp as a numeric identifier
    df['State'] = 'PSM'
    
    # Extract feature columns (all columns except timestamp, Date, State, outbreak_label)
    feature_cols = [col for col in df.columns 
                    if col not in ['timestamp_(min)', 'Date', 'State', 'outbreak_label']]
    
    # Ensure all features are numeric
    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill().fillna(0)
    
    # Sort by timestamp
    df = df.sort_values('timestamp_(min)').reset_index(drop=True)
    
    return df, feature_cols


def create_sequences_with_mapping(state_data, feature_cols, window_size, stride=1, 
                                  target_time_col='timestamp_(min)'):
    """Create sequences with proper time tracking.
    
    Args:
        state_data: Dict mapping state name to DataFrame
        feature_cols: List of feature column names
        window_size: Window size for sequences
        stride: Stride for sequence creation
        target_time_col: Column name for target time (for mapping)
    
    Returns:
        sequences: List of sequence dictionaries
        states_order: List of state names in order
    """
    sequences = []
    states_order = sorted(state_data.keys())
    state_to_id = {s: i for i, s in enumerate(states_order)}
    
    for state in states_order:
        df_state = state_data[state]
        state_id = state_to_id[state]
        
        X = df_state[feature_cols].values
        y = df_state["outbreak_label"].values if "outbreak_label" in df_state.columns else np.zeros(len(df_state))
        
        # Use timestamp as time identifier
        if target_time_col in df_state.columns:
            timestamps = df_state[target_time_col].values
        else:
            timestamps = np.arange(len(df_state))
        
        # Use Date if available, otherwise use timestamp
        if 'Date' in df_state.columns:
            dates = pd.to_datetime(df_state['Date'].values)
        else:
            # Create dummy dates from timestamps
            base_date = pd.Timestamp('1970-01-01')
            dates = [base_date + pd.Timedelta(minutes=float(ts)) for ts in timestamps]
        
        base_timestamp = timestamps[0] if len(timestamps) > 0 else 0
        T = len(df_state)
        
        r = df_state["target_return"].values if "target_return" in df_state.columns else np.zeros(len(df_state))
        
        for t in range(window_size, T, stride):
            X_win = X[t-window_size:t]
            timestamps_win = timestamps[t-window_size:t]
            dates_win = dates[t-window_size:t]
            
            # Use timestamp difference as time_id for mapping
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
                "target_timestamp": float(timestamps[t]),  # Store original timestamp for mapping
                "target_date": dates[t],
            })
    
    return sequences, states_order


def create_coarse_sequences_from_fine(fine_sequences, k, agg_func='mean'):
    """Create coarse sequences by aggregating fine sequences.
    
    This is a helper that creates coarse sequences directly from fine sequences
    for compatibility with the existing code structure.
    
    Args:
        fine_sequences: List of fine sequence dictionaries
        k: Aggregation factor (k fine steps -> 1 coarse step)
        agg_func: Aggregation function ('mean', 'sum', 'max', 'min')
    
    Returns:
        coarse_sequences: List of coarse sequence dictionaries
    """
    if len(fine_sequences) == 0:
        return []
    
    n_fine = len(fine_sequences)
    coarse_sequences = []
    
    # Process in groups of k
    for coarse_idx in range(0, n_fine, k):
        fine_group = fine_sequences[coarse_idx:coarse_idx + k]
        
        if len(fine_group) == 0:
            continue
        
        last_seq = fine_group[-1]
        
        # Aggregate X_next across the group
        X_next_group = np.array([s['X_next'] for s in fine_group])
        
        if agg_func == 'mean':
            X_next_agg = X_next_group.mean(axis=0)
        elif agg_func == 'sum':
            X_next_agg = X_next_group.sum(axis=0)
        elif agg_func == 'max':
            X_next_agg = X_next_group.max(axis=0)
        elif agg_func == 'min':
            X_next_agg = X_next_group.min(axis=0)
        else:
            X_next_agg = X_next_group.mean(axis=0)
        
        # For X_seq, use the last sequence's window (representative of the coarse time point)
        # This maintains temporal consistency
        X_seq_agg = last_seq['X_seq'].copy()
        
        # Aggregate labels: use max (any anomaly in group -> anomaly)
        y_agg = max([s['y_next'] for s in fine_group])
        
        # Aggregate return: use mean
        r_agg = np.mean([s['return_next'] for s in fine_group])
        
        # Use the last timestamp in the group as the coarse timestamp
        target_timestamp = fine_group[-1]['target_timestamp']
        target_date = fine_group[-1]['target_date']
        
        # Time IDs: use the last sequence's time IDs
        seq_time_ids = last_seq['seq_time_ids'].copy()
        target_time_id = last_seq['target_time_id']
        
        coarse_sequences.append({
            "state": last_seq['state'],
            "state_id": last_seq['state_id'],
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


def create_fine_to_coarse_mapping(fine_sequences, coarse_sequences, k):
    """Create mapping from fine sequence indices to coarse sequence indices.
    
    Mapping rule: coarse_t ↔ fine_[t*k : (t+1)*k]
    
    This means:
    - coarse_0 corresponds to fine_[0 : k]
    - coarse_1 corresponds to fine_[k : 2*k]
    - coarse_t corresponds to fine_[t*k : (t+1)*k]
    
    Args:
        fine_sequences: List of fine sequence dictionaries
        coarse_sequences: List of coarse sequence dictionaries
        k: Aggregation factor (k fine steps -> 1 coarse step)
    
    Returns:
        mapping: torch.Tensor of shape (len(fine_sequences),) with coarse indices
                 mapping[i] = coarse index that corresponds to fine sequence i
                 -1 if no mapping exists
    """
    N_fine = len(fine_sequences)
    N_coarse = len(coarse_sequences)
    mapping = torch.full((N_fine,), -1, dtype=torch.long)
    
    # Direct rule-based mapping: fine_idx -> coarse_idx = floor(fine_idx / k)
    for fine_idx in range(N_fine):
        coarse_idx = fine_idx // k
        
        # Check if this coarse index exists
        if coarse_idx < N_coarse:
            mapping[fine_idx] = coarse_idx
        else:
            # Fine sequence beyond available coarse sequences
            # Map to last coarse sequence if it exists
            if N_coarse > 0:
                mapping[fine_idx] = N_coarse - 1
    
    return mapping


def process_psm_data(
    data_dir='data/PSM/PSM',
    output_dir='data/PSM/processed',
    window_fine=60,
    window_coarse=12,
    k=5,  # Aggregation factor: k fine steps -> 1 coarse step
    train_ratio=0.6,
    val_ratio=0.2,
    agg_func='mean',
    feature_cols=None,
    max_features=None
):
    """Process PSM data into fine and coarse scales with explicit mapping.
    
    Args:
        data_dir: Directory containing PSM CSV files
        output_dir: Output directory for processed data
        window_fine: Window size for fine sequences
        window_coarse: Window size for coarse sequences
        k: Aggregation factor (k fine steps aggregate to 1 coarse step)
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        agg_func: Aggregation function for coarse scale ('mean', 'sum', 'max', 'min')
        feature_cols: Specific features to use (None = use all)
        max_features: Maximum number of features to use (None = use all)
    
    Returns:
        bundle_fine: Fine scale data bundle
        bundle_coarse: Coarse scale data bundle
        info: Dictionary with processing information
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train and test data
    print("Loading PSM data...")
    df_train, feature_cols_all = load_psm_data(data_dir, split='train')
    df_test, _ = load_psm_data(data_dir, split='test')
    
    # Combine train and test for processing (we'll split later)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.sort_values('timestamp_(min)').reset_index(drop=True)
    
    # Select features
    if feature_cols is None:
        if max_features is not None and max_features < len(feature_cols_all):
            # Select first max_features features (or could use feature selection)
            use_features = feature_cols_all[:max_features]
        else:
            use_features = feature_cols_all
    else:
        use_features = [f for f in feature_cols if f in feature_cols_all]
        if len(use_features) == 0:
            raise ValueError(f"None of the specified features found. Available: {feature_cols_all[:10]}")
    
    print(f"Using {len(use_features)} features")
    
    # Create target_return for regression (percentage change)
    # Use first feature as target for simplicity
    target_feature = use_features[0]
    df['target_return'] = df[target_feature].pct_change().fillna(0)
    
    # Prepare fine scale data
    fine_data = {'PSM': df.copy()}
    
    # Create fine sequences
    print("Creating fine sequences...")
    fine_sequences, fine_states = create_sequences_with_mapping(
        fine_data, use_features, window_fine, stride=1, target_time_col='timestamp_(min)'
    )
    print(f"Created {len(fine_sequences)} fine sequences")
    
    # Create coarse sequences by aggregating fine sequences
    print(f"Creating coarse sequences (k={k})...")
    coarse_sequences = create_coarse_sequences_from_fine(fine_sequences, k, agg_func=agg_func)
    print(f"Created {len(coarse_sequences)} coarse sequences")
    
    # Adjust window_coarse if needed (coarse sequences already have aggregated windows)
    # We need to ensure coarse sequences have the correct window size
    # For now, we'll use the same window_fine but aggregated
    
    # Split sequences (using temporal split)
    def split_sequences_temporal(seqs, train_ratio=0.6, val_ratio=0.2):
        dates = pd.to_datetime([s['target_date'] for s in seqs])
        order = dates.argsort()
        
        N = len(seqs)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        
        idx_train = order[:n_train]
        idx_val = order[n_train:n_train+n_val]
        idx_test = order[n_train+n_val:]
        
        return (
            torch.tensor(idx_train.tolist(), dtype=torch.long),
            torch.tensor(idx_val.tolist(), dtype=torch.long),
            torch.tensor(idx_test.tolist(), dtype=torch.long)
        )
    
    idx_tr_fine, idx_v_fine, idx_te_fine = split_sequences_temporal(fine_sequences, train_ratio, val_ratio)
    idx_tr_coarse, idx_v_coarse, idx_te_coarse = split_sequences_temporal(coarse_sequences, train_ratio, val_ratio)
    
    # Create bundles
    print("Creating bundles...")
    bundle_fine = sequences_to_bundle(
        fine_sequences, idx_tr_fine, idx_v_fine, idx_te_fine,
        fine_states, use_features, window_fine
    )
    
    bundle_coarse = sequences_to_bundle(
        coarse_sequences, idx_tr_coarse, idx_v_coarse, idx_te_coarse,
        fine_states, use_features, window_fine  # Window size is same but data is aggregated
    )
    
    # Standardize
    print("Standardizing...")
    bundle_fine = standardize(bundle_fine)
    bundle_coarse = standardize(bundle_coarse)
    
    # Create explicit mapping: fine_to_coarse_index
    print("Creating fine-to-coarse mapping...")
    mapping = create_fine_to_coarse_mapping(fine_sequences, coarse_sequences, k)
    bundle_fine['fine_to_coarse_index'] = mapping
    
    # Verify mapping
    n_valid = (mapping >= 0).sum().item()
    print(f"Mapping created: {n_valid}/{len(mapping)} fine sequences mapped to coarse sequences")
    
    # Save
    print(f"Saving to {output_dir}...")
    torch.save(bundle_fine, output_dir / 'psm_fine.pt')
    torch.save(bundle_coarse, output_dir / 'psm_coarse.pt')
    
    # Save info
    info = {
        'dataset': 'PSM',
        'target_feature': target_feature,
        'fine_shape': list(bundle_fine['X_seq'].shape),
        'coarse_shape': list(bundle_coarse['X_seq'].shape),
        'features': use_features,
        'n_features': len(use_features),
        'window_fine': window_fine,
        'window_coarse': window_coarse,
        'k': k,
        'agg_func': agg_func,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'n_fine_sequences': len(fine_sequences),
        'n_coarse_sequences': len(coarse_sequences),
        'n_train_fine': len(idx_tr_fine),
        'n_val_fine': len(idx_v_fine),
        'n_test_fine': len(idx_te_fine),
        'n_train_coarse': len(idx_tr_coarse),
        'n_val_coarse': len(idx_v_coarse),
        'n_test_coarse': len(idx_te_coarse),
        'mapping_valid_ratio': float(n_valid / len(mapping)),
    }
    
    with open(output_dir / 'psm_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("Processing complete!")
    return bundle_fine, bundle_coarse, info


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PSM data into fine/coarse scales')
    parser.add_argument('--data_dir', type=str, default='data/PSM/PSM')
    parser.add_argument('--output_dir', type=str, default='data/PSM/processed')
    parser.add_argument('--window_fine', type=int, default=60, help='Window size for fine sequences')
    parser.add_argument('--window_coarse', type=int, default=12, help='Window size for coarse sequences')
    parser.add_argument('--k', type=int, default=5, help='Aggregation factor (k fine steps -> 1 coarse step)')
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--agg_func', type=str, default='mean', choices=['mean', 'sum', 'max', 'min'])
    parser.add_argument('--max_features', type=int, default=None, help='Maximum number of features to use')
    
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
        max_features=args.max_features
    )
    
    print("\nProcessing Summary:")
    print(f"  Fine sequences: {info['n_fine_sequences']}")
    print(f"  Coarse sequences: {info['n_coarse_sequences']}")
    print(f"  Aggregation factor k: {info['k']}")
    print(f"  Mapping valid ratio: {info['mapping_valid_ratio']:.2%}")

