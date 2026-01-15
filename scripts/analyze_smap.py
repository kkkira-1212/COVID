"""
Analyze SMAP data to plan fine and coarse dimension definitions.
"""
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_smap_data():
    """Analyze SMAP data structure and propose fine/coarse dimensions."""
    
    # Load SMAP data
    data_dir = project_root / 'data' / 'SMAP'
    train = np.load(data_dir / 'SMAP_train.npy')
    test = np.load(data_dir / 'SMAP_test.npy')
    labels = np.load(data_dir / 'SMAP_test_label.npy')
    
    print("=" * 60)
    print("SMAP Data Analysis for Fine/Coarse Dimension Planning")
    print("=" * 60)
    
    # Basic statistics
    print(f"\n1. Basic Data Statistics:")
    print(f"   Train samples: {train.shape[0]:,}")
    print(f"   Test samples: {test.shape[0]:,}")
    print(f"   Total samples: {train.shape[0] + test.shape[0]:,}")
    print(f"   Features: {train.shape[1]}")
    
    # Label statistics
    print(f"\n2. Label Statistics (Test set only):")
    print(f"   Total test samples: {len(labels):,}")
    print(f"   Normal samples: {(~labels).sum():,} ({(~labels).sum() / len(labels) * 100:.2f}%)")
    print(f"   Anomaly samples: {labels.sum():,} ({labels.sum() / len(labels) * 100:.2f}%)")
    
    # Anomaly pattern analysis
    print(f"\n3. Anomaly Pattern Analysis:")
    anomaly_indices = np.where(labels)[0]
    if len(anomaly_indices) > 0:
        gaps = np.diff(anomaly_indices)
        consecutive_count = np.sum(gaps == 1)
        
        # Anomaly segments
        segments = []
        current_seg = [anomaly_indices[0]]
        for i in range(1, len(anomaly_indices)):
            if anomaly_indices[i] - anomaly_indices[i-1] == 1:
                current_seg.append(anomaly_indices[i])
            else:
                segments.append(current_seg)
                current_seg = [anomaly_indices[i]]
        segments.append(current_seg)
        segment_lengths = [len(s) for s in segments]
        
        print(f"   Number of anomaly segments: {len(segments):,}")
        print(f"   Min segment length: {min(segment_lengths)}")
        print(f"   Max segment length: {max(segment_lengths)}")
        print(f"   Mean segment length: {np.mean(segment_lengths):.1f}")
        print(f"   Median segment length: {np.median(segment_lengths):.1f}")
    
    # Combine train and test for full analysis
    print(f"\n4. Full Dataset Analysis (Train + Test):")
    full_data = np.concatenate([train, test], axis=0)
    # Train data has no labels (all normal), test has labels
    full_labels = np.concatenate([np.zeros(train.shape[0], dtype=bool), labels], axis=0)
    
    total_samples = len(full_data)
    print(f"   Total samples: {total_samples:,}")
    print(f"   Normal samples: {(~full_labels).sum():,} ({(~full_labels).sum() / total_samples * 100:.2f}%)")
    print(f"   Anomaly samples: {full_labels.sum():,} ({full_labels.sum() / total_samples * 100:.2f}%)")
    
    # Propose different window sizes and k values
    print(f"\n5. Proposed Fine/Coarse Dimension Configurations:")
    print(f"\n   Reference (PSM):")
    print(f"   - window_fine: 60")
    print(f"   - window_coarse: 12")
    print(f"   - k (aggregation factor): 5")
    print(f"   - Fine sequences: 220,262")
    print(f"   - Coarse sequences: 44,053 (ratio: {220262/44053:.2f})")
    
    # Test different configurations
    configs = [
        {'window_fine': 60, 'k': 5, 'name': 'Similar to PSM'},
        {'window_fine': 100, 'k': 5, 'name': 'Larger window'},
        {'window_fine': 60, 'k': 10, 'name': 'Larger k (more aggregation)'},
        {'window_fine': 100, 'k': 10, 'name': 'Large window + large k'},
        {'window_fine': 50, 'k': 5, 'name': 'Smaller window'},
    ]
    
    print(f"\n   Configuration Analysis:")
    print(f"   {'Config':<30} {'Fine Seqs':<12} {'Coarse Seqs':<12} {'Ratio':<10} {'Train Fine':<12} {'Test Fine':<12} {'Test Anomaly%':<12}")
    print(f"   {'-' * 100}")
    
    train_ratio = 0.6
    val_ratio = 0.2
    
    for config in configs:
        window_fine = config['window_fine']
        k = config['k']
        name = config['name']
        
        # Calculate sequences
        n_fine = total_samples - window_fine
        n_coarse = n_fine // k
        
        # Split fine sequences
        n_train_fine = int(n_fine * train_ratio)
        n_val_fine = int(n_fine * val_ratio)
        n_test_fine = n_fine - n_train_fine - n_val_fine
        
        # Calculate test anomaly ratio (approximate, considering window)
        # Test set starts after train set
        test_start_idx = train.shape[0] + window_fine
        test_end_idx = total_samples
        test_fine_labels = full_labels[test_start_idx:test_end_idx]
        test_anomaly_ratio = test_fine_labels.sum() / len(test_fine_labels) if len(test_fine_labels) > 0 else 0
        
        ratio = n_fine / n_coarse if n_coarse > 0 else 0
        
        print(f"   {name:<30} {n_fine:<12,} {n_coarse:<12,} {ratio:<10.2f} {n_train_fine:<12,} {n_test_fine:<12,} {test_anomaly_ratio*100:<12.2f}")
    
    # Detailed analysis for recommended configuration
    print(f"\n6. Detailed Analysis for Recommended Configuration:")
    recommended_window_fine = 60
    recommended_k = 5
    
    n_fine = total_samples - recommended_window_fine
    n_coarse = n_fine // recommended_k
    
    print(f"   Window fine: {recommended_window_fine}")
    print(f"   k (aggregation factor): {recommended_k}")
    print(f"   Fine sequences: {n_fine:,}")
    print(f"   Coarse sequences: {n_coarse:,}")
    print(f"   Ratio (fine/coarse): {n_fine/n_coarse:.2f}")
    
    # Split analysis
    n_train_fine = int(n_fine * train_ratio)
    n_val_fine = int(n_fine * val_ratio)
    n_test_fine = n_fine - n_train_fine - n_val_fine
    
    print(f"\n   Fine scale split (ratio {train_ratio:.1f}/{val_ratio:.1f}/{1-train_ratio-val_ratio:.1f}):")
    print(f"     Train: {n_train_fine:,}")
    print(f"     Val: {n_val_fine:,}")
    print(f"     Test: {n_test_fine:,}")
    
    n_train_coarse = int(n_coarse * train_ratio)
    n_val_coarse = int(n_coarse * val_ratio)
    n_test_coarse = n_coarse - n_train_coarse - n_val_coarse
    
    print(f"\n   Coarse scale split:")
    print(f"     Train: {n_train_coarse:,}")
    print(f"     Val: {n_val_coarse:,}")
    print(f"     Test: {n_test_coarse:,}")
    
    # Anomaly distribution in test set
    print(f"\n   Test set anomaly distribution (approximate):")
    test_start = train.shape[0] + recommended_window_fine
    test_labels_fine = full_labels[test_start:]
    test_fine_anomaly_count = test_labels_fine[:n_test_fine].sum()
    test_fine_anomaly_ratio = test_fine_anomaly_count / n_test_fine if n_test_fine > 0 else 0
    
    print(f"     Fine test sequences: {n_test_fine:,}")
    print(f"     Fine test anomalies: {test_fine_anomaly_count:,} ({test_fine_anomaly_ratio*100:.2f}%)")
    
    # Check if anomaly ratio is reasonable (similar to original)
    original_test_ratio = labels.sum() / len(labels)
    print(f"\n   Original test anomaly ratio: {original_test_ratio*100:.2f}%")
    print(f"   Fine test anomaly ratio: {test_fine_anomaly_ratio*100:.2f}%")
    print(f"   Difference: {abs(test_fine_anomaly_ratio - original_test_ratio)*100:.2f}%")
    
    print(f"\n7. Recommendation:")
    print(f"   Based on the analysis, we recommend:")
    print(f"   - window_fine: {recommended_window_fine}")
    print(f"   - window_coarse: {recommended_window_fine // recommended_k} (or same as fine, data is aggregated)")
    print(f"   - k: {recommended_k}")
    print(f"   - train_ratio: {train_ratio}")
    print(f"   - val_ratio: {val_ratio}")
    print(f"\n   This configuration:")
    print(f"   - Produces {n_fine:,} fine sequences and {n_coarse:,} coarse sequences")
    print(f"   - Maintains reasonable anomaly ratio ({test_fine_anomaly_ratio*100:.2f}% vs original {original_test_ratio*100:.2f}%)")
    print(f"   - Is similar to PSM configuration for consistency")
    print(f"   - Provides sufficient data for training ({n_train_fine:,} fine, {n_train_coarse:,} coarse)")


if __name__ == '__main__':
    analyze_smap_data()

