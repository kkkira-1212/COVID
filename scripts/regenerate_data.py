import os
import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_utils import loader, engineer_features, group_by_state, agg_map
from utils.labels import create_outbreak_labels
from utils.sequences import create_sequences, split_sequences, sequences_to_bundle, standardize
from utils.mapping import add_mapping

FEATURES_21 = [
    'NewCases', 'NewDeaths',
    'NewCases_MA7', 'NewDeaths_MA7',
    'Cases_GrowthRate', 'NewDeaths_return',
    'Patience_Count',
    'Vax_AllDoses', 'Vax_Dose1', 'Vax_Dose2',
    'Vax_Dose3',
    'Hosp_Count', 'Hosp_Deaths',
    'Ct_Value', 'Stringency_Index',
    'Aver_Hosp_Stay',
    'TotalDeaths_by_TotalCases', 'Hosp_Death_Rate',
    'TotalCases', 'TotalDeaths',
    'TotalCases_100k_inhab', 'TotalDeaths_100k_inhab'
]

FEATURES_6 = [
    'NewCases', 'NewDeaths',
    'NewCases_MA7', 'NewDeaths_MA7',
    'Cases_GrowthRate', 'NewDeaths_return'
]

def build_and_save_dataset(
    data_path,
    feature_cols,
    window_day,
    window_week,
    output_dir,
    output_prefix,
    mapping=True
):
    print(f"\n{'='*60}")
    print(f"Building dataset: {output_prefix}")
    print(f"Features: {len(feature_cols)} features")
    print(f"Window: day={window_day}, week={window_week}")
    print(f"{'='*60}\n")
    
    print("Step 1: Loading data...")
    df = loader(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    print("Step 2: Engineering features...")
    df = engineer_features(df)
    print(f"  Features engineered")
    
    print("Step 3: Creating labels...")
    df = create_outbreak_labels(df)
    num_outbreaks = df['outbreak_label'].sum()
    print(f"  Created labels: {num_outbreaks} outbreak samples out of {len(df)} total")
    
    print("Step 4: Grouping by state (day frequency)...")
    day_data = group_by_state(df, freq='D', agg_map=agg_map)
    print(f"  Grouped into {len(day_data)} states")
    
    print("Step 5: Grouping by state (week frequency)...")
    week_data = group_by_state(df, freq='W', agg_map=agg_map)
    print(f"  Grouped into {len(week_data)} states")
    
    print("Step 6: Creating sequences (day)...")
    seq_day, states_day = create_sequences(day_data, feature_cols, window_day, 1, 'D')
    print(f"  Created {len(seq_day)} day sequences")
    
    print("Step 7: Creating sequences (week)...")
    seq_week, states_week = create_sequences(week_data, feature_cols, window_week, 1, 'W')
    print(f"  Created {len(seq_week)} week sequences")
    
    print("Step 8: Splitting sequences...")
    idx_tr_d, idx_v_d, idx_te_d = split_sequences(seq_day, 0.7, 0.1)
    idx_tr_w, idx_v_w, idx_te_w = split_sequences(seq_week, 0.7, 0.1)
    print(f"  Day: train={len(idx_tr_d)}, val={len(idx_v_d)}, test={len(idx_te_d)}")
    print(f"  Week: train={len(idx_tr_w)}, val={len(idx_v_w)}, test={len(idx_te_w)}")
    
    print("Step 9: Converting to bundles...")
    bundle_day = sequences_to_bundle(seq_day, idx_tr_d, idx_v_d, idx_te_d, states_day, feature_cols, window_day)
    bundle_week = sequences_to_bundle(seq_week, idx_tr_w, idx_v_w, idx_te_w, states_week, feature_cols, window_week)
    print(f"  Bundle day: X_seq={bundle_day['X_seq'].shape}, X_next={bundle_day['X_next'].shape}")
    print(f"  Bundle week: X_seq={bundle_week['X_seq'].shape}, X_next={bundle_week['X_next'].shape}")
    
    print("Step 10: Standardizing...")
    bundle_day = standardize(bundle_day)
    bundle_week = standardize(bundle_week)
    print(f"  Standardized using training set statistics")
    print(f"  Feature means shape: {bundle_day['feature_means'].shape}")
    print(f"  Feature stds shape: {bundle_day['feature_stds'].shape}")
    
    if mapping:
        print("Step 11: Adding day-to-week mapping...")
        bundle_day = add_mapping(bundle_day, bundle_week)
        num_mapped = (bundle_day['fine_to_coarse_index'] >= 0).sum().item()
        print(f"  Mapped {num_mapped} day samples to week samples")
    
    print("Step 12: Saving bundles...")
    os.makedirs(output_dir, exist_ok=True)
    
    day_path = os.path.join(output_dir, f"{output_prefix}.pt")
    
    torch.save(bundle_day, day_path)
    
    print(f"  Saved day bundle to: {day_path}")
    
    print(f"\n{'='*60}")
    print(f"Dataset {output_prefix} completed successfully!")
    print(f"{'='*60}\n")
    
    return bundle_day, bundle_week

def verify_bundle(bundle, name):
    print(f"\nVerifying {name} bundle:")
    print(f"  X_seq shape: {bundle['X_seq'].shape}")
    print(f"  X_next shape: {bundle['X_next'].shape}")
    print(f"  X_seq and X_next feature dim match: {bundle['X_seq'].shape[2] == bundle['X_next'].shape[1]}")
    print(f"  Has feature_means: {'feature_means' in bundle}")
    print(f"  Has feature_stds: {'feature_stds' in bundle}")
    print(f"  Has NewDeaths_ret_next: {'NewDeaths_ret_next' in bundle}")
    print(f"  Number of features: {len(bundle['feature_cols'])}")
    
    if 'X_next' in bundle:
        X_seq_mean = bundle['X_seq'].mean(dim=(0,1))
        X_next_mean = bundle['X_next'].mean(dim=0)
        print(f"  X_seq mean (after std): {X_seq_mean.abs().max().item():.6f} (should be ~0)")
        print(f"  X_next mean (after std): {X_next_mean.abs().max().item():.6f} (should be ~0)")
        print(f"  X_seq std (after std): {bundle['X_seq'].std(dim=(0,1)).mean().item():.4f} (should be ~1)")
        print(f"  X_next std (after std): {bundle['X_next'].std(dim=0).mean().item():.4f} (should be ~1)")

if __name__ == "__main__":
    data_path = "data/3.1_3.2_Final_Dataset_State_Level.xlsx"
    output_dir = "data/processed"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Please ensure the State Level Excel file exists.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("COVID-19 Data Regeneration Script")
    print("="*60)
    print(f"Input: {data_path}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    datasets_to_generate = [
        {
            "features": FEATURES_21,
            "window_day": 14,
            "window_week": 8,
            "prefix": "day_21feat",
            "week_prefix": "week_21feat"
        },
        {
            "features": FEATURES_6,
            "window_day": 14,
            "window_week": 8,
            "prefix": "day_6feat",
            "week_prefix": "week_6feat"
        }
    ]
    
    for config in datasets_to_generate:
        bundle_day, bundle_week = build_and_save_dataset(
            data_path=data_path,
            feature_cols=config["features"],
            window_day=config["window_day"],
            window_week=config["window_week"],
            output_dir=output_dir,
            output_prefix=config["prefix"],
            mapping=True
        )
        
        verify_bundle(bundle_day, f"{config['prefix']} (day)")
        verify_bundle(bundle_week, f"{config['week_prefix']} (week)")
        
        week_path = os.path.join(output_dir, f"{config['week_prefix']}.pt")
        torch.save(bundle_week, week_path)
        print(f"  Saved week bundle to: {week_path}")
    
    print("\n" + "="*60)
    print("All datasets generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - {output_dir}/day_21feat.pt")
    print(f"  - {output_dir}/week_21feat.pt")
    print(f"  - {output_dir}/day_6feat.pt")
    print(f"  - {output_dir}/week_6feat.pt")
    print("="*60 + "\n")

