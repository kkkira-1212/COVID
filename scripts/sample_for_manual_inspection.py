#!/usr/bin/env python3
"""
为人工标注抽样：每个州2-3个anomaly，1-2个非anomaly对照
总数控制在30-50个window
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.evaluator import infer


def sample_for_manual_inspection(
    model_path,
    data_dir='data/processed',
    coarse_file='week_21feat.pt',
    fine_file=None,
    output_dir='analysis/manual_inspection',
    target_total=40,
    anomalies_per_state=2,
    normals_per_state=1,
    device='cuda',
    seed=42
):
    """
    抽样策略：
    - 每个州：2-3个anomaly，1-2个非anomaly（作为对照）
    - 总数控制在30-50个window
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 加载数据
    data_dir = Path(project_root) / data_dir
    bundle_coarse = torch.load(data_dir / coarse_file, weights_only=False, map_location='cpu')
    
    bundle_fine = None
    if fine_file and (data_dir / fine_file).exists():
        bundle_fine = torch.load(data_dir / fine_file, weights_only=False, map_location='cpu')
    elif not fine_file:
        # 尝试自动找到fine文件（匹配coarse文件）
        if 'week_21feat' in coarse_file:
            fine_file = 'day_21feat.pt'
        elif 'week_6feat' in coarse_file:
            fine_file = 'day_6feat.pt'
        if fine_file and (data_dir / fine_file).exists():
            bundle_fine = torch.load(data_dir / fine_file, weights_only=False, map_location='cpu')
            print(f"Auto-loaded fine file: {fine_file}")
    
    # 推理获取residual
    print("Running inference to get residual...")
    out = infer(
        model_path=model_path,
        bundle_coarse=bundle_coarse,
        bundle_fine=bundle_fine,
        device=device,
        use_postprocessing=False
    )
    
    residual = out['residual']
    y_true = out['y_true']
    idx_test = out['idx_test']
    meta = bundle_coarse['meta'].copy()
    
    # 获取test set的数据
    meta_test = meta.iloc[idx_test].copy()
    meta_test['target_date'] = pd.to_datetime(meta_test['target_date'])
    meta_test['residual'] = residual[idx_test]
    meta_test['y_true'] = y_true[idx_test]
    meta_test['original_index'] = idx_test
    
    # 按state分组
    states = sorted(meta_test['state'].unique())
    
    print(f"\n总共有 {len(states)} 个states")
    print(f"Test set总数: {len(meta_test)}")
    print(f"Test set中Anomaly: {meta_test['y_true'].sum()}, Normal: {(meta_test['y_true'] == 0).sum()}")
    
    # 抽样
    sampled_indices = []
    sampled_info = []
    
    for state in states:
        state_data = meta_test[meta_test['state'] == state]
        state_anomalies = state_data[state_data['y_true'] == 1]
        state_normals = state_data[state_data['y_true'] == 0]
        
        # 抽样anomaly（2-3个）
        n_anomalies_to_sample = min(anomalies_per_state, len(state_anomalies))
        if n_anomalies_to_sample > 0:
            if len(state_anomalies) <= n_anomalies_to_sample:
                sampled_anomalies = state_anomalies
            else:
                # 优先选择residual大的anomaly（更典型的anomaly）
                # 也随机选一些residual小的（可能有问题的）
                sorted_anomalies = state_anomalies.sort_values('residual', ascending=False)
                top_anomalies = sorted_anomalies.head(n_anomalies_to_sample - 1)
                if len(state_anomalies) > n_anomalies_to_sample:
                    remaining = sorted_anomalies.tail(len(state_anomalies) - (n_anomalies_to_sample - 1))
                    if len(remaining) > 0:
                        random_anomaly = remaining.sample(1)
                        sampled_anomalies = pd.concat([top_anomalies, random_anomaly])
                    else:
                        sampled_anomalies = top_anomalies
                else:
                    sampled_anomalies = top_anomalies
            
            for idx, row in sampled_anomalies.iterrows():
                sampled_indices.append(row['original_index'])
                sampled_info.append({
                    'state': state,
                    'target_date': row['target_date'],
                    'label': 'anomaly',
                    'residual': row['residual'],
                    'original_index': row['original_index'],
                    'human_judgment': '',  # 待标注
                })
        
        # 抽样normal（1-2个作为对照）
        n_normals_to_sample = min(normals_per_state, len(state_normals))
        if n_normals_to_sample > 0:
            # 选择residual范围不同的normal（有些residual大，有些小）
            if len(state_normals) <= n_normals_to_sample:
                sampled_normals = state_normals
            else:
                # 选择residual最大和最小的normal作为对照
                sorted_normals = state_normals.sort_values('residual', ascending=False)
                if n_normals_to_sample == 1:
                    # 随机选一个
                    sampled_normals = sorted_normals.sample(1)
                else:
                    # 选最大和最小的
                    top_normal = sorted_normals.head(1)
                    bottom_normal = sorted_normals.tail(1)
                    sampled_normals = pd.concat([top_normal, bottom_normal])
            
            for idx, row in sampled_normals.iterrows():
                sampled_indices.append(row['original_index'])
                sampled_info.append({
                    'state': state,
                    'target_date': row['target_date'],
                    'label': 'normal',
                    'residual': row['residual'],
                    'original_index': row['original_index'],
                    'human_judgment': '',  # 待标注
                })
    
    # 如果总数超过target_total，随机抽样
    if len(sampled_indices) > target_total:
        print(f"\n抽样了 {len(sampled_indices)} 个样本，超过目标 {target_total}，随机抽样到 {target_total} 个")
        selected_indices = random.sample(range(len(sampled_info)), target_total)
        sampled_info = [sampled_info[i] for i in selected_indices]
        sampled_indices = [item['original_index'] for item in sampled_info]
    
    # 创建DataFrame
    samples_df = pd.DataFrame(sampled_info)
    samples_df = samples_df.sort_values(['state', 'target_date'])
    
    print(f"\n最终抽样: {len(samples_df)} 个样本")
    print(f"  Anomaly: {(samples_df['label'] == 'anomaly').sum()}")
    print(f"  Normal: {(samples_df['label'] == 'normal').sum()}")
    print(f"  涉及 {samples_df['state'].nunique()} 个states")
    
    # 保存抽样结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV（用于标注）
    csv_path = output_dir / 'samples_for_annotation.csv'
    samples_df.to_csv(csv_path, index=False)
    print(f"\n抽样结果已保存到: {csv_path}")
    
    # 保存详细信息和数据
    # 需要保存每个样本的窗口数据（X_seq）用于可视化
    samples_data = {
        'samples_df': samples_df,
        'bundle_coarse': bundle_coarse,
        'sampled_indices': sampled_indices,
        'meta_test': meta_test,
    }
    
    torch.save(samples_data, output_dir / 'samples_data.pt')
    print(f"详细数据已保存到: {output_dir / 'samples_data.pt'}")
    
    return samples_df, samples_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample data for manual inspection')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--coarse_file', type=str, default='week_21feat.pt', help='Coarse data file')
    parser.add_argument('--fine_file', type=str, default=None, help='Fine data file (optional)')
    parser.add_argument('--output_dir', type=str, default='analysis/manual_inspection', help='Output directory')
    parser.add_argument('--target_total', type=int, default=40, help='Target total samples')
    parser.add_argument('--anomalies_per_state', type=int, default=2, help='Anomalies per state')
    parser.add_argument('--normals_per_state', type=int, default=1, help='Normals per state')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    sample_for_manual_inspection(
        model_path=args.model_path,
        data_dir=args.data_dir,
        coarse_file=args.coarse_file,
        fine_file=args.fine_file,
        output_dir=args.output_dir,
        target_total=args.target_total,
        anomalies_per_state=args.anomalies_per_state,
        normals_per_state=args.normals_per_state,
        device=args.device,
        seed=args.seed
    )

