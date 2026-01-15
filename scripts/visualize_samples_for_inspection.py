#!/usr/bin/env python3
"""
为人工标注可视化样本的时间序列
展示每个样本的窗口数据，方便人类直觉判断
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_samples_for_inspection(
    samples_data_path='analysis/manual_inspection/samples_data.pt',
    output_dir='analysis/manual_inspection/visualizations',
    n_cols=3
):
    """
    为每个样本创建时间序列可视化
    展示窗口内的特征变化
    """
    # 加载数据
    samples_data = torch.load(samples_data_path, weights_only=False, map_location='cpu')
    samples_df = samples_data['samples_df']
    bundle_coarse = samples_data['bundle_coarse']
    sampled_indices = samples_data['sampled_indices']
    
    # 获取特征名称
    feature_cols = bundle_coarse['feature_cols']
    window_size = bundle_coarse['window_size']
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"可视化 {len(samples_df)} 个样本...")
    
    # 为每个样本创建可视化
    for idx, row in samples_df.iterrows():
        sample_idx = row['original_index']
        state = row['state']
        target_date = row['target_date']
        label = row['label']
        residual = row['residual']
        
        # 获取窗口数据
        X_seq = bundle_coarse['X_seq'][sample_idx].numpy()  # (window_size, n_features)
        X_next = bundle_coarse['X_next'][sample_idx].numpy()  # (n_features,)
        
        # 选择几个关键特征进行可视化
        # 优先选择：NewCases, NewDeaths, NewCases_MA7, NewDeaths_MA7, Ct_Value等
        key_features = ['NewCases', 'NewDeaths', 'NewCases_MA7', 'NewDeaths_MA7', 
                       'Ct_Value', 'Stringency_Index', 'TotalCases', 'TotalDeaths']
        
        # 找出这些特征在feature_cols中的索引
        feature_indices = []
        feature_names = []
        for feat in key_features:
            if feat in feature_cols:
                feat_idx = feature_cols.index(feat)
                feature_indices.append(feat_idx)
                feature_names.append(feat)
        
        # 如果关键特征不够，补充其他特征
        if len(feature_indices) < 4:
            remaining = [i for i in range(len(feature_cols)) if i not in feature_indices]
            feature_indices.extend(remaining[:4-len(feature_indices)])
            feature_names.extend([feature_cols[i] for i in remaining[:4-len(feature_indices)]])
        
        feature_indices = feature_indices[:8]  # 最多8个特征
        feature_names = feature_names[:8]
        
        # 创建可视化
        n_features = len(feature_indices)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # 时间轴（窗口内的时间步）
        time_steps = np.arange(-window_size, 0)
        time_steps_with_next = np.arange(-window_size, 1)
        
        for i, (feat_idx, feat_name) in enumerate(zip(feature_indices, feature_names)):
            ax = axes[i]
            
            # 绘制窗口内的序列
            values = X_seq[:, feat_idx]
            ax.plot(time_steps, values, 'o-', color='blue', linewidth=2, markersize=4, label='Sequence')
            
            # 绘制target值（下一个时间点）
            next_value = X_next[feat_idx]
            ax.plot(0, next_value, 's', color='red', markersize=8, label='Target', zorder=5)
            
            # 标注target date
            ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Time Step (window)', fontsize=10)
            ax.set_ylabel(feat_name, fontsize=10)
            ax.set_title(feat_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # 隐藏多余的子图
        for i in range(len(feature_indices), len(axes)):
            axes[i].axis('off')
        
        # 添加标题信息
        title = f"{state} | {target_date.strftime('%Y-%m-%d')} | Label: {label} | Residual: {residual:.6f}"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        
        # 保存
        filename = f"{state}_{target_date.strftime('%Y%m%d')}_{label}_{sample_idx}.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1}/{len(samples_df)} 个样本")
    
    print(f"\n所有可视化已保存到: {output_dir}")
    print(f"总共 {len(samples_df)} 个样本的可视化")


def create_annotation_template(
    samples_csv_path='analysis/manual_inspection/samples_for_annotation.csv',
    output_path='analysis/manual_inspection/annotation_template.csv'
):
    """创建标注模板，包含human_judgment列"""
    
    samples_df = pd.read_csv(samples_csv_path)
    
    # 添加human_judgment列（如果不存在）
    if 'human_judgment' not in samples_df.columns:
        samples_df['human_judgment'] = ''
    
    # 重新排列列，human_judgment放在最后
    cols = [c for c in samples_df.columns if c != 'human_judgment'] + ['human_judgment']
    samples_df = samples_df[cols]
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(output_path, index=False)
    
    print(f"\n标注模板已保存到: {output_path}")
    print("请在human_judgment列中填入:")
    print("  1 = 明显有变化")
    print("  2 = 模糊")
    print("  3 = 完全看不出来")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize samples for manual inspection')
    parser.add_argument('--samples_data_path', type=str, 
                       default='analysis/manual_inspection/samples_data.pt',
                       help='Path to samples data file')
    parser.add_argument('--output_dir', type=str, 
                       default='analysis/manual_inspection/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--n_cols', type=int, default=3, help='Number of columns in subplot')
    parser.add_argument('--create_template', action='store_true', 
                       help='Create annotation template CSV')
    
    args = parser.parse_args()
    
    # 可视化
    visualize_samples_for_inspection(
        samples_data_path=args.samples_data_path,
        output_dir=args.output_dir,
        n_cols=args.n_cols
    )
    
    # 创建标注模板
    if args.create_template:
        samples_csv = Path(args.samples_data_path).parent / 'samples_for_annotation.csv'
        template_csv = Path(args.output_dir).parent / 'annotation_template.csv'
        create_annotation_template(
            samples_csv_path=samples_csv,
            output_path=template_csv
        )

