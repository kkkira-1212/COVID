#!/usr/bin/env python3
"""
COVID数据Residual可视化分析
针对COVID数据的特点进行全面的residual可视化
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import roc_curve, precision_recall_curve, auc

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.evaluator import infer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_residual_distribution(residual, y_true, idx_val, idx_test, save_path=None):
    """1. Residual分布对比 - 最核心的可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Validation set
    residual_val = residual[idx_val]
    y_val = y_true[idx_val]
    residual_normal_val = residual_val[y_val == 0]
    residual_anomaly_val = residual_val[y_val == 1]
    
    axes[0].hist(residual_normal_val, bins=50, alpha=0.7, label=f'Normal (n={len(residual_normal_val)})', 
                density=True, color='blue', edgecolor='black', linewidth=0.5)
    axes[0].hist(residual_anomaly_val, bins=50, alpha=0.7, label=f'Anomaly (n={len(residual_anomaly_val)})', 
                density=True, color='red', edgecolor='black', linewidth=0.5)
    axes[0].axvline(np.median(residual_normal_val), color='blue', linestyle='--', linewidth=2, label=f'Normal median: {np.median(residual_normal_val):.4f}')
    axes[0].axvline(np.median(residual_anomaly_val), color='red', linestyle='--', linewidth=2, label=f'Anomaly median: {np.median(residual_anomaly_val):.4f}')
    axes[0].set_xlabel('Residual', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Validation Set Residual Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Test set
    residual_test = residual[idx_test]
    y_test = y_true[idx_test]
    residual_normal_test = residual_test[y_test == 0]
    residual_anomaly_test = residual_test[y_test == 1]
    
    signal_ratio = np.median(residual_anomaly_test) / np.median(residual_normal_test) if np.median(residual_normal_test) > 0 else 0
    
    axes[1].hist(residual_normal_test, bins=50, alpha=0.7, label=f'Normal (n={len(residual_normal_test)})', 
                density=True, color='blue', edgecolor='black', linewidth=0.5)
    axes[1].hist(residual_anomaly_test, bins=50, alpha=0.7, label=f'Anomaly (n={len(residual_anomaly_test)})', 
                density=True, color='red', edgecolor='black', linewidth=0.5)
    axes[1].axvline(np.median(residual_normal_test), color='blue', linestyle='--', linewidth=2, label=f'Normal median: {np.median(residual_normal_test):.4f}')
    axes[1].axvline(np.median(residual_anomaly_test), color='red', linestyle='--', linewidth=2, label=f'Anomaly median: {np.median(residual_anomaly_test):.4f}')
    axes[1].set_xlabel('Residual', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title(f'Test Set Residual Distribution\nSignal Ratio: {signal_ratio:.2f}x', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_roc_pr_curves(residual, y_true, idx_val, idx_test, save_path=None):
    """2. ROC曲线和PR曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Validation set
    residual_val = residual[idx_val]
    y_val = y_true[idx_val]
    fpr_val, tpr_val, _ = roc_curve(y_val, residual_val)
    roc_auc_val = auc(fpr_val, tpr_val)
    precision_val, recall_val, _ = precision_recall_curve(y_val, residual_val)
    pr_auc_val = auc(recall_val, precision_val)
    
    axes[0].plot(fpr_val, tpr_val, label=f'Val (AUC={roc_auc_val:.4f})', linewidth=2, color='blue')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(recall_val, precision_val, label=f'Val (AUC={pr_auc_val:.4f})', linewidth=2, color='blue')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Test set
    residual_test = residual[idx_test]
    y_test = y_true[idx_test]
    fpr_test, tpr_test, _ = roc_curve(y_test, residual_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    precision_test, recall_test, _ = precision_recall_curve(y_test, residual_test)
    pr_auc_test = auc(recall_test, precision_test)
    
    axes[0].plot(fpr_test, tpr_test, label=f'Test (AUC={roc_auc_test:.4f})', linewidth=2, color='red', linestyle='--')
    axes[1].plot(recall_test, precision_test, label=f'Test (AUC={pr_auc_test:.4f})', linewidth=2, color='red', linestyle='--')
    axes[0].legend(fontsize=10)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_residual_timeseries(residual, y_true, meta, idx_val, idx_test, save_path=None):
    """3. Residual时间序列 - COVID特有"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 准备数据
    meta_df = meta.copy()
    meta_df['target_date'] = pd.to_datetime(meta_df['target_date'])
    meta_df['residual'] = residual
    meta_df['y_true'] = y_true
    meta_df['split'] = 'train'
    meta_df.loc[idx_val, 'split'] = 'val'
    meta_df.loc[idx_test, 'split'] = 'test'
    
    meta_df = meta_df.sort_values('target_date')
    
    # 上：所有数据的时间序列
    for split, color, label in [('train', 'gray', 'Train'), ('val', 'blue', 'Val'), ('test', 'red', 'Test')]:
        split_data = meta_df[meta_df['split'] == split]
        axes[0].scatter(split_data['target_date'], split_data['residual'], 
                       alpha=0.5, s=10, color=color, label=label)
    
    # 标注anomaly点
    anomaly_data = meta_df[meta_df['y_true'] == 1]
    axes[0].scatter(anomaly_data['target_date'], anomaly_data['residual'], 
                   alpha=0.8, s=30, color='red', marker='x', label='Anomaly', linewidths=2)
    
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Residual', fontsize=12)
    axes[0].set_title('Residual Time Series (All Data)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # 下：Test set的详细视图
    test_data = meta_df[meta_df['split'] == 'test']
    normal_test = test_data[test_data['y_true'] == 0]
    anomaly_test = test_data[test_data['y_true'] == 1]
    
    axes[1].scatter(normal_test['target_date'], normal_test['residual'], 
                   alpha=0.6, s=15, color='blue', label='Normal')
    axes[1].scatter(anomaly_test['target_date'], anomaly_test['residual'], 
                   alpha=0.8, s=30, color='red', marker='x', label='Anomaly', linewidths=2)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Residual', fontsize=12)
    axes[1].set_title('Test Set Residual Time Series (Detailed)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_residual_by_state(residual, y_true, meta, idx_test, save_path=None):
    """4. 不同State的residual分布"""
    meta_df = meta.copy()
    meta_df['residual'] = residual
    meta_df['y_true'] = y_true
    
    test_data = meta_df.iloc[idx_test].copy()
    
    # 按state统计
    state_stats = []
    for state in test_data['state'].unique():
        state_data = test_data[test_data['state'] == state]
        normal_data = state_data[state_data['y_true'] == 0]
        anomaly_data = state_data[state_data['y_true'] == 1]
        
        state_stats.append({
            'state': state,
            'normal_median': np.median(normal_data['residual']) if len(normal_data) > 0 else 0,
            'anomaly_median': np.median(anomaly_data['residual']) if len(anomaly_data) > 0 else 0,
            'normal_count': len(normal_data),
            'anomaly_count': len(anomaly_data),
        })
    
    state_df = pd.DataFrame(state_stats)
    state_df = state_df.sort_values('normal_median', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 左：Normal median by state
    axes[0].barh(state_df['state'], state_df['normal_median'], color='blue', alpha=0.7)
    axes[0].set_xlabel('Normal Residual Median', fontsize=12)
    axes[0].set_ylabel('State', fontsize=12)
    axes[0].set_title('Normal Residual Median by State', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 右：Anomaly median by state (only states with anomalies)
    state_with_anomaly = state_df[state_df['anomaly_count'] > 0]
    if len(state_with_anomaly) > 0:
        axes[1].barh(state_with_anomaly['state'], state_with_anomaly['anomaly_median'], color='red', alpha=0.7)
        axes[1].set_xlabel('Anomaly Residual Median', fontsize=12)
        axes[1].set_ylabel('State', fontsize=12)
        axes[1].set_title('Anomaly Residual Median by State', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_train_val_test_comparison(residual, y_true, idx_train, idx_val, idx_test, save_path=None):
    """5. Train/Val/Test的residual分布对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, idx, title, color in zip(axes, [idx_train, idx_val, idx_test], 
                                     ['Train', 'Validation', 'Test'], 
                                     ['gray', 'blue', 'red']):
        residual_split = residual[idx]
        y_split = y_true[idx]
        residual_normal = residual_split[y_split == 0]
        residual_anomaly = residual_split[y_split == 1]
        
        ax.hist(residual_normal, bins=50, alpha=0.7, label=f'Normal (n={len(residual_normal)})', 
               density=True, color='blue', edgecolor='black', linewidth=0.5)
        if len(residual_anomaly) > 0:
            ax.hist(residual_anomaly, bins=50, alpha=0.7, label=f'Anomaly (n={len(residual_anomaly)})', 
                   density=True, color='red', edgecolor='black', linewidth=0.5)
            signal_ratio = np.median(residual_anomaly) / np.median(residual_normal) if np.median(residual_normal) > 0 else 0
            ax.set_title(f'{title}\nSignal Ratio: {signal_ratio:.2f}x', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{title}\n(No Anomaly)', fontsize=12, fontweight='bold')
        
        ax.axvline(np.median(residual_normal), color='blue', linestyle='--', linewidth=2)
        if len(residual_anomaly) > 0:
            ax.axvline(np.median(residual_anomaly), color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Residual', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_residual_scatter(residual, y_true, idx_test, save_path=None):
    """6. Residual vs 标签的散点图"""
    residual_test = residual[idx_test]
    y_test = y_true[idx_test]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    normal_indices = y_test == 0
    anomaly_indices = y_test == 1
    
    ax.scatter(np.arange(len(residual_test))[normal_indices], 
              residual_test[normal_indices], 
              alpha=0.6, s=10, color='blue', label=f'Normal (n={np.sum(normal_indices)})')
    ax.scatter(np.arange(len(residual_test))[anomaly_indices], 
              residual_test[anomaly_indices], 
              alpha=0.8, s=30, color='red', marker='x', label=f'Anomaly (n={np.sum(anomaly_indices)})', linewidths=2)
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Residual', fontsize=12)
    ax.set_title('Residual vs Sample Index (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize COVID residual analysis')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--coarse_file', type=str, default='week_21feat.pt', help='Coarse data file')
    parser.add_argument('--fine_file', type=str, default='day_21feat.pt', help='Fine data file (optional)')
    parser.add_argument('--output_dir', type=str, default='visualizations/covid_residual', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("Loading data...")
    data_dir = Path(project_root) / args.data_dir
    bundle_coarse = torch.load(data_dir / args.coarse_file, weights_only=False, map_location='cpu')
    
    bundle_fine = None
    if args.fine_file and (data_dir / args.fine_file).exists():
        bundle_fine = torch.load(data_dir / args.fine_file, weights_only=False, map_location='cpu')
        print(f"Loaded fine scale data: {bundle_fine['X_seq'].shape}")
    
    print(f"Loaded coarse scale data: {bundle_coarse['X_seq'].shape}")
    
    # 推理获取residual
    print("Running inference...")
    out = infer(
        model_path=args.model_path,
        bundle_coarse=bundle_coarse,
        bundle_fine=bundle_fine,
        device=args.device,
        use_postprocessing=False
    )
    
    residual = out['residual']
    y_true = out['y_true']
    idx_val = out['idx_val']
    idx_test = out['idx_test']
    # idx_train is not returned by infer, get it from bundle
    idx_train = bundle_coarse['idx_train'].numpy()
    meta = bundle_coarse['meta']
    
    print(f"Residual shape: {residual.shape}")
    print(f"Train/Val/Test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")
    
    # 生成所有可视化
    print("\nGenerating visualizations...")
    
    # 1. Residual分布对比
    print("1. Plotting residual distribution...")
    plot_residual_distribution(
        residual, y_true, idx_val, idx_test,
        save_path=output_dir / '01_residual_distribution.png'
    )
    
    # 2. ROC和PR曲线
    print("2. Plotting ROC and PR curves...")
    plot_roc_pr_curves(
        residual, y_true, idx_val, idx_test,
        save_path=output_dir / '02_roc_pr_curves.png'
    )
    
    # 3. Residual时间序列
    print("3. Plotting residual time series...")
    plot_residual_timeseries(
        residual, y_true, meta, idx_val, idx_test,
        save_path=output_dir / '03_residual_timeseries.png'
    )
    
    # 4. 不同State的residual分布
    print("4. Plotting residual by state...")
    plot_residual_by_state(
        residual, y_true, meta, idx_test,
        save_path=output_dir / '04_residual_by_state.png'
    )
    
    # 5. Train/Val/Test对比
    print("5. Plotting train/val/test comparison...")
    plot_train_val_test_comparison(
        residual, y_true, idx_train, idx_val, idx_test,
        save_path=output_dir / '05_train_val_test_comparison.png'
    )
    
    # 6. Residual散点图
    print("6. Plotting residual scatter...")
    plot_residual_scatter(
        residual, y_true, idx_test,
        save_path=output_dir / '06_residual_scatter.png'
    )
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Residual Statistics Summary")
    print("="*60)
    
    residual_test = residual[idx_test]
    y_test = y_true[idx_test]
    residual_normal = residual_test[y_test == 0]
    residual_anomaly = residual_test[y_test == 1]
    
    print(f"\nTest Set:")
    print(f"  Normal: n={len(residual_normal)}, median={np.median(residual_normal):.6f}, mean={np.mean(residual_normal):.6f}, std={np.std(residual_normal):.6f}")
    print(f"  Anomaly: n={len(residual_anomaly)}, median={np.median(residual_anomaly):.6f}, mean={np.mean(residual_anomaly):.6f}, std={np.std(residual_anomaly):.6f}")
    signal_ratio = np.median(residual_anomaly) / np.median(residual_normal) if np.median(residual_normal) > 0 else 0
    print(f"  Signal Ratio (Anomaly/Normal median): {signal_ratio:.2f}x")
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

