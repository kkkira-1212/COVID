#!/usr/bin/env python3
"""
Multi-scale Residual Quality Analysis
按照文档要求执行三步分析：
1. 获取R_c和A_f序列
2. 计算Spearman相关性和MAE对齐误差
3. 30秒sanity check
"""
import torch
import numpy as np
from pathlib import Path
import sys
from scipy.stats import spearmanr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.evaluator import infer
from model.trainer import to_device


def get_residual_sequences(model_path, bundle_coarse, bundle_fine, device='cuda', use_val=True, use_test=False, return_train_stats=False):
    """
    步骤1: 运行multi-scale模型，得到两条序列
    
    Args:
        return_train_stats: 如果True，同时返回训练集的统计量用于归一化
    
    Returns:
        R_c: coarse-scale residuals序列 [r_c(1), r_c(2), ..., r_c(N)]
        A_f: aggregated fine-scale residuals序列 [a(1), a(2), ..., a(N)]
        idx_coarse: 使用的coarse索引
        set_name: 数据集名称
        train_stats: (可选) 训练集统计量 {'med_rc', 'mad_rc', 'med_af', 'mad_af', 'mean_rc', 'std_rc', 'mean_af', 'std_af'}
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    if config.get('coarse_only', False):
        raise ValueError("This script is for multi-scale models only. Model is coarse-only.")
    
    data_coarse = to_device(bundle_coarse, device)
    data_fine = to_device(bundle_fine, device)
    
    from model.encoder import TransformerSeqEncoder, RegressionHeadWithRelation
    
    enc_fine = TransformerSeqEncoder(
        input_dim=data_fine['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_fine['X_seq'].shape[1] + 5
    ).to(device)
    enc_fine.pooling = config['pooling']
    
    enc_coarse = TransformerSeqEncoder(
        input_dim=data_coarse['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_coarse['X_seq'].shape[1] + 5
    ).to(device)
    enc_coarse.pooling = config['pooling']
    
    head = RegressionHeadWithRelation(config['d_model'], data_coarse['X_seq'].shape[2]).to(device)
    
    enc_fine.load_state_dict(checkpoint['enc_fine'])
    enc_coarse.load_state_dict(checkpoint['enc_coarse'])
    head.load_state_dict(checkpoint['head'])
    
    enc_fine.eval()
    enc_coarse.eval()
    head.eval()
    
    # 确定使用哪个数据集
    if use_val:
        idx_set_name = 'val'
        idx_coarse = data_coarse['idx_val'].cpu().numpy() if isinstance(data_coarse['idx_val'], torch.Tensor) else data_coarse['idx_val']
    elif use_test:
        idx_set_name = 'test'
        idx_coarse = data_coarse['idx_test'].cpu().numpy() if isinstance(data_coarse['idx_test'], torch.Tensor) else data_coarse['idx_test']
    else:
        idx_set_name = 'all'
        idx_coarse = np.arange(len(data_coarse['X_seq']))
    
    # 获取mapping
    mapping = data_fine.get('fine_to_coarse_index', None)
    if mapping is None:
        raise ValueError("fine_to_coarse_index mapping not found in bundle_fine")
    
    if isinstance(mapping, torch.Tensor):
        mapping = mapping.cpu().numpy()
    else:
        mapping = np.array(mapping)
    
    print(f"\n{'='*60}")
    print(f"Step 1: Generating Residual Sequences ({idx_set_name.upper()} set)")
    print(f"{'='*60}")
    print(f"Coarse samples in set: {len(idx_coarse)}")
    
    with torch.no_grad():
        # Forward pass
        z_fine = enc_fine(data_fine['X_seq'])
        z_coarse = enc_coarse(data_coarse['X_seq'])
        pred_fine_all, pred_coarse_all = head(z_fine, z_coarse)
        
        # 计算residuals
        residual_vec_fine = (data_fine['X_next'] - pred_fine_all).abs()
        residual_vec_coarse = (data_coarse['X_next'] - pred_coarse_all).abs()
        
        # Coarse residual: mean over features, then we get sequence for selected indices
        residual_coarse_mean = residual_vec_coarse.mean(dim=1).cpu().numpy()  # Shape: [N_coarse]
        
        # Fine residual: mean over features
        residual_fine_mean = residual_vec_fine.mean(dim=1).cpu().numpy()  # Shape: [N_fine]
        
        # 获取R_c序列（按coarse索引顺序）
        R_c = residual_coarse_mean[idx_coarse]
        
        # 获取A_f序列（aggregate fine residuals for each coarse sample）
        # 对于每个coarse样本c_idx，找到所有映射到它的fine样本，取median
        A_f = np.zeros(len(idx_coarse))
        
        # 获取fine set索引（用于过滤）
        if use_val:
            idx_fine_set = data_fine['idx_val'].cpu().numpy() if isinstance(data_fine['idx_val'], torch.Tensor) else data_fine['idx_val']
        elif use_test:
            idx_fine_set = data_fine['idx_test'].cpu().numpy() if isinstance(data_fine['idx_test'], torch.Tensor) else data_fine['idx_test']
        else:
            idx_fine_set = np.arange(len(data_fine['X_seq']))
        
        fine_counts = []
        for i, c_idx in enumerate(idx_coarse):
            # 找到所有映射到这个coarse样本的fine样本
            fine_mask = (mapping == c_idx)
            fine_indices_all = np.where(fine_mask)[0]
            
            # 过滤到对应set中的fine样本
            fine_indices = fine_indices_all[np.isin(fine_indices_all, idx_fine_set)]
            fine_counts.append(len(fine_indices))
            
            if len(fine_indices) > 0:
                # 使用median聚合（文档指定）
                A_f[i] = np.median(residual_fine_mean[fine_indices])
            elif len(fine_indices_all) > 0:
                # 如果set内没有fine样本，但mapping存在，使用所有fine样本的median
                A_f[i] = np.median(residual_fine_mean[fine_indices_all])
            else:
                # 如果没有fine样本映射，使用coarse residual
                A_f[i] = residual_coarse_mean[c_idx]
        
        print(f"R_c shape: {R_c.shape}")
        print(f"A_f shape: {A_f.shape}")
        print(f"R_c range: [{np.min(R_c):.4f}, {np.max(R_c):.4f}], mean: {np.mean(R_c):.4f}, median: {np.median(R_c):.4f}")
        print(f"A_f range: [{np.min(A_f):.4f}, {np.max(A_f):.4f}], mean: {np.mean(A_f):.4f}, median: {np.median(A_f):.4f}")
        print(f"Fine samples per coarse (min/max/mean): {np.min(fine_counts)}/{np.max(fine_counts)}/{np.mean(fine_counts):.1f}")
        
        # 检查scale差异
        scale_ratio = np.mean(R_c) / (np.mean(A_f) + 1e-8)
        print(f"Scale ratio (R_c_mean / A_f_mean): {scale_ratio:.2f}x")
        if scale_ratio > 2.0 or scale_ratio < 0.5:
            print(f"⚠ Warning: Large scale difference between coarse and fine residuals")
        
        # 如果需要，计算训练集统计量
        train_stats = None
        if return_train_stats:
            idx_train_coarse = data_coarse['idx_train'].cpu().numpy() if isinstance(data_coarse['idx_train'], torch.Tensor) else data_coarse['idx_train']
            idx_train_fine = data_fine['idx_train'].cpu().numpy() if isinstance(data_fine['idx_train'], torch.Tensor) else data_fine['idx_train']
            
            # 计算训练集的R_c
            R_c_train = residual_coarse_mean[idx_train_coarse]
            
            # 计算训练集的A_f
            A_f_train = np.zeros(len(idx_train_coarse))
            for i, c_idx in enumerate(idx_train_coarse):
                fine_mask = (mapping == c_idx)
                fine_indices_all = np.where(fine_mask)[0]
                fine_indices = fine_indices_all[np.isin(fine_indices_all, idx_train_fine)]
                
                if len(fine_indices) > 0:
                    A_f_train[i] = np.median(residual_fine_mean[fine_indices])
                elif len(fine_indices_all) > 0:
                    A_f_train[i] = np.median(residual_fine_mean[fine_indices_all])
                else:
                    A_f_train[i] = residual_coarse_mean[c_idx]
            
            # 计算统计量
            med_rc_train = np.median(R_c_train)
            med_af_train = np.median(A_f_train)
            mean_rc_train = np.mean(R_c_train)
            mean_af_train = np.mean(A_f_train)
            
            # MAD (Median Absolute Deviation)
            mad_rc_train = np.median(np.abs(R_c_train - med_rc_train))
            mad_af_train = np.median(np.abs(A_f_train - med_af_train))
            
            # Std
            std_rc_train = np.std(R_c_train)
            std_af_train = np.std(A_f_train)
            
            train_stats = {
                'med_rc': med_rc_train,
                'mad_rc': mad_rc_train,
                'med_af': med_af_train,
                'mad_af': mad_af_train,
                'mean_rc': mean_rc_train,
                'std_rc': std_rc_train,
                'mean_af': mean_af_train,
                'std_af': std_af_train
            }
            
            print(f"\nTraining Set Statistics:")
            print(f"  R_c: median={med_rc_train:.4f}, MAD={mad_rc_train:.4f}, mean={mean_rc_train:.4f}, std={std_rc_train:.4f}")
            print(f"  A_f: median={med_af_train:.4f}, MAD={mad_af_train:.4f}, mean={mean_af_train:.4f}, std={std_af_train:.4f}")
    
    if return_train_stats:
        return R_c, A_f, idx_coarse, idx_set_name, train_stats
    else:
        return R_c, A_f, idx_coarse, idx_set_name


def normalize_residuals(R_c, A_f, train_stats, use_mad=True, eps=1e-8):
    """
    使用训练集统计量归一化residuals
    
    Args:
        R_c: coarse residuals
        A_f: aggregated fine residuals
        train_stats: 训练集统计量字典
        use_mad: 如果True使用MAD，否则使用std
        eps: 防止除零的小值
    
    Returns:
        R_c_norm: 归一化后的coarse residuals
        A_f_norm: 归一化后的fine residuals
    """
    if use_mad:
        # 使用median和MAD归一化
        R_c_norm = (R_c - train_stats['med_rc']) / (train_stats['mad_rc'] + eps)
        A_f_norm = (A_f - train_stats['med_af']) / (train_stats['mad_af'] + eps)
    else:
        # 使用mean和std归一化
        R_c_norm = (R_c - train_stats['mean_rc']) / (train_stats['std_rc'] + eps)
        A_f_norm = (A_f - train_stats['mean_af']) / (train_stats['std_af'] + eps)
    
    return R_c_norm, A_f_norm


def calculate_metrics(R_c, A_f, R_c_norm=None, A_f_norm=None):
    """
    步骤2: 计算两个指标（原始和归一化后的）
    
    Args:
        R_c, A_f: 原始residuals
        R_c_norm, A_f_norm: 归一化后的residuals（可选）
    
    Returns:
        包含原始和归一化指标的字典
    """
    print(f"\n{'='*60}")
    print("Step 2: Calculating Metrics (Acceptance Test)")
    print(f"{'='*60}")
    
    # 原始指标
    spearman_rho, spearman_p = spearmanr(R_c, A_f)
    mae_align = np.mean(np.abs(R_c - A_f))
    pearson_r = np.corrcoef(R_c, A_f)[0, 1]
    
    print(f"\nOriginal (Raw) Residuals:")
    print(f"  Spearman Rank Correlation (rho): {spearman_rho:.6f} (p-value: {spearman_p:.6f})")
    print(f"  Alignment Error MAE: {mae_align:.6f}")
    print(f"  Pearson Correlation: {pearson_r:.6f}")
    
    results = {
        'raw': {
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'mae_align': mae_align,
            'pearson_r': pearson_r
        }
    }
    
    # 归一化后的指标
    if R_c_norm is not None and A_f_norm is not None:
        spearman_rho_norm, spearman_p_norm = spearmanr(R_c_norm, A_f_norm)
        mae_align_norm = np.mean(np.abs(R_c_norm - A_f_norm))
        pearson_r_norm = np.corrcoef(R_c_norm, A_f_norm)[0, 1]
        
        print(f"\nNormalized Residuals (using train statistics):")
        print(f"  Spearman Rank Correlation (rho): {spearman_rho_norm:.6f} (p-value: {spearman_p_norm:.6f})")
        print(f"  Alignment Error MAE: {mae_align_norm:.6f}")
        print(f"  Pearson Correlation: {pearson_r_norm:.6f}")
        
        results['normalized'] = {
            'spearman_rho': spearman_rho_norm,
            'spearman_p': spearman_p_norm,
            'mae_align': mae_align_norm,
            'pearson_r': pearson_r_norm
        }
        
        # 解释
        print(f"\nInterpretation:")
        if spearman_rho_norm > 0.7:
            print(f"  ✓ Strong positive correlation - Multi-scale residuals are well aligned")
        elif spearman_rho_norm > 0.4:
            print(f"  ○ Moderate correlation - Some alignment exists")
        else:
            print(f"  ✗ Weak correlation - Multi-scale residuals may not be well aligned")
        
        # 对比
        print(f"\nImprovement after normalization:")
        print(f"  Spearman: {spearman_rho:.4f} -> {spearman_rho_norm:.4f} ({spearman_rho_norm - spearman_rho:+.4f})")
        print(f"  MAE: {mae_align:.4f} -> {mae_align_norm:.4f} ({mae_align_norm - mae_align:+.4f})")
    else:
        # 解释原始结果
        if spearman_rho > 0.7:
            print(f"  ✓ Strong positive correlation - Multi-scale residuals are well aligned")
        elif spearman_rho > 0.4:
            print(f"  ○ Moderate correlation - Some alignment exists")
        else:
            print(f"  ✗ Weak correlation - Multi-scale residuals may not be well aligned")
    
    return results


def sanity_check(R_c, A_f, idx_coarse, n_samples=10, seed=42, R_c_norm=None, A_f_norm=None):
    """
    步骤3: 30秒sanity check
    
    随机选10个hour windows，打印t, r_c(t), a(t)
    检查：当r_c(t)大时，a(t)是否也大；当r_c(t)小时，a(t)是否也小
    """
    print(f"\n{'='*60}")
    print(f"Step 3: 30-Second Sanity Check (Random {n_samples} samples)")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    n_total = len(R_c)
    if n_samples > n_total:
        n_samples = n_total
    
    # 随机选择索引
    selected_indices = np.random.choice(n_total, size=n_samples, replace=False)
    selected_indices = np.sort(selected_indices)  # 按顺序排列方便查看
    
    print(f"\n{'t (idx)':>8} | {'r_c(t)':>12} | {'a(t)':>12} | {'Consistent?':>12}")
    print(f"{'-'*50}")
    
    consistent_count = 0
    large_both = 0
    small_both = 0
    large_small_mismatch = 0
    small_large_mismatch = 0
    
    # 定义"大"和"小"的阈值（基于中位数）
    median_r_c = np.median(R_c)
    median_a_f = np.median(A_f)
    
    for idx in selected_indices:
        t = idx_coarse[idx]
        r_c_val = R_c[idx]
        a_val = A_f[idx]
        
        # 判断一致性
        r_c_large = r_c_val > median_r_c
        a_large = a_val > median_a_f
        
        if r_c_large == a_large:
            consistent = "✓ Yes"
            consistent_count += 1
            if r_c_large:
                large_both += 1
            else:
                small_both += 1
        else:
            consistent = "✗ No"
            if r_c_large:
                large_small_mismatch += 1
            else:
                small_large_mismatch += 1
        
        print(f"{t:8d} | {r_c_val:12.6f} | {a_val:12.6f} | {consistent:>12}")
    
    print(f"\n{'='*50}")
    print(f"Sanity Check Summary:")
    print(f"  Consistent (both large or both small): {consistent_count}/{n_samples} ({100*consistent_count/n_samples:.1f}%)")
    print(f"  Both large: {large_both}")
    print(f"  Both small: {small_both}")
    print(f"  Large r_c but small a: {large_small_mismatch}")
    print(f"  Small r_c but large a: {small_large_mismatch}")
    
    if consistent_count >= n_samples * 0.7:
        print(f"\n✓ PASS: Residuals are intuitively consistent")
        print(f"  When hour residual is large, aggregated fine residual is also large")
        print(f"  When hour residual is small, aggregated fine residual is also small")
    else:
        print(f"\n✗ FAIL: Residuals may not be well aligned")
        print(f"  Consider checking mapping and aggregation logic")
    
    # 如果有归一化数据，也做sanity check
    if R_c_norm is not None and A_f_norm is not None:
        print(f"\n{'='*60}")
        print(f"Step 3b: Sanity Check on Normalized Residuals (Random {n_samples} samples)")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        selected_indices = np.random.choice(n_total, size=n_samples, replace=False)
        selected_indices = np.sort(selected_indices)
        
        print(f"\n{'t (idx)':>8} | {'r_c_norm':>12} | {'a_norm(t)':>12} | {'Consistent?':>12}")
        print(f"{'-'*50}")
        
        consistent_count_norm = 0
        median_r_c_norm = np.median(R_c_norm)
        median_a_f_norm = np.median(A_f_norm)
        
        for idx in selected_indices:
            t = idx_coarse[idx]
            r_c_val = R_c_norm[idx]
            a_val = A_f_norm[idx]
            
            r_c_large = r_c_val > median_r_c_norm
            a_large = a_val > median_a_f_norm
            
            if r_c_large == a_large:
                consistent = "✓ Yes"
                consistent_count_norm += 1
            else:
                consistent = "✗ No"
            
            print(f"{t:8d} | {r_c_val:12.6f} | {a_val:12.6f} | {consistent:>12}")
        
        print(f"\nNormalized Sanity Check: {consistent_count_norm}/{n_samples} ({100*consistent_count_norm/n_samples:.1f}%) consistent")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-scale Residual Quality Analysis")
    parser.add_argument('--model_path', type=str, required=True, help='Path to multi-scale model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/SWaT/processed')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_val', action='store_true', default=True, help='Use validation set (default)')
    parser.add_argument('--use_test', action='store_true', help='Use test set instead of validation')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples for sanity check')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sanity check')
    
    args = parser.parse_args()
    
    if args.use_test:
        args.use_val = False
    
    data_dir = Path(args.data_dir)
    bundle_hour = torch.load(data_dir / 'swat_hour.pt', weights_only=False)
    bundle_minute = torch.load(data_dir / 'swat_minute.pt', weights_only=False)
    
    print("="*60)
    print("Multi-scale Residual Quality Analysis")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Hour scale: {bundle_hour['X_seq'].shape}")
    print(f"Minute scale: {bundle_minute['X_seq'].shape}")
    
    # Step 1: 获取序列（包括训练集统计量）
    # 先获取训练集统计量
    _, _, _, _, train_stats = get_residual_sequences(
        args.model_path, bundle_hour, bundle_minute,
        device=args.device, use_val=False, use_test=False, return_train_stats=True
    )
    
    # 获取当前数据集的序列
    R_c, A_f, idx_coarse, set_name = get_residual_sequences(
        args.model_path, bundle_hour, bundle_minute, 
        device=args.device, use_val=args.use_val, use_test=args.use_test,
        return_train_stats=False
    )
    
    # 归一化
    R_c_norm, A_f_norm = normalize_residuals(R_c, A_f, train_stats, use_mad=True)
    
    # Step 2: 计算指标（原始和归一化）
    metrics = calculate_metrics(R_c, A_f, R_c_norm, A_f_norm)
    
    # Step 3: Sanity check（原始和归一化）
    sanity_check(R_c, A_f, idx_coarse, n_samples=args.n_samples, seed=args.seed,
                 R_c_norm=R_c_norm, A_f_norm=A_f_norm)
    
    print(f"\n{'='*60}")
    print("Analysis Complete")
    print(f"{'='*60}")
    print(f"\nSummary for {set_name.upper()} set:")
    print(f"\nRaw Residuals:")
    print(f"  Spearman ρ: {metrics['raw']['spearman_rho']:.6f}")
    print(f"  MAE Alignment: {metrics['raw']['mae_align']:.6f}")
    print(f"  Pearson r: {metrics['raw']['pearson_r']:.6f}")
    if 'normalized' in metrics:
        print(f"\nNormalized Residuals:")
        print(f"  Spearman ρ: {metrics['normalized']['spearman_rho']:.6f}")
        print(f"  MAE Alignment: {metrics['normalized']['mae_align']:.6f}")
        print(f"  Pearson r: {metrics['normalized']['pearson_r']:.6f}")


if __name__ == '__main__':
    main()

