#!/usr/bin/env python3
"""
分析residual质量和F1波动原因
- 检查residual分布
- 分析阈值选择的影响
- 对比不同聚合方式
- 检查fine vs coarse scale的residual差异
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.evaluator import infer
from model.trainer import to_device


def analyze_residual_distribution(residual, y_true, idx_val, idx_test, title="Residual Distribution"):
    """分析residual的分布"""
    residual_val = residual[idx_val]
    residual_test = residual[idx_test]
    y_val = y_true[idx_val]
    y_test = y_true[idx_test]
    
    residual_normal_val = residual_val[y_val == 0]
    residual_attack_val = residual_val[y_val == 1]
    residual_normal_test = residual_test[y_test == 0]
    residual_attack_test = residual_test[y_test == 1]
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"\nValidation Set ({len(idx_val)} samples, {y_val.sum()} attacks):")
    print(f"  Normal - Mean: {np.mean(residual_normal_val):.6f}, Std: {np.std(residual_normal_val):.6f}, "
          f"Median: {np.median(residual_normal_val):.6f}")
    print(f"  Attack - Mean: {np.mean(residual_attack_val):.6f}, Std: {np.std(residual_attack_val):.6f}, "
          f"Median: {np.median(residual_attack_val):.6f}")
    if len(residual_attack_val) > 0:
        signal_ratio = np.median(residual_attack_val) / np.median(residual_normal_val) if np.median(residual_normal_val) > 0 else 0
        print(f"  Signal Ratio (Attack/Normal Median): {signal_ratio:.2f}x")
    
    print(f"\nTest Set ({len(idx_test)} samples, {y_test.sum()} attacks):")
    print(f"  Normal - Mean: {np.mean(residual_normal_test):.6f}, Std: {np.std(residual_normal_test):.6f}, "
          f"Median: {np.median(residual_normal_test):.6f}")
    print(f"  Attack - Mean: {np.mean(residual_attack_test):.6f}, Std: {np.std(residual_attack_test):.6f}, "
          f"Median: {np.median(residual_attack_test):.6f}")
    if len(residual_attack_test) > 0:
        signal_ratio = np.median(residual_attack_test) / np.median(residual_normal_test) if np.median(residual_normal_test) > 0 else 0
        print(f"  Signal Ratio (Attack/Normal Median): {signal_ratio:.2f}x")
    
    return {
        'val_normal': residual_normal_val,
        'val_attack': residual_attack_val,
        'test_normal': residual_normal_test,
        'test_attack': residual_attack_test
    }


def analyze_threshold_sensitivity(residual, y_true, idx_val, idx_test, title="Threshold Analysis"):
    """分析阈值选择的敏感性"""
    res_val = residual[idx_val]
    y_val = y_true[idx_val]
    res_test = residual[idx_test]
    y_test = y_true[idx_test]
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # 动态阈值（当前方法）
    percentiles = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    print(f"\nDynamic Threshold (基于Validation Residual分位数):")
    print(f"  Percentile Range | Threshold | Val F1 | Test F1 | Test AUPRC")
    print(f"  {'-'*60}")
    
    best_val_f1 = 0.0
    best_test_f1 = 0.0
    best_th = None
    
    for p_low, p_high in zip(percentiles[:-1], percentiles[1:]):
        ths = np.linspace(np.percentile(res_val, p_low), np.percentile(res_val, p_high), 25)
        val_f1s = [f1_score(y_val, (res_val >= t).astype(int), zero_division=0) for t in ths]
        test_f1s = [f1_score(y_test, (res_test >= t).astype(int), zero_division=0) for t in ths]
        
        max_val_f1_idx = np.argmax(val_f1s)
        max_val_f1 = val_f1s[max_val_f1_idx]
        best_th_for_range = ths[max_val_f1_idx]
        test_f1_at_best = test_f1s[max_val_f1_idx]
        
        test_auprc = average_precision_score(y_test, res_test)
        
        print(f"  {p_low:3d}-{p_high:3d}%        | {best_th_for_range:8.4f} | {max_val_f1:6.4f} | {test_f1_at_best:7.4f} | {test_auprc:.4f}")
        
        if max_val_f1 > best_val_f1:
            best_val_f1 = max_val_f1
            best_test_f1 = test_f1_at_best
            best_th = best_th_for_range
    
    print(f"\n  Best Val F1: {best_val_f1:.4f} at threshold {best_th:.4f}")
    print(f"  Test F1 at Best Val Threshold: {best_test_f1:.4f}")
    
    # 固定阈值（基于训练集）
    print(f"\nFixed Threshold (基于固定分位数):")
    fixed_percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
    print(f"  Percentile | Threshold | Val F1 | Test F1")
    print(f"  {'-'*50}")
    for p in fixed_percentiles:
        th = np.percentile(res_val, p)
        val_f1 = f1_score(y_val, (res_val >= th).astype(int), zero_division=0)
        test_f1 = f1_score(y_test, (res_test >= th).astype(int), zero_division=0)
        print(f"  {p:3d}%      | {th:8.4f} | {val_f1:6.4f} | {test_f1:7.4f}")


def compare_aggregation_methods(model_path, bundle_coarse, bundle_fine=None, device='cuda'):
    """对比不同的residual聚合方式"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    data_coarse = to_device(bundle_coarse, device)
    
    from model.encoder import TransformerSeqEncoder, RegressionHeadWithRelation
    
    coarse_only = config.get('coarse_only', False)
    
    enc_coarse = TransformerSeqEncoder(
        input_dim=data_coarse['X_seq'].shape[2],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=data_coarse['X_seq'].shape[1] + 5
    ).to(device)
    enc_coarse.pooling = config['pooling']
    
    head = RegressionHeadWithRelation(config['d_model'], data_coarse['X_seq'].shape[2]).to(device)
    
    enc_coarse.load_state_dict(checkpoint['enc_coarse'])
    head.load_state_dict(checkpoint['head'])
    
    if not coarse_only and bundle_fine is not None:
        data_fine = to_device(bundle_fine, device)
        enc_fine = TransformerSeqEncoder(
            input_dim=data_fine['X_seq'].shape[2],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_seq_len=data_fine['X_seq'].shape[1] + 5
        ).to(device)
        enc_fine.pooling = config['pooling']
        enc_fine.load_state_dict(checkpoint['enc_fine'])
    
    enc_coarse.eval()
    head.eval()
    
    idx_val_coarse_np = data_coarse['idx_val'].cpu().numpy() if isinstance(data_coarse['idx_val'], torch.Tensor) else data_coarse['idx_val']
    idx_test_coarse_np = data_coarse['idx_test'].cpu().numpy() if isinstance(data_coarse['idx_test'], torch.Tensor) else data_coarse['idx_test']
    y_val_coarse = data_coarse['y_next'][data_coarse['idx_val']].cpu().numpy() if isinstance(data_coarse['y_next'], torch.Tensor) else data_coarse['y_next'][idx_val_coarse_np]
    y_test_coarse = data_coarse['y_next'][data_coarse['idx_test']].cpu().numpy() if isinstance(data_coarse['y_next'], torch.Tensor) else data_coarse['y_next'][idx_test_coarse_np]
    
    print(f"\n{'='*60}")
    print("Residual Aggregation Comparison")
    print(f"{'='*60}")
    
    with torch.no_grad():
        if not coarse_only and bundle_fine is not None:
            enc_fine.eval()
            z_fine = enc_fine(data_fine['X_seq'])
            z_coarse = enc_coarse(data_coarse['X_seq'])
            pred_fine_all, pred_coarse_all = head(z_fine, z_coarse)
        else:
            z_coarse = enc_coarse(data_coarse['X_seq'])
            _, pred_coarse_all = head(z_coarse, z_coarse)
        
        # Fine scale residual (only for multi-scale)
        if not coarse_only and bundle_fine is not None:
            residual_vec_fine = (data_fine['X_next'] - pred_fine_all).abs()
            residual_fine_mean = residual_vec_fine.mean(dim=1).cpu().numpy()  # Mean over features
            residual_fine_max = residual_vec_fine.max(dim=1)[0].cpu().numpy()  # Max over features
            residual_fine_median = residual_vec_fine.median(dim=1)[0].cpu().numpy()  # Median over features
            residual_fine_norm = torch.norm(residual_vec_fine, dim=1).cpu().numpy()  # L2 norm
        
        # Coarse scale residual
        residual_vec_coarse = (data_coarse['X_next'] - pred_coarse_all).abs()
        residual_coarse_mean = residual_vec_coarse.mean(dim=1).cpu().numpy()
        residual_coarse_max = residual_vec_coarse.max(dim=1)[0].cpu().numpy()
        residual_coarse_median = residual_vec_coarse.median(dim=1)[0].cpu().numpy()
        residual_coarse_norm = torch.norm(residual_vec_coarse, dim=1).cpu().numpy()
        
        # Evaluate on coarse scale
        print(f"\nCoarse Scale (Hour) - Validation Set:")
        print(f"  Method      | Normal Median | Attack Median | Signal | Val F1 | Test F1 | Test AUPRC")
        print(f"  {'-'*75}")
        
        methods_coarse = {
            'Mean': residual_coarse_mean,
            'Max': residual_coarse_max,
            'Median': residual_coarse_median,
            'L2 Norm': residual_coarse_norm
        }
        
        for name, res in methods_coarse.items():
            res_val = res[idx_val_coarse_np]
            res_test = res[idx_test_coarse_np]
            
            normal_median = np.median(res_val[y_val_coarse == 0])
            attack_median = np.median(res_val[y_val_coarse == 1]) if y_val_coarse.sum() > 0 else 0
            signal = attack_median / normal_median if normal_median > 0 else 0
            
            # Find best threshold
            ths = np.linspace(np.percentile(res_val, 10), np.percentile(res_val, 95), 25)
            val_f1s = [f1_score(y_val_coarse, (res_val >= t).astype(int), zero_division=0) for t in ths]
            best_idx = np.argmax(val_f1s)
            best_th = ths[best_idx]
            best_val_f1 = val_f1s[best_idx]
            
            test_f1 = f1_score(y_test_coarse, (res_test >= best_th).astype(int), zero_division=0)
            test_auprc = average_precision_score(y_test_coarse, res_test)
            
            print(f"  {name:10s} | {normal_median:13.6f} | {attack_median:13.6f} | {signal:6.2f}x | {best_val_f1:6.4f} | {test_f1:7.4f} | {test_auprc:9.4f}")
        
        # Evaluate on fine scale (if we aggregate fine samples to coarse)
        if not coarse_only and bundle_fine is not None:
            print(f"\nFine Scale (Minute) - Aggregated to Coarse Level:")
            mapping = data_fine.get('fine_to_coarse_index', None)
            if mapping is not None:
                if isinstance(mapping, torch.Tensor):
                    mapping = mapping.cpu().numpy()
                else:
                    mapping = np.array(mapping)
                
                methods_fine = {
                    'Mean': residual_fine_mean,
                    'Max': residual_fine_max,
                    'Median': residual_fine_median,
                    'L2 Norm': residual_fine_norm
                }
                
                for name, res_fine in methods_fine.items():
                    # Aggregate fine to coarse
                    res_coarse_agg = np.zeros(len(residual_coarse_mean))
                    for c_idx in range(len(res_coarse_agg)):
                        fine_indices = np.where(mapping == c_idx)[0]
                        if len(fine_indices) > 0:
                            res_coarse_agg[c_idx] = res_fine[fine_indices].mean()
                        else:
                            res_coarse_agg[c_idx] = residual_coarse_mean[c_idx]
                    
                    res_val = res_coarse_agg[idx_val_coarse_np]
                    res_test = res_coarse_agg[idx_test_coarse_np]
                    
                    normal_median = np.median(res_val[y_val_coarse == 0])
                    attack_median = np.median(res_val[y_val_coarse == 1]) if y_val_coarse.sum() > 0 else 0
                    signal = attack_median / normal_median if normal_median > 0 else 0
                    
                    ths = np.linspace(np.percentile(res_val, 10), np.percentile(res_val, 95), 25)
                    val_f1s = [f1_score(y_val_coarse, (res_val >= t).astype(int), zero_division=0) for t in ths]
                    best_idx = np.argmax(val_f1s)
                    best_th = ths[best_idx]
                    best_val_f1 = val_f1s[best_idx]
                    
                    test_f1 = f1_score(y_test_coarse, (res_test >= best_th).astype(int), zero_division=0)
                    test_auprc = average_precision_score(y_test_coarse, res_test)
                    
                    print(f"  {name:10s} | {normal_median:13.6f} | {attack_median:13.6f} | {signal:6.2f}x | {best_val_f1:6.4f} | {test_f1:7.4f} | {test_auprc:9.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/SWaT/processed')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    bundle_hour = torch.load(data_dir / 'swat_hour.pt', weights_only=False)
    bundle_minute = torch.load(data_dir / 'swat_minute.pt', weights_only=False)
    
    print("="*60)
    print("Residual Quality Analysis")
    print("="*60)
    print(f"\nModel: {args.model_path}")
    print(f"Hour scale: {bundle_hour['X_seq'].shape}")
    print(f"Minute scale: {bundle_minute['X_seq'].shape}")
    
    # 检查是否是coarse-only模型
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    coarse_only = checkpoint['config'].get('coarse_only', False)
    
    # 1. 使用infer获取residual
    if coarse_only:
        result = infer(args.model_path, bundle_hour, bundle_fine=None, device=args.device)
    else:
        result = infer(args.model_path, bundle_hour, bundle_minute, device=args.device)
    residual = result['residual']
    y_true = result['y_true']
    idx_val = result['idx_val']
    idx_test = result['idx_test']
    
    # 2. 分析residual分布
    dist_info = analyze_residual_distribution(
        residual, y_true, idx_val, idx_test,
        title="Current Residual (Coarse Scale, Mean Aggregation)"
    )
    
    # 3. 分析阈值敏感性
    analyze_threshold_sensitivity(
        residual, y_true, idx_val, idx_test,
        title="Threshold Selection Analysis"
    )
    
    # 4. 对比不同聚合方式
    if coarse_only:
        compare_aggregation_methods(
            args.model_path, bundle_hour, bundle_fine=None, device=args.device
        )
    else:
        compare_aggregation_methods(
            args.model_path, bundle_hour, bundle_minute, device=args.device
        )
    
    print(f"\n{'='*60}")
    print("Analysis Complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

