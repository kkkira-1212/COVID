#!/usr/bin/env python3
"""
根据Residual分析Label，找出可能需要调整的样本
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.evaluator import infer


def analyze_label_by_residual(model_path, data_dir, coarse_file, fine_file=None, device='cuda', output_file=None):
    """根据residual分析label，找出可疑样本"""
    
    # 加载数据
    data_dir = Path(project_root) / data_dir
    bundle_coarse = torch.load(data_dir / coarse_file, weights_only=False, map_location='cpu')
    
    bundle_fine = None
    if fine_file and (data_dir / fine_file).exists():
        bundle_fine = torch.load(data_dir / fine_file, weights_only=False, map_location='cpu')
    
    # 推理获取residual
    print("Running inference...")
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
    meta = bundle_coarse['meta']
    
    # 获取test set的数据
    residual_test = residual[idx_test]
    y_test = y_true[idx_test]
    meta_test = meta.iloc[idx_test].copy()
    meta_test['residual'] = residual_test
    meta_test['y_true'] = y_test
    
    # 计算统计量
    normal_indices = y_test == 0
    anomaly_indices = y_test == 1
    
    residual_normal = residual_test[normal_indices]
    residual_anomaly = residual_test[anomaly_indices]
    
    normal_median = np.median(residual_normal)
    anomaly_median = np.median(residual_anomaly)
    normal_q95 = np.percentile(residual_normal, 95)
    anomaly_q5 = np.percentile(residual_anomaly, 5)
    
    print("\n" + "="*60)
    print("Residual Statistics (Test Set)")
    print("="*60)
    print(f"Normal - Median: {normal_median:.6f}, Q95: {normal_q95:.6f}")
    print(f"Anomaly - Median: {anomaly_median:.6f}, Q5: {anomaly_q5:.6f}")
    print(f"Signal Ratio: {anomaly_median/normal_median:.2f}x")
    
    # 找出可疑样本
    print("\n" + "="*60)
    print("可疑样本分析")
    print("="*60)
    
    # 1. Residual小的Anomaly（可能不应该标记为anomaly）
    suspicious_anomalies = meta_test[
        (meta_test['y_true'] == 1) & 
        (meta_test['residual'] < normal_median)
    ].copy()
    
    print(f"\n1. Residual小的Anomaly (residual < normal_median={normal_median:.6f})")
    print(f"   数量: {len(suspicious_anomalies)}")
    if len(suspicious_anomalies) > 0:
        print(f"   Residual范围: [{suspicious_anomalies['residual'].min():.6f}, {suspicious_anomalies['residual'].max():.6f}]")
        print(f"   这些样本的residual比normal还小，可能需要重新检查label")
        print(f"\n   前10个样本:")
        for idx, row in suspicious_anomalies.head(10).iterrows():
            print(f"     {row['state']} | {row['target_date']} | residual={row['residual']:.6f}")
    
    # 2. Residual大的Normal（可能应该标记为anomaly）
    suspicious_normals = meta_test[
        (meta_test['y_true'] == 0) & 
        (meta_test['residual'] > anomaly_median)
    ].copy()
    
    print(f"\n2. Residual大的Normal (residual > anomaly_median={anomaly_median:.6f})")
    print(f"   数量: {len(suspicious_normals)}")
    if len(suspicious_normals) > 0:
        print(f"   Residual范围: [{suspicious_normals['residual'].min():.6f}, {suspicious_normals['residual'].max():.6f}]")
        print(f"   这些样本的residual比anomaly还大，可能需要标记为anomaly")
        print(f"\n   前10个样本:")
        for idx, row in suspicious_normals.head(10).iterrows():
            print(f"     {row['state']} | {row['target_date']} | residual={row['residual']:.6f}")
    
    # 3. 边界情况（residual在normal和anomaly之间）
    boundary_cases = meta_test[
        (meta_test['residual'] >= normal_q95) & 
        (meta_test['residual'] <= anomaly_q5)
    ].copy()
    
    print(f"\n3. 边界情况 (normal_q95 < residual < anomaly_q5)")
    print(f"   数量: {len(boundary_cases)}")
    if len(boundary_cases) > 0:
        normal_boundary = boundary_cases[boundary_cases['y_true'] == 0]
        anomaly_boundary = boundary_cases[boundary_cases['y_true'] == 1]
        print(f"   Normal: {len(normal_boundary)}, Anomaly: {len(anomaly_boundary)}")
        print(f"   这些样本的residual在边界，label定义可能需要更清晰")
    
    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'suspicious_anomalies': suspicious_anomalies,
            'suspicious_normals': suspicious_normals,
            'boundary_cases': boundary_cases,
            'stats': {
                'normal_median': normal_median,
                'anomaly_median': anomaly_median,
                'signal_ratio': anomaly_median / normal_median,
            }
        }
        
        # 保存为CSV
        if len(suspicious_anomalies) > 0:
            suspicious_anomalies.to_csv(output_path.parent / 'suspicious_anomalies.csv', index=False)
        if len(suspicious_normals) > 0:
            suspicious_normals.to_csv(output_path.parent / 'suspicious_normals.csv', index=False)
        if len(boundary_cases) > 0:
            boundary_cases.to_csv(output_path.parent / 'boundary_cases.csv', index=False)
        
        print(f"\n结果已保存到: {output_path.parent}")
    
    # 建议
    print("\n" + "="*60)
    print("调整建议")
    print("="*60)
    
    if len(suspicious_anomalies) > len(anomaly_indices) * 0.3:
        print(f"\n⚠ 警告: {len(suspicious_anomalies)}/{len(anomaly_indices)} ({len(suspicious_anomalies)/len(anomaly_indices)*100:.1f}%) 的anomaly样本residual很小")
        print("   建议: 检查anomaly定义规则，可能需要更严格的标准")
    
    if len(suspicious_normals) > len(normal_indices) * 0.1:
        print(f"\n⚠ 警告: {len(suspicious_normals)}/{len(normal_indices)} ({len(suspicious_normals)/len(normal_indices)*100:.1f}%) 的normal样本residual很大")
        print("   建议: 检查这些normal样本，可能需要标记为anomaly")
    
    if anomaly_median / normal_median < 1.2:
        print(f"\n⚠ 警告: Signal Ratio ({anomaly_median/normal_median:.2f}x) 较低")
        print("   建议: 考虑调整anomaly定义，或者训练时过滤掉anomaly样本")
    
    return {
        'suspicious_anomalies': suspicious_anomalies,
        'suspicious_normals': suspicious_normals,
        'boundary_cases': boundary_cases,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze label by residual')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--coarse_file', type=str, default='week_21feat.pt', help='Coarse data file')
    parser.add_argument('--fine_file', type=str, default=None, help='Fine data file (optional)')
    parser.add_argument('--output_file', type=str, default='analysis/label_analysis_results.csv', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    analyze_label_by_residual(
        model_path=args.model_path,
        data_dir=args.data_dir,
        coarse_file=args.coarse_file,
        fine_file=args.fine_file,
        device=args.device,
        output_file=args.output_file
    )


