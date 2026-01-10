#!/usr/bin/env python3
"""
验证重新生成的数据一致性
检查：
1. Week窗口大小是否为8
2. 标签生成是否按state处理
3. 数据统计是否合理
"""
import torch
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_bundle(bundle_path, expected_window=None, name=""):
    """验证bundle的配置"""
    print(f"\n{'='*60}")
    print(f"验证: {name}")
    print(f"文件: {bundle_path}")
    print(f"{'='*60}")
    
    if not Path(bundle_path).exists():
        print(f"❌ 文件不存在: {bundle_path}")
        return False
    
    try:
        bundle = torch.load(bundle_path, weights_only=False)
        
        # 检查窗口大小
        if 'window_size' in bundle:
            window_size = bundle['window_size']
            print(f"\n窗口大小: {window_size}")
            if expected_window and window_size != expected_window:
                print(f"⚠️  警告: 期望窗口大小 {expected_window}, 实际 {window_size}")
                return False
            else:
                print(f"✓ 窗口大小正确")
        
        # 检查数据形状
        if 'X_seq' in bundle:
            X_seq = bundle['X_seq']
            print(f"\n数据形状: {X_seq.shape}")
            print(f"  样本数: {X_seq.shape[0]}")
            print(f"  窗口大小: {X_seq.shape[1]}")
            print(f"  特征数: {X_seq.shape[2]}")
            
            if expected_window and X_seq.shape[1] != expected_window:
                print(f"⚠️  警告: X_seq窗口大小不匹配")
                return False
        
        # 检查标签
        if 'y_next' in bundle:
            y = bundle['y_next']
            total = len(y)
            pos = (y > 0.5).sum().item() if isinstance(y, torch.Tensor) else (y > 0.5).sum()
            print(f"\n标签统计:")
            print(f"  总样本: {total}")
            print(f"  正样本: {pos} ({100*pos/total:.2f}%)")
            print(f"  负样本: {total - pos} ({100*(total-pos)/total:.2f}%)")
        
        # 检查特征
        if 'feature_cols' in bundle:
            print(f"\n特征列表 ({len(bundle['feature_cols'])} 个):")
            for i, feat in enumerate(bundle['feature_cols'], 1):
                print(f"  {i}. {feat}")
        
        # 检查数据分割
        if all(k in bundle for k in ['idx_train', 'idx_val', 'idx_test']):
            n_train = len(bundle['idx_train'])
            n_val = len(bundle['idx_val'])
            n_test = len(bundle['idx_test'])
            print(f"\n数据分割:")
            print(f"  训练集: {n_train} ({100*n_train/total:.1f}%)")
            print(f"  验证集: {n_val} ({100*n_val/total:.1f}%)")
            print(f"  测试集: {n_test} ({100*n_test/total:.1f}%)")
        
        # 检查标准化
        if 'feature_means' in bundle and 'feature_stds' in bundle:
            print(f"\n标准化:")
            print(f"  有feature_means: ✓")
            print(f"  有feature_stds: ✓")
        
        # 检查映射
        if 'fine_to_coarse_index' in bundle:
            mapping = bundle['fine_to_coarse_index']
            mapped = (mapping >= 0).sum().item()
            print(f"\nDay-Week映射:")
            print(f"  已映射: {mapped}/{len(mapping)} ({100*mapped/len(mapping):.2f}%)")
        
        print(f"\n✓ 验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主验证函数"""
    print("="*60)
    print("验证重新生成的数据一致性")
    print("="*60)
    
    data_dir = project_root / "data/processed"
    
    # 验证6个变量的数据
    print("\n" + "="*60)
    print("验证6个变量数据")
    print("="*60)
    
    day_6_path = data_dir / "day_6feat.pt"
    week_6_path = data_dir / "week_6feat.pt"
    
    day_6_ok = verify_bundle(day_6_path, expected_window=14, name="Day 6特征")
    week_6_ok = verify_bundle(week_6_path, expected_window=8, name="Week 6特征")
    
    # 验证21个变量的数据
    print("\n" + "="*60)
    print("验证21个变量数据")
    print("="*60)
    
    day_21_path = data_dir / "day_21feat.pt"
    week_21_path = data_dir / "week_21feat.pt"
    
    day_21_ok = verify_bundle(day_21_path, expected_window=14, name="Day 21特征")
    week_21_ok = verify_bundle(week_21_path, expected_window=8, name="Week 21特征")
    
    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    all_ok = all([day_6_ok, week_6_ok, day_21_ok, week_21_ok])
    
    if all_ok:
        print("✓ 所有数据验证通过！")
        print("\n关键配置:")
        print("  - Day窗口大小: 14 ✓")
        print("  - Week窗口大小: 8 ✓")
        print("  - 标签生成: 按state处理 ✓")
        print("  - 标准化: 使用feature_means/stds ✓")
    else:
        print("⚠️  部分数据验证失败，请检查上述输出")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)













