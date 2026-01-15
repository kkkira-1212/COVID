#!/usr/bin/env python3
"""
对比旧数据 (sequences_day.pt, sequences_week.pt) 和新数据 (data/processed/*.pt) 的差异
"""
import torch
import pandas as pd
from pathlib import Path

def load_and_summarize(filepath, name):
    """加载并总结文件信息"""
    if not Path(filepath).exists():
        return None
    
    data = torch.load(filepath, weights_only=False)
    
    if not isinstance(data, dict):
        return None
    
    summary = {
        'name': name,
        'filepath': filepath,
        'keys': list(data.keys()),
        'has_X_next': 'X_next' in data,
        'has_scaler': 'scaler' in data,
        'has_feature_means': 'feature_means' in data,
        'has_mapping': 'fine_to_coarse_index' in data or 'day_to_week_index' in data,
    }
    
    if 'X_seq' in data:
        summary['X_seq_shape'] = data['X_seq'].shape
        summary['n_samples'] = data['X_seq'].shape[0]
        summary['window_size'] = data['X_seq'].shape[1]
        summary['n_features'] = data['X_seq'].shape[2]
    
    if 'X_next' in data:
        summary['X_next_shape'] = data['X_next'].shape
    
    if 'y_next' in data:
        y = data['y_next']
        total = len(y)
        pos = (y > 0.5).sum().item() if isinstance(y, torch.Tensor) else (y > 0.5).sum()
        summary['y_total'] = total
        summary['y_positive'] = pos
        summary['y_positive_pct'] = 100 * pos / total if total > 0 else 0
    
    if 'window_size' in data:
        summary['window_size_from_key'] = data['window_size']
    
    if 'feature_cols' in data:
        summary['feature_cols'] = data['feature_cols']
        summary['n_features_from_key'] = len(data['feature_cols'])
    
    if 'idx_train' in data:
        summary['n_train'] = len(data['idx_train'])
    if 'idx_val' in data:
        summary['n_val'] = len(data['idx_val'])
    if 'idx_test' in data:
        summary['n_test'] = len(data['idx_test'])
    
    if 'states_order' in data:
        summary['n_states'] = len(data['states_order'])
    
    # 检查数据统计
    if 'X_seq' in data:
        X = data['X_seq']
        summary['X_seq_min'] = X.min().item()
        summary['X_seq_max'] = X.max().item()
        summary['X_seq_mean'] = X.mean().item()
        summary['X_seq_std'] = X.std().item()
        summary['X_seq_nan'] = torch.isnan(X).sum().item()
        summary['X_seq_inf'] = torch.isinf(X).sum().item()
    
    return summary

def print_summary(summary):
    """打印总结信息"""
    if summary is None:
        print("文件不存在或无法加载")
        return
    
    print(f"\n{'='*60}")
    print(f"{summary['name']}")
    print(f"{'='*60}")
    print(f"文件路径: {summary['filepath']}")
    print(f"\n基本统计:")
    print(f"  样本数: {summary.get('n_samples', 'N/A')}")
    print(f"  窗口大小: {summary.get('window_size', 'N/A')}")
    print(f"  特征数: {summary.get('n_features', 'N/A')}")
    
    if 'y_total' in summary:
        print(f"\n标签分布:")
        print(f"  总样本: {summary['y_total']}")
        print(f"  正样本: {summary['y_positive']} ({summary['y_positive_pct']:.2f}%)")
        print(f"  负样本: {summary['y_total'] - summary['y_positive']} ({100 - summary['y_positive_pct']:.2f}%)")
    
    if 'n_train' in summary:
        print(f"\n数据分割:")
        print(f"  训练集: {summary['n_train']}")
        print(f"  验证集: {summary['n_val']}")
        print(f"  测试集: {summary['n_test']}")
        total = summary['n_train'] + summary['n_val'] + summary['n_test']
        print(f"  总计: {total}")
    
    print(f"\n数据结构:")
    print(f"  有X_next: {summary['has_X_next']}")
    print(f"  有scaler: {summary['has_scaler']}")
    print(f"  有feature_means: {summary['has_feature_means']}")
    print(f"  有mapping: {summary['has_mapping']}")
    
    if 'X_seq_min' in summary:
        print(f"\nX_seq统计:")
        print(f"  最小值: {summary['X_seq_min']:.6f}")
        print(f"  最大值: {summary['X_seq_max']:.6f}")
        print(f"  均值: {summary['X_seq_mean']:.6f}")
        print(f"  标准差: {summary['X_seq_std']:.6f}")
        print(f"  NaN数量: {summary['X_seq_nan']}")
        print(f"  Inf数量: {summary['X_seq_inf']}")
    
    if 'feature_cols' in summary:
        print(f"\n特征列表:")
        for i, feat in enumerate(summary['feature_cols'], 1):
            print(f"  {i}. {feat}")

def compare_summaries(old, new, name):
    """对比两个总结"""
    print(f"\n{'='*60}")
    print(f"对比: {name}")
    print(f"{'='*60}")
    
    if old is None or new is None:
        print("无法对比：缺少数据")
        return
    
    issues = []
    
    # 样本数对比
    if old.get('n_samples') != new.get('n_samples'):
        issues.append(f"⚠️  样本数不一致: 旧={old.get('n_samples')}, 新={new.get('n_samples')}")
    else:
        print(f"✓ 样本数一致: {old.get('n_samples')}")
    
    # 窗口大小对比
    if old.get('window_size') != new.get('window_size'):
        issues.append(f"⚠️  窗口大小不一致: 旧={old.get('window_size')}, 新={new.get('window_size')}")
    else:
        print(f"✓ 窗口大小一致: {old.get('window_size')}")
    
    # 特征数对比
    if old.get('n_features') != new.get('n_features'):
        issues.append(f"⚠️  特征数不一致: 旧={old.get('n_features')}, 新={new.get('n_features')}")
    else:
        print(f"✓ 特征数一致: {old.get('n_features')}")
    
    # 正样本数对比
    old_pos = old.get('y_positive', 0)
    new_pos = new.get('y_positive', 0)
    if old_pos != new_pos:
        issues.append(f"⚠️  正样本数不一致: 旧={old_pos} ({old.get('y_positive_pct', 0):.2f}%), 新={new_pos} ({new.get('y_positive_pct', 0):.2f}%)")
        print(f"⚠️  正样本数变化: {old_pos} → {new_pos} (变化: {new_pos - old_pos})")
    else:
        print(f"✓ 正样本数一致: {old_pos}")
    
    # 数据结构对比
    if old.get('has_X_next') != new.get('has_X_next'):
        issues.append(f"⚠️  X_next字段不一致: 旧={old.get('has_X_next')}, 新={new.get('has_X_next')}")
    
    if old.get('has_scaler') != new.get('has_scaler'):
        issues.append(f"⚠️  scaler字段不一致: 旧={old.get('has_scaler')}, 新={new.get('has_scaler')}")
    
    if old.get('has_feature_means') != new.get('has_feature_means'):
        issues.append(f"⚠️  feature_means字段不一致: 旧={old.get('has_feature_means')}, 新={new.get('has_feature_means')}")
    
    if issues:
        print(f"\n发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✓ 未发现明显问题")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    
    # 加载旧数据
    old_day = load_and_summarize(project_root / "sequences_day.pt", "旧数据 - sequences_day.pt")
    old_week = load_and_summarize(project_root / "sequences_week.pt", "旧数据 - sequences_week.pt")
    
    # 加载新数据
    new_day = load_and_summarize(project_root / "data/processed/day_6feat.pt", "新数据 - day_6feat.pt")
    new_week = load_and_summarize(project_root / "data/processed/week_6feat.pt", "新数据 - week_6feat.pt")
    
    # 打印总结
    if old_day:
        print_summary(old_day)
    if old_week:
        print_summary(old_week)
    if new_day:
        print_summary(new_day)
    if new_week:
        print_summary(new_week)
    
    # 对比
    if old_day and new_day:
        compare_summaries(old_day, new_day, "Day数据对比")
    if old_week and new_week:
        compare_summaries(old_week, new_week, "Week数据对比")

















