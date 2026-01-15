#!/usr/bin/env python3
"""
对比旧6.py和当前标签生成方式的差异
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_utils import loader, engineer_features
from utils.labels import create_outbreak_labels

def old_label_method_6py(df, threshold_percentile=90):
    """
    旧6.py的标签生成方法（简单百分位数，全局处理）
    """
    df_label = df.copy()
    
    if 'NewDeaths_return' in df_label.columns:
        returns = df_label['NewDeaths_return'].values
        # ⚠️ 这里是全局处理，不是按state单独处理！
        threshold = np.percentile(returns[returns > 0], threshold_percentile)
        df_label['outbreak_label'] = (returns > threshold).astype(int)
    else:
        raise ValueError("NewDeaths_return column not found!")
    
    return df_label, threshold

def current_label_method(df):
    """
    当前的标签生成方法（混合方法，按state单独处理）
    """
    return create_outbreak_labels(df)

def compare_label_methods():
    """
    对比两种标签生成方法
    """
    print("="*70)
    print("对比标签生成方法")
    print("="*70)
    
    # 加载数据
    data_path = project_root / "data/3.1_3.2_Final_Dataset_State_Level.xlsx"
    if not data_path.exists():
        print(f"数据文件不存在: {data_path}")
        return
    
    print("\n1. 加载数据...")
    df = loader(str(data_path))
    print(f"   原始数据: {len(df)} 行")
    
    print("\n2. 特征工程...")
    df = engineer_features(df)
    print(f"   特征工程后: {len(df)} 行")
    
    print("\n3. 使用旧6.py方法生成标签（全局百分位数）...")
    df_old, threshold_old = old_label_method_6py(df, threshold_percentile=90)
    old_pos = df_old['outbreak_label'].sum()
    old_total = len(df_old)
    print(f"   阈值: {threshold_old:.6f}")
    print(f"   正样本: {old_pos} ({100*old_pos/old_total:.2f}%)")
    
    print("\n4. 使用当前方法生成标签（按state混合方法）...")
    df_current = current_label_method(df)
    current_pos = df_current['outbreak_label'].sum()
    current_total = len(df_current)
    print(f"   正样本: {current_pos} ({100*current_pos/current_total:.2f}%)")
    
    print("\n5. 按state对比...")
    print(f"{'State':<10} {'旧方法':<10} {'当前方法':<10} {'差异':<10}")
    print("-" * 50)
    
    states = sorted(df['State'].unique())
    for state in states[:10]:  # 只显示前10个
        old_state_pos = df_old[df_old['State'] == state]['outbreak_label'].sum()
        current_state_pos = df_current[df_current['State'] == state]['outbreak_label'].sum()
        diff = current_state_pos - old_state_pos
        print(f"{state:<10} {old_state_pos:<10} {current_state_pos:<10} {diff:<10}")
    
    print("\n6. 关键差异分析:")
    print(f"   旧方法（全局）: 所有state使用同一个阈值 {threshold_old:.6f}")
    print(f"   当前方法（按state）: 每个state单独计算阈值和模式")
    print(f"   差异: 当前方法正样本数 {'增加' if current_pos > old_pos else '减少'} {abs(current_pos - old_pos)} 个")
    
    # 检查Week聚合后的标签
    print("\n7. Week数据聚合后的标签对比...")
    from utils.data_utils import group_by_state, agg_map
    
    # 旧方法的week数据
    week_data_old = group_by_state(df_old, freq='W', agg_map=agg_map)
    old_week_pos = sum(df['outbreak_label'].sum() for df in week_data_old.values())
    old_week_total = sum(len(df) for df in week_data_old.values())
    
    # 当前方法的week数据
    week_data_current = group_by_state(df_current, freq='W', agg_map=agg_map)
    current_week_pos = sum(df['outbreak_label'].sum() for df in week_data_current.values())
    current_week_total = sum(len(df) for df in week_data_current.values())
    
    print(f"   旧方法Week正样本: {old_week_pos} / {old_week_total} ({100*old_week_pos/old_week_total:.2f}%)")
    print(f"   当前方法Week正样本: {current_week_pos} / {current_week_total} ({100*current_week_pos/current_week_total:.2f}%)")
    
    return df_old, df_current

if __name__ == "__main__":
    compare_label_methods()

















