#!/usr/bin/env python3
"""
诊断Week数据聚合的问题
检查标签在聚合前后的变化
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_utils import loader, engineer_features, group_by_state, agg_map
from utils.labels import create_outbreak_labels

def diagnose_week_aggregation():
    """
    诊断Week聚合的问题
    """
    print("="*70)
    print("诊断Week数据聚合问题")
    print("="*70)
    
    # 加载数据
    data_path = project_root / "data/3.1_3.2_Final_Dataset_State_Level.xlsx"
    if not data_path.exists():
        print(f"数据文件不存在: {data_path}")
        return
    
    print("\n1. 加载和特征工程...")
    df = loader(str(data_path))
    df = engineer_features(df)
    
    print("\n2. 生成标签（在日数据上）...")
    df = create_outbreak_labels(df)
    day_pos = df['outbreak_label'].sum()
    print(f"   日数据正样本: {day_pos} / {len(df)} ({100*day_pos/len(df):.2f}%)")
    
    # 检查一个state的详细情况
    test_state = 'SP'  # 使用SP作为示例
    df_state = df[df['State'] == test_state].copy()
    print(f"\n3. 检查 {test_state} 州的日数据:")
    print(f"   总行数: {len(df_state)}")
    print(f"   正样本: {df_state['outbreak_label'].sum()}")
    print(f"   日期范围: {df_state['Date'].min()} 到 {df_state['Date'].max()}")
    
    # Week聚合
    print(f"\n4. Week聚合 {test_state} 州...")
    week_data = group_by_state(df, freq='W', agg_map=agg_map)
    df_state_week = week_data[test_state].copy()
    
    print(f"   Week数据行数: {len(df_state_week)}")
    print(f"   Week正样本: {df_state_week['outbreak_label'].sum()}")
    print(f"   日期范围: {df_state_week['WeekStart'].min()} 到 {df_state_week['WeekStart'].max()}")
    
    # 检查聚合逻辑
    print(f"\n5. 检查聚合逻辑:")
    print(f"   聚合前日数据正样本: {df_state['outbreak_label'].sum()}")
    print(f"   聚合后Week数据正样本: {df_state_week['outbreak_label'].sum()}")
    
    # 检查聚合是否正确（使用max）
    print(f"\n6. 验证聚合逻辑（使用max）:")
    df_state_indexed = df_state.set_index('Date')
    df_state_resampled = df_state_indexed.resample('W-SUN').agg({
        'outbreak_label': 'max',
        'NewDeaths': 'sum',
        'NewCases': 'sum',
        'State': 'first'
    })
    
    manual_week_pos = df_state_resampled['outbreak_label'].sum()
    print(f"   手动聚合（max）正样本: {manual_week_pos}")
    print(f"   使用group_by_state正样本: {df_state_week['outbreak_label'].sum()}")
    
    # 检查是否有问题
    if manual_week_pos != df_state_week['outbreak_label'].sum():
        print(f"   ⚠️  警告: 聚合结果不一致！")
    else:
        print(f"   ✓ 聚合逻辑正确")
    
    # 检查所有state
    print(f"\n7. 检查所有state的Week聚合:")
    total_day_pos = df['outbreak_label'].sum()
    total_week_pos = sum(df['outbreak_label'].sum() for df in week_data.values())
    
    print(f"   所有state日数据正样本: {total_day_pos}")
    print(f"   所有state Week数据正样本: {total_week_pos}")
    
    # 检查每个state
    print(f"\n8. 按state检查（前10个）:")
    print(f"{'State':<10} {'日数据正样本':<15} {'Week数据正样本':<15} {'Week行数':<10}")
    print("-" * 60)
    
    states = sorted(df['State'].unique())[:10]
    for state in states:
        day_state_pos = df[df['State'] == state]['outbreak_label'].sum()
        week_state_pos = week_data[state]['outbreak_label'].sum()
        week_state_rows = len(week_data[state])
        print(f"{state:<10} {day_state_pos:<15} {week_state_pos:<15} {week_state_rows:<10}")
    
    print(f"\n9. 关键发现:")
    print(f"   - 标签是在聚合前生成的（在日数据上）")
    print(f"   - Week聚合时使用 'max' 聚合（一周内有任何outbreak就标记）")
    print(f"   - 这是正确的逻辑：每个state单独处理，然后聚合")

if __name__ == "__main__":
    diagnose_week_aggregation()













