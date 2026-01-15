#!/usr/bin/env python3
"""
分析人工标注结果，理解residual-based anomaly到底在抓什么
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_manual_annotations(
    annotation_csv='analysis/manual_inspection/annotation_template.csv',
    output_dir='analysis/manual_inspection/results'
):
    """
    分析人工标注结果
    
    关键问题：
    1. residual-based anomaly到底在抓什么？
    2. Label定义是否合理？
    3. 是否需要调整evaluation方式？
    4. 是否需要重新定义anomaly？
    """
    
    # 加载标注结果
    df = pd.read_csv(annotation_csv)
    
    # 检查是否有标注
    if 'human_judgment' not in df.columns or df['human_judgment'].isna().all():
        print("⚠ 警告: 未找到human_judgment标注，请先在CSV中填入标注")
        print("   标注选项: 1=明显有变化, 2=模糊, 3=完全看不出来")
        return
    
    # 过滤掉未标注的
    df_labeled = df[df['human_judgment'].notna() & (df['human_judgment'] != '')].copy()
    
    # 转换标注格式（支持1/2/3或文字）
    def normalize_judgment(val):
        if pd.isna(val) or val == '':
            return ''
        val_str = str(val).strip()
        # 如果是数字，转换为文字
        if val_str == '1':
            return '明显有变化'
        elif val_str == '2':
            return '模糊'
        elif val_str == '3':
            return '完全看不出来'
        # 如果已经是文字，直接返回
        elif val_str in ['明显有变化', '模糊', '完全看不出来']:
            return val_str
        else:
            return val_str  # 保留原值，后续处理
    
    df_labeled['human_judgment'] = df_labeled['human_judgment'].apply(normalize_judgment)
    df_labeled = df_labeled[df_labeled['human_judgment'] != ''].copy()
    
    if len(df_labeled) == 0:
        print("⚠ 警告: 没有找到标注的样本")
        return
    
    print(f"分析 {len(df_labeled)} 个已标注样本")
    print(f"原始label: Anomaly={df_labeled[df_labeled['label']=='anomaly'].shape[0]}, Normal={df_labeled[df_labeled['label']=='normal'].shape[0]}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 统计分析：原始label vs 人工判断
    print("\n" + "="*60)
    print("1. 原始Label vs 人工判断的对比")
    print("="*60)
    
    # 交叉表
    crosstab = pd.crosstab(df_labeled['label'], df_labeled['human_judgment'], margins=True)
    print("\n交叉表:")
    print(crosstab)
    
    # 2. Residual分布 vs 人工判断
    print("\n" + "="*60)
    print("2. Residual分布 vs 人工判断")
    print("="*60)
    
    judgment_residual_stats = df_labeled.groupby('human_judgment')['residual'].agg(['mean', 'median', 'std', 'count'])
    print("\n各判断类别的Residual统计:")
    print(judgment_residual_stats)
    
    # 3. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 3.1 Residual分布 vs 人工判断
    judgments = ['明显有变化', '模糊', '完全看不出来']
    colors = ['red', 'orange', 'blue']
    
    for i, (judgment, color) in enumerate(zip(judgments, colors)):
        data = df_labeled[df_labeled['human_judgment'] == judgment]['residual']
        if len(data) > 0:
            axes[0, 0].hist(data, bins=20, alpha=0.6, label=f'{judgment} (n={len(data)})', 
                          color=color, edgecolor='black')
    axes[0, 0].set_xlabel('Residual', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Residual分布 vs 人工判断', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 3.2 原始Label vs 人工判断的散点图
    label_color_map = {'anomaly': 'red', 'normal': 'blue'}
    judgment_marker_map = {'明显有变化': 'o', '模糊': 's', '完全看不出来': '^'}
    
    for label in ['anomaly', 'normal']:
        label_data = df_labeled[df_labeled['label'] == label]
        for judgment in judgments:
            judgment_data = label_data[label_data['human_judgment'] == judgment]
            if len(judgment_data) > 0:
                axes[0, 1].scatter(judgment_data['residual'], 
                                  [judgments.index(judgment)] * len(judgment_data),
                                  c=label_color_map[label], 
                                  marker=judgment_marker_map[judgment],
                                  alpha=0.6, s=100, label=f'{label}-{judgment}')
    
    axes[0, 1].set_xlabel('Residual', fontsize=12)
    axes[0, 1].set_ylabel('人工判断', fontsize=12)
    axes[0, 1].set_yticks(range(len(judgments)))
    axes[0, 1].set_yticklabels(judgments)
    axes[0, 1].set_title('Residual vs 人工判断（按原始Label分类）', fontsize=14, fontweight='bold')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3.3 一致性分析
    # 如果原始label是anomaly，人工判断"明显有变化" = 一致
    # 如果原始label是normal，人工判断"完全看不出来" = 一致
    df_labeled['consistency'] = ''
    for idx, row in df_labeled.iterrows():
        if row['label'] == 'anomaly' and row['human_judgment'] == '明显有变化':
            df_labeled.at[idx, 'consistency'] = '一致'
        elif row['label'] == 'normal' and row['human_judgment'] == '完全看不出来':
            df_labeled.at[idx, 'consistency'] = '一致'
        elif row['human_judgment'] == '模糊':
            df_labeled.at[idx, 'consistency'] = '模糊'
        else:
            df_labeled.at[idx, 'consistency'] = '不一致'
    
    consistency_counts = df_labeled['consistency'].value_counts()
    axes[1, 0].bar(consistency_counts.index, consistency_counts.values, 
                   color=['green', 'orange', 'red'], alpha=0.7)
    axes[1, 0].set_xlabel('一致性', fontsize=12)
    axes[1, 0].set_ylabel('数量', fontsize=12)
    axes[1, 0].set_title('原始Label vs 人工判断的一致性', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 3.4 Residual分布（按一致性分类）
    for consistency in ['一致', '模糊', '不一致']:
        data = df_labeled[df_labeled['consistency'] == consistency]['residual']
        if len(data) > 0:
            axes[1, 1].hist(data, bins=20, alpha=0.6, label=f'{consistency} (n={len(data)})', 
                          edgecolor='black')
    axes[1, 1].set_xlabel('Residual', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Residual分布 vs 一致性', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'annotation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化已保存到: {output_dir / 'annotation_analysis.png'}")
    
    # 4. 关键发现总结
    print("\n" + "="*60)
    print("3. 关键发现总结")
    print("="*60)
    
    # 4.1 不一致的样本
    inconsistent = df_labeled[df_labeled['consistency'] == '不一致']
    print(f"\n不一致的样本 (n={len(inconsistent)}):")
    if len(inconsistent) > 0:
        print("这些样本的原始label和人工判断不一致")
        print("\n原始Label是Anomaly，但人工判断不是'明显有变化':")
        anomaly_inconsistent = inconsistent[(inconsistent['label'] == 'anomaly') & 
                                          (inconsistent['human_judgment'] != '明显有变化')]
        print(f"  数量: {len(anomaly_inconsistent)}")
        if len(anomaly_inconsistent) > 0:
            print(f"  平均Residual: {anomaly_inconsistent['residual'].mean():.6f}")
            print(f"  这些样本可能不应该标记为anomaly，或者anomaly定义需要调整")
        
        print("\n原始Label是Normal，但人工判断是'明显有变化':")
        normal_inconsistent = inconsistent[(inconsistent['label'] == 'normal') & 
                                         (inconsistent['human_judgment'] == '明显有变化')]
        print(f"  数量: {len(normal_inconsistent)}")
        if len(normal_inconsistent) > 0:
            print(f"  平均Residual: {normal_inconsistent['residual'].mean():.6f}")
            print(f"  这些样本可能应该标记为anomaly")
    
    # 4.2 Residual和人工判断的关系
    print("\nResidual和人工判断的关系:")
    for judgment in judgments:
        judgment_data = df_labeled[df_labeled['human_judgment'] == judgment]
        if len(judgment_data) > 0:
            print(f"  {judgment}: 平均Residual = {judgment_data['residual'].mean():.6f}, "
                  f"Median = {judgment_data['residual'].median():.6f}")
    
    # 4.3 回答核心问题
    print("\n" + "="*60)
    print("4. 回答核心问题：residual-based anomaly到底在抓什么？")
    print("="*60)
    
    # 如果"明显有变化"的residual明显大于"完全看不出来"
    change_residual = df_labeled[df_labeled['human_judgment'] == '明显有变化']['residual'].median()
    no_change_residual = df_labeled[df_labeled['human_judgment'] == '完全看不出来']['residual'].median()
    
    if change_residual > no_change_residual * 1.2:
        print(f"\n✅ Residual和人工判断一致: '明显有变化'的Residual ({change_residual:.6f}) "
              f"明显大于'完全看不出来' ({no_change_residual:.6f})")
        print("   → Residual-based方法在抓'明显的变化'")
    else:
        print(f"\n⚠ Residual和人工判断不完全一致: '明显有变化'的Residual ({change_residual:.6f}) "
              f"和'完全看不出来' ({no_change_residual:.6f}) 差异不大")
        print("   → 可能需要调整label定义，或者residual-based方法在抓其他模式")
    
    # 保存详细分析结果
    df_labeled.to_csv(output_dir / 'annotated_samples_analysis.csv', index=False)
    print(f"\n详细分析结果已保存到: {output_dir / 'annotated_samples_analysis.csv'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze manual annotations')
    parser.add_argument('--annotation_csv', type=str,
                       default='analysis/manual_inspection/annotation_template.csv',
                       help='Path to annotation CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='analysis/manual_inspection/results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    analyze_manual_annotations(
        annotation_csv=args.annotation_csv,
        output_dir=args.output_dir
    )

