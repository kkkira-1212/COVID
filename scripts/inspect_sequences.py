#!/usr/bin/env python3
"""
检查 sequences_day.pt 和 sequences_week.pt 文件的信息
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def inspect_pt_file(filepath, name):
    """检查 .pt 文件的详细信息"""
    print(f"\n{'='*60}")
    print(f"检查文件: {name}")
    print(f"路径: {filepath}")
    print(f"{'='*60}")
    
    if not Path(filepath).exists():
        print(f"❌ 文件不存在: {filepath}")
        return None
    
    try:
        # 使用 weights_only=False 因为文件包含 pandas DataFrame
        data = torch.load(filepath, weights_only=False)
        
        print(f"\n文件类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"\n字典键: {list(data.keys())}")
            print(f"\n详细信息:")
            
            for key, value in data.items():
                print(f"\n  [{key}]")
                print(f"    类型: {type(value)}")
                
                if isinstance(value, torch.Tensor):
                    print(f"    形状: {value.shape}")
                    print(f"    数据类型: {value.dtype}")
                    if value.dtype in (torch.float32, torch.float64, torch.float16):
                        print(f"    最小值: {value.min().item():.6f}")
                        print(f"    最大值: {value.max().item():.6f}")
                        print(f"    均值: {value.mean().item():.6f}")
                        print(f"    标准差: {value.std().item():.6f}")
                        if value.numel() > 0:
                            nan_count = torch.isnan(value).sum().item()
                            inf_count = torch.isinf(value).sum().item()
                            print(f"    NaN数量: {nan_count}")
                            print(f"    Inf数量: {inf_count}")
                    else:
                        print(f"    最小值: {value.min().item()}")
                        print(f"    最大值: {value.max().item()}")
                        if value.numel() > 0:
                            unique_count = len(torch.unique(value))
                            print(f"    唯一值数量: {unique_count}")
                
                elif isinstance(value, (list, tuple)):
                    print(f"    长度: {len(value)}")
                    if len(value) > 0:
                        print(f"    第一个元素类型: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"    第一个元素的键: {list(value[0].keys())}")
                
                elif isinstance(value, pd.DataFrame):
                    print(f"    形状: {value.shape}")
                    print(f"    列: {list(value.columns)}")
                    print(f"    前5行:")
                    print(value.head())
                
                elif isinstance(value, np.ndarray):
                    print(f"    形状: {value.shape}")
                    print(f"    数据类型: {value.dtype}")
                    print(f"    最小值: {value.min():.6f}")
                    print(f"    最大值: {value.max():.6f}")
                    print(f"    均值: {value.mean():.6f}")
                    print(f"    标准差: {value.std():.6f}")
                
                elif isinstance(value, (int, float, str, bool)):
                    print(f"    值: {value}")
                
                else:
                    print(f"    值: {str(value)[:100]}")
            
            # 检查关键字段
            print(f"\n关键字段检查:")
            if 'X_seq' in data:
                X_seq = data['X_seq']
                print(f"  X_seq: {X_seq.shape} (N, T, D)")
            if 'X_next' in data:
                X_next = data['X_next']
                print(f"  X_next: {X_next.shape} (N, D)")
            if 'y_next' in data:
                y_next = data['y_next']
                print(f"  y_next: {y_next.shape}, 正样本数: {(y_next > 0.5).sum().item() if isinstance(y_next, torch.Tensor) else (y_next > 0.5).sum()}")
            if 'idx_train' in data:
                print(f"  idx_train: {len(data['idx_train'])} 个样本")
            if 'idx_val' in data:
                print(f"  idx_val: {len(data['idx_val'])} 个样本")
            if 'idx_test' in data:
                print(f"  idx_test: {len(data['idx_test'])} 个样本")
            if 'feature_cols' in data:
                print(f"  feature_cols: {len(data['feature_cols'])} 个特征")
                print(f"    特征列表: {data['feature_cols']}")
            if 'window_size' in data:
                print(f"  window_size: {data['window_size']}")
            if 'states_order' in data:
                print(f"  states_order: {len(data['states_order'])} 个州")
                print(f"    州列表: {data['states_order']}")
            if 'feature_means' in data:
                print(f"  feature_means: {data['feature_means'].shape}")
            if 'feature_stds' in data:
                print(f"  feature_stds: {data['feature_stds'].shape}")
            if 'fine_to_coarse_index' in data:
                mapping = data['fine_to_coarse_index']
                mapped = (mapping >= 0).sum().item()
                print(f"  fine_to_coarse_index: {mapping.shape}, 已映射: {mapped}/{len(mapping)}")
        
        else:
            print(f"数据不是字典格式，而是: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"形状: {data.shape}")
        
        return data
        
    except Exception as e:
        print(f"❌ 加载文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    
    # 检查 sequences_day.pt
    day_path = project_root / "sequences_day.pt"
    day_data = inspect_pt_file(day_path, "sequences_day.pt")
    
    # 检查 sequences_week.pt
    week_path = project_root / "sequences_week.pt"
    week_data = inspect_pt_file(week_path, "sequences_week.pt")
    
    # 如果两个文件都存在，进行对比
    if day_data is not None and week_data is not None:
        print(f"\n{'='*60}")
        print("对比分析")
        print(f"{'='*60}")
        
        if isinstance(day_data, dict) and isinstance(week_data, dict):
            print(f"\n样本数量对比:")
            if 'X_seq' in day_data and 'X_seq' in week_data:
                print(f"  Day序列数: {day_data['X_seq'].shape[0]}")
                print(f"  Week序列数: {week_data['X_seq'].shape[0]}")
            
            print(f"\n特征维度对比:")
            if 'X_seq' in day_data and 'X_seq' in week_data:
                print(f"  Day特征数: {day_data['X_seq'].shape[2]}")
                print(f"  Week特征数: {week_data['X_seq'].shape[2]}")
                if day_data['X_seq'].shape[2] != week_data['X_seq'].shape[2]:
                    print(f"  ⚠️  警告: 特征数不匹配!")
            
            print(f"\n窗口大小对比:")
            if 'window_size' in day_data and 'window_size' in week_data:
                print(f"  Day窗口: {day_data['window_size']}")
                print(f"  Week窗口: {week_data['window_size']}")
            
            print(f"\n标签分布对比:")
            if 'y_next' in day_data and 'y_next' in week_data:
                day_pos = (day_data['y_next'] > 0.5).sum().item() if isinstance(day_data['y_next'], torch.Tensor) else (day_data['y_next'] > 0.5).sum()
                week_pos = (week_data['y_next'] > 0.5).sum().item() if isinstance(week_data['y_next'], torch.Tensor) else (week_data['y_next'] > 0.5).sum()
                day_total = len(day_data['y_next'])
                week_total = len(week_data['y_next'])
                print(f"  Day正样本: {day_pos}/{day_total} ({100*day_pos/day_total:.2f}%)")
                print(f"  Week正样本: {week_pos}/{week_total} ({100*week_pos/week_total:.2f}%)")
            
            print(f"\n特征列表对比:")
            if 'feature_cols' in day_data and 'feature_cols' in week_data:
                day_feats = set(day_data['feature_cols'])
                week_feats = set(week_data['feature_cols'])
                if day_feats == week_feats:
                    print(f"  ✓ 特征列表一致")
                else:
                    print(f"  ⚠️  特征列表不一致!")
                    print(f"  Day独有: {day_feats - week_feats}")
                    print(f"  Week独有: {week_feats - day_feats}")

