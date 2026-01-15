#!/usr/bin/env python3
"""Quick script to clear GPU memory cache"""
import torch
import gc

if torch.cuda.is_available():
    print(f"GPU devices available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"    Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    print("\nClearing GPU memory...")
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  GPU {i} cleared")
        print(f"    Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"    Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    print("GPU memory cleared successfully!")
else:
    print("CUDA not available")

