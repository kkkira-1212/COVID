import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.trainer import train_ours as train, to_device as todevice
from model.evaluator import infer


def evaluate_prediction_error(model_path, bundle_coarse, bundle_fine=None, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    result = infer(model_path, bundle_coarse, bundle_fine, device=device)
    
    residual = result['residual']
    y_true = result['y_true']
    idx_val = result['idx_val']
    idx_test = result['idx_test']
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    coarse_only = config.get('coarse_only', False)
    
    bundle = bundle_coarse
    if not coarse_only and bundle_fine is not None:
        bundle = bundle_fine
    
    X_seq = bundle['X_seq']
    X_next = bundle['X_next']
    
    checkpoint = torch.load(model_path, map_location=device)
    from model.encoder import TransformerSeqEncoder, RegressionHeadWithRelation
    
    data = todevice(bundle, device)
    num_vars = data['X_seq'].shape[2]
    
    if coarse_only:
        enc = TransformerSeqEncoder(
            input_dim=data['X_seq'].shape[2],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_seq_len=data['X_seq'].shape[1] + 5
        ).to(device)
        enc.pooling = config['pooling']
        enc.load_state_dict(checkpoint['enc_coarse'])
        
        head = RegressionHeadWithRelation(config['d_model'], num_vars).to(device)
        head.load_state_dict(checkpoint['head'])
        
        enc.eval()
        head.eval()
        
        with torch.no_grad():
            z = enc(data['X_seq'])
            _, pred_all = head(z, z)
            
            pred_train = pred_all[data['idx_train']].cpu().numpy()
            pred_val = pred_all[data['idx_val']].cpu().numpy()
            pred_test = pred_all[data['idx_test']].cpu().numpy()
            
            X_next_train = data['X_next'][data['idx_train']].cpu().numpy()
            X_next_val = data['X_next'][data['idx_val']].cpu().numpy()
            X_next_test = data['X_next'][data['idx_test']].cpu().numpy()
    else:
        from model.encoder import TransformerSeqEncoder, RegressionHeadWithRelation
        data_fine = todevice(bundle_fine, device)
        
        max_seq_len_fine = data_fine['X_seq'].shape[1] + 5
        max_seq_len_coarse = data['X_seq'].shape[1] + 5
        
        enc_fine = TransformerSeqEncoder(
            input_dim=data_fine['X_seq'].shape[2],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_seq_len=max_seq_len_fine
        ).to(device)
        enc_fine.pooling = config['pooling']
        
        state_dict_fine = checkpoint['enc_fine'].copy()
        if 'pos_encoding' in state_dict_fine:
            pos_encoding_shape = state_dict_fine['pos_encoding'].shape
            if pos_encoding_shape[1] != max_seq_len_fine:
                del state_dict_fine['pos_encoding']
        enc_fine.load_state_dict(state_dict_fine, strict=False)
        
        enc_coarse = TransformerSeqEncoder(
            input_dim=data['X_seq'].shape[2],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_seq_len=max_seq_len_coarse
        ).to(device)
        enc_coarse.pooling = config['pooling']
        
        state_dict_coarse = checkpoint['enc_coarse'].copy()
        if 'pos_encoding' in state_dict_coarse:
            pos_encoding_shape = state_dict_coarse['pos_encoding'].shape
            if pos_encoding_shape[1] != max_seq_len_coarse:
                del state_dict_coarse['pos_encoding']
        enc_coarse.load_state_dict(state_dict_coarse, strict=False)
        
        head = RegressionHeadWithRelation(config['d_model'], num_vars).to(device)
        head.load_state_dict(checkpoint['head'])
        
        enc_fine.eval()
        enc_coarse.eval()
        head.eval()
        
        with torch.no_grad():
            z_fine = enc_fine(data_fine['X_seq'])
            z_coarse = enc_coarse(data['X_seq'])
            pred_fine_all, pred_coarse_all = head(z_fine, z_coarse)
            
            pred_train = pred_fine_all[data_fine['idx_train']].cpu().numpy()
            pred_val = pred_fine_all[data_fine['idx_val']].cpu().numpy()
            pred_test = pred_fine_all[data_fine['idx_test']].cpu().numpy()
            
            X_next_train = data_fine['X_next'][data_fine['idx_train']].cpu().numpy()
            X_next_val = data_fine['X_next'][data_fine['idx_val']].cpu().numpy()
            X_next_test = data_fine['X_next'][data_fine['idx_test']].cpu().numpy()
    
    mae_train = np.mean(np.abs(pred_train - X_next_train))
    mse_train = np.mean((pred_train - X_next_train) ** 2)
    
    mae_val = np.mean(np.abs(pred_val - X_next_val))
    mse_val = np.mean((pred_val - X_next_val) ** 2)
    
    mae_test = np.mean(np.abs(pred_test - X_next_test))
    mse_test = np.mean((pred_test - X_next_test) ** 2)
    
    return {
        'mae_train': mae_train,
        'mse_train': mse_train,
        'mae_val': mae_val,
        'mse_val': mse_val,
        'mae_test': mae_test,
        'mse_test': mse_test,
        'residual': residual,
        'y_true': y_true,
        'idx_val': idx_val,
        'idx_test': idx_test
    }


def analyze_residual_distribution(residual, y_true, idx_val, idx_test, save_path=None):
    residual_val = residual[idx_val]
    residual_test = residual[idx_test]
    y_val = y_true[idx_val]
    y_test = y_true[idx_test]
    
    residual_normal_val = residual_val[y_val == 0]
    residual_attack_val = residual_val[y_val == 1]
    residual_normal_test = residual_test[y_test == 0]
    residual_attack_test = residual_test[y_test == 1]
    
    stats = {
        'val_normal_mean': np.mean(residual_normal_val),
        'val_normal_std': np.std(residual_normal_val),
        'val_normal_median': np.median(residual_normal_val),
        'val_normal_q90': np.percentile(residual_normal_val, 90),
        'val_normal_q95': np.percentile(residual_normal_val, 95),
        'val_attack_mean': np.mean(residual_attack_val) if len(residual_attack_val) > 0 else 0,
        'val_attack_std': np.std(residual_attack_val) if len(residual_attack_val) > 0 else 0,
        'val_attack_median': np.median(residual_attack_val) if len(residual_attack_val) > 0 else 0,
        'val_attack_q90': np.percentile(residual_attack_val, 90) if len(residual_attack_val) > 0 else 0,
        'val_attack_q95': np.percentile(residual_attack_val, 95) if len(residual_attack_val) > 0 else 0,
        'test_normal_mean': np.mean(residual_normal_test),
        'test_normal_std': np.std(residual_normal_test),
        'test_normal_median': np.median(residual_normal_test),
        'test_normal_q90': np.percentile(residual_normal_test, 90),
        'test_normal_q95': np.percentile(residual_normal_test, 95),
        'test_attack_mean': np.mean(residual_attack_test) if len(residual_attack_test) > 0 else 0,
        'test_attack_std': np.std(residual_attack_test) if len(residual_attack_test) > 0 else 0,
        'test_attack_median': np.median(residual_attack_test) if len(residual_attack_test) > 0 else 0,
        'test_attack_q90': np.percentile(residual_attack_test, 90) if len(residual_attack_test) > 0 else 0,
        'test_attack_q95': np.percentile(residual_attack_test, 95) if len(residual_attack_test) > 0 else 0,
    }
    
    if save_path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(residual_normal_val, bins=50, alpha=0.7, label='Normal', density=True)
        if len(residual_attack_val) > 0:
            axes[0].hist(residual_attack_val, bins=50, alpha=0.7, label='Attack', density=True)
        axes[0].set_xlabel('Residual')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Validation Set Residual Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(residual_normal_test, bins=50, alpha=0.7, label='Normal', density=True)
        if len(residual_attack_test) > 0:
            axes[1].hist(residual_attack_test, bins=50, alpha=0.7, label='Attack', density=True)
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Test Set Residual Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/SWaT/processed')
    parser.add_argument('--model_save_path', type=str, default='models/swat_stage1.pt')
    parser.add_argument('--coarse_only', action='store_true', help='Train only coarse scale')
    parser.add_argument('--use_lu', action='store_true', help='Use LU loss')
    parser.add_argument('--lambda_u', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--pooling', type=str, default='last')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only evaluate')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    bundle_hour = torch.load(data_dir / 'swat_hour.pt', weights_only=False)
    bundle_minute = torch.load(data_dir / 'swat_minute.pt', weights_only=False)
    
    print("=" * 60)
    print("SWaT Stage1 Training and Evaluation")
    print("=" * 60)
    print(f"\nData loaded:")
    print(f"  Hour scale: {bundle_hour['X_seq'].shape}")
    print(f"  Minute scale: {bundle_minute['X_seq'].shape}")
    
    if not args.skip_training:
        print(f"\nTraining model...")
        print(f"  Coarse only: {args.coarse_only}")
        print(f"  Use LU loss: {args.use_lu}")
        print(f"  Epochs: {args.epochs}")
        
        model_path = train(
            bundle_coarse=bundle_hour,
            bundle_fine=bundle_minute if not args.coarse_only else None,
            save_path=args.model_save_path,
            coarse_only=args.coarse_only,
            use_lu=args.use_lu,
            lambda_u=args.lambda_u,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            pooling=args.pooling,
            batch_size=args.batch_size
        )
        print(f"  Model saved to: {model_path}")
    else:
        model_path = args.model_save_path
        print(f"\nSkipping training, using existing model: {model_path}")
    
    print(f"\nEvaluating prediction error...")
    eval_results = evaluate_prediction_error(
        model_path,
        bundle_hour,
        bundle_minute if not args.coarse_only else None,
        device=args.device
    )
    
    print(f"\nPrediction Error (MAE/MSE):")
    print(f"  Train - MAE: {eval_results['mae_train']:.6f}, MSE: {eval_results['mse_train']:.6f}")
    print(f"  Val   - MAE: {eval_results['mae_val']:.6f}, MSE: {eval_results['mse_val']:.6f}")
    print(f"  Test  - MAE: {eval_results['mae_test']:.6f}, MSE: {eval_results['mse_test']:.6f}")
    
    print(f"\nAnalyzing residual distribution...")
    residual_stats = analyze_residual_distribution(
        eval_results['residual'],
        eval_results['y_true'],
        eval_results['idx_val'],
        eval_results['idx_test'],
        save_path='swat_residual_distribution.png'
    )
    
    print(f"\nResidual Statistics:")
    print(f"\nValidation Set:")
    print(f"  Normal - Mean: {residual_stats['val_normal_mean']:.6f}, Std: {residual_stats['val_normal_std']:.6f}, Median: {residual_stats['val_normal_median']:.6f}, Q90: {residual_stats['val_normal_q90']:.6f}, Q95: {residual_stats['val_normal_q95']:.6f}")
    if residual_stats['val_attack_mean'] > 0:
        print(f"  Attack - Mean: {residual_stats['val_attack_mean']:.6f}, Std: {residual_stats['val_attack_std']:.6f}, Median: {residual_stats['val_attack_median']:.6f}, Q90: {residual_stats['val_attack_q90']:.6f}, Q95: {residual_stats['val_attack_q95']:.6f}")
        print(f"  Signal: Attack residual is {residual_stats['val_attack_median'] / residual_stats['val_normal_median']:.2f}x higher than normal")
    
    print(f"\nTest Set:")
    print(f"  Normal - Mean: {residual_stats['test_normal_mean']:.6f}, Std: {residual_stats['test_normal_std']:.6f}, Median: {residual_stats['test_normal_median']:.6f}, Q90: {residual_stats['test_normal_q90']:.6f}, Q95: {residual_stats['test_normal_q95']:.6f}")
    if residual_stats['test_attack_mean'] > 0:
        print(f"  Attack - Mean: {residual_stats['test_attack_mean']:.6f}, Std: {residual_stats['test_attack_std']:.6f}, Median: {residual_stats['test_attack_median']:.6f}, Q90: {residual_stats['test_attack_q90']:.6f}, Q95: {residual_stats['test_attack_q95']:.6f}")
        signal_ratio = residual_stats['test_attack_median'] / residual_stats['test_normal_median'] if residual_stats['test_normal_median'] > 0 else 0
        print(f"  Signal: Attack residual is {signal_ratio:.2f}x higher than normal")
    
    print(f"\nResidual distribution plot saved to: swat_residual_distribution.png")
    print("=" * 60)


if __name__ == '__main__':
    main()

