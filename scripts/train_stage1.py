import torch
import numpy as np
from pathlib import Path
import sys
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.trainer import train_ours as train
from experiments.trainer_unified import train_unified
from utils.evaluator_stage1 import infer, evaluate
from utils.data_io import load_data_bundles


def main():
    parser = argparse.ArgumentParser(description='Train Stage 1 forecasting model')
    data_args = parser.add_argument_group('data')
    data_args.add_argument('--data_dir', type=str, required=True)
    data_args.add_argument('--coarse_file', type=str, required=True)
    data_args.add_argument('--fine_file', type=str, default=None)
    data_args.add_argument('--coarse_only', action='store_true')
    data_args.add_argument('--fine_only', action='store_true')

    model_args = parser.add_argument_group('model')
    model_args.add_argument('--d_model', type=int, default=64)
    model_args.add_argument('--nhead', type=int, default=4)
    model_args.add_argument('--num_layers', type=int, default=2)
    model_args.add_argument('--pooling', type=str, default='last', choices=['last', 'mean'])

    train_args = parser.add_argument_group('training')
    train_args.add_argument('--trainer', type=str, default='ours', choices=['ours', 'unified'])
    train_args.add_argument('--epochs', type=int, default=50)
    train_args.add_argument('--lr', type=float, default=3e-4)
    train_args.add_argument('--weight_decay', type=float, default=1e-4)
    train_args.add_argument('--batch_size', type=int, default=32)
    train_args.add_argument('--device', type=str, default='cuda')
    train_args.add_argument('--skip_training', action='store_true')
    train_args.add_argument('--resume_from', type=str, default=None)

    loss_args = parser.add_argument_group('loss')
    loss_args.add_argument('--use_lu', action='store_true', default=True)
    loss_args.add_argument('--no_lu', dest='use_lu', action='store_false')
    loss_args.add_argument('--lambda_u', type=float, default=1.0)
    loss_args.add_argument('--use_lr', action='store_true', default=True)
    loss_args.add_argument('--no_lr', dest='use_lr', action='store_false')
    loss_args.add_argument('--lambda_r', type=float, default=1.0)
    loss_args.add_argument('--lu_detach_coarse', action='store_true')

    io_args = parser.add_argument_group('io')
    io_args.add_argument('--model_save_path', type=str, required=True)
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Stage 1 Training")
    print("=" * 60)
    print(f"\nLoading data from {data_dir}...")
    
    fine_file = None if args.coarse_only else args.fine_file
    if args.fine_only and not args.fine_file:
        raise ValueError("Fine scale data file is required for fine_only training")
    if not args.coarse_only and not args.fine_only and not args.fine_file:
        raise ValueError("Fine scale data file is required for multi-scale training")

    bundle_coarse, bundle_fine = load_data_bundles(data_dir, args.coarse_file, fine_file)
    if args.fine_only and bundle_fine is None:
        raise ValueError("bundle_fine is required for fine_only training")
    
    print(f"\nData loaded:")
    print(f"  Coarse scale: {bundle_coarse['X_seq'].shape}")
    print(f"  Train samples: {len(bundle_coarse['idx_train'])}")
    print(f"  Val samples: {len(bundle_coarse['idx_val'])}")
    print(f"  Test samples: {len(bundle_coarse['idx_test'])}")
    
    if bundle_fine is not None:
        print(f"  Fine scale: {bundle_fine['X_seq'].shape}")
        print(f"  Fine train/val/test: {len(bundle_fine['idx_train'])}/{len(bundle_fine['idx_val'])}/{len(bundle_fine['idx_test'])}")
    
    y_all = bundle_coarse['y_next'].numpy()
    y_train = y_all[bundle_coarse['idx_train'].numpy()]
    y_val = y_all[bundle_coarse['idx_val'].numpy()]
    y_test = y_all[bundle_coarse['idx_test'].numpy()]
    
    print(f"\nLabel distribution:")
    print(f"  Train - Normal: {np.sum(y_train == 0)}, Anomaly: {np.sum(y_train == 1)}")
    print(f"  Val   - Normal: {np.sum(y_val == 0)}, Anomaly: {np.sum(y_val == 1)}")
    print(f"  Test  - Normal: {np.sum(y_test == 0)}, Anomaly: {np.sum(y_test == 1)}")
    
    val_has_anomalies = np.sum(y_val == 1) > 0
    
    if not args.skip_training:
        if args.trainer == 'unified' and args.fine_only:
            raise ValueError("trainer=unified does not support fine_only mode")

        print(f"\nTraining model...")
        print(f"  Coarse only: {args.coarse_only}")
        print(f"  Fine only: {args.fine_only}")
        print(f"  Trainer: {args.trainer}")
        if args.trainer == 'unified':
            print(f"  Use LR loss: {args.use_lr}")
            print(f"  Lambda LR: {args.lambda_r}")
        else:
            print(f"  Use LU loss: {args.use_lu}")
            print(f"  LU detach coarse: {args.lu_detach_coarse}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Device: {device}")

        if args.trainer == 'unified':
            model_path = train_unified(
                bundle_coarse=bundle_coarse,
                bundle_fine=bundle_fine if not args.coarse_only else None,
                save_path=args.model_save_path,
                coarse_only=args.coarse_only,
                use_lr=args.use_lr,
                lambda_r=args.lambda_r,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                pooling=args.pooling,
                batch_size=args.batch_size
            )
        else:
            model_path = train(
                bundle_coarse=bundle_coarse,
                bundle_fine=bundle_fine if (not args.coarse_only) or args.fine_only else None,
                save_path=args.model_save_path,
                coarse_only=args.coarse_only,
                fine_only=args.fine_only,
                use_classification=False,
                use_lu=args.use_lu,
                lu_detach_coarse=args.lu_detach_coarse,
                lambda_u=args.lambda_u,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                pooling=args.pooling,
                batch_size=args.batch_size,
                device=args.device,
                resume_from=args.resume_from
            )
        print(f"\nModel saved to: {model_path}")
    else:
        model_path = args.model_save_path
        print(f"\nSkipping training, using existing model: {model_path}")
    
    if not val_has_anomalies:
        print(f"\n⚠ Validation set has no anomalies. Skipping evaluation.")
        print(f"  (This is normal for datasets like SMAP where train/val are all normal)")
        print(f"  Model saved from final epoch. You can evaluate on test set separately.")
    else:
        print(f"\nEvaluating model and computing residuals...")
        inference_result = infer(
            model_path=model_path,
            bundle_coarse=bundle_coarse,
            bundle_fine=bundle_fine if (not args.coarse_only) or args.fine_only else None,
            device=args.device
        )
        
        residual = inference_result['residual']
        y_true = inference_result['y_true']
        idx_val = inference_result['idx_val']
        idx_test = inference_result['idx_test']
        
        print(f"\nComputing evaluation metrics...")
        eval_metrics = evaluate(residual, y_true, idx_val, idx_test)
        
        print(f"\nEvaluation Metrics:")
        print(f"  Threshold: {eval_metrics['threshold']:.6f}")
        print(f"  F1 Score: {eval_metrics['f1']:.4f}")
        print(f"  Precision: {eval_metrics['precision']:.4f}")
        print(f"  Recall: {eval_metrics['recall']:.4f}")
        print(f"  ROC-AUC: {eval_metrics['roc_auc']:.4f}")
        print(f"  AUPRC: {eval_metrics['auprc']:.4f}")
        
        print(f"\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Pipeline: OK (data loaded and processed)")
        print(f"Split: OK (train/val/test split maintained)")
        if eval_metrics['roc_auc'] > 0.5:
            print(f"Residual Separation: GOOD (ROC-AUC = {eval_metrics['roc_auc']:.4f} > 0.5)")
        else:
            print(f"Residual Separation: POOR (ROC-AUC = {eval_metrics['roc_auc']:.4f} <= 0.5)")
        print("=" * 60)
    
    if not val_has_anomalies:
        print(f"\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"✓ Pipeline: OK (data loaded and processed)")
        print(f"✓ Split: OK (train/val/test split maintained)")
        print(f"⚠ Evaluation: Skipped (validation set has no anomalies)")
        print(f"  Model saved from final epoch. Evaluate on test set separately if needed.")
        print("=" * 60)


if __name__ == '__main__':
    main()

