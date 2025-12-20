import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.patchtst import train_patchtst_forecast, inference_patchtst
from model.dlinear import train_dlinear_forecast, inference_dlinear
from model.lstm import train_lstm_forecast, inference_lstm
from model.AERCA import train_aerca_forecast, inference_aerca
from model.trainer import train_ours
from model.evaluator import evaluate_residual_scores, run_inference


def load_bundle_week(pt_path):
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    bundle_week = {
        'X_seq': data['X_seq'],
        'y_next': data['y_next'],
        'NewDeaths_ret_next': data['NewDeaths_ret_next'],
        'idx_train': data['idx_train'],
        'idx_val': data['idx_val'],
        'idx_test': data['idx_test'],
        'state_ids': data.get('state_ids', None),
        'seq_time_ids': data.get('seq_time_ids', None),
        'target_time_ids': data.get('target_time_ids', None),
    }
    if 'seq_meta' in data:
        bundle_week['meta'] = data['seq_meta']
    if 'states_order' in data:
        bundle_week['states_order'] = data['states_order']
    if 'window_size' in data:
        bundle_week['window_size'] = data['window_size']
    if 'num_features' in data:
        bundle_week['feature_cols'] = [f'feature_{i}' for i in range(data['num_features'])]
    print(f"Loaded bundle_week: {len(bundle_week['X_seq'])} sequences")
    print(f"  X_seq shape: {bundle_week['X_seq'].shape}")
    print(f"  Train: {len(bundle_week['idx_train'])}, Val: {len(bundle_week['idx_val'])}, Test: {len(bundle_week['idx_test'])}")
    return bundle_week


def load_bundle_day(pt_path):
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    bundle_day = {
        'X_seq': data['X_seq'],
        'y_next': data['y_next'],
        'NewDeaths_ret_next': data['NewDeaths_ret_next'],
        'idx_train': data['idx_train'],
        'idx_val': data['idx_val'],
        'idx_test': data['idx_test'],
        'day_to_week_index': data.get('day_to_week_index', None),
    }
    print(f"Loaded bundle_day: {len(bundle_day['X_seq'])} sequences")
    print(f"  X_seq shape: {bundle_day['X_seq'].shape}")
    print(f"  Train: {len(bundle_day['idx_train'])}, Val: {len(bundle_day['idx_val'])}, Test: {len(bundle_day['idx_test'])}")
    return bundle_day


def train_and_evaluate(model_name, bundle_week, bundle_day=None, save_dir="models/baselines",
                       epochs=200, lambda_u=1.0, alpha_cls=1.0, alpha_reg=0.1,
                       use_postprocessing=False, device='cuda', use_lu=None):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    
    if model_name == "patchtst":
        train_patchtst_forecast(
            bundle_week=bundle_week, save_path=model_path, epochs=epochs,
            lr=1e-3, weight_decay=1e-4, patch_len=4, d_model=64,
            dropout=0.1, patience_limit=30,
        )
    elif model_name == "dlinear":
        train_dlinear_forecast(
            bundle_week=bundle_week, save_path=model_path, epochs=epochs,
            lr=1e-3, weight_decay=1e-4, use_time_avg=True, patience_limit=30,
        )
    elif model_name == "lstm":
        train_lstm_forecast(
            bundle_week=bundle_week, save_path=model_path, epochs=epochs,
            lr=1e-3, weight_decay=1e-4, hidden_dim=64, num_layers=1,
            dropout=0.1, patience_limit=30,
        )
    elif model_name == "aerca":
        window_size = bundle_week.get('window_size', bundle_week['X_seq'].shape[1])
        train_aerca_forecast(
            bundle_week=bundle_week, save_path=model_path, epochs=epochs,
            lr=1e-4, weight_decay=1e-4, window_size=window_size, hidden_size=64,
            num_hidden_layers=3, patience_limit=30,
        )
    elif model_name.startswith("ours_"):
        if model_name == "ours_weekonly":
            train_ours(
                bundle_week=bundle_week, bundle_day=None, save_path=model_path,
                weekly_only=True, use_classification=False, use_lu=False,
                epochs=epochs, lr=3e-4, weight_decay=1e-4, patience_limit=30,
            )
        elif model_name == "ours_multiscale":
            if bundle_day is None:
                raise ValueError("bundle_day required for ours_multiscale")
            # Default to True if not explicitly set
            use_lu_flag = use_lu if use_lu is not None else True
            train_ours(
                bundle_week=bundle_week, bundle_day=bundle_day, save_path=model_path,
                weekly_only=False, use_classification=False, use_lu=use_lu_flag, lambda_u=lambda_u,
                epochs=epochs, lr=3e-4, weight_decay=1e-4, patience_limit=30,
            )
        elif model_name == "ours_multiscale_no_lu":
            if bundle_day is None:
                raise ValueError("bundle_day required for ours_multiscale_no_lu")
            train_ours(
                bundle_week=bundle_week, bundle_day=bundle_day, save_path=model_path,
                weekly_only=False, use_classification=False, use_lu=False, lambda_u=0.0,
                epochs=epochs, lr=3e-4, weight_decay=1e-4, patience_limit=30,
            )
        elif model_name == "ours_supervised":
            if bundle_day is None:
                raise ValueError("bundle_day required for ours_supervised")
            train_ours(
                bundle_week=bundle_week, bundle_day=bundle_day, save_path=model_path,
                weekly_only=False, use_classification=True, use_lu=True,
                alpha_cls=alpha_cls, alpha_reg=alpha_reg, lambda_u=lambda_u,
                epochs=epochs, lr=3e-4, weight_decay=1e-4, patience_limit=40,
            )
        else:
            raise ValueError(f"Unknown ours model: {model_name}")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    print(f"Model saved to: {model_path}")
    
    print(f"\nEvaluating {model_name.upper()}...")
    if model_name == "patchtst":
        results = inference_patchtst(model_path, bundle_week, device)
    elif model_name == "dlinear":
        results = inference_dlinear(model_path, bundle_week, device)
    elif model_name == "lstm":
        results = inference_lstm(model_path, bundle_week, device)
    elif model_name == "aerca":
        results = inference_aerca(model_path, bundle_week, device)
    elif model_name.startswith("ours_"):
        results = run_inference(
            model_path, bundle_week, bundle_day, device, use_postprocessing
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    metrics = evaluate_residual_scores(
        results["residual"], results["y_true"],
        results["idx_val"], results["idx_test"],
    )
    
    print(f"\n{model_name.upper()} Results:")
    print(f"  Threshold: {metrics['threshold']:.6f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Run baseline models on preprocessed data")
    parser.add_argument("--data_path", type=str, default="/home/kitsch/COVID/data/processed/week_21feat.pt",
                       help="Path to preprocessed weekly bundle (.pt file)")
    parser.add_argument("--save_dir", type=str, default="models/baselines",
                       help="Directory to save trained models")
    parser.add_argument("--models", type=str, nargs="+",
                       choices=["patchtst", "dlinear", "lstm", "aerca", "ours_weekonly",
                               "ours_multiscale", "ours_multiscale_no_lu", "ours_supervised", "all", "baselines",
                               "track_a_all", "track_b_all"],
                       default=["all"],
                       help="Which models to run")
    parser.add_argument("--data_path_day", type=str, default=None,
                       help="Path to preprocessed daily bundle (.pt file)")
    parser.add_argument("--lambda_u", type=float, default=1.0,
                       help="Weight for L_u loss")
    parser.add_argument("--no_use_lu", action='store_true', default=False,
                       help="Explicitly disable L_u loss for multiscale (default: enabled)")
    parser.add_argument("--alpha_cls", type=float, default=1.0,
                       help="Weight for classification loss")
    parser.add_argument("--alpha_reg", type=float, default=0.1,
                       help="Weight for regression loss")
    parser.add_argument("--use_postprocessing", action="store_true",
                       help="Use post-processing")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Handle use_lu flag: if no_use_lu is set, use_lu=False; otherwise None (use model default)
    args.use_lu = False if args.no_use_lu else None
    
    print(f"Loading data from: {args.data_path}")
    bundle_week = load_bundle_week(args.data_path)
    
    bundle_day = None
    if args.data_path_day:
        print(f"Loading day data from: {args.data_path_day}")
        bundle_day = load_bundle_day(args.data_path_day)
    
    if "all" in args.models:
        models_to_run = ["patchtst", "dlinear", "lstm", "aerca", "ours_weekonly", "ours_multiscale", "ours_supervised"]
    elif "baselines" in args.models:
        models_to_run = ["patchtst", "dlinear", "lstm", "aerca"]
    elif "track_a_all" in args.models:
        models_to_run = ["patchtst", "dlinear", "lstm", "ours_weekonly", "ours_multiscale"]
    elif "track_b_all" in args.models:
        models_to_run = ["ours_supervised"]
    else:
        models_to_run = args.models
    
    if bundle_day is None:
        models_need_day = ["ours_multiscale", "ours_multiscale_no_lu", "ours_supervised"]
        models_to_run = [m for m in models_to_run if m not in models_need_day]
    
    all_results = {}
    for model_name in models_to_run:
        try:
            metrics, results = train_and_evaluate(
                model_name=model_name, bundle_week=bundle_week, bundle_day=bundle_day,
                save_dir=args.save_dir, epochs=args.epochs, lambda_u=args.lambda_u,
                alpha_cls=args.alpha_cls, alpha_reg=args.alpha_reg,
                use_postprocessing=args.use_postprocessing, device=args.device,
                use_lu=args.use_lu,
            )
            all_results[model_name] = {'metrics': metrics, 'results': results}
        except Exception as e:
            print(f"\nError training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Summary Comparison")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'F1':<8} {'Precision':<10} {'Recall':<10} {'AUPRC':<8} {'ROC-AUC':<8}")
        print("-" * 60)
        for model_name, res in all_results.items():
            m = res['metrics']
            print(f"{model_name:<12} {m['f1']:<8.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['auprc']:<8.4f} {m['roc_auc']:<8.4f}")


if __name__ == "__main__":
    main()
