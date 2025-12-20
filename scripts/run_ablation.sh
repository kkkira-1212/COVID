#!/bin/bash
set -e

DATA_TYPE=${1:-6feat}
EXP_TYPE=${2:-all}

if [ "$DATA_TYPE" = "6feat" ]; then
    DATA_WEEK="data/processed/week_6feat.pt"
    DATA_DAY="data/processed/day_6feat.pt"
    SAVE_DIR="models/ablation/6feat"
elif [ "$DATA_TYPE" = "21feat" ]; then
    DATA_WEEK="data/processed/week_21feat.pt"
    DATA_DAY="data/processed/day_21feat.pt"
    SAVE_DIR="models/ablation/21feat"
else
    echo "Error: DATA_TYPE must be '6feat' or '21feat'"
    exit 1
fi

if [ "$EXP_TYPE" = "all" ] || [ "$EXP_TYPE" = "weekonly" ]; then
    python scripts/run_baselines.py \
        --data_path "$DATA_WEEK" \
        --models ours_weekonly \
        --save_dir "$SAVE_DIR/weekonly" \
        --epochs 200
fi

if [ "$EXP_TYPE" = "all" ] || [ "$EXP_TYPE" = "multiscale" ]; then
    python scripts/run_baselines.py \
        --data_path "$DATA_WEEK" \
        --data_path_day "$DATA_DAY" \
        --models ours_multiscale_no_lu \
        --save_dir "$SAVE_DIR/multiscale_no_lu" \
        --epochs 200
fi

if [ "$EXP_TYPE" = "all" ] || [ "$EXP_TYPE" = "lambda" ]; then
    for lambda_u in 0.3 0.5 1.0 1.5; do
        python scripts/run_baselines.py \
            --data_path "$DATA_WEEK" \
            --data_path_day "$DATA_DAY" \
            --models ours_multiscale \
            --lambda_u $lambda_u \
            --save_dir "$SAVE_DIR/lambda_${lambda_u}" \
            --epochs 200
    done
fi

