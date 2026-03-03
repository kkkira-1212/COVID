from pathlib import Path

import numpy as np
import torch

from model.encoder import TransformerSeqEncoder, RegressionHeadWithRelation


DATASET_FILE_CANDIDATES = {
    "psm": {
        "coarse": ["psm_coarse.pt"],
        "fine": ["psm_fine.pt"],
    },
    "swat": {
        "coarse": ["swat_hour.pt", "swat_coarse.pt"],
        "fine": ["swat_minute.pt", "swat_fine.pt"],
    },
    "smap": {
        "coarse": ["smap_coarse.pt"],
        "fine": ["smap_fine.pt"],
    },
}


def resolve_bundle_path(data_dir, dataset, scale, data_file=None):
    data_dir = Path(data_dir)
    if data_file is not None:
        bundle_path = Path(data_file)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Data file not found: {bundle_path}")
        return bundle_path, dataset if dataset != "auto" else "custom"

    if dataset == "auto":
        for candidate_dataset, scale_files in DATASET_FILE_CANDIDATES.items():
            for filename in scale_files.get(scale, []):
                if (data_dir / filename).exists():
                    return data_dir / filename, candidate_dataset
        raise ValueError(
            f"Cannot auto-detect dataset from {data_dir}. "
            f"Please pass --dataset or --data_file explicitly."
        )

    if dataset not in DATASET_FILE_CANDIDATES:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported: {sorted(DATASET_FILE_CANDIDATES.keys())}"
        )

    for filename in DATASET_FILE_CANDIDATES[dataset].get(scale, []):
        candidate = data_dir / filename
        if candidate.exists():
            return candidate, dataset

    raise FileNotFoundError(
        f"No {scale} bundle found for dataset '{dataset}' in {data_dir}."
    )


def load_model(model_path, num_vars, seq_len, device):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    enc = TransformerSeqEncoder(
        input_dim=num_vars,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        max_seq_len=seq_len + 5,
    ).to(device)
    enc.pooling = config.get("pooling", "last")
    head = RegressionHeadWithRelation(config["d_model"], num_vars).to(device)

    enc_key = "enc_coarse" if "enc_coarse" in checkpoint else "enc_fine"
    enc.load_state_dict(checkpoint[enc_key])
    head.load_state_dict(checkpoint["head"])

    enc.eval()
    head.eval()
    return enc, head


def select_normal_indices(y_true, idx_train, max_windows=None, seed=42, sample_method="random"):
    train_labels = y_true[idx_train]
    normal_mask = train_labels == 0
    normal_indices = idx_train[normal_mask]

    if len(normal_indices) == 0:
        raise ValueError("No normal windows found in training indices.")

    if max_windows is None or len(normal_indices) <= max_windows:
        return normal_indices

    if sample_method == "first":
        return normal_indices[:max_windows]

    rng = np.random.default_rng(seed)
    return rng.choice(normal_indices, size=max_windows, replace=False)
