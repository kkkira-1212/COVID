from pathlib import Path

import torch


def load_data_bundles(data_dir, coarse_file, fine_file=None):
    data_dir = Path(data_dir)
    bundle_coarse = torch.load(data_dir / coarse_file, weights_only=False)
    bundle_fine = None
    if fine_file is not None:
        bundle_fine = torch.load(data_dir / fine_file, weights_only=False)
    return bundle_coarse, bundle_fine
