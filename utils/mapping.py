import torch
import pandas as pd
from datetime import timedelta


def compute_fine_coarse_mapping(bundle_fine, bundle_coarse):
    fine = bundle_fine['meta']
    coarse = bundle_coarse['meta']

    fine_dates = pd.to_datetime(fine['target_date'])
    coarse_dates = pd.to_datetime(coarse['target_date'])
    fine_states = fine['state'].values
    coarse_states = coarse['state'].values

    N = len(fine)
    mapping = torch.full((N,), -1, dtype=torch.long)

    if len(coarse_dates) == 0:
        return mapping

    time_diff = (coarse_dates[1] - coarse_dates[0]).total_seconds() if len(coarse_dates) > 1 else 86400
    if time_diff < 3600:
        window = timedelta(minutes=60)
    elif time_diff < 86400:
        window = timedelta(hours=1)
    elif time_diff < 604800:
        window = timedelta(days=1)
    else:
        window = timedelta(days=7)

    for i in range(N):
        fs = fine_states[i]
        fd = fine_dates[i]
        idx = (coarse_states == fs).nonzero()[0]
        for j in idx:
            cs = coarse_dates[j]
            ce = cs + window
            if cs <= fd <= ce:
                mapping[i] = j
                break

    return mapping


def add_mapping(bundle_fine, bundle_coarse):
    mapping = compute_fine_coarse_mapping(bundle_fine, bundle_coarse)
    bundle_fine['fine_to_coarse_index'] = mapping
    return bundle_fine
