import torch
import pandas as pd
from datetime import timedelta

def compute_day_week_mapping(bundle_day, bundle_week):
    day = bundle_day['meta']
    week = bundle_week['meta']

    d_dates = pd.to_datetime(day['target_date'])
    w_dates = pd.to_datetime(week['target_date'])
    d_states = day['state'].values
    w_states = week['state'].values

    N = len(day)
    mapping = torch.full((N,), -1, dtype=torch.long)

    for i in range(N):
        ds = d_states[i]
        dd = d_dates[i]
        idx = (w_states == ds).nonzero()[0]
        for j in idx:
            ws = w_dates[j]
            we = ws + timedelta(days=6)
            if ws <= dd <= we:
                mapping[i] = j
                break

    return mapping

def add_mapping(bundle_day, bundle_week):
    mapping = compute_day_week_mapping(bundle_day, bundle_week)
    bundle_day['day_to_week_index'] = mapping
    return bundle_day