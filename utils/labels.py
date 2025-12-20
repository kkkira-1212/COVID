import numpy as np
import pandas as pd

def create_outbreak_labels(
    df, time_col="Date", state_col="State",
    target_col_ret="NewDeaths_return",
    target_col_level="NewDeaths",
    q_up=0.95, min_abs_level=5,
    w_pre=3, w_post=3,
    q_h=0.95, q_flat=0.2, q_trend=0.75
):
    df = df.copy()
    df["outbreak_label"] = 0

    for state in df[state_col].unique():
        g = df[df[state_col] == state].sort_values(time_col).copy()

        r = g[target_col_ret].astype(float)
        level = g[target_col_level].astype(float)

        r_pos = r[r > 0]
        th_extreme = np.percentile(r_pos, q_up * 100) if len(r_pos) else np.inf
        th_min_level = max(min_abs_level, level.quantile(0.20))
        spike_mask = ((r.abs() >= th_extreme) & (level >= th_min_level)).astype(int)

        h = r.diff()
        y_trend = pd.Series(0, index=g.index, dtype=int)

        h_valid = h.dropna()
        th_h = h_valid.abs().quantile(q_h) if len(h_valid) else np.inf

        abs_r = r.abs().dropna()
        th_flat = abs_r.quantile(q_flat) if len(abs_r) else 0.0
        th_trend = abs_r.quantile(q_trend) if len(abs_r) else 0.0
        th_trend = max(th_trend, th_flat * 1.1)

        def classify(mu):
            if abs(mu) <= th_flat: return "flat"
            if mu >= th_trend: return "up"
            if mu <= -th_trend: return "down"
            return "flat"

        for pos in range(1, len(r) - 1):
            if np.isnan(r.iloc[pos - 1]) or np.isnan(r.iloc[pos]) or np.isnan(h.iloc[pos]):
                continue

            turn = (r.iloc[pos - 1] * r.iloc[pos] <= 0) and (abs(h.iloc[pos]) >= th_h)
            if not turn:
                continue

            pre_mu = r.iloc[max(0, pos - w_pre):pos].mean()
            post_mu = r.iloc[pos + 1:min(len(r), pos + 1 + w_post)].mean()
            pre, post = classify(pre_mu), classify(post_mu)

            if (pre == "down" and post == "flat") or (pre == "flat" and post == "up"):
                y_trend.iloc[pos] = 1

        y_final = (spike_mask | y_trend).astype(int)
        df.loc[g.index, "outbreak_label"] = y_final

    return df
