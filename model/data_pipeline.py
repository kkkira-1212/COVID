from utils.data_utils import loader, engineer_features, group_by_state
from utils.sequences import create_sequences, split_sequences, sequences_to_bundle, standardize
from utils.mapping import add_mapping

def build_dataset(path, agg_map, feature_cols, window_day, window_week,mapping=True):
    df = loader(path)
    df = engineer_features(df)

    day_data = group_by_state(df, freq='D', agg_map=agg_map)
    week_data = group_by_state(df, freq='W', agg_map=agg_map)

    seq_day, states_day = create_sequences(day_data, feature_cols, window_day, 1, 'D')
    seq_week, states_week = create_sequences(week_data, feature_cols, window_week, 1, 'W')

    idx_tr_d, idx_v_d, idx_te_d = split_sequences(seq_day, 0.7, 0.1)
    idx_tr_w, idx_v_w, idx_te_w = split_sequences(seq_week, 0.7, 0.1)

    bundle_day = sequences_to_bundle(seq_day, idx_tr_d, idx_v_d, idx_te_d, states_day, feature_cols, window_day)
    bundle_week = sequences_to_bundle(seq_week, idx_tr_w, idx_v_w, idx_te_w, states_week, feature_cols, window_week)

    bundle_day = standardize(bundle_day)
    bundle_week = standardize(bundle_week)

    if mapping:
        bundle_day = add_mapping(bundle_day, bundle_week)

    return bundle_day, bundle_week
