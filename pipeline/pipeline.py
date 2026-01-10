import os
from utils.data_utils import loader, engineer_features, group_by_state, agg_map
from utils.labels import create_outbreak_labels
from utils.sequences import create_sequences, split_sequences, sequences_to_bundle, standardize
from utils.mapping import add_mapping
from model.trainer import train as train_with_lu
from model.evaluator import infer, evaluate
from model.patchtst import train_patchtst_forecast, inference_patchtst
from model.dlinear import train_dlinear_forecast, inference_dlinear
from model.lstm import train_lstm_forecast, inference_lstm

DEFAULT_FEATURES = [
    'NewCases', 'NewDeaths',
    'NewCases_MA7', 'NewDeaths_MA7',
    'Cases_GrowthRate', 'NewDeaths_return',
    'Patience_Count',
    'Vax_AllDoses', 'Vax_Dose1', 'Vax_Dose2',
    'Vax_Dose3',
    'Hosp_Count', 'Hosp_Deaths',
    'Ct_Value', 'Stringency_Index',
    'Aver_Hosp_Stay',
    'TotalDeaths_by_TotalCases', 'Hosp_Death_Rate',
    'TotalCases', 'TotalDeaths',
    'TotalCases_100k_inhab', 'TotalDeaths_100k_inhab'
]

class COVIDPipeline:
    def __init__(
        self,
        data_path,
        feature_cols=None,
        window_day=14,
        window_week=4,
        train_ratio=0.7,
        val_ratio=0.1,
        mapping=True,
        model_name="lu",
    ):
        self.data_path = data_path
        self.feature_cols = feature_cols or DEFAULT_FEATURES
        self.window_day = window_day
        self.window_week = window_week
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.mapping = mapping
        self.model_name = model_name
        self.bundle_day = None
        self.bundle_week = None

    def load_data(self):
        df = loader(self.data_path)
        df = engineer_features(df)
        return df

    def create_labels(self, df):
        df = create_outbreak_labels(df)
        return df

    def prepare_data(self, df):
        day_data = group_by_state(df, freq='D', agg_map=agg_map)
        week_data = group_by_state(df, freq='W', agg_map=agg_map)
        return day_data, week_data

    def build_sequences(self, day_data, week_data):
        seq_day, states_day = create_sequences(
            day_data, self.feature_cols, self.window_day, 1, 'D'
        )
        seq_week, states_week = create_sequences(
            week_data, self.feature_cols, self.window_week, 1, 'W'
        )
        return seq_day, states_day, seq_week, states_week

    def split_and_bundle(self, seq_day, states_day, seq_week, states_week):
        idx_tr_d, idx_v_d, idx_te_d = split_sequences(
            seq_day, self.train_ratio, self.val_ratio
        )
        idx_tr_w, idx_v_w, idx_te_w = split_sequences(
            seq_week, self.train_ratio, self.val_ratio
        )

        bundle_day = sequences_to_bundle(
            seq_day, idx_tr_d, idx_v_d, idx_te_d,
            states_day, self.feature_cols, self.window_day
        )
        bundle_week = sequences_to_bundle(
            seq_week, idx_tr_w, idx_v_w, idx_te_w,
            states_week, self.feature_cols, self.window_week
        )

        bundle_day = standardize(bundle_day)
        bundle_week = standardize(bundle_week)

        if self.mapping:
            bundle_day = add_mapping(bundle_day, bundle_week)

        return bundle_day, bundle_week

    def build_dataset(self):
        df = self.load_data()
        df = self.create_labels(df)
        day_data, week_data = self.prepare_data(df)
        seq_day, states_day, seq_week, states_week = self.build_sequences(
            day_data, week_data
        )
        self.bundle_day, self.bundle_week = self.split_and_bundle(
            seq_day, states_day, seq_week, states_week
        )
        return self.bundle_day, self.bundle_week

    def train(
        self,
        save_path,
        epochs=200,
        lambda_u=1.0,
        patience_limit=40,
        d_model=64,
        nhead=4,
        num_layers=2,
        pooling="last",
    ):
        if self.bundle_day is None or self.bundle_week is None:
            self.build_dataset()

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        if self.model_name == "lu":
            model_path = train_with_lu(
                self.bundle_day,
                self.bundle_week,
                save_path,
                epochs=epochs,
                lambda_u=lambda_u,
                patience_limit=patience_limit,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                pooling=pooling,
            )
        elif self.model_name == "patchtst":
            model_path = train_patchtst_forecast(
                self.bundle_week,
                save_path,
                epochs=epochs,
            )
        elif self.model_name == "dlinear":
            model_path = train_dlinear_forecast(
                self.bundle_week,
                save_path,
                epochs=epochs,
            )
        elif self.model_name == "lstm":
            model_path = train_lstm_forecast(
                self.bundle_week,
                save_path,
                epochs=epochs,
            )
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

        return model_path

    def evaluate(
        self,
        model_path,
        use_2b=False,
        device="cuda",
    ):
        if self.bundle_day is None or self.bundle_week is None:
            self.build_dataset()

        if self.model_name == "lu":
            out = infer(model_path, self.bundle_week, self.bundle_day, device)
        elif self.model_name == "patchtst":
            out = inference_patchtst(model_path, self.bundle_week, device)
        elif self.model_name == "dlinear":
            out = inference_dlinear(model_path, self.bundle_week, device)
        elif self.model_name == "lstm":
            out = inference_lstm(model_path, self.bundle_week, device)
        else:
            raise ValueError(f"Unknown model_name for evaluation: {self.model_name}")

        metrics = evaluate(
            out["residual"],
            out["y_true"],
            out["idx_val"],
            out["idx_test"],
        )
        return {"weekly_residual": metrics}, out


def run_full_pipeline(
    data_path,
    model_save_path,
    feature_cols=None,
    window_day=14,
    window_week=4,
    train_ratio=0.7,
    val_ratio=0.1,
    mapping=True,
    model_name="lu",
    epochs=200,
    lambda_u=1.0,
    patience_limit=40,
    d_model=64,
    nhead=4,
    num_layers=2,
    pooling="last",
    evaluate_model=True,
    use_2b=False,
    device='cuda'
):
    pipeline = COVIDPipeline(
        data_path=data_path,
        feature_cols=feature_cols,
        window_day=window_day,
        window_week=window_week,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        mapping=mapping,
        model_name=model_name,
    )

    print("Building dataset...")
    bundle_day, bundle_week = pipeline.build_dataset()
    print(f"Dataset built: {len(bundle_day['X_seq'])} day sequences, {len(bundle_week['X_seq'])} week sequences")

    print(f"Training model ({model_name})...")
    model_path = pipeline.train(
        save_path=model_save_path,
        epochs=epochs,
        lambda_u=lambda_u,
        patience_limit=patience_limit,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        pooling=pooling
    )
    print(f"Model saved to: {model_path}")

    if evaluate_model:
        print("Evaluating model...")
        eval_results, inference_results = pipeline.evaluate(
            model_path=model_path,
            use_2b=use_2b,
            device=device
        )
        print("\nEvaluation Results:")
        if model_name == "lu":
            print(f"Day - F1: {eval_results['day']['f1']:.4f}, AUPRC: {eval_results['day']['auprc']:.4f}, ROC-AUC: {eval_results['day']['roc_auc']:.4f}")
            print(f"Week - F1: {eval_results['week']['f1']:.4f}, AUPRC: {eval_results['week']['auprc']:.4f}, ROC-AUC: {eval_results['week']['roc_auc']:.4f}")
            if use_2b:
                print(f"Day (2B) - F1: {eval_results['day_2b']['f1']:.4f}, AUPRC: {eval_results['day_2b']['auprc']:.4f}, ROC-AUC: {eval_results['day_2b']['roc_auc']:.4f}")
                print(f"Best strength: {eval_results['strength']:.2f}")
        else:
            m = eval_results["weekly_residual"]
            print(f"Weekly residual - F1: {m['f1']:.4f}, AUPRC: {m['auprc']:.4f}, ROC-AUC: {m['roc_auc']:.4f}")
            print(f"Threshold: {m['threshold']:.4f}")

        return model_path, eval_results, inference_results

    return model_path, None, None


if __name__ == "__main__":
    data_path = "data/3.1_3.2_Final_Dataset_State_Level.xlsx"
    model_save_path = "models/covid_model.pt"

    model_path, eval_results, inference_results = run_full_pipeline(
        data_path=data_path,
        model_save_path=model_save_path,
        epochs=200,
        evaluate_model=True,
        use_2b=True
    )

