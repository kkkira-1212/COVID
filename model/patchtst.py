import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from .trainer import to_device


class PatchTSTBaseline(nn.Module):
    def __init__(self, input_dim, patch_len=4, d_model=64, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.embed = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=patch_len,
        )
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)          
        patches = self.embed(x)        
        patches = patches.mean(dim=-1)  
        out = self.proj(patches).squeeze(1)
        return out


def train_patchtst_forecast(
    bundle_week,
    save_path,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    patch_len=4,
    d_model=64,
    dropout=0.1,
    patience_limit=30,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = to_device(bundle_week, device)

    Xw = data["X_seq"]
    y_reg = data["NewDeaths_ret_next"]
    y_cls = data["y_next"]
    idx_tr = data["idx_train"]
    idx_val = data["idx_val"]

    model = PatchTSTBaseline(
        input_dim=Xw.shape[2],
        patch_len=patch_len,
        d_model=d_model,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    best_roc_auc = 0.0
    patience = 0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xw)
        loss = mse(pred[idx_tr], y_reg[idx_tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_all = model(Xw)
            residuals = (y_reg - pred_all).abs()
            res_val = residuals[idx_val].detach().cpu().numpy()
            y_val = y_cls[idx_val].detach().cpu().numpy()
            if len(np.unique(y_val)) >= 2:
                roc_auc = roc_auc_score(y_val, res_val)
            else:
                roc_auc = 0.0

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            patience = 0
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "config": {
                        "patch_len": patch_len,
                        "d_model": d_model,
                        "dropout": dropout,
                    },
                },
                save_path,
            )
        else:
            patience += 1
            if patience >= patience_limit:
                break

    return save_path


def inference_patchtst(model_path, bundle_week, device="cuda"):
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint["config"]

    data = to_device(bundle_week, device)
    Xw = data["X_seq"]

    model = PatchTSTBaseline(
        input_dim=Xw.shape[2],
        patch_len=cfg.get("patch_len", 4),
        d_model=cfg.get("d_model", 64),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        pred = model(Xw)

    residual = (data["NewDeaths_ret_next"] - pred).abs().cpu().numpy()
    # idx_train = data["idx_train"].cpu().numpy()
    # residual_mean = residual[idx_train].mean()
    # residual_std = residual[idx_train].std()
    # if residual_std > 1e-8:
    #     residual = (residual - residual_mean) / (residual_std + 1e-8)

    return {
        "y_pred": pred.cpu().numpy(),
        "residual": residual,
        "y_true": data["y_next"].cpu().numpy(),
        "idx_val": data["idx_val"].cpu().numpy(),
        "idx_test": data["idx_test"].cpu().numpy(),
    }


__all__ = ["PatchTSTBaseline", "train_patchtst_forecast", "inference_patchtst"]


