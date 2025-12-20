import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from .trainer import to_device
from utils.utils import compute_kl_divergence


class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device):
        super(SENNGC, self).__init__()
        self.coeff_nets = nn.ModuleList()
        for k in range(order):
            modules = [nn.Linear(num_vars, hidden_layer_size), nn.ReLU()]
            if num_hidden_layers > 1:
                for j in range(num_hidden_layers - 1):
                    modules.extend([nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()])
            modules.extend([nn.Linear(hidden_layer_size, num_vars**2), nn.Tanh()])
            self.coeff_nets.append(nn.Sequential(*modules))
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers
        self.device = device

    def forward(self, inputs: torch.Tensor):
        if inputs.shape[1] != self.order:
            raise ValueError(f"Input shape mismatch: expected order {self.order}, got {inputs.shape[1]}")
        coeffs = None
        preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        for k in range(self.order):
            coeff_net_k = self.coeff_nets[k]
            coeffs_k = coeff_net_k(inputs[:, k, :])
            coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
            if coeffs is None:
                coeffs = torch.unsqueeze(coeffs_k, 1)
            else:
                coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
            preds = preds + torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        return preds, coeffs


class AERCABaseline(nn.Module):
    def __init__(self, input_dim, window_size, hidden_size=64, num_hidden_layers=3, 
                 encoder_alpha=0.5, decoder_alpha=0.5, encoder_gamma=0.5, decoder_gamma=0.5,
                 encoder_lambda=0.5, decoder_lambda=0.5, beta=0.5):
        super().__init__()
        self.num_vars = input_dim
        self.window_size = window_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.encoder = SENNGC(input_dim, window_size, hidden_size, num_hidden_layers, device)
        self.decoder = SENNGC(input_dim, window_size, hidden_size, num_hidden_layers, device)
        self.decoder_prev = SENNGC(input_dim, window_size, hidden_size, num_hidden_layers, device)
        self.proj = nn.Linear(input_dim, 1)
        self.encoder_alpha = encoder_alpha
        self.decoder_alpha = decoder_alpha
        self.encoder_gamma = encoder_gamma
        self.decoder_gamma = decoder_gamma
        self.encoder_lambda = encoder_lambda
        self.decoder_lambda = decoder_lambda
        self.beta = beta

    def _sparsity_loss(self, coeffs, alpha):
        norm2 = torch.mean(torch.norm(coeffs, dim=1, p=2))
        norm1 = torch.mean(torch.norm(coeffs, dim=1, p=1))
        return (1 - alpha) * norm2 + alpha * norm1

    def _smoothness_loss(self, coeffs):
        if coeffs.shape[1] > 1:
            return torch.norm(coeffs[:, 1:, :, :] - coeffs[:, :-1, :, :], dim=1).mean()
        return torch.tensor(0.0, device=coeffs.device)

    def forward(self, x):
        device = next(self.parameters()).device
        preds_enc, encoder_coeffs = self.encoder(x)
        x_mean = x.mean(dim=1)
        us = preds_enc - x_mean
        kl_div = compute_kl_divergence(us, device) if us.shape[0] > 1 else torch.tensor(0.0, device=device)
        u_expanded = us.unsqueeze(1).expand(-1, self.window_size, -1)
        preds_dec, decoder_coeffs = self.decoder(u_expanded)
        prev_preds, prev_coeffs = self.decoder_prev(x)
        nexts_hat = preds_dec + prev_preds
        return nexts_hat, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us

    def predict(self, x):
        with torch.no_grad():
            nexts_hat, _, _, _, _, _ = self.forward(x)
            return self.proj(nexts_hat).squeeze(1)


def train_aerca_forecast(
    bundle_week,
    save_path,
    epochs=200,
    lr=1e-4,
    weight_decay=1e-4,
    window_size=28,
    hidden_size=64,
    num_hidden_layers=3,
    encoder_alpha=0.5,
    decoder_alpha=0.5,
    encoder_gamma=0.5,
    decoder_gamma=0.5,
    encoder_lambda=0.5,
    decoder_lambda=0.5,
    beta=0.5,
    patience_limit=30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = to_device(bundle_week, device)

    Xw = data["X_seq"]
    y_reg = data["NewDeaths_ret_next"]
    y_cls = data["y_next"]
    idx_tr = data["idx_train"]
    idx_val = data["idx_val"]

    if Xw.shape[1] != window_size:
        window_size = Xw.shape[1]

    model = AERCABaseline(
        input_dim=Xw.shape[2],
        window_size=window_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        encoder_alpha=encoder_alpha,
        decoder_alpha=decoder_alpha,
        encoder_gamma=encoder_gamma,
        decoder_gamma=decoder_gamma,
        encoder_lambda=encoder_lambda,
        decoder_lambda=decoder_lambda,
        beta=beta,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    best_f1 = 0.0
    patience = 0

    def eval_f1(residuals, labels, idx_val):
        res_val = residuals[idx_val].detach().cpu().numpy()
        y_val = labels[idx_val].detach().cpu().numpy()
        ths = np.linspace(np.percentile(res_val, 10), np.percentile(res_val, 95), 25)
        f1s = [f1_score(y_val, (res_val >= t).astype(int), zero_division=0) for t in ths]
        return max(f1s) if len(f1s) else 0.0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        nexts_hat, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = model(Xw[idx_tr])
        pred = model.proj(nexts_hat).squeeze(1)
        
        loss_recon = mse(pred, y_reg[idx_tr])
        loss_encoder_coeffs = model._sparsity_loss(encoder_coeffs, model.encoder_alpha)
        loss_decoder_coeffs = model._sparsity_loss(decoder_coeffs, model.decoder_alpha)
        loss_prev_coeffs = model._sparsity_loss(prev_coeffs, model.decoder_alpha)
        loss_encoder_smooth = model._smoothness_loss(encoder_coeffs)
        loss_decoder_smooth = model._smoothness_loss(decoder_coeffs)
        loss_prev_smooth = model._smoothness_loss(prev_coeffs)
        
        loss = (loss_recon +
                model.encoder_lambda * loss_encoder_coeffs +
                model.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
                model.encoder_gamma * loss_encoder_smooth +
                model.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
                model.beta * kl_div)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            nexts_hat_all, _, _, _, _, _ = model(Xw)
            pred_all = model.proj(nexts_hat_all).squeeze(1)
            residuals = (y_reg - pred_all).abs()
            f1 = eval_f1(residuals, y_cls, idx_val)

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "config": {
                        "window_size": window_size,
                        "hidden_size": hidden_size,
                        "num_hidden_layers": num_hidden_layers,
                        "encoder_alpha": encoder_alpha,
                        "decoder_alpha": decoder_alpha,
                        "encoder_gamma": encoder_gamma,
                        "decoder_gamma": decoder_gamma,
                        "encoder_lambda": encoder_lambda,
                        "decoder_lambda": decoder_lambda,
                        "beta": beta,
                    },
                },
                save_path,
            )
        else:
            patience += 1
            if patience >= patience_limit:
                break

    return save_path


def inference_aerca(model_path, bundle_week, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint["config"]

    data = to_device(bundle_week, device)
    Xw = data["X_seq"]

    model = AERCABaseline(
        input_dim=Xw.shape[2],
        window_size=cfg.get("window_size", Xw.shape[1]),
        hidden_size=cfg.get("hidden_size", 64),
        num_hidden_layers=cfg.get("num_hidden_layers", 3),
        encoder_alpha=cfg.get("encoder_alpha", 0.5),
        decoder_alpha=cfg.get("decoder_alpha", 0.5),
        encoder_gamma=cfg.get("encoder_gamma", 0.5),
        decoder_gamma=cfg.get("decoder_gamma", 0.5),
        encoder_lambda=cfg.get("encoder_lambda", 0.5),
        decoder_lambda=cfg.get("decoder_lambda", 0.5),
        beta=cfg.get("beta", 0.5),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        nexts_hat, _, _, _, _, _ = model(Xw)
        pred = model.proj(nexts_hat).squeeze(1)

    residual = (data["NewDeaths_ret_next"] - pred).abs().cpu().numpy()

    return {
        "y_pred": pred.cpu().numpy(),
        "residual": residual,
        "y_true": data["y_next"].cpu().numpy(),
        "idx_val": data["idx_val"].cpu().numpy(),
        "idx_test": data["idx_test"].cpu().numpy(),
    }


__all__ = ["AERCABaseline", "train_aerca_forecast", "inference_aerca"]
