import torch
import torch.nn as nn
import numpy as np

class TransformerSeqEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.3, max_seq_len=128,
                 pooling='last'):
        super().__init__()
        self.pooling = pooling
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.register_buffer(
            'pos_encoding',
            self._build_pos_encoding(max_seq_len, d_model)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _build_pos_encoding(self, max_len, d_model):
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :T]
        x = self.encoder(x)

        if self.pooling == 'last':
            h = x[:, -1]
        else:
            h = x.mean(dim=1)

        return self.output_proj(h)


class PredictionHeads(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_in // 2, 1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, 1)
        )


    def forward(self, z_day, z_week):
        logit_day = self.classifier(z_day).squeeze(1)
        logit_week = self.classifier(z_week).squeeze(1)
        pred_day = self.regressor(z_day).squeeze(1)
        pred_week = self.regressor(z_week).squeeze(1)
        return logit_day, logit_week, pred_day, pred_week


class RegressionHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.regressor_day = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_in, 1)
        )
        self.regressor_week = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_in, 1)
        )

    def forward(self, z_day, z_week):
        pred_day = self.regressor_day(z_day).squeeze(1)
        pred_week = self.regressor_week(z_week).squeeze(1)
        return pred_day, pred_week
