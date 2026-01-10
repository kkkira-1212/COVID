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
            self.build_pos_encoding(max_seq_len, d_model)
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

    def build_pos_encoding(self, max_len, d_model):
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


class RegressionHeadWithRelation(nn.Module):
    def __init__(self, d_in, num_vars):
        super().__init__()
        self.num_vars = num_vars
        self.var_relation_matrix = nn.Parameter(torch.randn(num_vars, num_vars) * 0.1)
        self.base_proj_fine = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_in, num_vars)
        )
        self.base_proj_coarse = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_in, num_vars)
        )

    def forward(self, z_fine, z_coarse):
        var_base_fine = self.base_proj_fine(z_fine)
        var_base_coarse = self.base_proj_coarse(z_coarse)
        pred_fine = torch.matmul(var_base_fine, self.var_relation_matrix.T)
        pred_coarse = torch.matmul(var_base_coarse, self.var_relation_matrix.T)
        return pred_fine, pred_coarse
