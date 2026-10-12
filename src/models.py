"""Arquitecturas CTC (BiGRU, TCN y Transformer) y contrato de inferencia para la aplicación."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.tokenizer import default_tokenizer


class BiGRUCTC(nn.Module):
    def __init__(self, n_features: int, vocab_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.config = dict(n_features=n_features, vocab_size=vocab_size,
                           hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.encoder = nn.GRU(n_features, hidden_size, num_layers=num_layers,
                              dropout=dropout if num_layers > 1 else 0.0,
                              bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        padded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.shape[1])
        return self.classifier(padded).log_softmax(dim=-1)  # [B, T, vocab]


def padding_mask(lengths: torch.Tensor, total_length: int) -> torch.Tensor:
    """True en las posiciones de relleno: [B, T]."""
    positions = torch.arange(total_length, device=lengths.device)
    return positions[None, :] >= lengths[:, None]


class TemporalBlock(nn.Module):
    """Convolución dilatada no causal con conexión residual; el relleno se mantiene en cero."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(torch.nn.functional.gelu(self.norm(y)))
        return (x + y).masked_fill(pad[..., None], 0.0)


class TCNCTC(nn.Module):
    """Red convolucional temporal: dilataciones crecientes amplían el contexto sin recurrencia."""

    def __init__(self, n_features: int, vocab_size: int, channels: int = 192,
                 levels: int = 6, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        self.config = dict(n_features=n_features, vocab_size=vocab_size, channels=channels,
                           levels=levels, kernel_size=kernel_size, dropout=dropout)
        self.input = nn.Linear(n_features, channels)
        # Dilataciones 1, 2, 4, ... reiniciadas cada 4 niveles para no dejar huecos en el campo receptivo
        self.blocks = nn.ModuleList(
            TemporalBlock(channels, kernel_size, 2 ** (level % 4), dropout) for level in range(levels))
        self.classifier = nn.Linear(channels, vocab_size)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pad = padding_mask(lengths.to(features.device), features.shape[1])
        x = self.input(features).masked_fill(pad[..., None], 0.0)
        for block in self.blocks:
            x = block(x, pad)
        return self.classifier(x).log_softmax(dim=-1)  # [B, T, vocab]


class TransformerCTC(nn.Module):
    """Encoder Transformer con codificación posicional sinusoidal y máscara de relleno."""

    def __init__(self, n_features: int, vocab_size: int, d_model: int = 192, num_layers: int = 4,
                 num_heads: int = 4, ff_size: int = 384, dropout: float = 0.1):
        super().__init__()
        self.config = dict(n_features=n_features, vocab_size=vocab_size, d_model=d_model,
                           num_layers=num_layers, num_heads=num_heads, ff_size=ff_size, dropout=dropout)
        self.input = nn.Sequential(nn.Linear(n_features, d_model), nn.LayerNorm(d_model))
        layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_size, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)

    @staticmethod
    def positional_encoding(length: int, d_model: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(length, d_model, device=device)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        return encoding

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pad = padding_mask(lengths.to(features.device), features.shape[1])
        x = self.input(features)
        x = x + self.positional_encoding(x.shape[1], x.shape[2], x.device)
        x = self.norm(self.encoder(x, src_key_padding_mask=pad))
        return self.classifier(x).log_softmax(dim=-1)  # [B, T, vocab]


# Clave del modelo -> (clase, nombre del archivo de pesos, etiqueta)
MODEL_REGISTRY = {
    "m1": (BiGRUCTC, "m1_bigru_ctc.pt", "M1 BiGRU + CTC"),
    "m2": (TCNCTC, "m2_tcn_ctc.pt", "M2 TCN + CTC"),
    "m3": (TransformerCTC, "m3_transformer_ctc.pt", "M3 Transformer + CTC"),
}
ALIASES = {"bigru": "m1", "bigru_ctc": "m1", "tcn": "m2", "tcn_ctc": "m2",
           "transformer": "m3", "transformer_ctc": "m3"}


def model_key(name: str) -> str:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Modelo no disponible: {name}")
    return key


def build_model(name: str, n_features: int, vocab_size: int, **params) -> nn.Module:
    return MODEL_REGISTRY[model_key(name)][0](n_features, vocab_size, **params)


def load_model(name: str, path: str | Path) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Los checkpoints anteriores a M2/M3 no guardaban la arquitectura: eran todos M1
    key = model_key(checkpoint.get("architecture", name))
    model = MODEL_REGISTRY[key][0](**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def predict(model: nn.Module, features: np.ndarray, mask: np.ndarray) -> str:
    valid = np.asarray(features, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    if len(valid) == 0:
        return ""
    device = next(model.parameters()).device
    model.eval()
    batch = torch.from_numpy(valid.copy()).unsqueeze(0).to(device)
    lengths = torch.tensor([len(valid)], dtype=torch.long)
    ids = model(batch, lengths)[0, :len(valid)].argmax(dim=-1).tolist()
    return default_tokenizer().decode(ids)
