"""Arquitectura inicial BiGRU + CTC y contrato de inferencia para la aplicación."""

from __future__ import annotations

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


def load_model(name: str, path: str | Path) -> nn.Module:
    if name.lower() not in {"m1", "bigru", "bigru_ctc"}:
        raise ValueError(f"Modelo no disponible: {name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BiGRUCTC(**checkpoint["config"])
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
