"""Preprocesamiento con diagnóstico y transcripción con alineación CTC."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src import preprocessing as pp
from src.tokenizer import BLANK_ID, default_tokenizer
from src.visualizers import dominant_hand, hand_tensor

from lib.data import fill_missing_columns, get_model

FACE_COLUMNS = 468 * 3


@dataclass
class Prepared:
    features: np.ndarray       # [T, N_FEATURES] tras el pipeline completo
    mask: np.ndarray           # [T]
    steps: list[tuple[str, str, str]]  # (número, paso, resultado)
    hand: str
    mirrored: bool
    raw_frames: int
    missing_columns: list[str]

    @property
    def valid(self) -> np.ndarray:
        return self.features[self.mask]


def prepare(raw: pd.DataFrame) -> Prepared:
    """Aplica preprocess_sequence (el mismo del entrenamiento) y reconstruye qué hizo cada paso."""
    raw, missing = fill_missing_columns(raw, pp.FEATURE_COLUMNS)
    raw = raw.sort_values("frame").reset_index(drop=True)
    features, mask = pp.preprocess_sequence(raw)

    hand = dominant_hand(raw)
    coords = hand_tensor(raw, hand)
    observed = np.isfinite(coords).all(axis=(1, 2))
    outliers = pp._velocity_outliers(coords)
    coords[outliers] = np.nan
    present = np.isfinite(coords).all(axis=(1, 2))
    n_raw = len(raw)
    if present.any():
        first, last = np.flatnonzero(present)[[0, -1]]
        trimmed = n_raw - (last - first + 1)
        present_in_window = int(present[first:last + 1].sum())
    else:
        trimmed, present_in_window = n_raw, 0
    interpolated = int(mask.sum()) - present_in_window
    face_dropped = sum(c.startswith(("x_face", "y_face", "z_face")) for c in raw.columns)
    read_cols = len(pp.FEATURE_COLUMNS) - 1 - len([c for c in missing if c != "frame"])
    side = "derecha" if hand == "right_hand" else "izquierda (se refleja)"
    coverage = pp.sequence_coverage(mask)

    steps = [
        ("1", "Canales", f"{read_cols} coordenadas de manos y pose"
                         + (f", {face_dropped} de cara descartadas" if face_dropped else "")),
        ("2", "Mano activa", f"{side}, visible en {observed.mean():.0%} de los frames"),
        ("3", "Picos imposibles invalidados", f"{int(outliers.sum())} frames"),
        ("4", "Recorte de extremos sin mano", f"{trimmed} frames"),
        ("5", f"Huecos de hasta {pp.MAX_GAP} frames interpolados", f"{max(interpolated, 0)} frames"),
        ("6", "Normalización", "muñeca al origen, escala de palma y hombros"),
        ("7", "Máscara de validez", f"{int(mask.sum())} de {len(mask)} frames ({coverage:.0%})"),
    ]
    return Prepared(features, mask, steps, hand, hand == "left_hand", n_raw, missing)


@dataclass
class Transcription:
    key: str
    text: str
    frame_ids: np.ndarray      # token argmax por frame válido
    frame_conf: np.ndarray     # probabilidad del token elegido
    milliseconds: float


@torch.inference_mode()
def transcribe(key: str, prepared: Prepared) -> Transcription:
    model = get_model(key)
    valid = np.ascontiguousarray(prepared.valid, dtype=np.float32)
    tokenizer = default_tokenizer()
    if len(valid) == 0:
        return Transcription(key, "", np.zeros(0, int), np.zeros(0), 0.0)
    batch = torch.from_numpy(valid).unsqueeze(0)
    lengths = torch.tensor([len(valid)])
    started = time.perf_counter()
    log_probs = model(batch, lengths)[0]
    best = log_probs.max(dim=-1)
    ids = best.indices.numpy()
    text = tokenizer.decode(ids.tolist())
    elapsed = (time.perf_counter() - started) * 1000
    return Transcription(key, text, ids, best.values.exp().numpy(), elapsed)


def emitted_so_far(ids: np.ndarray) -> list[str]:
    """Texto acumulado frame a frame con el decodificado greedy de CTC."""
    tokenizer = default_tokenizer()
    out, text, previous = [], "", None
    for token in ids:
        token = int(token)
        if token != previous and token != BLANK_ID:
            text += tokenizer.id_to_char[token]
        previous = token
        out.append(text)
    return out


def token_char(token: int) -> str:
    return "" if token == BLANK_ID else default_tokenizer().id_to_char[int(token)]
