"""Visualizaciones y métricas geométricas para landmarks de ASL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGERTIPS = {"Pulgar": 4, "Índice": 8, "Medio": 12, "Anular": 16, "Meñique": 20}


def hand_columns(hand: str) -> list[str]:
    """Nombres de las 63 columnas de una mano en el esquema Kaggle."""
    if hand not in {"left_hand", "right_hand"}:
        raise ValueError("hand debe ser 'left_hand' o 'right_hand'.")
    return [f"{axis}_{hand}_{i}" for axis in "xyz" for i in range(21)]


def dominant_hand(sequence: pd.DataFrame) -> str:
    """Estima la mano activa: la que posee mayor proporción de frames observados."""
    rates = {}
    for hand in ("left_hand", "right_hand"):
        cols = hand_columns(hand)
        available = [c for c in cols if c in sequence.columns]
        if not available:
            rates[hand] = 0.0
        else:
            rates[hand] = float(sequence[available].notna().all(axis=1).mean())
    return max(rates, key=rates.get)


def hand_tensor(sequence: pd.DataFrame, hand: str) -> np.ndarray:
    """Convierte una secuencia a tensor ``(frames, 21, 3)`` para una mano."""
    cols = hand_columns(hand)
    missing = [c for c in cols if c not in sequence.columns]
    if missing:
        raise KeyError(f"Faltan columnas de {hand}; ej.: {missing[0]}")
    return np.stack([
        sequence[[f"x_{hand}_{i}" for i in range(21)]].to_numpy(float),
        sequence[[f"y_{hand}_{i}" for i in range(21)]].to_numpy(float),
        sequence[[f"z_{hand}_{i}" for i in range(21)]].to_numpy(float),
    ], axis=-1)


def normalize_hand(coords: np.ndarray) -> np.ndarray:
    """Centra en muñeca y escala por la distancia muñeca–nudillo medio.

    Los frames sin muñeca o con escala nula conservan NaN para no inventar señal.
    """
    coords = np.asarray(coords, dtype=float)
    centered = coords - coords[:, :1, :]
    scale = np.linalg.norm(centered[:, 9, :], axis=1)
    valid = np.isfinite(scale) & (scale > 1e-8)
    result = np.full_like(centered, np.nan)
    result[valid] = centered[valid] / scale[valid, None, None]
    return result


def plot_hand_skeleton(ax, coords: np.ndarray, title: str = "", dimensions: int = 2):
    """Dibuja un frame de mano en 2D o 3D y devuelve el eje."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (21, 3):
        raise ValueError("coords debe tener forma (21, 3).")
    valid = np.isfinite(coords).all(axis=1)
    if dimensions == 3:
        ax.scatter(coords[valid, 0], coords[valid, 1], coords[valid, 2], s=28, c="#0b5fa5")
        for a, b in HAND_CONNECTIONS:
            if valid[a] and valid[b]:
                ax.plot(coords[[a, b], 0], coords[[a, b], 1], coords[[a, b], 2], c="#ef8a17", lw=1.6)
        ax.set_zlabel("z")
    else:
        ax.scatter(coords[valid, 0], coords[valid, 1], s=30, c="#0b5fa5")
        for a, b in HAND_CONNECTIONS:
            if valid[a] and valid[b]:
                ax.plot(coords[[a, b], 0], coords[[a, b], 1], c="#ef8a17", lw=1.7)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="x", ylabel="y", title=title)
    return ax


def fingertip_trajectory(sequence: pd.DataFrame, hand: str, landmark: int = 8) -> pd.DataFrame:
    """Devuelve la trayectoria temporal de una punta de dedo."""
    if not 0 <= landmark < 21:
        raise ValueError("landmark debe estar entre 0 y 20.")
    frame = sequence["frame"].to_numpy() if "frame" in sequence else np.arange(len(sequence))
    return pd.DataFrame({
        "frame": frame,
        "x": sequence[f"x_{hand}_{landmark}"].to_numpy(),
        "y": sequence[f"y_{hand}_{landmark}"].to_numpy(),
        "z": sequence[f"z_{hand}_{landmark}"].to_numpy(),
    })
