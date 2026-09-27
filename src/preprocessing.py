"""
preprocessing.py
Pipeline de limpieza y extracción de características para secuencias de fingerspelling.

Implementa las decisiones del análisis exploratorio (Parte II del EDA) como una sola
función, `preprocess_sequence`, que se usa tanto para preparar el conjunto de
entrenamiento como en la aplicación. Usar el mismo código en ambos lados garantiza
que el modelo reciba exactamente el mismo tipo de entrada al entrenar y al predecir.

Pasos (en orden):
1. Selección de canales: manos + subconjunto de pose; la malla facial se descarta.
2. Mano activa: la de mayor proporción de frames observados. Si es la izquierda, la
   secuencia se refleja horizontalmente para que todas parezcan diestras.
3. Invalidación de outliers biomecánicos: picos de velocidad marcados a la vez por el
   criterio robusto (MAD) y por el umbral físico.
4. Recorte de tramos ausentes al inicio y al final.
5. Interpolación lineal solo en huecos internos de hasta MAX_GAP frames.
6. Normalización: mano centrada en la muñeca y escalada por la palma; pose y posición
   de la muñeca referidas al centro y ancho de hombros.
7. Máscara de validez por frame. Los frames no recuperables se rellenan con 0 en el
   espacio normalizado solo para que el tensor sea numérico; la máscara indica al
   modelo que debe ignorarlos (no es imputación).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.visualizers import dominant_hand, hand_tensor, normalize_hand

# --------------------------------------------------------------------------------------
# PARÁMETROS DEL PIPELINE (justificados en el EDA, Parte II)
# --------------------------------------------------------------------------------------

MAX_GAP = 5                 # huecos internos más largos no se imputan
VELOCITY_THRESHOLD = 0.15   # umbral físico de velocidad (unidades normalizadas / frame)
MAD_Z_THRESHOLD = 3.5       # umbral del z-score robusto
MIN_COVERAGE = 0.5          # cobertura mínima de la mano activa para usar la secuencia

# Nariz, hombros, codos y muñecas de MediaPipe Pose
POSE_POINTS = (0, 11, 12, 13, 14, 15, 16)
# Permutación que intercambia izquierda/derecha dentro de POSE_POINTS al reflejar
POSE_MIRROR_ORDER = (0, 2, 1, 4, 3, 6, 5)
LEFT_SHOULDER, RIGHT_SHOULDER = 1, 2   # índices dentro de POSE_POINTS

HAND_POINTS = 21

FEATURE_COLUMNS: list[str] = (
    ["frame"]
    + [f"{axis}_{hand}_{i}" for hand in ("left_hand", "right_hand")
       for axis in "xyz" for i in range(HAND_POINTS)]
    + [f"{axis}_pose_{i}" for axis in "xy" for i in POSE_POINTS]
)

# Mano activa normalizada (21 x 3) + muñeca en coordenadas del cuerpo (2)
# + pose en coordenadas del cuerpo (7 x 2) + presencia de la otra mano (1)
N_FEATURES = HAND_POINTS * 3 + 2 + len(POSE_POINTS) * 2 + 1


# --------------------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# --------------------------------------------------------------------------------------

def preprocess_sequence(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Secuencia cruda (formato Kaggle, un solo sequence_id) ->
    (features float32 [T, N_FEATURES], valid_mask bool [T]).

    Si la mano activa no aparece en ningún frame, devuelve arreglos vacíos (T = 0).
    """
    raw = raw.sort_values("frame") if "frame" in raw.columns else raw
    raw = raw.reset_index(drop=True)

    hand = dominant_hand(raw)
    other = "left_hand" if hand == "right_hand" else "right_hand"
    mirror = hand == "left_hand"

    coords = hand_tensor(raw, hand)                                   # (T, 21, 3)
    other_present = np.isfinite(hand_tensor(raw, other)).all(axis=(1, 2))
    pose = _pose_tensor(raw)                                          # (T, 7, 2)

    coords[_velocity_outliers(coords)] = np.nan

    present = np.isfinite(coords).all(axis=(1, 2))
    if not present.any():
        return np.zeros((0, N_FEATURES), np.float32), np.zeros(0, bool)
    first, last = np.flatnonzero(present)[[0, -1]]
    window = slice(first, last + 1)
    coords, pose, other_present = coords[window], pose[window], other_present[window]

    coords = _interpolate_short_gaps(coords, MAX_GAP)
    mask = np.isfinite(coords).all(axis=(1, 2))

    hand_features = normalize_hand(coords)
    if mirror:
        hand_features[..., 0] *= -1

    center, scale = _body_frame(pose)
    wrist_body = (coords[:, 0, :2] - center) / scale[:, None]
    pose_body = (pose - center[:, None, :]) / scale[:, None, None]
    if mirror:
        wrist_body[:, 0] *= -1
        pose_body[..., 0] *= -1
        pose_body = pose_body[:, POSE_MIRROR_ORDER, :]

    features = np.concatenate([
        hand_features.reshape(len(coords), -1),
        wrist_body,
        pose_body.reshape(len(coords), -1),
        other_present[:, None].astype(float),
    ], axis=1)
    features[~mask] = 0.0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32), mask


def sequence_coverage(mask: np.ndarray) -> float:
    """Proporción de frames válidos tras la limpieza (0 si la secuencia quedó vacía)."""
    return float(mask.mean()) if len(mask) else 0.0


def is_usable(mask: np.ndarray) -> bool:
    """Criterio de descarte del EDA: la mano activa debe cubrir al menos MIN_COVERAGE."""
    return sequence_coverage(mask) >= MIN_COVERAGE


# --------------------------------------------------------------------------------------
# PASOS INTERNOS
# --------------------------------------------------------------------------------------

def _pose_tensor(raw: pd.DataFrame) -> np.ndarray:
    """Coordenadas (x, y) de POSE_POINTS como tensor (T, 7, 2)."""
    return np.stack([
        raw[[f"x_pose_{i}" for i in POSE_POINTS]].to_numpy(float),
        raw[[f"y_pose_{i}" for i in POSE_POINTS]].to_numpy(float),
    ], axis=-1)


def _velocity_outliers(coords: np.ndarray) -> np.ndarray:
    """Frames cuyo salto de centroide es outlier por MAD y por umbral físico a la vez."""
    with warnings.catch_warnings():
        # Los frames sin mano dan un centroide NaN; es el comportamiento esperado
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centroid = np.nanmean(coords[..., :2], axis=1)
    velocity = np.full(len(coords), np.nan)
    velocity[1:] = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
    if np.isnan(velocity).all():
        return np.zeros(len(coords), bool)

    median = np.nanmedian(velocity)
    mad = np.nanmedian(np.abs(velocity - median))
    z = 0.6745 * (velocity - median) / mad if mad > 0 else np.zeros_like(velocity)
    return (np.abs(np.nan_to_num(z)) > MAD_Z_THRESHOLD) & (np.nan_to_num(velocity) > VELOCITY_THRESHOLD)


def _interpolate_short_gaps(coords: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolación lineal de huecos internos completos de longitud <= max_gap.

    Se imputa el hueco entero o no se imputa: los huecos largos quedan en NaN.
    Supone que el primer y el último frame están presentes (ya recortados).
    """
    coords = coords.copy()
    missing = ~np.isfinite(coords).all(axis=(1, 2))
    edges = np.diff(np.concatenate(([False], missing, [False])).astype(np.int8))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        length = end - start
        if length > max_gap:
            continue
        before, after = coords[start - 1], coords[end]
        weights = (np.arange(1, length + 1) / (length + 1))[:, None, None]
        coords[start:end] = before + weights * (after - before)
    return coords


def _body_frame(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centro (punto medio de hombros) y escala (ancho de hombros) por frame.

    Si faltan los hombros en algún frame, se usa la mediana de la secuencia; si nunca
    aparecen, el origen queda en 0 con escala 1 (sin normalizar).
    """
    left, right = pose[:, LEFT_SHOULDER], pose[:, RIGHT_SHOULDER]
    center = (left + right) / 2
    scale = np.linalg.norm(left - right, axis=1)

    valid = np.isfinite(center).all(axis=1) & np.isfinite(scale) & (scale > 1e-6)
    if valid.any():
        center[~valid] = np.median(center[valid], axis=0)
        scale[~valid] = np.median(scale[valid])
    else:
        center[:] = 0.0
        scale[:] = 1.0
    return center, scale
