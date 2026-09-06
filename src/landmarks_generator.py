"""
landmarks_generator.py
Generador de archivos Parquet de *landmarks* representativos para el Proyecto 2
(Google - ASL Fingerspelling Recognition).

Cuando los archivos oficiales `train_landmarks/*.parquet` (≈190 GB en Kaggle) no están
disponibles localmente, este módulo construye una muestra que **replica fielmente el
esquema oficial y las propiedades estadísticas documentadas** del dataset:

* Esquema exacto: `sequence_id`, `frame` y 1,629 columnas de coordenadas
  (3 ejes × 543 puntos: 468 rostro + 21 mano izquierda + 33 pose + 21 mano derecha).
* Ausencias (`NaN`) **en ráfagas** (procesos de Markov de dos estados), no aleatorias
  independientes, tal como ocurre cuando MediaPipe pierde el *tracking*.
* Fuerte asimetría de lateralidad: la mano no dominante permanece ausente la mayor
  parte del tiempo durante el deletreo manual.
* Frames completamente vacíos (fallo total del detector) y *spikes* de jitter
  (saltos biomecánicamente imposibles) para el análisis de outliers.
* Duración de la secuencia condicionada por la longitud de la frase y por la
  velocidad idiosincrásica de cada firmante.

Uso típico:
    from src.landmarks_generator import ensure_landmark_sample
    index_df = ensure_landmark_sample(data_dir="data")
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Especificación oficial del esquema de landmarks
# --------------------------------------------------------------------------------------

#: Número de puntos clave por tipo de landmark (MediaPipe Holistic).
LANDMARK_COUNTS: Dict[str, int] = {
    "face": 468,
    "left_hand": 21,
    "pose": 33,
    "right_hand": 21,
}

#: Orden canónico de los tipos dentro de cada eje, tal como aparece en los Parquet oficiales.
LANDMARK_ORDER: List[str] = ["face", "left_hand", "pose", "right_hand"]

#: Ejes de coordenadas normalizadas de MediaPipe.
AXES: List[str] = ["x", "y", "z"]

TOTAL_LANDMARKS = sum(LANDMARK_COUNTS.values())          # 543
TOTAL_COORD_COLUMNS = TOTAL_LANDMARKS * len(AXES)        # 1,629


def build_landmark_columns() -> List[str]:
    """
    Construye la lista ordenada de las 1,629 columnas de coordenadas del esquema oficial.

    El patrón de nombre es `{eje}_{tipo}_{indice}` (p. ej. `x_right_hand_8`).
    """
    columns: List[str] = []
    for axis in AXES:
        for kind in LANDMARK_ORDER:
            columns.extend(f"{axis}_{kind}_{i}" for i in range(LANDMARK_COUNTS[kind]))
    return columns


# --------------------------------------------------------------------------------------
# Utilidades internas de simulación biomecánica
# --------------------------------------------------------------------------------------

def _smooth_random_walk(n_frames: int, n_dims: int, step: float, rng: np.random.Generator,
                        smooth_window: int = 9) -> np.ndarray:
    """
    Genera una trayectoria suave (paseo aleatorio filtrado por media móvil) de forma
    (n_frames, n_dims). Emula el desplazamiento global lento del cuerpo frente a la cámara.
    """
    steps = rng.normal(0.0, step, size=(n_frames, n_dims))
    walk = np.cumsum(steps, axis=0)
    if n_frames >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        walk = np.stack(
            [np.convolve(walk[:, d], kernel, mode="same") for d in range(n_dims)], axis=1
        )
    return walk


def _markov_missing_mask(n_frames: int, p_enter: float, p_stay: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Máscara booleana de ausencia con estructura de ráfaga (cadena de Markov de 2 estados).

    Parameters
    ----------
    p_enter : probabilidad de perder el tracking estando presente.
    p_stay  : probabilidad de continuar perdido estando ausente (controla el largo del hueco).

    Returns
    -------
    np.ndarray de bool, True = frame ausente (NaN).
    """
    mask = np.zeros(n_frames, dtype=bool)
    missing = rng.random() < 0.35  # el detector puede arrancar ya perdido
    draws = rng.random(n_frames)
    for t in range(n_frames):
        if missing:
            missing = draws[t] < p_stay
        else:
            missing = draws[t] < p_enter
        mask[t] = missing
    return mask


def _generate_group(kind: str, n_frames: int, center: np.ndarray, spread: float,
                    motion: float, rng: np.random.Generator) -> np.ndarray:
    """
    Genera el tensor de coordenadas (n_frames, n_points, 3) de un grupo de landmarks.

    La anatomía se modela como una nube fija de puntos alrededor de un ancla, y el
    movimiento como la suma de una deriva global suave más oscilaciones de mayor
    frecuencia (que en las manos representan la articulación de los dedos).
    """
    n_points = LANDMARK_COUNTS[kind]

    # Geometría anatómica estable del grupo (offsets relativos al ancla).
    offsets = rng.normal(0.0, spread, size=(n_points, 3))
    offsets[:, 2] *= 0.35  # el eje z tiene mucha menor dispersión que x, y

    # Deriva global suave del ancla.
    anchor = center[None, :] + _smooth_random_walk(n_frames, 3, step=motion * 0.02, rng=rng)

    # Oscilación articulatoria: fases y frecuencias distintas por punto.
    t = np.arange(n_frames)[:, None, None]
    freq = rng.uniform(0.08, 0.45, size=(1, n_points, 3))
    phase = rng.uniform(0.0, 2 * np.pi, size=(1, n_points, 3))
    articulation = motion * spread * 0.9 * np.sin(freq * t + phase)

    # Ruido de medición del sensor.
    noise = rng.normal(0.0, 0.0015, size=(n_frames, n_points, 3))

    coords = anchor[:, None, :] + offsets[None, :, :] + articulation + noise
    return coords.astype(np.float32)


def _inject_jitter_spikes(coords: np.ndarray, n_spikes: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Inserta *spikes* de jitter: frames aislados donde el tracking "teletransporta" el
    grupo completo, produciendo velocidades biomecánicamente imposibles.
    """
    n_frames = coords.shape[0]
    if n_frames < 5 or n_spikes <= 0:
        return coords
    idx = rng.choice(np.arange(2, n_frames - 2), size=min(n_spikes, n_frames - 4), replace=False)
    for i in idx:
        coords[i, :, :2] += rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.55)
    return coords


# --------------------------------------------------------------------------------------
# Generación de una secuencia completa
# --------------------------------------------------------------------------------------

def _generate_sequence(n_frames: int, dominant: str, columns: List[str],
                       rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Genera la matriz (n_frames, 1629) de una secuencia y devuelve además los parámetros
    latentes usados (útiles para validar el generador).
    """
    non_dominant = "left_hand" if dominant == "right_hand" else "right_hand"

    # Anclas anatómicas en el espacio normalizado de imagen de MediaPipe.
    face_center = np.array([0.50, 0.32, 0.00]) + rng.normal(0, 0.03, 3)
    pose_center = np.array([0.50, 0.62, 0.00]) + rng.normal(0, 0.04, 3)
    dom_x = 0.63 if dominant == "right_hand" else 0.37
    dom_center = np.array([dom_x, 0.52, 0.00]) + rng.normal(0, 0.04, 3)
    nd_center = np.array([1.0 - dom_x, 0.55, 0.00]) + rng.normal(0, 0.04, 3)

    groups = {
        "face": _generate_group("face", n_frames, face_center, spread=0.055, motion=0.25, rng=rng),
        "pose": _generate_group("pose", n_frames, pose_center, spread=0.140, motion=0.40, rng=rng),
        dominant: _generate_group(dominant, n_frames, dom_center, spread=0.038, motion=1.60, rng=rng),
        non_dominant: _generate_group(non_dominant, n_frames, nd_center, spread=0.038, motion=0.55, rng=rng),
    }

    # --- Spikes de jitter (outliers de tracking) sobre la mano dominante ---
    n_spikes = int(rng.random() < 0.42) * rng.integers(1, 4)
    groups[dominant] = _inject_jitter_spikes(groups[dominant], int(n_spikes), rng)

    # --- Máscaras de ausencia por grupo, con estructura de ráfaga ---
    masks = {
        "face": _markov_missing_mask(n_frames, p_enter=rng.uniform(0.010, 0.035),
                                     p_stay=rng.uniform(0.70, 0.90), rng=rng),
        "pose": _markov_missing_mask(n_frames, p_enter=rng.uniform(0.004, 0.018),
                                     p_stay=rng.uniform(0.60, 0.85), rng=rng),
        dominant: _markov_missing_mask(n_frames, p_enter=rng.uniform(0.010, 0.055),
                                       p_stay=rng.uniform(0.65, 0.88), rng=rng),
        non_dominant: _markov_missing_mask(n_frames, p_enter=rng.uniform(0.15, 0.45),
                                           p_stay=rng.uniform(0.93, 0.995), rng=rng),
    }

    # La mano no dominante está completamente ausente en ~30% de las secuencias.
    if rng.random() < 0.30:
        masks[non_dominant][:] = True

    # En ~6% de las secuencias la mano dominante entra tarde o sale antes del final.
    if rng.random() < 0.06:
        cut = int(n_frames * rng.uniform(0.10, 0.25))
        if rng.random() < 0.5:
            masks[dominant][:cut] = True
        else:
            masks[dominant][n_frames - cut:] = True

    # Frames globalmente vacíos: fallo total del detector en ese instante.
    empty_frames = rng.random(n_frames) < rng.uniform(0.003, 0.020)

    for kind, mask in masks.items():
        combined = mask | empty_frames
        groups[kind][combined, :, :] = np.nan

    # --- Ensamblado en el orden canónico de columnas ---
    matrix = np.empty((n_frames, TOTAL_COORD_COLUMNS), dtype=np.float32)
    col = 0
    for a, axis in enumerate(AXES):
        for kind in LANDMARK_ORDER:
            k = LANDMARK_COUNTS[kind]
            matrix[:, col:col + k] = groups[kind][:, :, a]
            col += k

    meta = {
        "dominant_hand": dominant,
        "empty_frame_rate": float(empty_frames.mean()),
    }
    return matrix, meta


# --------------------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------------------

def generate_landmark_sample(metadata: pd.DataFrame,
                             out_dir: str,
                             n_sequences: int = 120,
                             n_files: int = 3,
                             seed: int = 2026) -> pd.DataFrame:
    """
    Genera `n_sequences` secuencias de landmarks repartidas en `n_files` archivos Parquet
    y devuelve el índice de metadatos correspondiente (subconjunto de `metadata` con las
    rutas y el conteo real de frames).

    Parameters
    ----------
    metadata : DataFrame con al menos las columnas `sequence_id`, `participant_id`, `phrase`.
    out_dir  : directorio destino de los `.parquet`.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    columns = build_landmark_columns()

    sample = metadata.sample(n=min(n_sequences, len(metadata)), random_state=seed).copy()
    sample = sample.reset_index(drop=True)

    # Lateralidad por participante: ~88% de firmantes diestros (proporción poblacional).
    participants = sample["participant_id"].unique()
    handedness = {
        pid: ("right_hand" if rng.random() < 0.88 else "left_hand") for pid in participants
    }
    # Velocidad idiosincrásica de cada firmante (multiplicador de duración).
    speed = {pid: float(np.clip(rng.normal(1.0, 0.22), 0.55, 1.75)) for pid in participants}

    # Asignación de secuencias a archivos Parquet (varias secuencias por archivo, como en Kaggle).
    file_ids = [int(rng.integers(1_000_000, 9_999_999)) for _ in range(n_files)]
    sample["file_id"] = [file_ids[i % n_files] for i in range(len(sample))]
    sample["path"] = sample["file_id"].apply(lambda f: f"train_landmarks/{f}.parquet")

    records = []
    for file_id in file_ids:
        subset = sample[sample["file_id"] == file_id]
        blocks, frame_ids, seq_ids = [], [], []

        for _, row in subset.iterrows():
            pid = row["participant_id"]
            char_len = len(str(row["phrase"]))
            # Duración ≈ arranque + coste por carácter, modulada por la velocidad del firmante.
            mu = (18.0 + 7.4 * char_len) * speed[pid]
            n_frames = int(np.clip(rng.normal(mu, mu * 0.18), 4, 720))

            matrix, meta = _generate_sequence(n_frames, handedness[pid], columns, rng)
            blocks.append(matrix)
            frame_ids.append(np.arange(n_frames, dtype=np.int32))
            seq_ids.append(np.full(n_frames, row["sequence_id"], dtype=np.int64))

            records.append({
                "sequence_id": row["sequence_id"],
                "n_frames": n_frames,
                "true_dominant_hand": meta["dominant_hand"],
            })

        df_file = pd.DataFrame(np.vstack(blocks), columns=columns)
        df_file.insert(0, "frame", np.concatenate(frame_ids))
        df_file.insert(0, "sequence_id", np.concatenate(seq_ids))
        df_file.to_parquet(os.path.join(out_dir, f"{file_id}.parquet"),
                           engine="pyarrow", compression="snappy", index=False)

    index_df = sample.merge(pd.DataFrame(records), on="sequence_id", how="left")
    return index_df


def ensure_landmark_sample(data_dir: str = "data",
                           n_sequences: int = 120,
                           n_files: int = 3,
                           seed: int = 2026,
                           force: bool = False) -> pd.DataFrame:
    """
    Devuelve el índice de secuencias con landmarks disponibles, generando la muestra
    representativa solo si aún no existe en disco (operación idempotente).
    """
    out_dir = os.path.join(data_dir, "train_landmarks")
    index_path = os.path.join(data_dir, "train_landmarks_index.csv")

    if not force and os.path.exists(index_path):
        existing = pd.read_csv(index_path)
        if all(os.path.exists(os.path.join(out_dir, f"{fid}.parquet"))
               for fid in existing["file_id"].unique()):
            return existing

    meta_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(data_dir, "train_sample.csv")
    metadata = pd.read_csv(meta_path)

    index_df = generate_landmark_sample(metadata, out_dir=out_dir,
                                        n_sequences=n_sequences, n_files=n_files, seed=seed)
    index_df.to_csv(index_path, index=False)
    return index_df


if __name__ == "__main__":
    idx = ensure_landmark_sample(data_dir="data", force=True)
    print(idx.head())
    print(f"\nSecuencias generadas : {len(idx):,}")
    print(f"Archivos Parquet     : {idx['file_id'].nunique()}")
    print(f"Frames totales       : {idx['n_frames'].sum():,}")
