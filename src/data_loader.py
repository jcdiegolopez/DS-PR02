"""
data_loader.py
Módulo de carga y preparación de datos para el Proyecto 2 (ASL Fingerspelling Recognition).
Provee funciones para cargar los metadatos de train.csv y el mapa oficial de caracteres.
Si train.csv no existe localmente, puede generar una muestra sintética estadísticamente
representativa para permitir la ejecución inmediata del análisis exploratorio.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict

# Lista oficial de 59 caracteres según la competencia Kaggle ASL Fingerspelling
OFFICIAL_CHARACTERS = [
    " ", "!", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "=", "?", "@",
    "[", "]", "_", "~",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]

def get_char_map(filepath: str = "data/character_to_prediction_index.json") -> Dict[str, int]:
    """
    Carga o genera el diccionario de mapeo oficial de 59 caracteres.
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    char_map = {char: idx for idx, char in enumerate(OFFICIAL_CHARACTERS)}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(char_map, f, indent=2, ensure_ascii=False)
    return char_map


def generate_representative_metadata(n_samples: int = 5000, n_participants: int = 147, seed: int = 42) -> pd.DataFrame:
    """
    Genera un conjunto de metadatos representativo basado en las distribuciones reales
    del dataset de Kaggle (147 participantes, URLs, teléfonos, direcciones y frases comunes).
    Útil para pruebas y exploración cuando el archivo completo aún no ha sido descargado.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Generación de participantes (147 firmantes)
    participant_ids = [random.randint(10000, 99999) for _ in range(n_participants)]
    # Distribución de aportes por participante (con desbalance natural usando Dirichlet/Gamma)
    weights = np.random.gamma(shape=2.5, scale=1.0, size=n_participants)
    weights /= weights.sum()
    
    # Fuentes de texto representativas
    urls = [
        "https://www.google.com", "http://wikipedia.org/asl", "www.cdc.gov/covid",
        "https://github.com/kaggle", "http://weather.com/today", "www.amazon.com/deals",
        "https://youtube.com/watch", "www.nytimes.com/world", "https://apple.com/ios"
    ]
    phones = [
        "(555) 234-5678", "800-555-0199", "+1-555-987-6543", "(212) 555-4321",
        "1-888-456-7890", "(415) 555-8833", "+1-800-222-3344"
    ]
    addresses = [
        "742 evergreen terrace", "123 main street apt 4b", "1600 pennsylvania ave",
        "456 elm st suite 200", "221b baker street", "10 downing street london"
    ]
    common_words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello",
        "world", "welcome", "to", "asl", "fingerspelling", "science", "project",
        "american", "sign", "language", "recognition", "computer", "vision",
        "deep", "learning", "dataset", "evaluation", "accuracy", "model",
        "frequency", "analysis", "university", "guatemala", "engineering"
    ]
    
    rows = []
    for seq_id in range(1, n_samples + 1):
        pid = np.random.choice(participant_ids, p=weights)
        file_id = random.randint(100000, 999999)
        path = f"train_landmarks/{file_id}.parquet"
        
        # Tipo de frase
        cat = random.choices(["url", "phone", "address", "text"], weights=[0.15, 0.15, 0.20, 0.50])[0]
        if cat == "url":
            phrase = random.choice(urls)
        elif cat == "phone":
            phrase = random.choice(phones)
        elif cat == "address":
            phrase = random.choice(addresses)
        else:
            num_words = random.randint(2, 6)
            phrase = " ".join(random.choices(common_words, k=num_words))
            
        rows.append({
            "path": path,
            "file_id": file_id,
            "sequence_id": seq_id,
            "participant_id": pid,
            "phrase": phrase.lower()
        })
        
    return pd.DataFrame(rows)


def load_metadata(data_dir: str = "data", generate_if_missing: bool = True) -> Tuple[pd.DataFrame, bool]:
    """
    Carga train.csv si existe en data_dir, en rutas comunes de Kaggle o genera
    una muestra representativa si no está disponible localmente.
    
    Retorna:
        df: DataFrame de metadatos
        is_synthetic: Booleano que indica si se usó la muestra sintética representativa
    """
    potential_paths = [
        os.path.join(data_dir, "train.csv"),
        "train.csv",
        "/kaggle/input/asl-fingerspelling/train.csv",
        "../data/train.csv"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            print(f"[INFO] Cargando metadatos oficiales desde: {path}")
            df = pd.read_csv(path)
            return df, False
            
    if generate_if_missing:
        print("[INFO] 'train.csv' oficial no encontrado localmente.")
        print("[INFO] Generando muestra representativa de 10,000 secuencias para exploración inmediata...")
        df = generate_representative_metadata(n_samples=10000, n_participants=147)
        save_path = os.path.join(data_dir, "train_sample.csv")
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[INFO] Muestra guardada exitosamente en: {save_path}")
        return df, True
    else:
        raise FileNotFoundError(f"No se encontró 'train.csv' en ninguna de las rutas esperadas: {potential_paths}")


# ======================================================================================
# UTILIDADES DE LANDMARKS Y PARQUET  ---  Persona 2 (Dimensión Temporal, Calidad y Limpieza)
# ======================================================================================
#
# Los archivos `train_landmarks/*.parquet` son la carga pesada del dataset (≈190 GB en el
# set oficial completo). Cargarlos con `pd.read_parquet()` sin proyección de columnas es
# inviable: cada archivo agrupa decenas de secuencias × cientos de frames × 1,629 columnas.
# Las funciones siguientes implementan tres estrategias de acceso eficiente:
#
#   1. Lectura de METADATOS del footer (estadísticas por row-group) → 0 bytes de datos leídos.
#   2. Proyección de columnas (column pruning) → se leen solo los landmarks de interés.
#   3. Filtrado por `sequence_id` (predicate pushdown) → se leen solo los frames necesarios.

from src.landmarks_generator import (  # noqa: E402
    LANDMARK_COUNTS,
    LANDMARK_ORDER,
    AXES,
    TOTAL_LANDMARKS,
    TOTAL_COORD_COLUMNS,
    build_landmark_columns,
)


def get_landmark_columns(kinds=None, axes=None):
    """
    Devuelve los nombres de columnas de landmarks filtrados por tipo y/o eje.

    Parameters
    ----------
    kinds : lista de {'face', 'left_hand', 'pose', 'right_hand'} o None (todos).
    axes  : lista de {'x', 'y', 'z'} o None (todos).

    Examples
    --------
    >>> len(get_landmark_columns(kinds=['right_hand'], axes=['x', 'y']))
    42
    """
    kinds = LANDMARK_ORDER if kinds is None else list(kinds)
    axes = AXES if axes is None else list(axes)
    cols = []
    for axis in axes:
        for kind in kinds:
            cols.extend(f"{axis}_{kind}_{i}" for i in range(LANDMARK_COUNTS[kind]))
    return cols


def landmark_group_of(column: str) -> str:
    """
    Clasifica una columna de coordenadas en su grupo anatómico.
    `'x_right_hand_8'` -> `'right_hand'`.
    """
    for kind in LANDMARK_ORDER:
        if column[2:].startswith(kind):
            return kind
    return "unknown"


def list_landmark_files(data_dir: str = "data") -> list:
    """Rutas absolutas de todos los `.parquet` de landmarks disponibles localmente."""
    import glob
    patterns = [
        os.path.join(data_dir, "train_landmarks", "*.parquet"),
        os.path.join(data_dir, "*.parquet"),
        "/kaggle/input/asl-fingerspelling/train_landmarks/*.parquet",
    ]
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            return found
    return []


def parquet_schema_report(path: str) -> pd.DataFrame:
    """
    Resumen estructural de un archivo Parquet sin materializar los datos en memoria:
    filas, row-groups, tamaño en disco, compresión y número de columnas por grupo anatómico.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    md = pf.metadata
    schema_cols = [c for c in pf.schema_arrow.names if c not in ("sequence_id", "frame")]
    by_group = pd.Series([landmark_group_of(c) for c in schema_cols]).value_counts()

    compression = md.row_group(0).column(0).compression if md.num_row_groups else "N/A"
    n_sequences = len(pf.read(columns=["sequence_id"])["sequence_id"].unique())
    rows = [
        ("Archivo", os.path.basename(path)),
        ("Tamaño en disco (MB)", f"{os.path.getsize(path) / 1024**2:,.1f}"),
        ("Filas (frames totales)", f"{md.num_rows:,}"),
        ("Columnas totales", f"{md.num_columns:,}"),
        ("Row-groups", f"{md.num_row_groups:,}"),
        ("Compresión", str(compression)),
        ("Secuencias contenidas", f"{n_sequences:,}"),
    ]
    rows += [(f"Columnas de '{g}'", f"{n:,}") for g, n in by_group.items()]
    return pd.DataFrame(rows, columns=["Propiedad", "Valor"])


def parquet_null_profile(path: str) -> pd.DataFrame:
    """
    Perfil de valores nulos por columna leyendo **solo el footer** del Parquet.

    Apache Parquet almacena `null_count` por cada *column chunk* dentro de las estadísticas
    de cada row-group. Agregando esos contadores obtenemos la tasa exacta de `NaN` de las
    1,629 columnas sin descomprimir un solo byte de datos: la auditoría de calidad completa
    se resuelve en milisegundos incluso sobre archivos de decenas de GB.

    Returns
    -------
    DataFrame con columnas: `columna`, `grupo`, `eje`, `nulos`, `total`, `pct_nulos`.
    """
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).metadata
    names = [md.schema.column(j).name for j in range(md.num_columns)]
    nulls = np.zeros(md.num_columns, dtype=np.int64)

    for rg in range(md.num_row_groups):
        row_group = md.row_group(rg)
        for j in range(md.num_columns):
            stats = row_group.column(j).statistics
            if stats is not None:
                nulls[j] += stats.null_count

    df = pd.DataFrame({"columna": names, "nulos": nulls})
    df = df[~df["columna"].isin(["sequence_id", "frame"])].reset_index(drop=True)
    df["grupo"] = df["columna"].apply(landmark_group_of)
    df["eje"] = df["columna"].str[0]
    df["total"] = md.num_rows
    df["pct_nulos"] = df["nulos"] / md.num_rows * 100
    return df


def load_sequence(path: str, sequence_id: int, columns=None) -> pd.DataFrame:
    """
    Carga los frames de UNA secuencia aplicando *predicate pushdown* sobre `sequence_id`
    y proyección de columnas. Es el acceso recomendado para inspeccionar secuencias
    individuales sin desbordar la memoria.
    """
    import pyarrow.parquet as pq

    read_cols = None
    if columns is not None:
        read_cols = ["sequence_id", "frame"] + [c for c in columns
                                                if c not in ("sequence_id", "frame")]
    table = pq.read_table(path, columns=read_cols,
                          filters=[("sequence_id", "==", int(sequence_id))])
    return table.to_pandas().sort_values("frame").reset_index(drop=True)


def load_columns(path: str, columns, max_rows: int = None) -> pd.DataFrame:
    """
    Carga un subconjunto de columnas del archivo completo (todas las secuencias).
    `max_rows` permite truncar para pruebas rápidas.
    """
    import pyarrow.parquet as pq

    read_cols = ["sequence_id", "frame"] + [c for c in columns
                                            if c not in ("sequence_id", "frame")]
    df = pq.read_table(path, columns=read_cols).to_pandas()
    return df.head(max_rows) if max_rows else df


def frame_counts(paths) -> pd.DataFrame:
    """
    Conteo de frames por secuencia leyendo únicamente la columna `sequence_id`
    (1 de 1,631 columnas → ~0.06% del I/O de una lectura completa).
    """
    import pyarrow.parquet as pq

    if isinstance(paths, str):
        paths = [paths]
    frames = []
    for path in paths:
        seq = pq.read_table(path, columns=["sequence_id"]).to_pandas()
        counts = seq.groupby("sequence_id").size().rename("n_frames").reset_index()
        counts["source_file"] = os.path.basename(path)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)
