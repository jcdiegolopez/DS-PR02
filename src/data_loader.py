"""
data_loader.py
Módulo oficial de carga, inspección y descarga de datos para el Proyecto 2
(Google - American Sign Language Fingerspelling Recognition).

Garantiza reproducibilidad total trabajando ÚNICAMENTE con los datos oficiales de Kaggle:
- Metadatos oficiales de train.csv y diccionario character_to_prediction_index.json.
- Descarga automática de archivos Parquet oficiales desde Kaggle si no existen localmente.
- Proyección de columnas y predicate pushdown para acceso eficiente a coordenadas MediaPipe.
"""

from __future__ import annotations

import os
import json
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------------------
# ESPECIFICACIÓN OFICIAL DEL ESQUEMA MEDIAPIPE HOLISTIC
# --------------------------------------------------------------------------------------

LANDMARK_COUNTS: Dict[str, int] = {
    "face": 468,
    "left_hand": 21,
    "pose": 33,
    "right_hand": 21,
}

LANDMARK_ORDER: List[str] = ["face", "left_hand", "pose", "right_hand"]
AXES: List[str] = ["x", "y", "z"]
TOTAL_LANDMARKS: int = sum(LANDMARK_COUNTS.values())      # 543
TOTAL_COORD_COLUMNS: int = TOTAL_LANDMARKS * len(AXES)    # 1,629

# 59 caracteres oficiales según la competencia Kaggle ASL Fingerspelling
OFFICIAL_CHARACTERS: List[str] = [
    " ", "!", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "=", "?", "@",
    "[", "_",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "~"
]


def build_landmark_columns() -> List[str]:
    """Construye las 1,629 columnas de coordenadas en el orden canónico oficial."""
    cols = []
    for axis in AXES:
        for kind in LANDMARK_ORDER:
            cols.extend(f"{axis}_{kind}_{i}" for i in range(LANDMARK_COUNTS[kind]))
    return cols


def get_landmark_columns(kinds: List[str] = None, axes: List[str] = None) -> List[str]:
    """Devuelve los nombres de columnas filtrados por grupo anatómico y/o eje."""
    kinds = LANDMARK_ORDER if kinds is None else list(kinds)
    axes = AXES if axes is None else list(axes)
    cols = []
    for axis in axes:
        for kind in kinds:
            cols.extend(f"{axis}_{kind}_{i}" for i in range(LANDMARK_COUNTS[kind]))
    return cols


def landmark_group_of(column: str) -> str:
    """Clasifica una columna de coordenadas en su grupo anatómico (face, hand, pose)."""
    for kind in LANDMARK_ORDER:
        if column[2:].startswith(kind):
            return kind
    return "unknown"


# --------------------------------------------------------------------------------------
# CARGA Y DESCARGA OFICIAL DE METADATOS
# --------------------------------------------------------------------------------------

def get_char_map(filepath: str = "data/character_to_prediction_index.json") -> Dict[str, int]:
    """Carga el diccionario oficial de 59 caracteres de predicción."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    char_map = {char: idx for idx, char in enumerate(OFFICIAL_CHARACTERS)}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(char_map, f, indent=2, ensure_ascii=False)
    return char_map


def load_metadata(data_dir: str = "data") -> pd.DataFrame:
    """
    Carga el catálogo oficial de metadatos train.csv (67,208 registros).
    Si no existe localmente, lo descarga automáticamente usando kagglehub.
    Retorna el DataFrame oficial de la competencia garantizando reproducibilidad.
    """
    potential_paths = [
        os.path.join(data_dir, "train.csv"),
        "train.csv",
        os.path.join("..", data_dir, "train.csv"),
        os.path.join("data", "train.csv"),
        "/kaggle/input/asl-fingerspelling/train.csv",
    ]
    
    for path in potential_paths:
        if os.path.exists(path) and not os.path.isdir(path):
            print(f"[INFO] Cargando metadatos oficiales desde: {path}")
            return pd.read_csv(path)
            
    print("[INFO] 'train.csv' no encontrado localmente. Descargando desde Kaggle...")
    try:
        import kagglehub
        import zipfile
        import shutil
        
        p = kagglehub.competition_download("asl-fingerspelling", path="train.csv")
        target_csv = os.path.join(data_dir, "train.csv")
        os.makedirs(data_dir, exist_ok=True)
        
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p, "r") as z:
                z.extract("train.csv", data_dir)
        elif os.path.isfile(p):
            shutil.copy(p, target_csv)
            
        print(f"[INFO] train.csv oficial descargado en: {target_csv}")
        return pd.read_csv(target_csv)
    except Exception as e:
        print(f"[ERROR] No se pudo descargar train.csv automáticamente de Kaggle: {e}")
        raise


# --------------------------------------------------------------------------------------
# CARGA Y DESCARGA OFICIAL DE ARCHIVOS PARQUET
# --------------------------------------------------------------------------------------

def download_official_parquet(parquet_name: str = "5414471.parquet", data_dir: str = "data") -> str:
    """
    Descarga un archivo Parquet oficial directamente de la competencia de Kaggle
    y lo almacena en data/train_landmarks/.
    """
    target_dir = os.path.join(data_dir, "train_landmarks")
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, parquet_name)
    
    if os.path.exists(target_file):
        return target_file

    # Verificar si ya existe en la caché de kagglehub del sistema
    cache_path = os.path.expanduser(f"~/.cache/kagglehub/competitions/asl-fingerspelling/train_landmarks/{parquet_name}")
    if os.path.exists(cache_path):
        print(f"[INFO] Archivo oficial encontrado en caché de kagglehub: {cache_path}")
        import shutil
        shutil.copy(cache_path, target_file)
        return target_file
        
    print(f"[INFO] Descargando archivo oficial '{parquet_name}' desde Kaggle...")
    try:
        import kagglehub
        import zipfile
        import shutil
        
        p = kagglehub.competition_download("asl-fingerspelling", path=f"train_landmarks/{parquet_name}")
        if os.path.abspath(p) == os.path.abspath(target_file):
            return target_file
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p, "r") as z:
                z.extractall(target_dir)
        elif os.path.isdir(p):
            subfile = os.path.join(p, parquet_name)
            if os.path.exists(subfile):
                shutil.copy(subfile, target_file)
            else:
                for item in os.listdir(p):
                    if item.endswith(".parquet"):
                        shutil.copy(os.path.join(p, item), os.path.join(target_dir, item))
        else:
            shutil.copy(p, target_file)
            
        print(f"[INFO] Archivo oficial Parquet listo en: {target_file}")
        return target_file
    except Exception as e:
        print(f"[ERROR] No se pudo descargar automáticamente el Parquet oficial '{parquet_name}': {e}")
        print("[INSTRUCCIONES] Inicia sesión en Kaggle con 'kagglehub.login()' o configura ~/.kaggle/kaggle.json")
        raise


def list_landmark_files(data_dir: str = "data", auto_download: bool = True) -> List[str]:
    """
    Rutas absolutas de todos los archivos .parquet oficiales disponibles.
    Si no encuentra ninguno y auto_download=True, descarga automáticamente
    el archivo oficial 5414471.parquet desde Kaggle.
    """
    home_cache = os.path.expanduser("~/.cache/kagglehub/competitions/asl-fingerspelling/train_landmarks/*.parquet")
    patterns = [
        os.path.join(data_dir, "train_landmarks", "*.parquet"),
        os.path.join(data_dir, "*.parquet"),
        os.path.join("..", data_dir, "train_landmarks", "*.parquet"),
        "/kaggle/input/asl-fingerspelling/train_landmarks/*.parquet",
        home_cache,
    ]
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            return found
            
    if auto_download:
        try:
            downloaded = download_official_parquet(data_dir=data_dir)
            return [downloaded]
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo descargar automáticamente el Parquet: {e}")
            
    return []


def get_official_landmarks_index(data_dir: str = "data") -> pd.DataFrame:
    """
    Construye el índice de secuencias con coordenadas reales leyendo los sequence_id
    del archivo Parquet oficial disponible y cruzándolo con train.csv.
    """
    files = list_landmark_files(data_dir, auto_download=True)
    if not files:
        raise FileNotFoundError("No hay archivos Parquet oficiales disponibles.")
        
    import pyarrow.parquet as pq
    seq_records = []
    for f in files:
        fid = os.path.splitext(os.path.basename(f))[0]
        fid_num = int(fid) if fid.isdigit() else fid
        table = pq.read_table(f, columns=["sequence_id"])
        seq_series = table["sequence_id"].to_pandas()
        counts = seq_series.value_counts()
        for s_id, n_frames in counts.items():
            seq_records.append({
                "sequence_id": int(s_id),
                "file_id": fid_num,
                "n_frames": int(n_frames),
                "parquet_path": f
            })
            
    df_seqs = pd.DataFrame(seq_records)
    df_meta = load_metadata(data_dir)
    merged = df_seqs.merge(df_meta, on=["sequence_id", "file_id"], how="inner")
    if merged.empty:
        merged = df_seqs.merge(df_meta, on="sequence_id", how="inner")
    return merged


# --------------------------------------------------------------------------------------
# OPERACIONES DE LECTURA EFICIENTE SOBRE PARQUET
# --------------------------------------------------------------------------------------

def parquet_schema_report(path: str) -> pd.DataFrame:
    """Reporte estructural de un archivo Parquet leyendo solo el footer."""
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
    """Perfil exacto de valores nulos leyendo estadísticas del footer del Parquet."""
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


def load_sequence(path: str, sequence_id: int, columns: List[str] = None) -> pd.DataFrame:
    """Carga los frames de una secuencia específica aplicando predicate pushdown."""
    import pyarrow.parquet as pq

    read_cols = None
    if columns is not None:
        read_cols = ["sequence_id", "frame"] + [c for c in columns if c not in ("sequence_id", "frame")]
    table = pq.read_table(path, columns=read_cols, filters=[("sequence_id", "==", int(sequence_id))])
    df = table.to_pandas()
    if "sequence_id" not in df.columns:
        df = df.reset_index()
    return df.sort_values("frame").reset_index(drop=True)


def load_columns(path: str, columns: List[str], max_rows: int = None) -> pd.DataFrame:
    """Carga columnas proyectadas de un archivo Parquet oficial asegurando sequence_id en columnas."""
    import pyarrow.parquet as pq

    read_cols = ["sequence_id", "frame"] + [c for c in columns if c not in ("sequence_id", "frame")]
    df = pq.read_table(path, columns=read_cols).to_pandas()
    if "sequence_id" not in df.columns:
        df = df.reset_index()
    return df.head(max_rows) if max_rows else df


def frame_counts(paths) -> pd.DataFrame:
    """Conteo eficiente de frames por secuencia leyendo únicamente la columna sequence_id."""
    import pyarrow.parquet as pq

    if isinstance(paths, str):
        paths = [paths]
    frames = []
    for path in paths:
        seq = pq.read_table(path, columns=["sequence_id"]).to_pandas()
        if "sequence_id" not in seq.columns:
            seq = seq.reset_index()
        counts = seq.groupby("sequence_id").size().rename("n_frames").reset_index()
        counts["source_file"] = os.path.basename(path)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)
