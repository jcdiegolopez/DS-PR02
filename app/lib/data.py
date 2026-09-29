"""Carga de datos, métricas y modelos para la app (todo en caché de Streamlit)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "app" / "sample_sequences"

# Los tres modelos del plan. Solo se muestran como disponibles cuando existen sus pesos.
MODEL_INFO = {
    "m1": {"label": "M1 BiGRU", "family": "Recurrente bidireccional"},
    "m2": {"label": "M2 TCN", "family": "Convolución temporal"},
    "m3": {"label": "M3 Transformer", "family": "Atención (Transformer / Conformer)"},
}

# Misma partición que el notebook de resultados (Parte I, sección 1)
SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


class InputError(ValueError):
    """Error de entrada con un mensaje pensado para el usuario."""


@st.cache_data(show_spinner=False)
def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    rng = np.random.default_rng(SPLIT_SEED)
    participants = np.array(sorted(meta["participant_id"].unique()))
    rng.shuffle(participants)
    n_train = int(round(len(participants) * SPLIT_FRACTIONS["train"]))
    n_val = int(round(len(participants) * SPLIT_FRACTIONS["val"]))
    split_of = {p: "train" for p in participants[:n_train]}
    split_of.update({p: "val" for p in participants[n_train:n_train + n_val]})
    split_of.update({p: "test" for p in participants[n_train + n_val:]})
    meta["split"] = meta["participant_id"].map(split_of)
    meta["phrase_length"] = meta["phrase"].str.len()
    return meta


@st.cache_data(show_spinner=False)
def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(SAMPLES_DIR / "manifest.csv", keep_default_na=False)
    return manifest.sort_values("phrase", key=lambda s: s.str.len()).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_sample(file_name: str) -> pd.DataFrame:
    return pd.read_parquet(SAMPLES_DIR / file_name)


def read_upload(name: str, payload: bytes) -> pd.DataFrame:
    """Lee un parquet o CSV subido y valida que tenga el formato de Kaggle."""
    try:
        if name.lower().endswith(".csv"):
            raw = pd.read_csv(io.BytesIO(payload))
        else:
            raw = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de lectura es un archivo inválido
        raise InputError(f"No se pudo leer el archivo como {'CSV' if name.lower().endswith('.csv') else 'Parquet'}.") from exc
    if raw.index.name == "sequence_id":
        raw = raw.reset_index()
    if "sequence_id" in raw.columns and raw["sequence_id"].nunique() > 1:
        first = raw["sequence_id"].iloc[0]
        raw = raw[raw["sequence_id"] == first]
        st.info(f"El archivo trae varias secuencias; se usa la primera ({first}).")
    hand_cols = [f"{a}_{h}_{i}" for h in ("left_hand", "right_hand") for a in "xyz" for i in range(21)]
    present = [c for c in hand_cols if c in raw.columns]
    if len(present) < 63:
        raise InputError("El archivo no tiene las columnas de landmarks de mano del formato Kaggle "
                         "(por ejemplo x_right_hand_0 ... z_right_hand_20).")
    if len(raw) == 0:
        raise InputError("El archivo no tiene frames.")
    return raw


def fill_missing_columns(raw: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Agrega como NaN las columnas que falten (p. ej. pose), para que el pipeline las trate como ausentes."""
    missing = [c for c in columns if c not in raw.columns and c != "frame"]
    if missing:
        raw = raw.assign(**{c: np.nan for c in missing})
    if "frame" not in raw.columns:
        raw = raw.assign(frame=np.arange(len(raw)))
    return raw, missing


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_history(key: str) -> pd.DataFrame | None:
    path = MODELS_DIR / f"{key}_history.csv"
    return pd.read_csv(path) if path.exists() else None


def weights_path(key: str) -> Path | None:
    candidates = sorted(p for p in MODELS_DIR.glob(f"{key}_*.pt") if not p.stem.endswith("_last"))
    return candidates[0] if candidates else None


def available_models() -> list[str]:
    """Modelos con pesos que además carga src.models.load_model; los demás quedan como pendientes."""
    ready = []
    for key in MODEL_INFO:
        if weights_path(key) is None:
            continue
        try:
            get_model(key)
        except Exception:  # noqa: BLE001 - pesos sin soporte en el loader o archivo dañado
            continue
        ready.append(key)
    return ready


@st.cache_resource(show_spinner=False)
def get_model(key: str):
    from src.models import load_model

    path = weights_path(key)
    if path is None:
        raise InputError(f"No hay pesos para {MODEL_INFO[key]['label']}.")
    return load_model(key, path)
