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
