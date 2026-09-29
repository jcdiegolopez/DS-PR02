"""Lectura de secuencias procesadas y lotes de longitud variable para CTC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.tokenizer import CharacterTokenizer, default_tokenizer


def ctc_min_length(text: str) -> int:
    """Frames mínimos: los caracteres consecutivos iguales requieren un blank entre ellos."""
    return len(text) + sum(left == right for left, right in zip(text, text[1:]))


class ProcessedSequenceDataset(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        tokenizer: CharacterTokenizer | None = None,
        min_coverage: float = 0.5,
        max_sequences: int | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Split desconocido: {split}")
        root = Path(processed_dir)
        self.features = np.load(root / "features.npy", mmap_mode="r")
        self.masks = np.load(root / "masks.npy", mmap_mode="r")
        self.tokenizer = tokenizer or default_tokenizer()
        index = pd.read_csv(root / "index.csv", keep_default_na=False)
        selected = index[(index["split"] == split) & (index["coverage"] >= min_coverage)].copy()
        selected = selected[selected["phrase"].astype(str).str.len() > 0]

        records = []
        for row in selected.itertuples(index=False):
            start, end = int(row.offset), int(row.offset) + int(row.length)
            if start < 0 or end > len(self.masks) or end > len(self.features):
                raise ValueError(f"Offset fuera de los arreglos: {row.sequence_id}")
            target = self.tokenizer.encode(str(row.phrase))
            valid = int(np.count_nonzero(self.masks[start:end]))
            if valid >= ctc_min_length(str(row.phrase)):
                records.append((start, end, target, str(row.phrase), int(row.sequence_id), valid))
        if max_sequences is not None:
            records = records[:max_sequences]
        self.records = records
        if not records:
            raise ValueError(f"No hay secuencias aptas para CTC en el split {split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        start, end, target, phrase, sequence_id, _ = self.records[index]
        valid_features = np.asarray(self.features[start:end], dtype=np.float32)[self.masks[start:end]]
        return {
            "features": torch.from_numpy(valid_features.copy()),
            "target": torch.tensor(target, dtype=torch.long),
            "phrase": phrase,
            "sequence_id": sequence_id,
        }


def collate_ctc(samples: list[dict]) -> dict:
    features = [sample["features"] for sample in samples]
    targets = [sample["target"] for sample in samples]
    return {
        "features": pad_sequence(features, batch_first=True),
        "input_lengths": torch.tensor([len(item) for item in features], dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor([len(item) for item in targets], dtype=torch.long),
        "phrases": [sample["phrase"] for sample in samples],
        "sequence_ids": [sample["sequence_id"] for sample in samples],
    }
