"""Codificación de caracteres para CTC con el vocabulario oficial del reto."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

BLANK_ID = 0
DEFAULT_MAP = Path(__file__).resolve().parents[1] / "data" / "character_to_prediction_index.json"


class CharacterTokenizer:
    def __init__(self, mapping_path: str | Path = DEFAULT_MAP):
        with open(mapping_path, encoding="utf-8") as stream:
            official = json.load(stream)
        if sorted(official.values()) != list(range(len(official))):
            raise ValueError("El mapa de caracteres debe tener índices consecutivos desde 0")
        # El índice 0 del archivo oficial representa un espacio, no el blank de CTC.
        self.char_to_id = {char: int(index) + 1 for char, index in official.items()}
        self.id_to_char = {index: char for char, index in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id) + 1

    def encode(self, phrase: str) -> list[int]:
        try:
            return [self.char_to_id[char] for char in phrase]
        except KeyError as exc:
            raise ValueError(f"Carácter fuera del vocabulario: {exc.args[0]!r}") from exc

    def decode(self, ids: list[int]) -> str:
        """Decodificación greedy: primero colapsa repeticiones, después quita blanks."""
        result: list[str] = []
        previous = None
        for token in ids:
            token = int(token)
            if token != previous and token != BLANK_ID:
                if token not in self.id_to_char:
                    raise ValueError(f"ID fuera del vocabulario: {token}")
                result.append(self.id_to_char[token])
            previous = token
        return "".join(result)


@lru_cache(maxsize=1)
def default_tokenizer() -> CharacterTokenizer:
    return CharacterTokenizer()


def encode(text: str) -> list[int]:
    return default_tokenizer().encode(text)


def decode(ids: list[int]) -> str:
    return default_tokenizer().decode(ids)
