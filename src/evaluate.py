"""Métricas de transcripción y evaluación exclusiva del conjunto de validación."""

from __future__ import annotations

import time

import torch

from src.tokenizer import CharacterTokenizer, default_tokenizer


def edit_operations(reference: str, prediction: str) -> list[tuple[str, str, str]]:
    """Alineación mínima con operaciones (tipo, carácter real, carácter predicho)."""
    rows, cols = len(reference) + 1, len(prediction) + 1
    distances = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distances[i][0] = i
    for j in range(cols):
        distances[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = int(reference[i - 1] != prediction[j - 1])
            distances[i][j] = min(distances[i - 1][j] + 1,
                                  distances[i][j - 1] + 1,
                                  distances[i - 1][j - 1] + cost)
    operations = []
    i, j = len(reference), len(prediction)
    while i or j:
        if i and j and distances[i][j] == distances[i - 1][j - 1] + int(reference[i - 1] != prediction[j - 1]):
            kind = "correct" if reference[i - 1] == prediction[j - 1] else "substitution"
            operations.append((kind, reference[i - 1], prediction[j - 1]))
            i -= 1
            j -= 1
        elif i and distances[i][j] == distances[i - 1][j] + 1:
            operations.append(("deletion", reference[i - 1], ""))
            i -= 1
        else:
            operations.append(("insertion", "", prediction[j - 1]))
            j -= 1
    return operations[::-1]


def character_category(char: str) -> str:
    if char.isalpha():
        return "letters"
    if char.isdigit():
        return "digits"
    if char.isspace():
        return "spaces"
    return "symbols"


def transcription_metrics(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        raise ValueError("No hay predicciones para evaluar")
    distance_sum = 0
    reference_chars = 0
    sequence_errors = []
    exact = 0
    categories = {name: {"reference": 0, "errors": 0, "insertions": 0}
                  for name in ("letters", "digits", "symbols", "spaces")}
    for reference, prediction in pairs:
        ops = edit_operations(reference, prediction)
        distance = sum(kind != "correct" for kind, _, _ in ops)
        distance_sum += distance
        reference_chars += len(reference)
        sequence_errors.append(distance / max(1, len(reference)))
        exact += reference == prediction
        for kind, true_char, predicted_char in ops:
            if kind == "insertion":
                categories[character_category(predicted_char)]["insertions"] += 1
            else:
                category = categories[character_category(true_char)]
                category["reference"] += 1
                category["errors"] += kind != "correct"
    for category in categories.values():
        category["error_rate"] = category["errors"] / max(1, category["reference"])
    cer = distance_sum / max(1, reference_chars)
    return {
        "sequences": len(pairs),
        "normalized_edit_distance": sum(sequence_errors) / len(sequence_errors),
        "cer": cer,
        "competition_score": max(0.0, 1.0 - cer),
        "exact_match": exact / len(pairs),
        "total_edit_distance": distance_sum,
        "reference_characters": reference_chars,
        "by_character_type": categories,
    }


@torch.inference_mode()
def evaluate_loader(model, loader, device: torch.device,
                    tokenizer: CharacterTokenizer | None = None) -> dict:
    tokenizer = tokenizer or default_tokenizer()
    model.eval()
    pairs = []
    inference_seconds = 0.0
    for batch in loader:
        features = batch["features"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(features, batch["input_lengths"])
        predictions = logits.argmax(dim=-1).cpu()
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        for i, reference in enumerate(batch["phrases"]):
            ids = predictions[i, :int(batch["input_lengths"][i])].tolist()
            pairs.append((reference, tokenizer.decode(ids)))
    metrics = transcription_metrics(pairs)
    metrics["inference_seconds_total"] = inference_seconds
    metrics["inference_ms_per_sequence"] = 1000 * inference_seconds / len(pairs)
    metrics["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return metrics
