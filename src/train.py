"""Entrenamiento reproducible de los modelos CTC (M1, M2, M3) sobre datos ya procesados."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ProcessedSequenceDataset, collate_ctc
from src.evaluate import evaluate_loader
from src.models import MODEL_REGISTRY, build_model, load_model, model_key
from src.tokenizer import default_tokenizer

# Hiperparámetros por defecto de cada arquitectura
DEFAULT_PARAMS = {
    "m1": {"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
    "m2": {"channels": 192, "levels": 6, "kernel_size": 5, "dropout": 0.2},
    "m3": {"d_model": 192, "num_layers": 4, "num_heads": 4, "ff_size": 384, "dropout": 0.1},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_metrics_file(path: str | Path, key: str, metrics: dict) -> None:
    """Agrega o reemplaza las métricas de un modelo sin borrar las de los demás."""
    path = Path(path)
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current[key] = metrics
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def run_training(processed_dir: str | Path, output_dir: str | Path, *, model: str = "m1",
                 model_params: dict | None = None, epochs: int = 12, batch_size: int = 16,
                 learning_rate: float = 1e-3, min_coverage: float = 0.5,
                 max_sequences: int | None = None, seed: int = 42, patience: int = 4,
                 resume: bool = False, max_minutes: float | None = None,
                 write_metrics: bool = True, verbose: bool = True,
                 hidden_size: int | None = None, num_layers: int | None = None,
                 dropout: float | None = None) -> dict:
    """Entrena con train, selecciona el checkpoint con val y nunca consulta test.

    `max_minutes` corta el entrenamiento al terminar la época en que se supera el tiempo;
    se conserva el mejor checkpoint. `hidden_size`, `num_layers` y `dropout` se mantienen
    como atajos del primer modelo (M1).
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs y batch_size deben ser positivos")
    key = model_key(model)
    params = {**DEFAULT_PARAMS[key], **(model_params or {})}
    legacy = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}
    params.update({name: value for name, value in legacy.items() if value is not None and name in params})

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(processed_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split_index = pd.read_csv(root / "index.csv", usecols=["participant_id", "split"])
    train_participants = set(split_index.loc[split_index.split == "train", "participant_id"])
    val_participants = set(split_index.loc[split_index.split == "val", "participant_id"])
    if train_participants & val_participants:
        raise ValueError("Hay participantes compartidos entre train y val")
    tokenizer = default_tokenizer()
    training = ProcessedSequenceDataset(root, "train", tokenizer, min_coverage, max_sequences)
    validation = ProcessedSequenceDataset(root, "val", tokenizer, min_coverage, max_sequences)
    n_features = training.features.shape[1]
    config_path = root / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as stream:
            expected = json.load(stream)["n_features"]
        if n_features != expected:
            raise ValueError(f"Dimensión de features: {n_features}; se esperaban {expected}")
    loader_options = dict(batch_size=batch_size, collate_fn=collate_ctc, num_workers=0)
    train_loader = DataLoader(training, shuffle=True, **loader_options)
    val_loader = DataLoader(validation, shuffle=False, **loader_options)
    network = build_model(key, n_features, tokenizer.vocab_size, **params).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
    history = []
    best_loss = float("inf")
    remaining = patience
    checkpoint_path = destination / MODEL_REGISTRY[key][1]
    last_path = destination / f"{key}_last.pt"
    start_epoch = 1
    optimizer_reinitialized = False

    if resume:
        source = last_path if last_path.exists() else checkpoint_path
        if not source.exists():
            raise FileNotFoundError(f"No hay checkpoint para reanudar en {destination}")
        saved = torch.load(source, map_location=device, weights_only=True)
        if saved["config"] != network.config:
            raise ValueError("La arquitectura solicitada no coincide con el checkpoint")
        network.load_state_dict(saved["state_dict"])
        if "optimizer_state_dict" in saved:
            optimizer.load_state_dict(saved["optimizer_state_dict"])
        else:
            optimizer_reinitialized = True
            print("Checkpoint antiguo: se conservan los pesos y se reinicia AdamW", flush=True)
        start_epoch = int(saved["epoch"]) + 1
        best_loss = float(saved["best_loss"] if "best_loss" in saved else saved["val_loss"])
        remaining = int(saved.get("remaining", patience))
        history = saved.get("history", [{"epoch": int(saved["epoch"]),
                                         "train_loss": None, "val_loss": best_loss}])
        print(f"Reanudando desde la época {start_epoch}", flush=True)

    started = time.perf_counter()
    stopped_by_time = False
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        losses = {}
        for split, loader in (("train", train_loader), ("val", val_loader)):
            network.train(split == "train")
            weighted_loss = 0.0
            total = 0
            for batch in loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                with torch.set_grad_enabled(split == "train"):
                    log_probs = network(features, batch["input_lengths"])
                    loss = loss_function(log_probs.transpose(0, 1), targets,
                                         batch["input_lengths"], batch["target_lengths"])
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                        optimizer.step()
                weighted_loss += float(loss.item()) * len(batch["phrases"])
                total += len(batch["phrases"])
            losses[split] = weighted_loss / total
        seconds = time.perf_counter() - epoch_started
        history.append({"epoch": epoch, "train_loss": losses["train"], "val_loss": losses["val"],
                        "seconds": seconds})
        if verbose:
            print(f"[{key}] Época {epoch:02d}: train={losses['train']:.4f} val={losses['val']:.4f} "
                  f"({seconds:.0f} s)", flush=True)
        if losses["val"] < best_loss:
            best_loss = losses["val"]
            remaining = patience
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({"state_dict": network.state_dict(), "config": network.config,
                        "architecture": key, "epoch": epoch, "val_loss": best_loss}, temporary)
            temporary.replace(checkpoint_path)
        else:
            remaining -= 1
        temporary = last_path.with_suffix(".tmp")
        torch.save({"state_dict": network.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "config": network.config, "architecture": key, "epoch": epoch,
                    "best_loss": best_loss, "remaining": remaining, "history": history}, temporary)
        temporary.replace(last_path)
        pd.DataFrame(history).to_csv(destination / f"{key}_history.csv", index=False)
        if remaining <= 0:
            break
        if max_minutes is not None and (time.perf_counter() - started) / 60 >= max_minutes:
            stopped_by_time = epoch < epochs
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint["state_dict"])
    metrics = evaluate_loader(network, val_loader, device, tokenizer)
    metrics.update({"model": MODEL_REGISTRY[key][2], "split": "val", "best_epoch": checkpoint["epoch"],
                    "epochs_run": history[-1]["epoch"] if history else 0,
                    "stopped_by_time_limit": stopped_by_time,
                    "training_minutes": (time.perf_counter() - started) / 60,
                    "best_val_loss": best_loss, "train_sequences": len(training),
                    "val_sequences": len(validation), "seed": seed,
                    "resumed_from_epoch": start_epoch - 1 if resume else None,
                    "optimizer_reinitialized_on_resume": optimizer_reinitialized,
                    "training_config": {"epochs_requested": epochs, "batch_size": batch_size,
                                        "learning_rate": learning_rate, "min_coverage": min_coverage,
                                        **params}})
    if write_metrics:
        update_metrics_file(destination / "metrics.json", key, metrics)
    return metrics


def evaluate_checkpoint(name: str, checkpoint_path: str | Path, processed_dir: str | Path,
                        split: str = "test", min_coverage: float = 0.5,
                        batch_size: int = 32) -> tuple[dict, pd.DataFrame]:
    """Evalúa un checkpoint guardado sobre un conjunto y devuelve (métricas, predicciones)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = default_tokenizer()
    dataset = ProcessedSequenceDataset(processed_dir, split, tokenizer, min_coverage)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ctc, num_workers=0)
    network = load_model(name, checkpoint_path).to(device)
    rows: list[dict] = []
    metrics = evaluate_loader(network, loader, device, tokenizer, records=rows)
    metrics.update({"split": split, "min_coverage": min_coverage})
    return metrics, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="m1", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--resume", action="store_true", help="Reanudar desde <modelo>_last.pt o el mejor checkpoint")
    args = parser.parse_args()
    print(json.dumps(run_training(args.processed_dir, args.output_dir, model=args.model,
                                  epochs=args.epochs, batch_size=args.batch_size,
                                  max_sequences=args.max_sequences, max_minutes=args.max_minutes,
                                  resume=args.resume),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
