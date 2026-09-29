"""Entrenamiento reproducible del primer modelo CTC sobre datos ya procesados."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ProcessedSequenceDataset, collate_ctc
from src.evaluate import evaluate_loader
from src.models import BiGRUCTC
from src.tokenizer import default_tokenizer


def run_training(processed_dir: str | Path, output_dir: str | Path, *,
                 epochs: int = 12, batch_size: int = 16, learning_rate: float = 1e-3,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2,
                 min_coverage: float = 0.5, max_sequences: int | None = None,
                 seed: int = 42, patience: int = 4, resume: bool = False) -> dict:
    """Entrena con train, selecciona checkpoint con val y nunca consulta test."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs y batch_size deben ser positivos")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
    model = BiGRUCTC(n_features, tokenizer.vocab_size, hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
    history = []
    best_loss = float("inf")
    remaining = patience
    checkpoint_path = destination / "m1_bigru_ctc.pt"
    last_path = destination / "m1_last.pt"
    start_epoch = 1
    optimizer_reinitialized = False

    if resume:
        source = last_path if last_path.exists() else checkpoint_path
        if not source.exists():
            raise FileNotFoundError(f"No hay checkpoint para reanudar en {destination}")
        saved = torch.load(source, map_location=device, weights_only=True)
        if saved["config"] != model.config:
            raise ValueError("La arquitectura solicitada no coincide con el checkpoint")
        model.load_state_dict(saved["state_dict"])
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

    for epoch in range(start_epoch, epochs + 1):
        losses = {}
        for split, loader in (("train", train_loader), ("val", val_loader)):
            model.train(split == "train")
            weighted_loss = 0.0
            total = 0
            for batch in loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                with torch.set_grad_enabled(split == "train"):
                    log_probs = model(features, batch["input_lengths"])
                    loss = loss_function(log_probs.transpose(0, 1), targets,
                                         batch["input_lengths"], batch["target_lengths"])
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                weighted_loss += float(loss.item()) * len(batch["phrases"])
                total += len(batch["phrases"])
            losses[split] = weighted_loss / total
        history.append({"epoch": epoch, "train_loss": losses["train"], "val_loss": losses["val"]})
        print(f"Época {epoch:02d}: train={losses['train']:.4f} val={losses['val']:.4f}", flush=True)
        if losses["val"] < best_loss:
            best_loss = losses["val"]
            remaining = patience
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({"state_dict": model.state_dict(), "config": model.config,
                        "epoch": epoch, "val_loss": best_loss}, temporary)
            temporary.replace(checkpoint_path)
        else:
            remaining -= 1
        temporary = last_path.with_suffix(".tmp")
        torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "config": model.config, "epoch": epoch, "best_loss": best_loss,
                    "remaining": remaining, "history": history}, temporary)
        temporary.replace(last_path)
        pd.DataFrame(history).to_csv(destination / "m1_history.csv", index=False)
        if remaining <= 0:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = evaluate_loader(model, val_loader, device, tokenizer)
    metrics.update({"model": "M1 BiGRU + CTC", "split": "val", "best_epoch": checkpoint["epoch"],
                    "best_val_loss": best_loss, "train_sequences": len(training),
                    "val_sequences": len(validation), "seed": seed,
                    "resumed_from_epoch": start_epoch - 1 if resume else None,
                    "optimizer_reinitialized_on_resume": optimizer_reinitialized,
                    "training_config": {"epochs_requested": epochs, "batch_size": batch_size,
                                        "learning_rate": learning_rate, "hidden_size": hidden_size,
                                        "num_layers": num_layers, "dropout": dropout,
                                        "min_coverage": min_coverage}})
    with open(destination / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump({"m1": metrics}, stream, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Reanudar desde m1_last.pt o el mejor checkpoint")
    args = parser.parse_args()
    print(json.dumps(run_training(args.processed_dir, args.output_dir, epochs=args.epochs,
                                  batch_size=args.batch_size, max_sequences=args.max_sequences,
                                  resume=args.resume),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
