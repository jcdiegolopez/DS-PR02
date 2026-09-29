"""Contratos críticos de CTC, carga de datos e inferencia."""

import json
import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.tokenizer import BLANK_ID, CharacterTokenizer


class TokenizerTests(unittest.TestCase):
    def test_blank_does_not_replace_official_space(self):
        tokenizer = CharacterTokenizer()
        self.assertEqual(BLANK_ID, 0)
        self.assertNotEqual(tokenizer.encode(" ")[0], BLANK_ID)
        a = tokenizer.encode("a")[0]
        self.assertEqual(tokenizer.decode([a, a, BLANK_ID, a]), "aa")
        self.assertEqual(tokenizer.decode([BLANK_ID, *tokenizer.encode("a a"), BLANK_ID]), "a a")

    def test_unknown_character_is_rejected(self):
        with self.assertRaises(ValueError):
            CharacterTokenizer().encode("Á")


class TrainingSmokeTests(unittest.TestCase):
    def test_processed_data_to_checkpoint_and_metrics(self):
        import torch
        from src.dataset import ProcessedSequenceDataset, collate_ctc, ctc_min_length
        from src.evaluate import transcription_metrics
        from src.models import load_model, predict
        from src.train import run_training

        N_FEATURES = 80  # mano (63) + muñeca (2) + pose (14) + presencia (1)
        self.assertEqual(ctc_min_length("aa"), 3)
        metrics = transcription_metrics([("aa", "a"), ("a1", "a1")])
        self.assertEqual(metrics["total_edit_distance"], 1)
        self.assertEqual(metrics["by_character_type"]["letters"]["errors"], 1)
        self.assertEqual(metrics["by_character_type"]["digits"]["errors"], 0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            processed.mkdir()
            rows = []
            features = []
            masks = []
            for i, (split, phrase) in enumerate([
                ("train", "ab"), ("train", "aa"),
                ("val", "ab"), ("val", "aa"),
            ]):
                rng = np.random.default_rng(i)
                chunk = rng.normal(size=(8, N_FEATURES)).astype(np.float16)
                mask = np.array([True, True, False, True, True, True, True, True])
                rows.append({"sequence_id": i + 1, "participant_id": i + 1,
                             "phrase": phrase, "split": split, "offset": i * 8,
                             "length": 8, "coverage": float(mask.mean())})
                features.append(chunk)
                masks.append(mask)
            np.save(processed / "features.npy", np.concatenate(features))
            np.save(processed / "masks.npy", np.concatenate(masks))
            pd.DataFrame(rows).to_csv(processed / "index.csv", index=False)

            dataset = ProcessedSequenceDataset(processed, "train")
            batch = collate_ctc([dataset[0], dataset[1]])
            self.assertEqual(batch["input_lengths"].tolist(), [7, 7])
            self.assertEqual(batch["target_lengths"].tolist(), [2, 2])
            self.assertTrue(torch.all(batch["targets"] > 0))

            output = root / "models"
            result = run_training(processed, output, epochs=1, batch_size=2,
                                  hidden_size=8, num_layers=1, dropout=0)
            self.assertEqual(result["split"], "val")
            self.assertEqual(result["val_sequences"], 2)
            self.assertTrue((output / "m1_bigru_ctc.pt").exists())
            self.assertTrue((output / "m1_last.pt").exists())
            self.assertIn("m1", json.loads((output / "metrics.json").read_text(encoding="utf-8")))
            resumed = run_training(processed, output, epochs=2, batch_size=2,
                                   hidden_size=8, num_layers=1, dropout=0, resume=True)
            self.assertEqual(resumed["resumed_from_epoch"], 1)
            self.assertEqual(pd.read_csv(output / "m1_history.csv")["epoch"].tolist(), [1, 2])
            model = load_model("m1", output / "m1_bigru_ctc.pt")
            text = predict(model, features[0].astype(np.float32), masks[0])
            self.assertIsInstance(text, str)
            del dataset, batch, model
            gc.collect()  # Windows mantiene abiertos los .npy mientras exista el memmap


if __name__ == "__main__":
    unittest.main()
