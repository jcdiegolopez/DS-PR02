"""Contratos del pipeline de limpieza que comparten el entrenamiento y la aplicación."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import (FEATURE_COLUMNS, MAX_GAP, N_FEATURES, _interpolate_short_gaps,
                               is_usable, preprocess_sequence)

SAMPLES = Path(__file__).resolve().parents[1] / "app" / "sample_sequences"


def synthetic_sequence(frames: int = 40, hand: str = "right_hand", seed: int = 0) -> pd.DataFrame:
    """Secuencia con una sola mano visible y pose completa, en el esquema de Kaggle."""
    rng = np.random.default_rng(seed)
    data = {column: np.full(frames, np.nan) for column in FEATURE_COLUMNS}
    data["frame"] = np.arange(frames)
    for axis in "xyz":
        for i in range(21):
            base = 0.5 + 0.02 * i + (0.1 if axis == "y" else 0.0)
            data[f"{axis}_{hand}_{i}"] = base + rng.normal(0, 0.002, frames)
    for axis in "xy":
        for i in (0, 11, 12, 13, 14, 15, 16):
            data[f"{axis}_pose_{i}"] = 0.3 + 0.01 * i + rng.normal(0, 0.001, frames)
    return pd.DataFrame(data)


def mirrored(raw: pd.DataFrame) -> pd.DataFrame:
    """La misma secuencia vista en espejo: x -> 1 - x e izquierda <-> derecha."""
    out = raw.copy()
    swap = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 0: 0}
    for i in range(21):
        for axis in "yz":
            out[f"{axis}_left_hand_{i}"], out[f"{axis}_right_hand_{i}"] = (
                raw[f"{axis}_right_hand_{i}"], raw[f"{axis}_left_hand_{i}"])
        out[f"x_left_hand_{i}"] = 1 - raw[f"x_right_hand_{i}"]
        out[f"x_right_hand_{i}"] = 1 - raw[f"x_left_hand_{i}"]
    for i, j in swap.items():
        out[f"x_pose_{i}"] = 1 - raw[f"x_pose_{j}"]
        out[f"y_pose_{i}"] = raw[f"y_pose_{j}"]
    return out


class PreprocessingTests(unittest.TestCase):
    def test_shape_dtype_and_mask(self):
        features, mask = preprocess_sequence(synthetic_sequence())
        self.assertEqual(features.shape, (40, N_FEATURES))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask.all())
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(is_usable(mask))

    def test_wrist_is_the_origin_of_the_hand(self):
        features, _ = preprocess_sequence(synthetic_sequence())
        np.testing.assert_allclose(features[:, :3], 0.0, atol=1e-6)

    def test_left_handed_signer_matches_mirrored_right_handed(self):
        raw = synthetic_sequence()
        right, mask_right = preprocess_sequence(raw)
        left, mask_left = preprocess_sequence(mirrored(raw))
        np.testing.assert_array_equal(mask_right, mask_left)
        np.testing.assert_allclose(right, left, atol=1e-4)

    def test_edges_without_hand_are_trimmed(self):
        raw = synthetic_sequence()
        hand_columns = [c for c in raw.columns if "right_hand" in c]
        raw.loc[:4, hand_columns] = np.nan
        raw.loc[35:, hand_columns] = np.nan
        features, mask = preprocess_sequence(raw)
        self.assertEqual(len(features), 30)
        self.assertTrue(mask.all())

    def test_only_short_gaps_are_interpolated(self):
        raw = synthetic_sequence(frames=60)
        hand_columns = [c for c in raw.columns if "right_hand" in c]
        raw.loc[10:10 + MAX_GAP - 1, hand_columns] = np.nan      # hueco de MAX_GAP: se rellena
        raw.loc[30:30 + MAX_GAP, hand_columns] = np.nan          # hueco de MAX_GAP + 1: se enmascara
        features, mask = preprocess_sequence(raw)
        self.assertTrue(mask[10:10 + MAX_GAP].all())
        self.assertFalse(mask[30:31 + MAX_GAP].any())
        self.assertTrue((features[~mask] == 0).all())

    def test_interpolation_is_linear(self):
        coords = np.zeros((4, 21, 3))
        coords[3] = 3.0
        coords[1:3] = np.nan
        filled = _interpolate_short_gaps(coords, MAX_GAP)
        np.testing.assert_allclose(filled[:, 0, 0], [0.0, 1.0, 2.0, 3.0])

    def test_sequence_without_hands_is_empty(self):
        raw = synthetic_sequence()
        raw[[c for c in raw.columns if "hand" in c]] = np.nan
        features, mask = preprocess_sequence(raw)
        self.assertEqual(features.shape, (0, N_FEATURES))
        self.assertEqual(len(mask), 0)
        self.assertFalse(is_usable(mask))

    def test_frame_order_does_not_matter(self):
        raw = synthetic_sequence()
        shuffled = raw.sample(frac=1.0, random_state=3)
        np.testing.assert_array_equal(preprocess_sequence(raw)[0], preprocess_sequence(shuffled)[0])

    def test_app_samples_match_their_manifest(self):
        manifest = pd.read_csv(SAMPLES / "manifest.csv", keep_default_na=False)
        for row in manifest.head(5).itertuples():
            features, mask = preprocess_sequence(pd.read_parquet(SAMPLES / row.file))
            self.assertEqual(features.shape[1], N_FEATURES)
            self.assertAlmostEqual(float(mask.mean()), row.coverage, places=3)


if __name__ == "__main__":
    unittest.main()
