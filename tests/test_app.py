"""Entrada de archivos, inferencia compartida y páginas con los artefactos disponibles."""

import io
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from lib.data import InputError, load_manifest, load_metrics, load_sample, read_upload
from lib.inference import prepare, transcribe


class InputTests(unittest.TestCase):
    def setUp(self):
        self.raw = load_sample(load_manifest().iloc[0]["file"])

    def upload(self, raw):
        return read_upload("entrada.csv", raw.to_csv(index=False).encode())

    def test_csv_and_parquet_keep_the_same_features(self):
        csv = prepare(self.upload(self.raw))
        buffer = io.BytesIO()
        self.raw.to_parquet(buffer)
        parquet = prepare(read_upload("entrada.parquet", buffer.getvalue()))
        np.testing.assert_allclose(csv.features, parquet.features, atol=1e-5)
        np.testing.assert_array_equal(csv.mask, parquet.mask)

    def test_multiple_sequences_remain_available_for_selection(self):
        raw = self.raw.assign(sequence_id=1)
        another = self.raw.assign(sequence_id=2)
        self.assertEqual(self.upload(pd.concat([raw, another]))["sequence_id"].nunique(), 2)

    def test_invalid_inputs_have_actionable_errors(self):
        cases = [self.raw.iloc[:0], self.raw.assign(x_right_hand_0="texto"),
                 self.raw.assign(frame=np.nan), pd.concat([self.raw, self.raw.iloc[:1]]),
                 self.raw.assign(x_right_hand_0=np.inf)]
        for raw in cases:
            with self.subTest(shape=raw.shape), self.assertRaises(InputError):
                self.upload(raw)
        with self.assertRaises(InputError):
            read_upload("entrada.csv", b"no,landmarks\n1,2")
        with self.assertRaises(InputError):
            read_upload("entrada.parquet", b"invalid")
        with self.assertRaises(InputError):
            read_upload("entrada.txt", b"invalid")

    def test_mixed_partial_hands_do_not_count_as_one_complete_hand(self):
        columns = [f"{a}_right_hand_{i}" for a in "xyz" for i in range(11)]
        columns += [f"{a}_left_hand_{i}" for a in "xyz" for i in range(10)]
        with self.assertRaises(InputError):
            self.upload(pd.DataFrame(np.zeros((5, 63)), columns=columns))

    def test_missing_pose_and_absent_hand_are_supported(self):
        hand = [c for c in self.raw.columns if "_hand_" in c]
        prepared = prepare(self.upload(self.raw[hand]))
        self.assertTrue(np.isfinite(prepared.features).all())
        absent = prepare(self.upload(self.raw[hand] * np.nan))
        self.assertEqual(int(absent.mask.sum()), 0)


class AppSmokeTests(unittest.TestCase):
    def test_all_architectures_load_through_app_and_share_decoding(self):
        import torch
        import lib.data as data
        from src.models import build_model, predict
        from src.tokenizer import default_tokenizer

        prepared = prepare(load_sample(load_manifest().iloc[0]["file"]))
        configs = {"m1": {"hidden_size": 8, "num_layers": 1},
                   "m2": {"channels": 8, "levels": 1, "kernel_size": 3},
                   "m3": {"d_model": 8, "num_layers": 1, "num_heads": 2, "ff_size": 16}}
        # Checkpoints temporales prueban el contrato; no se entregan como modelos entrenados.
        with tempfile.TemporaryDirectory() as folder:
            for key, config in configs.items():
                model = build_model(key, 80, default_tokenizer().vocab_size, **config)
                torch.save({"architecture": key, "config": model.config, "state_dict": model.state_dict()},
                           Path(folder) / f"{key}_test.pt")
            data.get_model.clear()
            try:
                with patch.object(data, "MODELS_DIR", Path(folder)):
                    self.assertEqual(data.available_models(), ["m1", "m2", "m3"])
                    for key in configs:
                        result = transcribe(key, prepared)
                        self.assertEqual(result.text, predict(data.get_model(key), prepared.features, prepared.mask))
                        self.assertEqual(len(result.frame_ids), int(prepared.mask.sum()))
            finally:
                data.get_model.clear()

    def test_metrics_of_different_splits_are_not_compared_together(self):
        from streamlit.testing.v1 import AppTest

        real = load_metrics()["m1"]
        fixture = {"m1": dict(real), "m2": dict(real, split="test")}
        with patch("lib.data.load_metrics", return_value=fixture):
            app = AppTest.from_file(str(ROOT / "app/views/rendimiento.py"), default_timeout=60).run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.multiselect[0].value, ["m1"])
            app.selectbox[0].set_value(("test", real["sequences"])).run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.multiselect[0].value, ["m2"])
            app.multiselect[0].set_value([]).run()
            self.assertEqual(list(app.exception), [])

    def test_real_checkpoint_matches_shared_prediction(self):
        from lib.data import get_model
        from src.models import predict

        prepared = prepare(load_sample(load_manifest().iloc[0]["file"]))
        result = transcribe("m1", prepared)
        self.assertEqual(result.text, predict(get_model("m1"), prepared.features, prepared.mask))
        self.assertEqual(len(result.frame_ids), int(prepared.mask.sum()))
        self.assertGreater(result.milliseconds, 0)
        self.assertEqual(load_metrics()["m1"]["split"], "val")

    def test_all_pages_render_and_toggles_work(self):
        from streamlit.testing.v1 import AppTest

        for name in ("explorar", "predecir", "rendimiento"):
            with self.subTest(page=name):
                app = AppTest.from_file(str(ROOT / "app" / "views" / f"{name}.py"), default_timeout=60).run()
                self.assertEqual(list(app.exception), [])
                for toggle in list(app.toggle):
                    toggle.set_value(not toggle.value)
                app.run()
                self.assertEqual(list(app.exception), [])
                if name == "rendimiento":
                    self.assertTrue(any("no son la evaluación final" in c.value for c in app.caption))


if __name__ == "__main__":
    unittest.main()
