import json
import os
import tempfile
import unittest
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_ppg2abp.data import load_mimic3_jsonl
from ecg_ppg2abp.model import ECGPPGCLIPTransformer
from ecg_ppg2abp.train import TrainerConfig, train_model


def _make_signal(length: int, scale: float) -> list:
    return [scale * (i / max(length - 1, 1)) for i in range(length)]


class PipelineTests(unittest.TestCase):
    def test_model_predict_shape(self) -> None:
        model = ECGPPGCLIPTransformer(seq_len=16, embed_dim=8, abp_len=12, seed=0)
        pred = model.predict_abp(_make_signal(16, 1.0), _make_signal(16, 0.8))
        self.assertEqual(len(pred), 12)

    def test_load_and_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, "mimic3.jsonl")
            with open(data_file, "w", encoding="utf-8") as f:
                for k in range(5):
                    sample = {
                        "ecg": _make_signal(16, 1.0 + k * 0.01),
                        "ppg": _make_signal(16, 0.9 + k * 0.01),
                        "abp": _make_signal(8, 1.1 + k * 0.01),
                    }
                    f.write(json.dumps(sample) + "\n")

            samples = load_mimic3_jsonl(data_file)
            self.assertEqual(len(samples), 5)

            metrics = train_model(
                data_file,
                TrainerConfig(epochs=2, batch_size=2, learning_rate=5e-3, embed_dim=8),
            )
            self.assertIn("final_loss", metrics)
            self.assertGreaterEqual(metrics["final_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
