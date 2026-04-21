import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_ppg2abp.train import TrainerConfig, train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ECG+PPG multimodal ABP model (Transformer + CLIP-style)"
    )
    parser.add_argument("--dataset", required=True, help="Path to MIMIC3 JSONL dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-weight", type=float, default=0.05)
    parser.add_argument("--embed-dim", type=int, default=16)
    args = parser.parse_args()

    config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_weight=args.clip_weight,
        embed_dim=args.embed_dim,
    )
    metrics = train_model(args.dataset, config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
