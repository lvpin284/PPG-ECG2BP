from dataclasses import dataclass
from typing import Dict, List

from .data import Sample, iter_batches, load_mimic3_jsonl
from .model import ECGPPGCLIPTransformer


@dataclass
class TrainerConfig:
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    clip_weight: float = 0.05
    embed_dim: int = 16


def _batch_train_step(
    model: ECGPPGCLIPTransformer,
    batch: List[Sample],
    learning_rate: float,
    clip_weight: float,
) -> float:
    loss_sum = 0.0
    for sample in batch:
        loss_sum += model.train_regression_step(
            sample.ecg,
            sample.ppg,
            sample.abp,
            lr=learning_rate,
            clip_weight=clip_weight,
        )
    return loss_sum / max(len(batch), 1)


def train_model(dataset_path: str, config: TrainerConfig) -> Dict[str, float]:
    samples = load_mimic3_jsonl(dataset_path)
    seq_len = len(samples[0].ecg)
    abp_len = len(samples[0].abp)
    model = ECGPPGCLIPTransformer(seq_len=seq_len, embed_dim=config.embed_dim, abp_len=abp_len)

    final_loss = 0.0
    for _ in range(config.epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in iter_batches(samples, config.batch_size):
            epoch_loss += _batch_train_step(model, batch, config.learning_rate, config.clip_weight)
            num_batches += 1
        final_loss = epoch_loss / max(num_batches, 1)

    return {"final_loss": final_loss, "samples": float(len(samples)), "epochs": float(config.epochs)}
