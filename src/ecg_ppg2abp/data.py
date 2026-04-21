import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Sample:
    ecg: List[float]
    ppg: List[float]
    abp: List[float]


def load_mimic3_jsonl(path: str) -> List[Sample]:
    """Load JSONL samples with required keys: ecg, ppg, abp.

    Raises ValueError for malformed rows or empty datasets.
    """
    records: List[Sample] = []
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                sample = Sample(
                    ecg=[float(x) for x in payload["ecg"]],
                    ppg=[float(x) for x in payload["ppg"]],
                    abp=[float(x) for x in payload["abp"]],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sample at line {line_no}: {exc}") from exc
            records.append(sample)
    if not records:
        raise ValueError(f"No valid samples found in {path}")
    return records


def iter_batches(samples: List[Sample], batch_size: int) -> Iterable[List[Sample]]:
    """Yield mini-batches; the last batch may be smaller than batch_size."""
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]
