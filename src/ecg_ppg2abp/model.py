import math
import random
from typing import List, Tuple


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(values: List[float]) -> List[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    denom = sum(exps)
    if denom == 0:
        return [1.0 / len(values)] * len(values)
    return [v / denom for v in exps]


def _cosine(a: List[float], b: List[float]) -> float:
    denom = math.sqrt(_dot(a, a) * _dot(b, b))
    if denom == 0:
        return 0.0
    return _dot(a, b) / denom


class ECGPPGCLIPTransformer:
    SIMILARITY_SATURATION_THRESHOLD = 20.0

    """
    Lightweight ECG+PPG multimodal fusion model for ABP prediction.
    - Modality encoder: simplified Transformer self-attention
    - Cross-modal alignment: CLIP-style cosine alignment
    - Regression head: predicts ABP waveform

    Args:
        seq_len: Fixed length of ECG/PPG input windows.
        embed_dim: Embedding dimension for each modality encoder.
        abp_len: Output ABP waveform length.
        seed: Random seed used for reproducible initialization.

    轻量级 ECG+PPG 多模态融合模型：
    - 模态编码：简化 Transformer Self-Attention
    - 跨模态对齐：CLIP 风格 cosine alignment
    - 回归头：输出 ABP 波形
    """

    def __init__(self, seq_len: int, embed_dim: int = 16, abp_len: int = 32, seed: int = 42):
        if seq_len <= 0 or embed_dim <= 0 or abp_len <= 0:
            raise ValueError("seq_len, embed_dim, abp_len must be positive")
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.abp_len = abp_len

        rnd = random.Random(seed)
        self.ecg_proj = [rnd.uniform(-0.15, 0.15) for _ in range(embed_dim)]
        self.ppg_proj = [rnd.uniform(-0.15, 0.15) for _ in range(embed_dim)]

        self.w_q = [[rnd.uniform(-0.1, 0.1) for _ in range(embed_dim)] for _ in range(embed_dim)]
        self.w_k = [[rnd.uniform(-0.1, 0.1) for _ in range(embed_dim)] for _ in range(embed_dim)]
        self.w_v = [[rnd.uniform(-0.1, 0.1) for _ in range(embed_dim)] for _ in range(embed_dim)]

        fused_dim = embed_dim * 2
        self.reg_w = [[rnd.uniform(-0.1, 0.1) for _ in range(abp_len)] for _ in range(fused_dim)]
        self.reg_b = [0.0 for _ in range(abp_len)]

    def _mat_vec(self, matrix: List[List[float]], vector: List[float]) -> List[float]:
        return [_dot(row, vector) for row in matrix]

    def _encode_signal(self, signal: List[float], projection: List[float]) -> List[float]:
        if len(signal) != self.seq_len:
            raise ValueError(f"Expected signal length {self.seq_len}, got {len(signal)}")

        tokens: List[List[float]] = []
        for idx, value in enumerate(signal):
            pos_bias = math.sin(idx / self.seq_len * math.pi)
            token = [value * projection[d] + pos_bias for d in range(self.embed_dim)]
            tokens.append(token)

        q = [self._mat_vec(self.w_q, t) for t in tokens]
        k = [self._mat_vec(self.w_k, t) for t in tokens]
        v = [self._mat_vec(self.w_v, t) for t in tokens]

        attended: List[List[float]] = []
        scale = math.sqrt(self.embed_dim)
        for qi in q:
            scores = [_dot(qi, kj) / scale for kj in k]
            attn = _softmax(scores)
            mixed = [0.0] * self.embed_dim
            for w, vj in zip(attn, v):
                for d in range(self.embed_dim):
                    mixed[d] += w * vj[d]
            attended.append(mixed)

        pooled = [0.0] * self.embed_dim
        for vec in attended:
            for d in range(self.embed_dim):
                pooled[d] += vec[d]
        return [x / self.seq_len for x in pooled]

    def encode_pair(self, ecg: List[float], ppg: List[float]) -> Tuple[List[float], List[float]]:
        return self._encode_signal(ecg, self.ecg_proj), self._encode_signal(ppg, self.ppg_proj)

    def clip_alignment_loss(
        self, ecg_emb: List[float], ppg_emb: List[float], alignment_temperature: float = 0.07
    ) -> float:
        """Return CLIP-style pairwise alignment loss using temperature-scaled cosine similarity."""
        similarity = _cosine(ecg_emb, ppg_emb) / max(alignment_temperature, 1e-6)
        # Single-pair approximation: maximize similarity with a stable logistic objective.
        if similarity > self.SIMILARITY_SATURATION_THRESHOLD:
            return math.exp(-similarity)
        return math.log1p(math.exp(-similarity))

    def predict_abp(self, ecg: List[float], ppg: List[float]) -> List[float]:
        ecg_emb, ppg_emb = self.encode_pair(ecg, ppg)
        fused = ecg_emb + ppg_emb
        pred = [0.0 for _ in range(self.abp_len)]
        for j in range(self.abp_len):
            pred[j] = self.reg_b[j] + sum(fused[i] * self.reg_w[i][j] for i in range(len(fused)))
        return pred

    def train_regression_step(
        self,
        ecg: List[float],
        ppg: List[float],
        abp_target: List[float],
        lr: float = 1e-3,
        clip_weight: float = 0.05,
    ) -> float:
        if len(abp_target) != self.abp_len:
            raise ValueError(f"Expected ABP length {self.abp_len}, got {len(abp_target)}")
        ecg_emb, ppg_emb = self.encode_pair(ecg, ppg)
        fused = ecg_emb + ppg_emb
        pred = self.predict_abp(ecg, ppg)

        mse = 0.0
        grad_out = [0.0 for _ in range(self.abp_len)]
        for i in range(self.abp_len):
            err = pred[i] - abp_target[i]
            mse += err * err
            grad_out[i] = 2.0 * err / self.abp_len
        mse /= self.abp_len

        for i in range(len(fused)):
            for j in range(self.abp_len):
                self.reg_w[i][j] -= lr * fused[i] * grad_out[j]
        for j in range(self.abp_len):
            self.reg_b[j] -= lr * grad_out[j]

        clip_loss = self.clip_alignment_loss(ecg_emb, ppg_emb)
        return mse + clip_weight * clip_loss
