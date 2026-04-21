# PPG-ECG2BP

A minimal runnable ECG + PPG multimodal ABP prediction pipeline for extracted MIMIC-III data.  
（面向已提取 MIMIC-III 数据的 ECG + PPG 多模态 ABP 预测最小实现）

## 模型设计（轻量级）

- **Transformer 风格模态编码器**：分别对 ECG、PPG 做 self-attention 编码；
- **CLIP 风格对齐损失**：通过 cosine alignment 约束 ECG/PPG 表征对齐；
- **融合回归头**：拼接双模态 embedding，预测 ABP 波形。

Implementation:

- `src/ecg_ppg2abp/model.py`
- `src/ecg_ppg2abp/train.py`
- `train.py`

## 数据格式（JSONL）

每行一个样本：

```json
{"ecg":[...], "ppg":[...], "abp":[...]}
```

要求：
- `ecg`、`ppg` 长度固定且一致（例如窗口长度 256）
- `abp` 长度固定（例如输出长度 64）

## 训练命令

```bash
python train.py \
  --dataset /path/to/mimic3_extracted.jsonl \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --clip-weight 0.05 \
  --embed-dim 16
```

输出训练指标（JSON），包含 `final_loss`。

## 测试

```bash
python -m unittest discover -s tests -v
```
