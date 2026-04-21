# PPG-ECG2BP

一个可直接运行的 ECG + PPG 多模态融合预测 ABP 的最小实现，面向已提取好的 MIMIC-III 数据。

## 模型设计（轻量级）

- **Transformer 风格模态编码器**：分别对 ECG、PPG 做 self-attention 编码；
- **CLIP 风格对齐损失**：通过 cosine alignment 约束 ECG/PPG 表征对齐；
- **融合回归头**：拼接双模态 embedding，预测 ABP 波形。

实现位于：

- `/home/runner/work/PPG-ECG2BP/PPG-ECG2BP/src/ecg_ppg2abp/model.py`
- `/home/runner/work/PPG-ECG2BP/PPG-ECG2BP/src/ecg_ppg2abp/train.py`
- `/home/runner/work/PPG-ECG2BP/PPG-ECG2BP/train.py`

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
python /home/runner/work/PPG-ECG2BP/PPG-ECG2BP/train.py \
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
cd /home/runner/work/PPG-ECG2BP/PPG-ECG2BP
python -m unittest discover -s tests -v
```
