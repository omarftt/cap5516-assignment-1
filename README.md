
Readme · MD
Copy

# Pneumonia Detection with ViT-Tiny

## Folder Structure

```
assignment1/
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── checkpoints/
├── logs/
├── results/
├── plots/
├── utils/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── gradcam.py
│   └── utils.py
├── train.py
├── evaluate.py
├── main.py
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

---

## Experiments

| Exp | Configuration       | Pretrained | Warmup | Weighted Loss | Label Smoothing | Rand. Erasing |
|-----|---------------------|------------|--------|---------------|-----------------|---------------|
| 1   | Scratch             | ✗          | ✗      | ✗             | 0.1             | ✗             |
| 2   | Fine-tune           | ✓          | ✗      | ✗             | 0.1             | ✗             |
| 3   | + Warmup            | ✓          | ✓      | ✗             | 0.1             | ✗             |
| 4   | + Weighted Loss     | ✓          | ✗      | ✓             | 0.1             | ✗             |
| 5   | No Label Smoothing  | ✓          | ✗      | ✗             | 0.0             | ✗             |
| 6   | Warmup + Weighted   | ✓          | ✓      | ✓             | 0.1             | ✗             |
| 7   | + Rand. Erasing     | ✓          | ✓      | ✓             | 0.1             | ✓             |

---

## Training Commands

```bash
# Exp 1 — Scratch
python main.py --run_name exp1_scratch --mode train --batch_size 128 --epochs 50

# Exp 2 — Fine-tune
python main.py --run_name exp2_finetune --pretrained --mode train --batch_size 128 --epochs 50

# Exp 3 — Warmup
python main.py --run_name exp3_warmup --pretrained --use_warmup --mode train --batch_size 128 --epochs 50

# Exp 4 — Weighted Loss
python main.py --run_name exp4_weighted --pretrained --use_weighted_loss --mode train --batch_size 128 --epochs 50

# Exp 5 — No Label Smoothing
python main.py --run_name exp5_no_smooth --pretrained --label_smoothing 0.0 --mode train --batch_size 128 --epochs 50

# Exp 6 — Warmup + Weighted Loss
python main.py --run_name exp6_weighted_warmup --pretrained --use_warmup --use_weighted_loss --mode train --batch_size 128 --epochs 50

# Exp 7 — Random Erasing
python main.py --run_name exp7_erasing --pretrained --use_warmup --use_weighted_loss --use_random_erasing --mode train --batch_size 128 --epochs 50
```

## Testing Commands

```bash
# Add --gradcam to any command to generate Grad-CAM failure case plots
python main.py --run_name exp1_scratch --mode test
python main.py --run_name exp2_finetune --mode test --gradcam
python main.py --run_name exp3_warmup --mode test
python main.py --run_name exp4_weighted --mode test
python main.py --run_name exp5_no_smooth --mode test
python main.py --run_name exp6_weighted_warmup --mode test
python main.py --run_name exp7_erasing --mode test
```