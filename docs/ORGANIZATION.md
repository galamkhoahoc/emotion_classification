# Tổ chức Logs và Checkpoints - ViEmoText

## 📋 Tổng quan

Tài liệu này mô tả cách tổ chức và quản lý logs, checkpoints, và các artifacts khác trong dự án ViEmoText.

---

## 📁 Cấu trúc thư mục

```
ViEmoText/
├── outputs/              # Kết quả training tổng hợp
│   ├── experiment_1/
│   │   ├── config.json
│   │   ├── final_model.pt
│   │   ├── training_metrics.json
│   │   └── results.txt
│   └── experiment_2/
│
├── logs/                 # Training logs
│   ├── training_20260828_154300.log
│   ├── training_20260829_091500.log
│   └── tensorboard/      # TensorBoard logs (optional)
│       ├── run_1/
│       └── run_2/
│
├── checkpoints/          # Model checkpoints
│   ├── experiment_1/
│   │   ├── best_model.pt
│   │   ├── checkpoint_epoch_5.pt
│   │   └── checkpoint_epoch_10.pt
│   └── experiment_2/
│       └── best_model.pt
│
├── data/                 # Dataset cache
│   ├── .cache/          # HuggingFace cache
│   └── processed/       # Processed data (if any)
│
└── evaluation_results/   # Evaluation outputs
    ├── test_results_exp1/
    │   ├── confusion_matrix.png
    │   ├── metrics.txt
    │   └── predictions.npz
    └── test_results_exp2/
```

---

## 🎯 Best Practices

### 1. **Naming Convention**

#### Training runs
```
Format: {experiment_name}_{YYYYMMDD}_{HHMMSS}
Example: phobert_baseline_20260828_154300
```

#### Checkpoints
```
best_model.pt              # Best model theo validation metric
final_model.pt             # Model ở epoch cuối cùng
checkpoint_epoch_{N}.pt    # Checkpoint tại epoch N
```

#### Logs
```
Format: training_{YYYYMMDD}_{HHMMSS}.log
Example: training_20260828_154300.log
```

---

## 💾 Checkpoint Management

### Checkpoint Structure

Mỗi checkpoint nên chứa:

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_metric': best_metric,
    'config': config.to_dict(),
    'training_time': time.time(),
    'loss_history': loss_history
}
```

### Save Strategy

**Option 1: Save Best Only** (Recommended cho production)
- Chỉ lưu model tốt nhất theo validation metric
- Tiết kiệm disk space
- Đủ cho deployment

```python
if val_metric > best_metric:
    torch.save(checkpoint, 'checkpoints/best_model.pt')
```

**Option 2: Save Every N Epochs**
- Lưu checkpoint mỗi N epochs
- Cho phép resume training
- Hữu ích cho debugging

```python
if epoch % save_every == 0:
    torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pt')
```

**Option 3: Save Top K** (Recommended cho experimentation)
- Giữ K checkpoints tốt nhất
- Balance giữa space và flexibility

```python
# Giữ top 3 models
save_total_limit = 3
```

---

## 📊 Logging Strategy

### 1. File Logging

**Training log** - Chi tiết mọi thứ:
```
logs/training_20260828_154300.log
```

Content:
```
2026-08-28 15:43:00 - INFO - Starting training...
2026-08-28 15:43:01 - INFO - Epoch 1/10
2026-08-28 15:45:32 - INFO - Train Loss: 1.2345
2026-08-28 15:47:15 - INFO - Val Loss: 1.1234 | F1: 0.6543
```

### 2. Metrics Tracking

**training_metrics.json** - Structured metrics:
```json
{
  "train_loss": [1.5, 1.3, 1.2, ...],
  "val_loss": [1.4, 1.2, 1.1, ...],
  "val_metrics": [
    {"accuracy": 0.65, "f1_macro": 0.63},
    {"accuracy": 0.68, "f1_macro": 0.66}
  ]
}
```

### 3. Console Output

- Progress bars (tqdm)
- Summary sau mỗi epoch
- Final results

---

## 🔄 Workflow Examples

### Experiment 1: Baseline

```bash
python scripts/train.py \
    --output_dir outputs/baseline_exp1 \
    --checkpoint_dir checkpoints/baseline_exp1 \
    --log_dir logs
```

**Tạo ra:**
```
outputs/baseline_exp1/
├── config.json
├── final_model.pt
└── training_metrics.json

checkpoints/baseline_exp1/
└── best_model.pt

logs/
└── training_20260828_154300.log
```

### Experiment 2: With Focal Loss

```bash
python scripts/train.py \
    --output_dir outputs/focal_loss_exp2 \
    --checkpoint_dir checkpoints/focal_loss_exp2
```

**Tạo ra:**
```
outputs/focal_loss_exp2/
checkpoints/focal_loss_exp2/
logs/training_20260828_163045.log
```

---

## 🧹 Cleanup Strategy

### 1. Automatic Cleanup

Trong training script, tự động xóa old checkpoints:

```python
# Keep only top 3 checkpoints
save_total_limit = 3

# Hoặc delete checkpoints older than N days
max_checkpoint_age_days = 30
```

### 2. Manual Cleanup

Script để cleanup:

```bash
# Xóa checkpoints cũ hơn 30 ngày
find checkpoints/ -name "*.pt" -mtime +30 -delete

# Xóa logs cũ hơn 60 ngày
find logs/ -name "*.log" -mtime +60 -delete

# Keep only best models
find checkpoints/ -name "checkpoint_epoch_*.pt" -delete
```

### 3. Selective Archiving

Cho experiments quan trọng, archive vào cloud storage:

```bash
# Upload to cloud
aws s3 sync outputs/important_exp1/ s3://bucket/experiments/exp1/
# Or
rclone copy outputs/important_exp1/ gdrive:experiments/exp1/
```

---

## 📈 Monitoring Tools

### 1. TensorBoard (Optional)

Nếu muốn visualization tốt hơn:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/tensorboard/run_1')
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Metrics/f1', f1_score, epoch)
```

View:
```bash
tensorboard --logdir logs/tensorboard
```

### 2. Weights & Biases (Optional)

Cho team collaboration:

```python
import wandb

wandb.init(project="viemotext", name="experiment_1")
wandb.log({"train_loss": train_loss, "val_f1": val_f1})
```

### 3. MLflow (Optional)

Cho experiment tracking:

```python
import mlflow

mlflow.log_params(config.to_dict())
mlflow.log_metric("val_f1", val_f1)
mlflow.pytorch.log_model(model, "model")
```

---

## 🎨 Visualization

### Compare Experiments

Script để so sánh results:

```python
# scripts/compare_experiments.py
import json
import matplotlib.pyplot as plt

# Load metrics từ nhiều experiments
exp1 = json.load(open('outputs/exp1/training_metrics.json'))
exp2 = json.load(open('outputs/exp2/training_metrics.json'))

# Plot comparison
plt.plot(exp1['val_loss'], label='Baseline')
plt.plot(exp2['val_loss'], label='Focal Loss')
plt.legend()
plt.savefig('outputs/comparison.png')
```

---

## 💡 Tips & Tricks

### 1. Unique Experiment Names

Sử dụng timestamp để tránh overwrite:

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs/exp_{timestamp}"
```

### 2. Save Config với Results

Luôn lưu config để reproduce:

```python
import json

with open(f'{output_dir}/config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)
```

### 3. Git Commit Hash

Lưu git commit để track code version:

```python
import subprocess

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
metadata = {'git_commit': git_hash, 'timestamp': timestamp}
```

### 4. Quick Resume Training

```python
# Save với resume info
checkpoint = {
    ...,
    'epoch': epoch,
    'global_step': global_step,
    'rng_state': torch.get_rng_state()
}

# Resume
checkpoint = torch.load('checkpoints/checkpoint_epoch_5.pt')
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 📦 Disk Space Management

### Estimated Sizes

- **PhoBERT model**: ~500MB per checkpoint
- **Logs**: ~10MB per training run
- **Metrics**: <1MB per experiment
- **HuggingFace cache**: ~2GB (dataset + model)

### Storage Recommendations

- **Minimum**: 10GB (cho 5-10 experiments)
- **Recommended**: 50GB (cho nhiều experiments)
- **Optimal**: 100GB+ với external backup

---

## 🔐 Security & Privacy

### Don't Commit

- ❌ Model checkpoints (.pt, .pth)
- ❌ Training logs (.log)
- ❌ Large datasets
- ❌ API keys / credentials

### Do Commit

- ✅ Config files
- ✅ Scripts
- ✅ Directory structure (.gitkeep)
- ✅ Documentation

### Sharing Models

Cho public sharing:
1. Upload lên HuggingFace Model Hub
2. Hoặc release qua GitHub Releases
3. Bao gồm model card với usage instructions

---

## 📝 Example: Complete Training Session

```bash
# 1. Start training
python scripts/train.py \
    --output_dir outputs/exp_emoji_baseline \
    --checkpoint_dir checkpoints/exp_emoji_baseline \
    --batch_size 32 \
    --num_epochs 15

# Generates:
# - logs/training_20260828_154300.log
# - checkpoints/exp_emoji_baseline/best_model.pt
# - checkpoints/exp_emoji_baseline/final_model.pt
# - outputs/exp_emoji_baseline/config.json
# - outputs/exp_emoji_baseline/training_metrics.json

# 2. Evaluate
python scripts/evaluate.py \
    --checkpoint checkpoints/exp_emoji_baseline/best_model.pt \
    --output_dir evaluation_results/exp_emoji_baseline \
    --plot_cm \
    --save_predictions

# Generates:
# - evaluation_results/exp_emoji_baseline/metrics.txt
# - evaluation_results/exp_emoji_baseline/confusion_matrix.png
# - evaluation_results/exp_emoji_baseline/predictions.npz

# 3. Archive if satisfied
mkdir -p archives/exp_emoji_baseline
cp -r outputs/exp_emoji_baseline archives/
cp -r checkpoints/exp_emoji_baseline archives/
cp -r evaluation_results/exp_emoji_baseline archives/
```

---

## 🚀 Next Steps

1. **Implement automatic cleanup** trong training script
2. **Add TensorBoard support** cho real-time monitoring
3. **Create comparison script** để so sánh experiments
4. **Setup cloud backup** cho important checkpoints
5. **Document best hyperparameters** trong mỗi experiment folder

---

**💡 Remember: Good organization saves time and prevents confusion!**
