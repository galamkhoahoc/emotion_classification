# ViEmoText - Vietnamese Emotion Text Classification

**Phiên bản 2.0 - Cấu trúc mã nguồn được tổ chức lại hoàn toàn**

Dự án phân loại cảm xúc văn bản tiếng Việt sử dụng mô hình **PhoBERT** và **BamiBERT** với hỗ trợ emoji thông qua kỹ thuật **Emoji Embedding Transfer**. Hỗ trợ đa mô hình thông qua Factory Pattern, cho phép dễ dàng chuyển đổi giữa các mô hình.

> **Phát triển bởi:** Nhóm Gà làm khoa học (HCMUS)  
> **Kế thừa từ:** PhoBERT-Emoji v2 - Đồ án nhóm ban đầu

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Dataset](#dataset)
- [Phương pháp](#phương-pháp)
- [Kết quả](#kết-quả)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Tổng quan

ViEmoText là một hệ thống phân loại cảm xúc văn bản tiếng Việt được xây dựng trên nền tảng PhoBERT (Vietnamese BERT). Điểm đặc biệt của dự án là khả năng hiểu và xử lý emoji thông qua kỹ thuật **Emoji Embedding Transfer** - ánh xạ vector embedding từ các từ tiếng Việt tương ứng sang emoji.

### Các loại cảm xúc được phân loại:

1. **Other** (Khác)
2. **Disgust** (Ghê tởm)
3. **Enjoyment** (Vui vẻ)
4. **Sadness** (Buồn)
5. **Fear** (Sợ hãi)
6. **Surprise** (Ngạc nhiên)
7. **Anger** (Tức giận)

---

## Tính năng

- ✅ **Multi-Model Support**: Hỗ trợ PhoBERT và BamiBERT với Factory Pattern
- ✅ **PhoBERT-based**: Sử dụng mô hình pre-trained PhoBERT cho tiếng Việt
- ✅ **BamiBERT-based**: Hỗ trợ BamiBERT của Qualcomm với context length 2048
- ✅ **Emoji Support**: Hỗ trợ emoji trong văn bản thông qua Emoji Embedding Transfer
- ✅ **Multiple Loss Functions**: Hỗ trợ Cross Entropy, Weighted CE, và Focal Loss
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 (Macro & Weighted)
- ✅ **Training Utilities**: Early stopping, checkpointing, learning rate scheduling
- ✅ **Evaluation Tools**: Confusion matrix, classification report, prediction saving
- ✅ **Model Comparison**: So sánh hiệu suất giữa các mô hình
- ✅ **Modular Architecture**: Cấu trúc code rõ ràng, dễ mở rộng và bảo trì

---

## Cấu trúc dự án

```
ViEmoText/
├── docs/                           # Tài liệu phiên bản
│   └── VERSION_280826_1543.md     # Version log
│
├── configs/                        # Cấu hình
│   ├── __init__.py
│   └── config.py                  # File cấu hình chính
│
├── src/                           # Mã nguồn chính
│   ├── __init__.py
│   │
│   ├── data/                      # Xử lý dữ liệu
│   │   ├── __init__.py
│   │   └── dataset.py             # Dataset classes và data loading
│   │
│   ├── models/                    # Kiến trúc mô hình
│   │   ├── __init__.py
│   │   ├── base_classifier.py    # Abstract base class
│   │   ├── phobert_emotion.py    # PhoBERT classifier
│   │   ├── bamibert_emotion.py   # BamiBERT classifier
│   │   ├── model_factory.py      # Model factory
│   │   └── emoji_embeddings.py   # Emoji embedding utilities
│   │
│   ├── losses/                    # Hàm loss
│   │   ├── __init__.py
│   │   ├── focal_loss.py         # Focal Loss
│   │   └── weighted_cross_entropy.py
│   │
│   └── utils/                     # Tiện ích
│       ├── __init__.py
│       ├── metrics.py             # Metrics computation
│       └── logger.py              # Logging utilities
│
├── scripts/                       # Scripts thực thi
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── compare_experiments.py    # Model comparison script
│
├── requirements.txt               # Dependencies
├── README.md                      # File này
├── LICENSE                        # MIT License
└── phobert_emoji_update_v2.ipynb # Notebook phiên bản cũ (legacy)
```

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- CUDA 11.0+ (nếu sử dụng GPU)
- 8GB RAM (16GB khuyến nghị)

### Cài đặt dependencies

```bash
# Clone repository
git clone <repository-url>
cd emotion_classification

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Cài đặt VnCoreNLP (Optional)

Nếu muốn sử dụng word segmentation:

```bash
pip install vncorenlp
# Download VnCoreNLP models
# Xem hướng dẫn tại: https://github.com/vncorenlp/VnCoreNLP
```

---

## Sử dụng

### Training

#### 1. Training cơ bản

```bash
python scripts/train.py
```

#### 2. Training với tùy chỉnh tham số

```bash
python scripts/train.py \
    --batch_size 32 \
    --num_epochs 20 \
    --learning_rate 3e-5 \
    --output_dir ./outputs
```

#### 3. Training không sử dụng emoji embeddings

```bash
python scripts/train.py --no_emoji
```

#### 4. Training với BamiBERT

```bash
python scripts/train.py --model_type bamibert
```

#### 5. Training BamiBERT với tùy chỉnh

```bash
python scripts/train.py \
    --model_type bamibert \
    --batch_size 8 \
    --num_epochs 15 \
    --learning_rate 2e-5
```

### Evaluation

#### 1. Đánh giá trên test set

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --split test \
    --plot_cm \
    --save_predictions
```

#### 2. Đánh giá trên validation set

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --split val \
    --output_dir evaluation_results
```

#### 3. Đánh giá model BamiBERT

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model_bamibert.pt \
    --model_type bamibert \
    --split test
```

### So sánh mô hình

So sánh kết quả giữa PhoBERT và BamiBERT:

```bash
python scripts/compare_experiments.py \
    outputs/phobert_experiment \
    outputs/bamibert_experiment \
    --output comparison_results
```

Script sẽ tạo:
- `comparison_table.csv` - Bảng so sánh dạng CSV
- `comparison_table.md` - Bảng so sánh dạng Markdown
- `training_curves.png` - Biểu đồ training curves

### Sử dụng trong code

```python
from configs.config import Config
from src.models.model_factory import create_model, create_model_from_checkpoint
from src.models.emoji_embeddings import apply_emoji_embeddings
import torch

# === Sử dụng PhoBERT (mặc định) ===
config = Config(model_type="phobert")
model, tokenizer = create_model(config)
model = apply_emoji_embeddings(model, tokenizer)

# === Sử dụng BamiBERT ===
config = Config(model_type="bamibert")
model, tokenizer = create_model(config)

# === Load model từ checkpoint ===
config = Config(model_type="phobert")
model, tokenizer = create_model_from_checkpoint('checkpoints/best_model.pt', config)
model.eval()

# Predict
text = "Tôi rất vui hôm nay 😊"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs['logits'], dim=1)
    
print(f"Predicted emotion: {config.emotion_labels[prediction]}")
```

---

## Dataset

Dự án sử dụng dataset **UIT-VSMEC** (Vietnamese Social Media Emotion Corpus):

- **Train set**: 5,548 câu
- **Validation set**: 686 câu  
- **Test set**: 693 câu

Dataset được tải tự động từ HuggingFace: `uit-nlp/vietnamese_students_feedback`

### Phân bố cảm xúc

| Cảm xúc | Số lượng mẫu |
|---------|--------------|
| Enjoyment | ~2,000 |
| Sadness | ~1,500 |
| Anger | ~800 |
| Fear | ~600 |
| Surprise | ~500 |
| Disgust | ~400 |
| Other | ~1,000 |

---

## Phương pháp

### 1. PhoBERT Model

PhoBERT là mô hình BERT được pre-train đặc biệt cho tiếng Việt, sử dụng:
- **Tokenization**: BPE (Byte Pair Encoding) với từ điển 64K tokens
- **Architecture**: BERT-base (12 layers, 768 hidden size, 12 attention heads)
- **Pre-training**: 20GB Vietnamese text từ Wikipedia và tin tức
- **Max context length**: 256 tokens
- **Word segmentation**: Cần thiết (sử dụng VnCoreNLP)

### 2. BamiBERT Model

BamiBERT là mô hình BERT tiếng Việt do Qualcomm AI Research phát triển:
- **Architecture**: BERT-base tương thích
- **Max context length**: 2048 tokens (dài hơn PhoBERT 8 lần)
- **Word segmentation**: Không cần thiết (hoạt động với raw text)
- **Lợi ích**: Xử lý được văn bản dài hơn, không cần preprocessing phức tạp

### So sánh PhoBERT vs BamiBERT

| Đặc điểm | PhoBERT | BamiBERT |
|-----------|---------|----------|
| Max tokens | 256 | 2048 |
| Word segmentation | Cần thiết | Không cần |
| Model name | vinai/phobert-base | Qualcomm-AI-Research/BamiBERT |
| Nguồn | VinAI Research | Qualcomm AI Research |

### 2. Emoji Embedding Transfer

Kỹ thuật độc đáo của dự án - chuyển ngữ nghĩa từ từ tiếng Việt sang emoji:

```python
# Mapping định nghĩa
emoji_mapping = {
    "😊": "vui",
    "😢": "buồn", 
    "😡": "giận",
    ...
}

# Transfer process
for emoji, word in emoji_mapping.items():
    emoji_id = tokenizer.encode(emoji)
    word_id = tokenizer.encode(word)
    embeddings[emoji_id] = embeddings[word_id].copy()
```

**Lợi ích:**
- Model hiểu được ngữ nghĩa của emoji mà không cần retrain
- Giảm data sparsity cho emoji tokens
- Cải thiện performance trên social media text

### 3. Training Strategy

- **Optimizer**: AdamW với weight decay
- **Learning Rate**: 2e-5 với linear warmup
- **Batch Size**: 16 (có thể điều chỉnh)
- **Early Stopping**: Patience = 3 epochs
- **Loss Functions**: Cross Entropy / Focal Loss / Weighted CE

---

## Kết quả

### Baseline Performance (PhoBERT gốc)

| Metric | Score |
|--------|-------|
| Accuracy | 0.6811 |
| F1 Macro | 0.6496 |
| F1 Weighted | 0.6806 |
| Precision Macro | 0.6608 |
| Recall Macro | 0.6431 |

### Performance với Emoji Embeddings

*(Cập nhật sau khi training với cấu trúc mới)*

---

## Cấu hình

Tất cả cấu hình được quản lý trong `configs/config.py`:

```python
# === PhoBERT configuration ===
config = Config(
    model_type="phobert",        # Chọn PhoBERT
    num_labels=7,
    max_length=256,              # Tự động set cho PhoBERT
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=10,
    enable_emoji_embedding=True,
    loss_type="cross_entropy",
    output_dir="outputs",
    checkpoint_dir="checkpoints"
)

# === BamiBERT configuration ===
config = Config(
    model_type="bamibert",       # Chọn BamiBERT
    num_labels=7,
    max_length=2048,             # Tự động set cho BamiBERT
    batch_size=8,                # BamiBERT dùng nhiều memory hơn
    learning_rate=2e-5,
    num_epochs=10,
    enable_emoji_embedding=True,
    loss_type="cross_entropy",
    output_dir="outputs_bamibert",
    checkpoint_dir="checkpoints_bamibert"
)
```

> **Lưu ý**: Khi sử dụng BamiBERT:
> - `model_name` tự động được set thành `Qualcomm-AI-Research/BamiBERT`
> - `max_length` mặc định 2048 (có thể giảm để tiết kiệm memory)
> - `use_word_segmentation` tự động disabled
> - Nên giảm `batch_size` do context length lớn hơn

---

## Tài liệu tham khảo

1. **PhoBERT**: Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. *Findings of EMNLP 2020*. [[Paper]](https://arxiv.org/abs/2003.00744)

2. **BamiBERT**: Qualcomm AI Research. BamiBERT: A Vietnamese BERT model. [[Model]](https://huggingface.co/Qualcomm-AI-Research/BamiBERT)

3. **UIT-VSMEC**: Vietnamese Social Media Emotion Corpus. [[Dataset]](https://huggingface.co/datasets/uit-nlp/vietnamese_students_feedback)

4. **Transformers**: Wolf, T., et al. (2020). Transformers: State-of-the-art Natural Language Processing. *EMNLP 2020*. [[Paper]](https://arxiv.org/abs/1910.03771)

5. **Focal Loss**: Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*. [[Paper]](https://arxiv.org/abs/1708.02002)

---

## Đóng góp

Nhóm nghiên cứu **Gà làm khoa học** - Trường Đại học Khoa học Tự nhiên TPHCM

### Thành viên:
- Trần Khánh Duy (MSSV: 23110078)
- Lê Minh Huy (MSSV: 23110018)
- Lê Văn Quý (MSSV: 23110040)
- Đoàn Phú Quý (MSSV: 23110112)

---

## License

Dự án được phát hành dưới [MIT License](LICENSE).

---

## Liên hệ

**Nhóm nghiên cứu Gà làm khoa học**  
Khu đô thị Đại học Quốc gia, Phường Đông Hòa, TP.HCM  
Email: galamkhoahoc@gmail.com

---

## Lời cảm ơn

- VinAI Research cho PhoBERT model
- Qualcomm AI Research cho BamiBERT model
- UIT-NLP Lab cho dataset UIT-VSMEC
- HuggingFace cho Transformers library
- Cộng đồng NLP Việt Nam

---

**⭐ Nếu dự án hữu ích, hãy cho chúng tôi một star trên GitHub!**
