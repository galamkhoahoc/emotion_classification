# ViEmoText - Vietnamese Emotion Text Classification

**Phiên bản 2.0 - Cấu trúc mã nguồn được tổ chức lại hoàn toàn**

Dự án phân loại cảm xúc văn bản tiếng Việt sử dụng mô hình **PhoBERT** với hỗ trợ emoji thông qua kỹ thuật **Emoji Embedding Transfer**.

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

- ✅ **PhoBERT-based**: Sử dụng mô hình pre-trained PhoBERT cho tiếng Việt
- ✅ **Emoji Support**: Hỗ trợ emoji trong văn bản thông qua Emoji Embedding Transfer
- ✅ **Multiple Loss Functions**: Hỗ trợ Cross Entropy, Weighted CE, và Focal Loss
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 (Macro & Weighted)
- ✅ **Training Utilities**: Early stopping, checkpointing, learning rate scheduling
- ✅ **Evaluation Tools**: Confusion matrix, classification report, prediction saving
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
│   │   ├── phobert_emotion.py    # PhoBERT classifier
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
│   └── evaluate.py               # Evaluation script
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

### Sử dụng trong code

```python
from configs.config import Config
from src.models.phobert_emotion import PhoBERTEmotionClassifier
from src.models.emoji_embeddings import apply_emoji_embeddings
from transformers import AutoTokenizer
import torch

# Load config
config = Config()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = PhoBERTEmotionClassifier(
    model_name=config.model_name,
    num_labels=config.num_labels
)

# Apply emoji embeddings
model = apply_emoji_embeddings(model, tokenizer)

# Load trained weights
model.load_state_dict(torch.load('path/to/model.pt'))
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
config = Config(
    # Model
    model_name="vinai/phobert-base",
    num_labels=7,
    max_length=256,
    
    # Training
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=10,
    
    # Emoji
    enable_emoji_embedding=True,
    
    # Loss
    loss_type="cross_entropy",  # hoặc "focal_loss", "weighted_ce"
    
    # Paths
    output_dir="outputs",
    checkpoint_dir="checkpoints"
)
```

---

## Tài liệu tham khảo

1. **PhoBERT**: Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. *Findings of EMNLP 2020*. [[Paper]](https://arxiv.org/abs/2003.00744)

2. **UIT-VSMEC**: Vietnamese Social Media Emotion Corpus. [[Dataset]](https://huggingface.co/datasets/uit-nlp/vietnamese_students_feedback)

3. **Transformers**: Wolf, T., et al. (2020). Transformers: State-of-the-art Natural Language Processing. *EMNLP 2020*. [[Paper]](https://arxiv.org/abs/1910.03771)

4. **Focal Loss**: Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*. [[Paper]](https://arxiv.org/abs/1708.02002)

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
- UIT-NLP Lab cho dataset UIT-VSMEC
- HuggingFace cho Transformers library
- Cộng đồng NLP Việt Nam

---

**⭐ Nếu dự án hữu ích, hãy cho chúng tôi một star trên GitHub!**
