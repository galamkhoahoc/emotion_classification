# PhoBERT-Emoji v2

**Phiên bản đồ án nhóm xây dựng pipeline và mô hình Emoji Weights ban đầu. Nhóm Gà làm khoa học (HCMUS) kế thừa hợp pháp phiên bản này và mở rộng phát triển nó trong các giai đoạn tiếp theo.**

Dự án phân loại cảm xúc tiếng Việt sử dụng mô hình PhoBERT với hỗ trợ emoji.

##  Mô tả

Dự án này sử dụng mô hình **PhoBERT** (Vietnamese BERT) để phân loại cảm xúc trong văn bản tiếng Việt. Điểm đặc biệt của phiên bản v2 là khả năng xử lý và hiểu ngữ nghĩa của emoji thông qua việc ánh xạ vector embedding từ các từ tiếng Việt tương ứng.

### Tính năng chính

-  Phân loại 7 loại cảm xúc: Enjoyment, Sadness, Anger, Fear, Disgust, Surprise, Other
-  Sử dụng PhoBERT pre-trained model cho tiếng Việt
-  Hỗ trợ emoji trong văn bản thông qua kỹ thuật embedding transfer
-  Sử dụng VnCoreNLP cho việc word segmentation
-  Trained trên dataset UIT-VSMEC

## Cài đặt

### Yêu cầu

- Python 3.7+
- CUDA (nếu sử dụng GPU)
- Java JDK 8+ (cho VnCoreNLP)


## Dataset

Dự án sử dụng dataset **UIT-VSMEC** (Vietnamese Social Media Emotion Corpus) từ Hugging Face:

- **Train set**: 5,548 câu
- **Validation set**: 686 câu
- **Test set**: 693 câu

Dataset bao gồm các cảm xúc:
- Other (Khác)
- Disgust (Ghê tởm)
- Enjoyment (Vui vẻ)
- Sadness (Buồn)
- Fear (Sợ hãi)
- Surprise (Ngạc nhiên)
- Anger (Tức giận)

## Sử dụng

### Training

Mở và chạy notebook `phobert_emoji_update_v2.ipynb` trong Jupyter hoặc Google Colab:

1. **Load dataset**: Dataset tự động được tải từ Hugging Face
2. **Preprocessing**: Sử dụng VnCoreNLP để word segmentation
3. **Emoji Embedding**: Ánh xạ emoji sang vector từ các từ tiếng Việt tương ứng
4. **Training**: Fine-tune PhoBERT model
5. **Evaluation**: Đánh giá trên test set


## Phương pháp

### Emoji Embedding Transfer

Một trong những đóng góp chính của dự án là kỹ thuật transfer embedding cho emoji:

1. **Mapping**: Tạo dictionary ánh xạ emoji → từ tiếng Việt
   - 😊 → "vui"
   - 😢 → "buồn"
   - 😡 → "giận"
   - ...

2. **Transfer**: Copy embedding vector từ từ tiếng Việt sang emoji token
   ```python
   embeddings[icon_id] = embeddings[vietnamese_id]
   ```

3. **Benefit**: Model có thể hiểu ngữ nghĩa của emoji mà không cần retrain từ đầu

## Cấu trúc dự án

```
PhoBERT-Emoji_v2/
├── phobert_emoji_update_v2.ipynb  # Notebook chính
├── LICENSE                         # MIT License
└── README.md                       # File này
```

## Kết quả

Model đạt được hiệu suất tốt trên dataset UIT-VSMEC với khả năng:
- Phân loại chính xác các cảm xúc trong văn bản tiếng Việt
- Xử lý được emoji trong context
- Hiểu được ngữ nghĩa của social media text 
- Đánh giá trên test set: 
  
      - Accuracy: 0.6811
      - F1 Macro: 0.6496
      - F1 Weighted: 0.6806
      - Precision Macro: 0.6608
      - Recall Macro: 0.6431




## Tài liệu tham khảo

1. PhoBERT: Pre-trained language models for Vietnamese (Nguyen & Nguyen, 2020)
2. UIT-VSMEC: A Vietnamese Social Media Emotion Corpus for Emotion Recognition
3. Transformers: State-of-the-art Natural Language Processing (Hugging Face)

## Liên hệ

**Nhóm nghiên cứu Gà làm khoa học**<br>
Khu đô thị Đại học Quốc gia, Phường Đông Hòa, TP.HCM<br>
Email: galamkhoahoc@gmail.com
