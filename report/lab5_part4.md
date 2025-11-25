# Lab 5 Part 4: NER với Bi-LSTM


## 1. 🗂️ Cấu trúc file code
```
├── Import thư viện (torch, datasets, seqeval) & định nghĩa siêu tham số 
├── Task 1: Load dataset CoNLL-2003 + xây vocab thủ công 
├── Task 2: Dataset class + collate_fn (padding) 
├── Task 3: Định nghĩa model Bi-LSTM (không tích hợp CRF) 
├── Task 4: Vòng lặp train (CrossEntropyLoss) + Early Stopping 
├── Task 5: Dự đoán thử nghiệm (Inference) 
└── Hàm evaluate chi tiết (SeqEval metrics)
```
## 2. 🔧 Chi tiết từng phần triển khai

### **1. Imports & Hyperparameters**
- PyTorch core, HuggingFace datasets, seqeval.
- Tham số chính:
- `BATCH_SIZE = 16`
- `EMBEDDING_DIM = 100`
- `HIDDEN_DIM = 256`
- `PATIENCE = 3`

### **2. Load dữ liệu**
- Dataset: **CoNLL-2003**.
- Xây **vocab thủ công** từ tập train.
- Token OOV → `<UNK>`.


### **3. Dataset + DataLoader**
- Class `NERDataset` kế thừa `torch.utils.data.Dataset`.
- Hàm `collate_fn` dùng `pad_sequence(batch_first=True)`.
- Padding:
- Token = `0`
- Label = `PAD_TAG = -1`


### **4. Bi-LSTM Model**
- Kiến trúc model:
- `nn.Embedding(vocab_size, 100)`
- `nn.LSTM(..., bidirectional=True, batch_first=True)`
- `nn.Linear(hidden_dim * 2, num_labels)`
- Không có lớp CRF.


### **5. Huấn luyện**
- Loss: `nn.CrossEntropyLoss(ignore_index=PAD_TAG)`.
- Optimizer: Adam.
- Early Stopping theo `val_loss` (patience = 3).


### **6. Đánh giá**
- Metrics:
    - Loss
    - Token-level Accuracy
    - Entity-level Precision/Recall/F1 (seqeval)


### **7. Dự đoán**
- `predict_sentence()`:
- Preprocess → Model → Argmax.


## 3. 📊 Kết quả huấn luyện & đánh giá


### **Kết quả huấn luyện**


| Epoch | Train Loss | Val Loss | Val Acc | Ghi chú |
|-------|-----------|----------|----------|---------|
| 1 | 0.3705 | 0.2642 | 0.9262 | Loss giảm nhanh |
| 2 | 0.1588 | 0.2051 | 0.9431 | Học tốt |
| 3 | 0.0716 | 0.1841 | 0.9483 | **Best model** |
| 4 | 0.0270 | 0.1942 | 0.9519 | Overfitting nhẹ |
| 5 | 0.0089 | 0.2128 | 0.9525 | Overfitting mạnh |
| 6 | 0.0035 | 0.2415 | 0.9526 | Early Stopping |


---


### **Kết quả dự đoán thực tế**
- **Ưu điểm:**
- Nhận diện tốt các thực thể rõ ràng: *New York City (LOC), Microsoft (ORG)*.
- **Nhược điểm (thiếu CRF):**
- Sai logic nhãn: *I-PER đứng đầu chuỗi*.
- Miss entity: *Malala → O*.


---


### **SeqEval (Test Set)**


| Entity | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|----------|
| LOC | 0.86 | 0.68 | 0.76 | 1668 |
| MISC | 0.45 | 0.64 | 0.53 | 702 |
| ORG | 0.74 | 0.57 | 0.64 | 1661 |
| PER | 0.65 | 0.67 | 0.66 | 1617 |
| **Macro Avg** | **0.68** | **0.64** | **0.65** | **5648** |


🔎 **Nhận xét chuyên môn:**
- ORG recall thấp → model bỏ sót nhiều tổ chức.
- MISC precision thấp → model dự đoán nhầm nhiều.
- F1 tổng chỉ ~0.66 → mức trung bình cho mô hình không có pretrained embedding.


---


## 4. ⚠️ Hạn chế & Vấn đề kỹ thuật


### **1. Thiếu CRF Layer**
- CrossEntropyLoss không học được **transition rules**.
- Dẫn đến chuỗi nhãn không hợp lệ.


### **2. Embedding ngẫu nhiên**
- Không dùng GloVe/BERT → mô hình học chậm và yếu.


### **3. Tokenization & OOV**
- Word-level thủ công → mất thông tin ở entity hiếm.

## 5. 📚 Tài liệu 
- Notebook: *lab5-part4.ipynb*
- Dataset: CoNLL-2003
- Model: Gemini (Pro)