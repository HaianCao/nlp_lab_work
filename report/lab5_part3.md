# Lab 5: Part-of-Speech Tagging sử dụng RNN

## 📁 Cấu trúc dự án

```text
data/
└── ud-english-ewt/
    ├── en_ewt-ud-train.conllu    # Dữ liệu huấn luyện
    ├── en_ewt-ud-dev.conllu      # Dữ liệu validation
    └── en_ewt-ud-test.conllu     # Dữ liệu kiểm thử

notebook/
└── lab5-part3.ipynb
```

## 🔧 Các thành phần chính

### 1. Xử lý dữ liệu (Data Processing)

- **CoNLL-U Parser**:

  - Đọc định dạng chuẩn CoNLL-U từ file raw text.
  - Tách lấy cột `FORM` (từ gốc) và `UPOS` (nhãn từ loại Universal).
  - Loại bỏ các dòng comment (bắt đầu bằng `#`) và xử lý các câu cách nhau bởi dòng trống.

- **Vocabulary (Từ điển)**:

  - Xây dựng lớp `Vocabulary` để ánh xạ hai chiều `Token <-> Index`.
  - **Word Dictionary**: Tự động thêm token đặc biệt `<PAD>` (index 0) và `<UNK>` (index 1) để xử lý các từ không có trong tập huấn luyện (Out-of-Vocabulary).
  - **Tag Dictionary**: Chỉ thêm `<PAD>` (index 0) và các nhãn POS chuẩn (NOUN, VERB, ADJ, ...).

- **Padding Strategy**:
  - Sử dụng `pad_sequence` với tham số `batch_first=True`.
  - Cơ chế **Dynamic Padding**: Trong mỗi batch, các câu được đệm (pad) về độ dài của câu dài nhất _trong batch đó_ (thay vì câu dài nhất toàn bộ dataset). Điều này giúp tiết kiệm bộ nhớ và tăng tốc độ tính toán.

### 2. Kiến trúc Mô hình (Model Architecture)

- **Class**: `SimpleRNNTagger`
- **Loại mô hình**: Vanilla RNN (RNN thuần) cho bài toán Sequence Labeling.
- **Luồng xử lý dữ liệu (Forward Pass)**:
  1.  **Input**: Batch các chỉ số từ (indices) có kích thước `(Batch_Size, Seq_Len)`.
  2.  **Embedding Layer**: Chuyển đổi indices thành dense vectors kích thước `(Batch_Size, Seq_Len, Embedding_Dim)`.
    3.  **RNN Layer (Vanilla RNN)**:
      - Xử lý chuỗi theo chiều thuận (left-to-right).
      - Output dimension tại mỗi bước thời gian là: `hidden_dim`.
    4.  **Linear Layer (Fully Connected)**: Chiếu output của RNN về không gian nhãn (`num_tags`), tạo ra logits để tính xác suất.

### 3. Huấn luyện & Đánh giá (Training & Evaluation)

- **Loss Function**: Sử dụng `CrossEntropyLoss`.
  - Cấu hình quan trọng: `ignore_index=PAD_IDX`. Tham số này đảm bảo mô hình không bị phạt (không tính loss) khi dự đoán sai tại các vị trí đệm (padding), giúp gradient tập trung vào các từ thực.
- **Optimizer**: Sử dụng `Adam` với learning rate `0.001`, cho khả năng hội tụ nhanh hơn SGD thuần.
- **Metric**: Accuracy (Độ chính xác).
  - Được tính toán bằng thư viện **Numpy** để tối ưu hiệu năng.
  - Chỉ tính độ chính xác trên các token khác padding (masking strategy).
- **Model Selection**:
  - Theo dõi độ chính xác trên tập Validation (Dev set) sau mỗi epoch.
  - Chỉ lưu checkpoint `best_model.pth` khi `Val Accuracy` đạt đỉnh mới.
- **Monitoring**: Tích hợp thư viện `tqdm` để hiển thị thanh tiến trình (progress bar), loss, và accuracy theo thời gian thực.

### 2. Chạy huấn luyện và kiểm thử

Luồng thực thi của chương trình (chi tiết):

- Load Data: Tải dữ liệu `train/dev/test` từ thư mục `data/`.
- Build Vocab: Xây dựng bộ từ điển từ tập train (word2index, tag2index), thêm token đặc biệt `<PAD>` và `<UNK>`.
- Train: Huấn luyện mô hình qua 15 epochs, sử dụng checkpoint để tự động lưu model tốt nhất trên tập dev.
- Evaluate: Tải model tốt nhất (`best_model.pth`) và đánh giá trên tập Test.
- Demo: Thực hiện dự đoán nhãn cho các câu ví dụ tiếng Anh.

### 📊 Kết quả thử nghiệm

#### Tham số cấu hình (Configuration)

| Tham số       | Giá trị | Mô tả                                       |
| ------------- | ------: | ------------------------------------------- |
| Embedding Dim |   20000 | Kích thước vector biểu diễn từ              |
| Hidden Dim    |    1024 | Kích thước trạng thái ẩn của RNN           |
| Batch Size    |      16 | Số lượng mẫu dữ liệu trong một lần cập nhật |
| Epochs        |       5 | Tổng số vòng lặp huấn luyện                 |

#### Log quá trình huấn luyện (Mẫu)

```text
Loaded: Train(12543), Dev(2002), Test(2077)
Vocab Size: Word=16654, Tag=18

Epoch 1/5 | Train Loss: 0.4328 | Train Acc: 86.15% | Dev Acc: 85.90%
--> Saved Best Model!

Epoch 2/5 | Train Loss: 0.1990 | Train Acc: 92.51% | Dev Acc: 85.66%

Epoch 3/5 | Train Loss: 0.1697 | Train Acc: 93.33% | Dev Acc: 85.70%

Epoch 4/5 | Train Loss: 0.1622 | Train Acc: 93.45% | Dev Acc: 86.23%
--> Saved Best Model!

Epoch 5/5 | Train Loss: 0.1582 | Train Acc: 93.64% | Dev Acc: 85.19%

Final Best Dev Acc: 86.23%
```

#### Kết quả cuối cùng

- Best Validation Accuracy: ~86.23%

#### Demo Prediction (ví dụ)

Input Sentence: "The quick brown fox jumps over the lazy dog"

Predicted Output (JSON):

```json
[
  ["The", "DET"],
  ["quick", "ADJ"],
  ["brown", "ADJ"],
  ["fox", "NOUN"],
  ["jumps", "VERB"],
  ["over", "ADP"],
  ["the", "DET"],
  ["lazy", "ADJ"],
  ["dog", "NOUN"]
]
```

Input Sentence: "I love NLP"

Predicted Output (JSON):

```json
[
  ["I", "PRON"],
  ["love", "VERB"],
  ["NLP", "ADV"]
]
```

### 💡 Phân tích & Đánh giá

1. Mô hình hiện tại sử dụng RNN thuần (Vanilla RNN)

- Hạn chế của RNN: RNN truyền thống (Vanilla RNN) gặp vấn đề Vanishing Gradient (tiêu biến đạo hàm) khiến nó khó học được sự phụ thuộc dài hạn. RNN cũng chỉ xử lý theo chiều một chiều (quá khứ -> hiện tại) và thiếu cơ chế cổng (gates) để điều phối thông tin.
- Hệ quả thực tiễn: Với RNN thuần, mô hình có thể vẫn học được các mẫu ngắn hạn và đạt kết quả chấp nhận được trên tập dữ liệu này, nhưng sẽ kém hơn các mô hình có cơ chế ghi nhớ dài hạn (như LSTM/GRU) khi cần xử lý phụ thuộc xa.

2. Xử lý từ chưa biết (OOV - Out of Vocabulary)

- Trong thực tế, tập Test luôn chứa những từ chưa từng xuất hiện trong tập Train.
- Giải pháp: sử dụng token `<UNK>` (Unknown) giúp hệ thống không bị crash. Mô hình học cách biểu diễn vector cho `<UNK>` dựa trên các từ tần suất thấp trong tập train, từ đó có thể đưa ra dự đoán hợp lý cho từ lạ dựa trên ngữ cảnh (các từ xung quanh).

3. Tối ưu hiệu năng

- Numpy Metrics: Chuyển việc tính toán Accuracy từ Tensor (GPU) sang Numpy (CPU) giúp giảm tải cho GPU và tận dụng tốc độ xử lý mảng của Numpy.
- Dynamic Padding: Thay vì padding toàn bộ dataset theo câu dài nhất (có thể lên tới 100-200 từ), ta chỉ padding theo độ dài lớn nhất trong từng batch (ví dụ: 30-40 từ). Điều này giúp giảm đáng kể lượng tính toán vô ích trên các token `<PAD>`.

### ⚠️ Khó khăn và Giải pháp

1. Vấn đề Padding và Loss Function

- Vấn đề: Các câu ngắn được điền thêm token `<PAD>` (index 0). Nếu tính toán Loss trên cả các token này, mô hình sẽ bị nhiễu vì phải học cách dự đoán nhãn cho `<PAD>`, làm giảm độ chính xác trên các từ thật.
- Giải pháp: Sử dụng tham số `ignore_index=PAD_IDX` trong `CrossEntropyLoss`. PyTorch sẽ tự động bỏ qua các vị trí có nhãn là 0 khi tính gradient, giúp mô hình chỉ tập trung học các từ có nghĩa.

2. Hiện tượng Overfitting

- Vấn đề: Với kích thước Embedding (100) và Hidden Dim (256), mô hình có số lượng tham số khá lớn so với lượng dữ liệu train (~12k câu), dẫn đến việc mô hình "học vẹt" (Acc trên Train rất cao nhưng trên Dev không tăng).
- Giải pháp:
  - Áp dụng Dropout (p=0.5) tại các lớp Embedding và RNN để ngẫu nhiên tắt các nơ-ron, buộc mô hình phải học các đặc trưng mạnh mẽ hơn.
  - Sử dụng cơ chế Model Checkpointing: Luôn lưu lại phiên bản mô hình có độ chính xác cao nhất trên tập Dev, thay vì lấy mô hình ở epoch cuối cùng (thường đã bị overfit).
