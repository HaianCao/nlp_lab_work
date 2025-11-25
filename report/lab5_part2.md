# Lab 5: Sử dụng RNN (LSTM) cho bài toán phân loại văn bản

So sánh bốn phương pháp khác nhau:

1. **TF-IDF + Logistic Regression** — đơn giản, nhanh
2. **Word2Vec + Dense Layer** — sử dụng embedding word2vec
3. **Embedding (Pre-trained) + LSTM** — LSTM với embedding được pre-train từ Word2Vec
4. **Embedding (Scratch) + LSTM** — LSTM với embedding được học từ đầu

---

## Các bước thực hiện

### 1. **Chuẩn bị dữ liệu (Data Preprocessing)**

- Đọc ba tập dữ liệu: `train.csv`, `val.csv`, `test.csv`
- Sử dụng `LabelEncoder` để chuyển đổi nhãn category thành dạng số (0, 1, 2, ...)

### 2. **Pipeline 1: TF-IDF + Logistic Regression**

- Chuyển đổi văn bản thành vector TF-IDF (max 5000 features)
- Huấn luyện mô hình Logistic Regression với `max_iter=1000`
- Đánh giá bằng loss (log_loss) và classification_report
- **Kết quả**: Bảo toàn tốt nhất so với các mô hình khác (Test Loss: 1.0502)

### 3. **Pipeline 2: Word2Vec + Dense Layer**

- Huấn luyện Word2Vec model với `vector_size=100`, `window=5`, `min_count=1`
- Chuyển đổi mỗi câu thành vector trung bình của các từ
- Xây dựng neural network: `Input(100) → Dense(128, relu) → Dropout(0.5) → Dense(64, softmax)`
- Huấn luyện 500 epochs với `batch_size=16`
- **Kết quả**: Kém hơn TF-IDF (Test Loss: 1.9769)

### 4. **Pipeline 3: Embedding (Pre-trained) + LSTM**

- Tạo tokenizer với `vocab_size=10000`
- Xây dựng embedding matrix từ Word2Vec weights (những từ có trong model được copy vào)
- Xây dựng LSTM model với embedding pre-trained, đóng băng embedding (`trainable=False`)
- **Quan trọng**: Đặt `recurrent_dropout=0.0` để kích hoạt CuDNN (GPU acceleration)
- Sử dụng EarlyStopping để dừng sớm khi validation loss không cải thiện
- **Kết quả**: Kết quả cho kém hơn nhiều (Test Loss: 2.7085)

### 5. **Pipeline 4: Embedding (Scratch) + LSTM**

- Tương tự Pipeline 3 nhưng embedding không pre-trained mà được học từ đầu
- Đặt `recurrent_dropout=0.0` để tối ưu GPU
- **Kết quả**: Tệ nhất (Test Loss: 4.1236)

### 6. **Đánh giá và trực quan hóa**

- Tính confusion matrix, chuẩn hóa theo hàng
- Vẽ heatmap để so sánh dự báo vs thực tế
- Thử nghiệm trên ba câu test mẫu

---

## Hướng dẫn chạy mã

### Chuẩn bị môi trường

Chương trình chạy hoàn toàn trên google colab với GPU được sử dụng là T4 GPU miễn phí.

---

## Khó khăn gặp phải và cách giải quyết

### Thách thức 1: **LSTM chậm trên GPU (hoặc không dùng GPU)**

**Vấn đề**: Khi chạy LSTM, huấn luyện rất chậm. Ngay cả khi bạn có GPU, TensorFlow vẫn dùng CPU.

**Nguyên nhân**:

- Nếu `recurrent_dropout != 0.0`, TensorFlow **không thể** sử dụng kernel CuDNN tối ưu trên GPU
- Thay vào đó, TensorFlow dùng fallback implementation chậm hơn

**Giải pháp**:

```python
# ❌ SAI — sẽ chạy chậm trên GPU hoặc CPU
LSTM(128, dropout=0.2, recurrent_dropout=0.1)

# ✅ ĐÚNG — cho phép sử dụng CuDNN kernel
LSTM(128, dropout=0.2, recurrent_dropout=0.0)
```

**Kết quả**: Giảm thời gian huấn luyện đi hàng chục lần

---

### Thách thức 2: **Mô hình LSTM không học tốt**

**Vấn đề**: Kết quả của LSTM (Pipeline 3, 4) kém hơn TF-IDF + LR. F1-score thấp, nhiều class không được dự báo.

**Nguyên nhân**:

- Dữ liệu có thể không cân bằng hoặc quá nhỏ
- Tham số mô hình không tối ưu (embedding_dim, hidden_size, ...)
- Overfitting do mô hình quá phức tạp
- Kích thước dữ liệu nhỏ không đủ để huấn luyện LSTM hiệu quả

---

## Kết quả và phân tích / Results and Analysis

### Bảng so sánh: Độ đo F1 và Loss

| Pipeline                       | iot_hue_lighton | iot_wemo_on | music_settings | audio_volume_down | datetime_convert | email_addcontact | iot_wemo_off | Test Loss |
| ------------------------------ | --------------- | ----------- | -------------- | ----------------- | ---------------- | ---------------- | ------------ | --------- |
| TF-IDF + Logistic Regression   | 0.67            | 0.88        | 0.73           | 0.86              | 0.71             | 0.82             | 0.84         | 1.0502    |
| Word2Vec + Dense         | 0               | 0.60        | 0              | 0.33              | 0.50             | 0.76             | 0.61         | 1.9769    |
| Embedding (Pre-trained) + LSTM | 0               | 0.62        | 0              | 0.22              | 0                | 0                | 0.44         | 2.7085    |
| Embedding (Scratch) + LSTM     | 0               | 0           | 0              | 0                 | 0                | 0                | 0            | 4.1236    |

### Test thử trên dữ liệu thực
Các câu
- can you remind me to not call my mom (1)
- is it going to be sunny or rainy tomorrow (2)
- find a flight from new york to london but not through paris (3)

| Câu | TF-IDF + Logistic Regression | Word2Vec + Dense | Embedding (Pre-trained) + LSTM | Embedding (Scratch) + LSTM | Nhãn đúng |
|--|--|--|--|--|--|
| (1) | calendar_set | social_post | alarm_query | general_quirky | reminder_create |
| (2) | weather_query | weather_query | alarm_query | general_quirky | weather_query |
| (3) | general_negate | social_post | alarm_set | general_quirky | flight_search |

### Nhận xét chi tiết dựa trên bảng

#### 1. **TF-IDF + Logistic Regression — Tối ưu nhất ✅**

- **Test Loss thấp nhất: 1.0502** — cho thấy mô hình này phù hợp nhất với bài toán
- **F1-score cân bằng** trên tất cả các class:
  - Cao nhất: `iot_wemo_on` (0.88), `audio_volume_down` (0.86)
  - Thấp nhất: `iot_hue_lighton` (0.67), `datetime_convert` (0.71)
  - Trung bình: ~0.75 (rất tốt)
- **Kết luận**: Mô hình đơn giản, nhanh, ổn định. Khuyên dùng cho bài toán này.

#### 2. **Word2Vec + Dense — Kém hơn nhưng chấp nhận được**

- **Test Loss: 1.9769** — gấp ~1.9 lần so với TF-IDF
- **Vấn đề**: Không dự báo được 3 class: `iot_hue_lighton`, `music_settings`, `iot_wemo_off` (F1=0)
- **Điểm tốt**: Dự báo tốt cho `email_addcontact` (0.76), `iot_wemo_on` (0.60)
- **Nguyên nhân**: Phương pháp trung bình từ embedding không giữ được thông tin tầm quan trọng của từng từ
- **Kết luận**: Sử dụng embedding trung bình không đủ hiệu quả.

#### 3. **Embedding (Pre-trained) + LSTM — Hiệu suất thấp 😞**

- **Test Loss cao: 2.7085** — gấp 2.7 lần so với TF-IDF
- **Vấn đề nghiêm trọng**: Không dự báo được 4 class hoàn toàn:
  - `iot_hue_lighton`, `music_settings`, `datetime_convert`, `email_addcontact` (F1=0)
  - Chỉ dự báo được 3 class: `iot_wemo_on` (0.62), `iot_wemo_off` (0.44), `audio_volume_down` (0.22)
- **Nguyên nhân**:
  - Dữ liệu nhỏ
  - Embedding pre-trained có thể không thích hợp với dataset này
  - Mô hình quá phức tạp so với kích thước dữ liệu
- **Kết luận**: Pre-trained embedding từ Word2Vec không cải thiện hiệu suất; có thể dữ liệu quá nhỏ hoặc LSTM cần tuning tốt hơn.

#### 4. **Embedding (Scratch) + LSTM — Tệ nhất ❌**

- **Test Loss rất cao: 4.1236** — gấp 4 lần so với TF-IDF
- **Vấn đề cực đoan**: Không dự báo được BẤT KỲ class nào (tất cả F1=0)
- **Nguyên nhân**:
  - Embedding được học từ đầu + LSTM yêu cầu rất nhiều dữ liệu
  - Dữ liệu của dự án quá nhỏ để huấn luyện cả embedding + LSTM từ điểm khởi đầu
- **Kết luận**: Phương pháp này **KHÔNG phù hợp** với dữ liệu nhỏ. Chỉ nên dùng khi có dữ liệu lớn hơn.

---

### Tóm tắt so sánh

| Tiêu chí                | TF-IDF + LR | Word2Vec + Dense | LSTM Pre-trained           | LSTM Scratch         |
| ----------------------- | ----------- | ---------------- | -------------------------- | -------------------- |
| **Hiệu suất**           | ⭐⭐⭐⭐⭐  | ⭐⭐             | ⭐                         | ❌                   |
| **Tốc độ huấn luyện**   | Rất nhanh   | Nhanh            | Chậm                       | Chậm                 |
| **Yêu cầu dữ liệu**     | Ít          | Ít-Trung bình    | Trung bình                 | Nhiều                |
| **Khả năng mở rộng**    | Trung bình  | Tốt              | Tốt                        | Tốt                  |
| **Độ phức tạp mô hình** | Thấp        | Trung bình       | Cao                        | Cao                  |
| **Khuyến nghị**         | ✅ Sử dụng  | ⚠️ Tuning        | ⚠️ Tuning cần thêm dữ liệu | ❌ Không khuyến nghị |
