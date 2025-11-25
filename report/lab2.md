# Lab 2

## Tổng quan dự án

Dự án này xây dựng một pipeline xử lý văn bản hoàn chỉnh sử dụng Apache Spark, bao gồm:

- Tiền xử lý văn bản (tokenization, stop words removal)
- Vector hóa TF-IDF
- Tìm kiếm tài liệu tương tự sử dụng cosine similarity
- Hệ thống monitoring chi tiết với timing log

## Cấu trúc dự án

```
data/
├── c4-train.00000-of-01024-30K.json.gz  # Dữ liệu đầu vào (30K documents)
└── results/                              # Thư mục chứa kết quả
    ├── lab17_pipeline_output.txt         # Kết quả xử lý pipeline đầy đủ (Không được đẩy lên do giới hạn lưu trữ của Github)
    ├── lab17_pipeline_output_part1.txt   # File tách phần 1
    ├── lab17_pipeline_output_part2.txt   # File tách phần 2
    ├── similarity_5_results.txt          # Kết quả tìm kiếm tương tự top 5
    └── timing_log.txt                    # Log thời gian thực thi chi tiết

src/
├── config/
│   ├── io.py                     # Cấu hình pipeline và đường dẫn
│   └── output.py                 # Xử lý định dạng và lưu kết quả
├── core/
│   ├── pipeline.py               # Pipeline điều phối chính
│   ├── pipeline_process.py       # Xử lý các bước pipeline
│   └── find_similar_documents.py # Tìm kiếm tài liệu tương tự
├── interface/
│   └── pipeline_interface.py     # Interface cho các components
├── metrics/
│   └── similarity.py             # Tính toán độ tương tự (cosine similarity)
├── utils/
│   ├── timing.py                 # Hệ thống đo thời gian chi tiết
│   └── pipeline_items.py         # Các components xử lý pipeline
└── main.py                       # File chính để chạy chương trình


prompts.txt                     # File ghi chú prompts
requirements.txt                # Danh sách thư viện cần thiết
README.md                      # File tài liệu này
```

## Các thành phần chính

### 1. Pipeline Processing System

- **File**: `src/core/pipeline_process.py`
- **Mô tả**: Hệ thống xử lý pipeline với các bước:
  1. **RegexTokenizer**: Tách từ sử dụng regex pattern
  2. **StopWordsRemover**: Loại bỏ từ dừng tiếng Anh
  3. **HashingTF**: Chuyển đổi từ thành vector frequency
  4. **IDF**: Tính toán Inverse Document Frequency
  5. **Normalizer**: Chuẩn hóa vector (L2 norm)

### 2. Document Similarity System

- **File**: `src/core/find_similar_documents.py`
- **Mô tả**: Tìm kiếm tài liệu tương tự
- **Chức năng**:
  1. Chọn tài liệu query (ngẫu nhiên hoặc chỉ định - giới hạn RAM)
  2. Tính cosine similarity với tất cả tài liệu khác
  3. Trả về top K tài liệu tương tự nhất
  4. Lưu kết quả chi tiết với vector features

### 3. Metrics và Similarity

- **File**: `src/metrics/similarity.py`
- **Mô tả**: Tính toán độ tương tự cosine
- **Công thức**: `cosine_similarity = (A · B) / (||A|| * ||B||)`

### 4. Timing và Monitoring

- **File**: `src/utils/timing.py`
- **Mô tả**: Hệ thống đo thời gian chi tiết
- **Chức năng**:
  1. Đo thời gian từng stage và substage
  2. Tính toán tỷ lệ phần trăm thời gian
  3. Xuất báo cáo chi tiết và lưu log file

## Cách chạy chương trình

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Chạy chương trình chính

```bash
python -m src.main
```

## Cấu trúc dữ liệu

### Dữ liệu đầu vào

- **Định dạng**: JSON nén (gzip)
- **Kích thước**: 26.8 MB (nén)
- **Số lượng**: 30,000 tài liệu
- **Trường dữ liệu**:
  - `text`: Nội dung văn bản
  - `timestamp`: Thời gian tạo
  - `url`: Đường link nguồn

### Dữ liệu đầu ra

- **Định dạng**: Văn bản có cấu trúc
- **Kích thước**: 162.91 MB (toàn bộ 30,000 tài liệu)
- **Số dòng**: 375,098 dòng
- **Nội dung**:
  - Văn bản gốc (text)
  - Kích thước vector (20,000 chiều)
  - Chỉ số đặc trưng (vị trí HashingTF)
  - Trọng số TF-IDF tương ứng

### File được tách

- **Part 1**: `lab17_pipeline_output_part1.txt` (82.06 MB, records 1-15,000)
- **Part 2**: `lab17_pipeline_output_part2.txt` (80.85 MB, records 15,001-30,000)

## Quản lý cấu hình

### Các tham số chính

| Tham số                    | Giá trị          | Mô tả                            |
| -------------------------- | ---------------- | -------------------------------- |
| `NUM_FEATURES`             | 20,000           | Số chiều của vector              |
| `LIMIT_RECORDS`            | 30,000           | Số lượng tài liệu được lưu       |
| `INPUT_RAW`                | "text"           | Tên cột dữ liệu đầu vào          |
| `OUTPUT_TOKENIZER`         | "words"          | Tên cột kết quả tách từ          |
| `OUTPUT_STOPWORDS_REMOVED` | "filtered_words" | Tên cột sau khi loại từ dừng     |
| `OUTPUT_HASHING_TF`        | "raw_features"   | Tên cột kết quả HashingTF        |
| `OUTPUT_TFIDF`             | "features"       | Tên cột kết quả TF-IDF cuối cùng |

### Cấu hình Spark

```python
SparkSession.builder \
    .appName("TextProcessingPipeline") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()
```

### Tham số similarity search

| Tham số         | Giá trị | Mô tả                                    |
| --------------- | ------- | ---------------------------------------- |
| `TOP_K_SIMILAR` | 5       | Số lượng tài liệu tương tự trả về        |
| `SAMPLE_SIZE`   | 5,000   | Số tài liệu sample để tránh memory issue |

## Phân tích hiệu năng

### Thời gian thực thi mẫu (30,000 tài liệu)

| Giai đoạn                  | Thời gian  | Tỷ lệ | Mô tả                                 |
| -------------------------- | ---------- | ----- | ------------------------------------- |
| **Data loading**           | 13.0s      | 7.5%  | Tải và phân tích JSON                 |
| **RegexTokenizer**         | 2.5s       | 1.5%  | Tách từ sử dụng regex                 |
| **StopWordsRemover**       | 1.8s       | 1.0%  | Loại bỏ từ dừng                       |
| **HashingTF**              | 1.7s       | 1.0%  | Tính term frequency với hashing       |
| **IDF**                    | 29.1s      | 16.8% | Tính inverse document frequency       |
| **Normalizer**             | 1.3s       | 0.8%  | Chuẩn hóa vector L2                   |
| **Output saving**          | 45.0s      | 26.0% | Lưu kết quả chi tiết                  |
| **Find similar documents** | 78.2s      | 45.2% | Tìm kiếm tài liệu tương tự (sampling) |
| **Tổng cộng**              | **173.2s** | 100%  | Thời gian thực thi toàn bộ pipeline   |

### Nhận xét về hiệu năng

1. **Similarity search là bottleneck chính**: 45.2% tổng thời gian

   - Tính toán cosine similarity cho 5000 documents sample
   - Window operations cảnh báo performance degradation
   - Cần optimization cho large-scale similarity computation

2. **Output saving tốn thời gian**: 26.0% tổng thời gian

   - Ghi toàn bộ 30,000 tài liệu với vector features chi tiết
   - Định dạng và serialize vector TF-IDF

3. **IDF computation**: 16.8% tổng thời gian

   - Tính toán IDF trên toàn bộ corpus 30K documents
   - Cần scan toàn bộ dataset để tính document frequency

4. **Preprocessing rất hiệu quả**: ~4% tổng thời gian

   - Tokenization, stop words removal, hashing TF được tối ưu tốt
   - Spark MLlib transformers hoạt động rất nhanh

5. **Data loading ổn định**: 7.5% tổng thời gian
   - Đọc và parse JSON nén hiệu quả

## Cấu trúc dữ liệu

### Schema đầu vào

```
root
 |-- text: string (nullable = true)
 |-- timestamp: string (nullable = true)
 |-- url: string (nullable = true)
```

### Định dạng đầu ra

```
1. Beginners BBQ Class Taking Place in Missoula!
Do you want to get better at making delicious BBQ? You will have the opportunity...
Vector size: 20000
Indices (HashingTF positions): [264, 298, 673, 717, 829, 1271, 1466, 1499, ...]
Values (TF-IDF weights): [15.634, 2.694, 3.289, 3.112, 4.286, 6.358, ...]

2. Discussion in 'Mac OS X Lion (10.7)' started by axboi87, Jan 20, 2012.
...
```

## Kết quả và Output

### Thống kê tổng quan

- **Tổng số tài liệu được xử lý**: 30,000
- **Vector dimension**: 20,000 features
- **Thời gian xử lý trung bình**: ~173 giây
- **File outputs chính**:
  - `lab17_pipeline_output.txt`: Kết quả TF-IDF đầy đủ
  - `similarity_5_results.txt`: Top 5 similar documents với query info

### Document Similarity Results

Pipeline tự động:

1. Chọn ngẫu nhiên 1 document làm query
2. Sample 5,000 documents từ corpus để tính similarity
3. Tính cosine similarity giữa query và tất cả documents
4. Trả về top 5 documents tương tự nhất
5. Lưu kết quả chi tiết bao gồm:
   - Query document text và vector features
   - Similar documents với similarity scores
   - Vector indices và values chi tiết
   - Common features analysis

## Tính năng nổi bật

### 1. Complete Text Processing Pipeline

- **Full NLP workflow**: Từ raw text → cleaned tokens → TF-IDF vectors → similarity search
- **Robust error handling**: Spark job failures được xử lý gracefully
- **Memory optimization**: Sampling strategy để tránh memory issues với large datasets

### 2. Document Similarity Search

- **Cosine similarity computation**: Tính toán độ tương tự giữa documents
- **Top-K retrieval**: Tìm K documents tương tự nhất
- **Detailed analysis**: Vector features analysis và common indices tracking

### 3. Advanced Monitoring System

- **Comprehensive timing**: Đo thời gian chi tiết từng stage và substage
- **Performance analytics**: Tỷ lệ phần trăm thời gian, bottleneck identification
- **Detailed logging**: File logs chi tiết với timestamps

### 4. Production-Ready Architecture

- **Modular design**: Tách biệt rõ ràng các components
- **Configuration management**: Centralized config cho easy tuning
- **Error resilience**: Proper exception handling và cleanup

## Kết quả đạt được

### Thành công chính

1. **Xử lý thành công 30K documents**

   - Complete pipeline từ raw JSON → TF-IDF vectors → similarity results
   - Robust processing với proper error handling
   - Memory-efficient approach với sampling strategies

2. **High-quality similarity search**

   - Accurate cosine similarity computation
   - Meaningful similarity scores (0.1-0.3 range typical)
   - Detailed vector analysis với common features tracking

3. **Production-grade monitoring**
   - Comprehensive timing analysis
   - Clear performance bottleneck identification
   - Detailed logs cho debugging và optimization

### Điểm mạnh technical

1. **Spark MLlib integration**: Tận dụng optimized transformers
2. **Scalable architecture**: Có thể handle datasets lớn hơn
3. **Flexible configuration**: Dễ dàng adjust parameters
4. **Comprehensive output**: Chi tiết vector features cho analysis
