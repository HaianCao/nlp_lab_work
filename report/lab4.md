# Lab4 - Text Classification và Sentiment Analysis

## 🏗️ Cấu trúc dự án

```
lab_work/
├── src/                              # Source code chính
│   ├── models/                       # Các mô hình phân loại
│   │   ├── text_classifier.py        # LogisticRegression classifier
│   │   ├── naive_bayes.py           # Naive Bayes model
│   │   ├── neural_network.py        # Neural Network model
│   │   ├── gbts.py                  # Gradient Boosting model
│   │   └── model_interface.py       # Interface chung cho models
│   ├── preprocessing/               # Tiền xử lý văn bản
│   │   ├── noise_filtering.py       # Loại bỏ noise (URLs, HTML tags)
│   │   ├── vocab_reduction.py       # Stemming, lemmatization, stopwords
│   │   └── preprocessor_interface.py # Interface cho preprocessing
│   ├── tokenizer/                   # Tokenization
│   │   ├── regex_tokenizer.py       # Regex-based tokenizer
│   │   └── tokenize_interface.py    # Interface cho tokenizers
│   ├── vectorize/                   # Vector hóa văn bản
│   │   ├── tf_idf.py               # TF-IDF vectorizer
│   │   ├── glove.py                # GloVe embeddings
│   │   └── vectorize_interface.py   # Interface cho vectorizers
│   └── __init__.py                 # Package exports
├── test/                           # Test files và thử nghiệm
│   ├── lab5_test.py               # Test cơ bản với scikit-learn
│   ├── lab5_spark_sentiment_analysis.py # PySpark pipeline
│   └── lab5_model_improvement.py   # So sánh các phương pháp
└── sentiments.csv                  # Dataset
```

## 🚀 Các bước thực hiện

### 1. Xây dựng kiến trúc cơ bản
- **Bước 1**: Tạo TextClassifier với LogisticRegression làm baseline
- **Bước 2**: Implement interfaces chung cho models, preprocessing, tokenizers, vectorizers
- **Bước 3**: Xây dựng pipeline cơ bản với TF-IDF và scikit-learn

### 2. Mở rộng với PySpark
- **Bước 4**: Triển khai pipeline phân tích cảm xúc với PySpark MLlib
- **Bước 5**: Sử dụng Tokenizer, StopWordsRemover, HashingTF, IDF của Spark
- **Bước 6**: Đánh giá trên dataset lớn (5792 mẫu)

### 3. Phát triển preprocessing nâng cao
- **Bước 7**: Implement NoiseFiltering (loại bỏ URLs, HTML tags, lowercase)
- **Bước 8**: Xây dựng VocabReduction với NLTK (stemming, lemmatization, stopwords)
- **Bước 9**: Tạo RegexTokenizer linh hoạt

### 4. Tích hợp Word Embeddings
- **Bước 10**: Implement GloVeVectorizer với pre-trained model glove-wiki-gigaword-50
- **Bước 11**: Xây dựng vectorization pipeline cho embeddings
- **Bước 12**: Tích hợp với gensim library

### 5. Mở rộng models
- **Bước 13**: Thêm NaiveBayesModel với GaussianNB tự động
- **Bước 14**: Implement NeuralNetworkModel với MLPClassifier
- **Bước 15**: Xây dựng GBTSModel với GradientBoostingClassifier

### 6. Thử nghiệm và so sánh
- **Bước 16**: Tạo lab5_model_improvement.py để test tất cả combinations
- **Bước 17**: So sánh hiệu suất các phương pháp khác nhau

## 📋 Hướng dẫn chạy chương trình

### Tải các modules cần thiết
```bash
pip install -r requirements.txt
```
### Chạy modules riêng lẻ (test internal functionality)
```bash
# Test các vectorizers
python -m src.vectorize.glove
python -m src.vectorize.tf_idf

# Test các models  
python -m src.models.naive_bayes
python -m src.models.neural_network
python -m src.models.gbts

# Test preprocessing
python -m src.preprocessing.vocab_reduction
python -m src.preprocessing.noise_filtering
```

### Chạy test files
```bash
# Test cơ bản với LogisticRegression
python test/lab5_test.py

# Test PySpark pipeline
python test/lab5_spark_sentiment_analysis.py

# So sánh các phương pháp khác nhau
python test/lab5_model_improvement.py
```

## 📊 Kết quả thử nghiệm

### 1. Baseline - LogisticRegression (lab5_test.py)
- **Dataset**: 5792 samples (test set: ~1159 samples)  
- **Vectorizer**: TfidfVectorizer (scikit-learn)
- **Kết quả**:
  - Accuracy: **71.53%**
  - Precision: **73.16%** 
  - Recall: **86.75%**
  - F1-Score: **79.37%**

### 2. PySpark Pipeline (lab5_spark_sentiment_analysis.py)
- **Dataset**: 5792 samples (distributed processing)
- **Pipeline**: Tokenizer → StopWordsRemover → HashingTF → IDF → LogisticRegression
- **Kết quả**:
  - Accuracy: **73.22%**
  - Precision: **72.96%**
  - Recall: **73.22%**
  - F1-Score: **73.06%**

### 3. So sánh các phương pháp (lab5_model_improvement.py)

| Preprocessing | Tokenizer | Vectorizer | Model | Accuracy | Precision | Recall | F1-Score |
|---------------|-----------|------------|-------|----------|-----------|--------|----------|
| NoiseFiltering | RegexTokenizer | TFIDFVectorizer | NaiveBayes | **60.22%** | **77.26%** | **52.46%** | **62.49%** |
| NoiseFiltering | RegexTokenizer | **GloVeVectorizer** | NaiveBayes | **64.11%** | **67.87%** | **81.97%** | **74.26%** |
| VocabReduction | RegexTokenizer | TFIDFVectorizer | NaiveBayes | **57.55%** | **80.00%** | **43.72%** | **56.54%** |
| VocabReduction | RegexTokenizer | **GloVeVectorizer** | NaiveBayes | **61.35%** | **68.73%** | **71.17%** | **69.93%** |

## Phân tích kết quả

### Hiệu suất tổng thể
1. **PySpark Pipeline** đạt kết quả tốt nhất với **73.22% accuracy**
2. **Baseline LogisticRegression** cho kết quả ổn định với **71.53% accuracy**
3. **GloVe embeddings** thường cho kết quả tốt hơn TF-IDF với NaiveBayes

### So sánh preprocessing methods
- **NoiseFiltering**: Đơn giản nhưng hiệu quả, giữ lại nhiều thông tin
- **VocabReduction**: Giảm chiều dữ liệu nhưng có thể mất thông tin quan trọng

### So sánh vectorization methods  
- **TF-IDF**: Phù hợp với NB, cho precision cao nhưng recall thấp
- **GloVe**: Tốt hơn cho các tác vụ semantic, cân bằng precision-recall tốt hơn

### Ưu điểm GloVe embeddings
- Capture được semantic similarity giữa các từ
- Pre-trained trên large corpus (Wikipedia + Gigaword)
- Hoạt động tốt với GaussianNB cho embedding vectors

## ⚠️ Thách thức và giới hạn

### 1. Vấn đề hiệu suất
- **Neural Network**: Chạy rất chậm trên CPU, cần GPU để tăng tốc
- **Gradient Boosting**: Memory intensive, cần nhiều RAM cho large datasets

### 2. Interface inconsistency
- Các models có API khác nhau (fit/predict vs train/classify)
- Preprocessing methods có input/output formats khác nhau
- Vectorizers có method names không thống nhất

### 3. Giải pháp đã áp dụng
- **Tạo interfaces chung**: ModelInterface, VectorizeInterface, PreprocessorInterface
- **Standardize API**: Tất cả models implement fit(), predict(), evaluate()
- **Error handling**: Tự động detect và switch algorithms phù hợp
- **Modular design**: Dễ dàng thay đổi components trong pipeline

## 📚 Tài liệu tham khảo

1. **Scikit-learn Documentation**: Machine Learning algorithms và preprocessing
2. **PySpark MLlib Guide**: Distributed machine learning
3. **Gensim Documentation**: Word embeddings và topic modeling
4. **NLTK Documentation**: Natural language processing tools
5. **GloVe: Global Vectors for Word Representation** (Pennington et al., 2014)