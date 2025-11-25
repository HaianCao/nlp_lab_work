# Lab Work

## Cấu trúc dự án

```
lab_work/
├── src/                    # Source code chính
│   ├── __init__.py        # Package initialization
│   ├── config/            # Cấu hình
│   │   ├── io.py          # I/O configuration (lab1 + lab2)
│   │   └── output.py      # Output configuration (lab2)
│   ├── core/              # Core functionality
│   │   ├── dataset_loaders.py           # Data loading (lab1)
│   │   ├── interfaces.py                # Interfaces (lab1)
│   │   ├── classification_interfaces.py # ML Interfaces (lab4)
│   │   ├── pipeline.py                  # Pipeline core (lab2)
│   │   ├── pipeline_process.py          # Pipeline processing (lab2)
│   │   └── find_similar_documents.py    # Document similarity (lab2)
│   ├── models/            # Machine Learning Models (lab4)
│   │   ├── __init__.py           # Package initialization
│   │   ├── model_interface.py    # Model interface
│   │   ├── text_classifier.py    # Logistic Regression classifier
│   │   ├── naive_bayes.py        # Naive Bayes model
│   │   ├── neural_network.py     # Neural Network model
│   │   └── gbts.py              # Gradient Boosting model
│   ├── preprocessing/     # Text preprocessing
│   │   ├── __init__.py              # Package initialization
│   │   ├── simple_tokenizer.py      # Simple tokenization (lab1)
│   │   ├── regex_tokenizer.py       # Regex tokenization (lab1)
│   │   ├── embedding_tokenizer.py   # Tokenizer for embeddings (lab3)
│   │   ├── sentiment_tokenizer.py   # Sentiment-aware tokenizer (lab4)
│   │   ├── noise_filtering.py       # Noise removal (lab4)
│   │   └── vocab_reduction.py       # Vocabulary reduction (lab4)
│   ├── representations/   # Text representations
│   │   ├── __init__.py           # Package initialization
│   │   ├── count_vectorizer.py   # Count vectorization (lab1)
│   │   └── word_embedder.py      # Word embeddings (lab3)
│   ├── vectorize/         # Advanced vectorization (lab4)
│   │   ├── __init__.py           # Package initialization
│   │   ├── tf_idf_vectorizer.py  # Custom TF-IDF vectorizer
│   │   └── glove_vectorizer.py   # GloVe embeddings vectorizer
│   ├── interface/         # User interfaces
│   │   └── pipeline_interface.py  # Pipeline interface (lab2)
│   ├── metrics/           # Evaluation metrics
│   │   └── similarity.py          # Similarity metrics (lab2)
│   └── utils/             # Utilities
│       ├── pipeline_items.py      # Pipeline items (lab2)
│       └── timing.py              # Timing utilities (lab2)
├── test/                  # Test files và demos
│   ├── main.py                    # Main test file (lab1)
│   ├── lab2_main.py              # Lab2 main test
│   ├── lab2_test.py              # Lab2 pipeline tests
│   ├── lab4_test.py              # Lab3/4 embedding tests
│   ├── lab5_test.py              # Lab4/5 classification tests
│   ├── lab4_embedding_training_demo.py    # Embedding training demo
│   ├── lab4_spark_word2vec_demo.py       # Spark Word2Vec demo
│   ├── lab5_spark_sentiment_analysis.py  # Spark sentiment analysis
│   ├── lab5_model_improvement.py         # Model comparison & optimization
│   └── split_output_simple.py            # Output splitting utility
├── report/                # Báo cáo và documentation
│   ├── lab1.md           # Lab1 report (Tokenization & Vectorization)
│   ├── lab2.md           # Lab2 report (Spark NLP Pipeline)
│   ├── lab3.md           # Lab3 report (Word Embeddings)
│   ├── lab4.md           # Lab4 report (Text Classification)
│   ├── lab5_part2.md     # Lab5 report (Intent Classification)
│   ├── lab5_part4.md     # Lab5 report (NER with Bi-LSTM)
│   └── prompts_lab2.txt  # Lab2 prompts và queries
├── data/                 # Data files và datasets
│   ├── en_ewt-ud-train.txt       # Universal Dependencies train set (lab1)
│   ├── en_ewt-ud-dev.txt         # Universal Dependencies dev set (lab1)
│   ├── en_ewt-ud-test.txt        # Universal Dependencies test set (lab1)
│   └── results/                 # Output và kết quả (tracked by git)
│       ├── lab17_pipeline_output.txt    # Lab2 pipeline results
│       ├── similarity_5_results.txt     # Document similarity results
│       ├── timing_log.txt              # Performance timing logs
│       └── spark_word2vec_model/       # Spark Word2Vec model files
│           ├── data/                   # Model data files
│           │   └── _SUCCESS            # Spark success marker
│           └── metadata/               # Model metadata
│               ├── _SUCCESS            # Spark success marker
│               └── part-00000          # Model metadata partition
├── notebook/             # Jupyter notebooks và báo cáo PDF
│   ├── lab5.ipynb                    # Lab5 Intent Classification notebook
│   ├── lab5-part4.ipynb              # Lab5 NER với Bi-LSTM notebook
│   ├── lab6.ipynb                    # Lab6 experiments notebook
│   ├── 23001818_Cao_Hai_An_lab_3.pdf       # Lab3 report PDF
│   ├── 23001818_Cao_Hai_An_lab_5.pdf       # Lab5 report PDF
│   ├── 23001818_Cao_Hai_An_lab5_part4.pdf  # Lab5 Part4 NER report PDF
│   └── 23001818_Cao_Hai_An_lab_6.pdf       # Lab6 report PDF
├── scripts/              # Utility scripts cho setup
│   ├── setup_glove.sh    # GloVe setup script (Linux/Mac)
│   ├── setup_glove.bat   # GloVe setup script (Windows batch)
│   ├── setup_glove.ps1   # GloVe setup script (PowerShell)
│   ├── cleanup_glove.sh  # GloVe cleanup script (Linux/Mac)
│   ├── cleanup_glove.bat # GloVe cleanup script (Windows batch)
│   └── cleanup_glove.ps1 # GloVe cleanup script (PowerShell)
├── requirements.txt      # Dependencies
└── README.md            # Project documentation
```

## Các lab đã tích hợp

### Lab 1: Tokenization và Vectorization

- **Mô tả**: Xử lý văn bản cơ bản với tokenization và count vectorization
- **Modules**: `preprocessing/simple_tokenizer.py`, `preprocessing/regex_tokenizer.py`, `representations/count_vectorizer.py`
- **Test**: `test/lab1_test.py`
- **Trạng thái**: ✅ Hoàn thành và đã test

### Lab 2: Spark NLP Pipeline

- **Mô tả**: Pipeline xử lý văn bản với Spark, tìm kiếm tương tự văn bản
- **Modules**: `core/pipeline*.py`, `interface/`, `metrics/similarity.py`, `utils/`
- **Test**: `test/lab2_test.py`
- **Trạng thái**: ✅ Hoàn thành và đã test (30K docs trong 167s)

### Lab 3: Word Embeddings

- **Mô tả**: Word embeddings với GloVe và Word2Vec training
- **Modules**: `preprocessing/embedding_tokenizer.py`, `representations/word_embedder.py`
- **Test**: `test/lab4_test.py`, `test/lab4_embedding_training_demo.py`, `test/lab4_spark_word2vec_demo.py`
- **Trạng thái**: ✅ Hoàn thành tích hợp

### Lab 4: Text Classification & Sentiment Analysis

- **Mô tả**: Phân loại văn bản với nhiều thuật toán ML và preprocessing nâng cao
- **Tính năng**:
  - 4 thuật toán ML: Logistic Regression, Naive Bayes, Neural Network, Gradient Boosting
  - Advanced preprocessing: Noise filtering, vocabulary reduction, sentiment tokenization
  - Multiple vectorization methods: Custom TF-IDF, GloVe embeddings
  - Spark MLlib integration cho big data processing
- **Modules**: `models/`, `preprocessing/noise_filtering.py`, `preprocessing/vocab_reduction.py`, `vectorize/`
- **Test**: `test/lab5_test.py`, `test/lab5_spark_sentiment_analysis.py`, `test/lab5_model_improvement.py`
- **Dataset**: sentiments.csv (5792 samples) hoặc synthetic data
- **Kết quả**: Accuracy 71-73%, F1-Score 73-79% tùy theo pipeline
- **Trạng thái**: ✅ Hoàn thành tích hợp

### Lab 5: Intent Classification & Named Entity Recognition với Deep Learning

- **Mô tả**: Ứng dụng RNN/LSTM cho các bài toán NLP phức tạp với so sánh đa phương pháp
- **Part 1-3: Intent Classification**:
  - **4 phương pháp**: TF-IDF + Logistic Regression, Word2Vec + Dense Layer, Embedding (Pre-trained) + LSTM, Embedding (Scratch) + LSTM
  - **Dataset**: HWU64 intent dataset với train/val/test splits
  - **Kết quả so sánh**:
    - TF-IDF + LR: **Best performance** (Test Loss: 1.0502, F1: ~0.75)
    - Word2Vec + Dense: Trung bình (Test Loss: 1.9769)
    - LSTM Pre-trained: Kém (Test Loss: 2.7085)
    - LSTM Scratch: Tệ nhất (Test Loss: 4.1236)
  - **Insights**: Mô hình đơn giản thường hiệu quả hơn với dữ liệu nhỏ
- **Part 4: Named Entity Recognition (NER)**:
  - **Architecture**: Bi-LSTM với embedding layer (không CRF)
  - **Dataset**: CoNLL-2003 NER dataset
  - **Kết quả**: F1-Score ~0.66 (LOC: 0.76, PER: 0.66, ORG: 0.64, MISC: 0.53)
  - **Kỹ thuật**: Custom vocabulary, padding sequences, Early Stopping

## Dependencies

Xem file `requirements.txt` cho danh sách đầy đủ các dependencies cần thiết.
