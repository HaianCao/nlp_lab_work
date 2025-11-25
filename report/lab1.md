# Lab 1

## Cấu trúc dự án

```
data/
├── en_ewt-ud-dev.txt
├── en_ewt-ud-test.txt
└── en_ewt-ud-train.txt

src/
├── config/
│   └── io.py                  # Định nghĩa các đường dẫn cơ bản của dự án
├── core/
│   └── interfaces.py          # Interface định nghĩa các class cơ bản
├── preprocessing/
│   ├── simple_tokenizer.py    # Simple Tokenizer implementation
│   └── regex_tokenizer.py     # Regex-based Tokenizer implementation
└── representations/
    └── count_vectorizer.py    # Count Vectorizer implementation

test/
├── lab2_test.py              # Test file cho Vectorizer
└── main.py                   # Test file cho Tokenizers

README.md                    # File tài liệu này
```

## Các bước triển khai

### 1. Simple Tokenizer

- **File**: `src/preprocessing/simple_tokenizer.py`
- **Mô tả**: Tokenizer đơn giản sử dụng phương pháp split cơ bản
- **Thuật toán**:
  1. Chuyển văn bản về chữ thường
  2. Tách theo khoảng trắng và dấu câu
- **Ưu điểm**: Đơn giản, nhanh, dễ hiểu
- **Nhược điểm**: Không xử lý tốt dấu câu và các trường hợp đặc biệt

### 2. Regex Tokenizer

- **File**: `src/preprocessing/regex_tokenizer.py`
- **Mô tả**: Tokenizer sử dụng regular expression để tách token chính xác hơn
- **Pattern**: `\w+|[^\w\s]`
  - `\w+`: Tìm một hoặc nhiều ký tự từ (chữ cái, số, gạch dưới)
  - `|`: Hoặc
  - `[^\w\s]`: Tìm ký tự không phải từ và không phải khoảng trắng (dấu câu)
- **Thuật toán**:
  1. Chuyển văn bản về chữ thường
  2. Áp dụng regex pattern để tách từ và dấu câu riêng biệt
  3. Trả về danh sách tokens đã được tách
- **Ưu điểm**: Xử lý tốt dấu câu, số, ký tự đặc biệt so với Simple Tokenizer
- **Nhược điểm**: Phức tạp hơn, cần hiểu regex

### 3. Count Vectorizer

- **File**: `src/representations/count_vectorizer.py`
- **Mô tả**: Chuyển đổi văn bản thành một vector dựa trên Bag of Words
- **Thuật toán**:
  1. **Fit**:
     - Tokenize văn bản
     - Tạo vocabulary (từ điển) từ tất cả unique tokens
  2. **Transform phase**:
     - Tokenize từng document
     - Đếm tần suất xuất hiện của mỗi token
     - Tạo vector đếm theo thứ tự vocabulary

## Cách chạy code

### Chạy test tokenizer

```bash
cd test
python main.py
```

### Chạy test vectorizer

```bash
cd test
python lab2_test.py
```

## Kết quả thử nghiệm

### Test sentences được sử dụng:

1. "Hello, world! This is a test."
2. "NLP is fascinating... isn't it?"
3. "Let's see how it handles 123 numbers and punctuation!"

### Kết quả chi tiết:

#### Tokenizer

```bash
Sentence 1: "Hello, world! This is a test."
==================================================
SimpleTokenizer output: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer output:  ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
```

```bash
Sentence 2: "NLP is fascinating... isn't it?"
==================================================
SimpleTokenizer output: ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
RegexTokenizer output:  ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']
```

```bash
Sentence 3: "Let's see how it handles 123 numbers and punctuation!"
==================================================
SimpleTokenizer output: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer output:  ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
```

#### Count Vectorizer

```python
Documents: [
    "Hello, world! This is a test.",
    "NLP is fascinating... isn't it?",
    "Let's see how it handles 123 numbers and punctuation!"
]
==============================
Vocabulary: {'hello': 0, ',': 1, 'world': 2,
             '!': 3, 'this': 4, 'is': 5,
             'a': 6, 'test': 7, '.': 8,
             'nlp': 9, 'fascinating': 10, 'isn': 11,
             "'": 12, 't': 13, 'it': 14,
             '?': 15, 'let': 16, 's': 17,
             'see': 18, 'how': 19, 'handles': 20,
             '123': 21, 'numbers': 22, 'and': 23,
             'punctuation': 24}

Document-Term Matrix:
[[1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 3 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1]]
```

### Giải thích kết quả:

#### Phân tích so sánh Tokenizers:

**1. Xử lý từ viết tắt:**

- **Simple Tokenizer**: Giữ nguyên "isn't", "let's" như một token
- **Regex Tokenizer**: Tách thành ["isn", "'", "t"] và ["let", "'", "s"]
- **Nhận xét**: Regex tokenizer tách chi tiết hơn do sử dụng cấu trúc regex để tách trong khi Simple tokenizer phải phụ thuộc vào tập các ký tự đặc biệt được định nghĩa trước, nếu nằm ngoài tập này thì không xử lý được

**2. Xử lý dấu câu:**

- **Simple Tokenizer**: Tách dấu ba chấm "..." thành ['.', '.', '.']
- **Regex Tokenizer**: Cũng tách thành ['.', '.', '.']
- **Nhận xét**: Cả hai đều xử lý dấu câu riêng lẻ giống nhau mặc dù việc xử lý như vậy đang là không tốt

**3. Xử lý số:**

- Cả hai tokenizer đều giữ nguyên số "123" như một token
- **Kết luận**: Hoạt động tốt với số nguyên

#### Phân tích Count Vectorizer:

**Vocabulary Construction:**

- Tổng cộng: 25 unique tokens từ 3 câu
- Bao gồm: từ thường (hello, world), số (123), dấu câu (!, ?, .)

## Khó khăn gặp phải và cách giải quyết

### 1. Đường dẫn cứng (Hard-coded Paths)

- **Vấn đề**: Sử dụng đường dẫn tuyệt đối hoặc relative paths không linh hoạt

  ```python
  # ❌ Cách cũ - đường dẫn cứng
  file_path = "C:/Users/Admin/Desktop/data/file.txt"  # Windows only
  file_path = "../data/file.txt"  # Phụ thuộc vào working directory
  ```

- **Hậu quả**:

  - Code không chạy được trên máy khác
  - Lỗi khi thay đổi cấu trúc thư mục
  - Khó bảo trì và scale

- **Giải pháp**: Sử dụng `pathlib` để tạo đường dẫn linh hoạt

  ```python
  # ✅ Cách mới - sử dụng pathlib
  from pathlib import Path

  # Tự động detect project root
  PROJECT_ROOT = Path(__file__).parent.parent
  DATA_DIR = PROJECT_ROOT / "data"

  def get_data_path(filename):
      return DATA_DIR / filename

  def get_test_data_path():
      return DATA_DIR / "en_ewt-ud-test.txt"
  ```
