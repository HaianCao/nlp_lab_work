import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.config.io import *
from src.core.dataset_loaders import load_raw_text_data

def test_vectorizer_simple_data(vectorizer, documents: list[str]):
    print("=== Vectorizer Evaluation ===\n")
    vocab = vectorizer.fit(documents)
    matrix = vectorizer.transform(documents)
    print("Vocabulary:")
    print(vocab)
    print("Document-Term Matrix:")
    print(matrix)

def test_vectorizer_file_data(vectorizer, file_path: str):
    text = load_raw_text_data(file_path)[:100]
    test_vectorizer_simple_data(vectorizer, [text])

def main():
    # Test sentences
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]
    
    # Test vectorizer
    regex_tokenizer = RegexTokenizer()
    count_vectorizer = CountVectorizer(tokenizer=regex_tokenizer)
    test_vectorizer_simple_data(count_vectorizer, sentences)
    test_vectorizer_file_data(count_vectorizer, get_test_data_path())

if __name__ == "__main__":
    main()