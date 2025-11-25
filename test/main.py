import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.interfaces import Tokenizer
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.config.io import *
from src.core.dataset_loaders import load_raw_text_data

def test_tokenizers_simple_data(tokenizers: list[Tokenizer], sentences: list[str]):
    print("=== Tokenizer Evaluation ===\n")
    for i, sentence in enumerate(sentences, 1):
        print(f"Sentence {i}: {sentence}")
        print("=" * 50)
        for tokenizer in tokenizers:
            tokens = tokenizer.tokenize(sentence)
            print(f"{tokenizer.__class__.__name__} output: {tokens}")
        print()

def test_tokenizers_file_data(tokenizers: list[Tokenizer], file_path: str):
    text = load_raw_text_data(file_path)[:100]
    test_tokenizers_simple_data(tokenizers, [text])

def main():
    # Test sentences
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]
    
    # # Test tokenizers
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()
    test_tokenizers_simple_data([simple_tokenizer, regex_tokenizer], sentences)
    test_tokenizers_file_data([simple_tokenizer, regex_tokenizer], get_test_data_path())

if __name__ == "__main__":
    main()