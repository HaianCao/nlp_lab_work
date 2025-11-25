"""
Lab4 Test Suite
This script contains comprehensive tests for the word embedding functionality.
"""

import sys
import os

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from preprocessing.embedding_tokenizer import EmbeddingTokenizer
from representations.word_embedder import WordEmbedder

def test_embedding_tokenizer():
    """Test the EmbeddingTokenizer class."""
    print("=== Testing EmbeddingTokenizer ===")
    
    tokenizer = EmbeddingTokenizer()
    
    # Test basic tokenization
    text = "Hello, world! This is a test."
    tokens = tokenizer.tokenize(text)
    expected = ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
    
    print(f"Input: {text}")
    print(f"Tokens: {tokens}")
    print(f"Expected: {expected}")
    print(f"Test passed: {tokens == expected}")
    
    # Test with numbers and special characters
    text2 = "Python 3.9 is great!"
    tokens2 = tokenizer.tokenize(text2)
    print(f"\nInput: {text2}")
    print(f"Tokens: {tokens2}")
    
    return tokens == expected

def test_word_embedder_training():
    """Test Word2Vec training functionality."""
    print("\n=== Testing WordEmbedder Training ===")
    
    tokenizer = EmbeddingTokenizer()
    embedder = WordEmbedder()
    
    # Create training data
    sentences = [
        "the cat sat on the mat",
        "the dog ran in the park", 
        "cats and dogs are pets",
        "pets need food and water",
        "water is essential for life"
    ]
    
    # Tokenize sentences
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    print(f"Training with {len(sentences)} sentences")
    
    # Train model
    embedder.train_word2vec(
        sentences=tokenized_sentences,
        vector_size=10,  # Small for testing
        window=2,
        min_count=1,
        workers=1,
        epochs=5
    )
    
    # Test vocabulary
    print(f"Vocabulary size: {len(embedder.vocab)}")
    print(f"Sample words in vocab: {list(embedder.vocab)[:5]}")
    
    # Test word embedding
    test_word = "cat"
    if test_word in embedder.vocab:
        embedding = embedder.get_embedding(test_word)
        print(f"Embedding for '{test_word}': shape={embedding.shape}")
        print(f"First 5 dimensions: {embedding[:5]}")
        
        # Test similar words
        similar_words = embedder.get_similar_words(test_word, top_n=3)
        print(f"Similar words to '{test_word}': {similar_words}")
        
        # Test word similarity
        if "dog" in embedder.vocab:
            similarity = embedder.word_similarity("cat", "dog")
            print(f"Similarity between 'cat' and 'dog': {similarity:.4f}")
    
    # Test document vector
    doc_words = ["the", "cat", "sat"]
    doc_vector = embedder.document_vector(doc_words)
    print(f"Document vector for {doc_words}: shape={doc_vector.shape}")
    
    return True

def test_model_save_load():
    """Test model saving and loading functionality."""
    print("\n=== Testing Model Save/Load ===")
    
    tokenizer = EmbeddingTokenizer()
    embedder = WordEmbedder()
    
    # Train a simple model
    sentences = [["hello", "world"], ["world", "peace"], ["peace", "love"]]
    embedder.train_word2vec(sentences, vector_size=5, epochs=3)
    
    # Save model
    model_path = "test_model.w2v"
    try:
        embedder.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        # Load model
        embedder2 = WordEmbedder()
        embedder2.load_model(model_path)
        print(f"Model loaded successfully")
        print(f"Loaded vocabulary size: {len(embedder2.vocab)}")
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        
        return True
    except Exception as e:
        print(f"Save/Load test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    embedder = WordEmbedder()
    
    # Test without loaded model
    try:
        embedder.get_embedding("test")
        print("ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Test empty document vector
    tokenizer = EmbeddingTokenizer()
    embedder.train_word2vec([["hello", "world"]], vector_size=5, epochs=1)
    
    # Document with no valid words
    doc_vector = embedder.document_vector(["nonexistent", "words"])
    print(f"Empty document vector shape: {doc_vector.shape}")
    print(f"All zeros: {np.allclose(doc_vector, 0)}")
    
    return True

def run_all_tests():
    """Run all tests."""
    print("🧪 Running Lab4 Test Suite")
    print("=" * 50)
    
    tests = [
        ("Tokenizer", test_embedding_tokenizer),
        ("Word2Vec Training", test_word_embedder_training),
        ("Model Save/Load", test_model_save_load),
        ("Edge Cases", test_edge_cases)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Please check the output above.")

if __name__ == "__main__":
    run_all_tests()