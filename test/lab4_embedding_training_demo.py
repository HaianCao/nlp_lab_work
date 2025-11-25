"""
Lab4 Embedding Training Demo
This script demonstrates word embedding training and usage.
"""

import sys
import os

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.embedding_tokenizer import EmbeddingTokenizer
from representations.word_embedder import WordEmbedder

def main():
    """
    Main function to demonstrate embedding training and usage.
    """
    print("=== Lab4 Embedding Training Demo ===")
    
    # Initialize tokenizer and embedder
    tokenizer = EmbeddingTokenizer()
    embedder = WordEmbedder()
    
    # Sample sentences for training
    sample_sentences = [
        "I love natural language processing",
        "Machine learning is fascinating",
        "Word embeddings capture semantic relationships",
        "Deep learning models are powerful",
        "Natural language understanding is important",
        "Text processing requires careful tokenization",
        "Semantic similarity can be measured using vectors",
        "Language models learn from large datasets"
    ]
    
    print(f"Training with {len(sample_sentences)} sentences...")
    
    # Tokenize sentences
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sample_sentences]
    
    print("Sample tokenized sentence:", tokenized_sentences[0])
    
    # Train Word2Vec model
    print("\n--- Training Word2Vec Model ---")
    embedder.train_word2vec(
        sentences=tokenized_sentences,
        vector_size=50,  # Small dimension for demo
        window=3,
        min_count=1,
        workers=1,
        epochs=10
    )
    
    # Test the trained model
    print("\n--- Testing Trained Model ---")
    test_word = "language"
    if test_word in embedder.vocab:
        embedding = embedder.get_embedding(test_word)
        print(f"Embedding for '{test_word}': {embedding[:5]}... (showing first 5 dimensions)")
        
        # Find similar words
        similar_words = embedder.get_similar_words(test_word, top_n=3)
        print(f"Words similar to '{test_word}': {similar_words}")
    else:
        print(f"'{test_word}' not found in vocabulary")
    
    # Test word similarity
    word1, word2 = "natural", "language"
    if word1 in embedder.vocab and word2 in embedder.vocab:
        similarity = embedder.word_similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
    
    # Test document vector
    print("\n--- Document Vector Demo ---")
    sample_doc = "natural language processing"
    doc_tokens = tokenizer.tokenize(sample_doc)
    doc_vector = embedder.document_vector(doc_tokens)
    print(f"Document: '{sample_doc}'")
    print(f"Document vector (first 5 dims): {doc_vector[:5]}")
    
    print("\n=== Demo Completed ===")

if __name__ == "__main__":
    main()