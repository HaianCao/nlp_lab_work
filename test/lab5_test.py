"""
Lab5 Test Suite - Basic Text Classification with scikit-learn
This script tests the basic text classification functionality using TF-IDF and Logistic Regression.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from models.text_classifier import TextClassifier
from preprocessing.noise_filtering import NoiseFiltering
from preprocessing.vocab_reduction import VocabReduction
from vectorize.tf_idf_vectorizer import TFIDFVectorizer

def load_sentiment_data():
    """Load sentiment data from CSV file."""
    try:
        # Try to find the sentiments.csv file
        possible_paths = [
            Path(__file__).parent.parent / "sentiments.csv",  # lab4/sentiments.csv
            Path(__file__).parent.parent.parent / "lab4" / "sentiments.csv",  # ../lab4/sentiments.csv
        ]
        
        df = None
        for path in possible_paths:
            if path.exists():
                df = pd.read_csv(path)
                print(f"✓ Loaded dataset from: {path}")
                break
        
        if df is None:
            # Create sample data if file not found
            print("⚠️ sentiments.csv not found. Creating sample data...")
            sample_data = {
                'text': [
                    "This movie is fantastic and I love it!",
                    "I hate this film, it's terrible.",
                    "The acting was superb, a truly great experience.",
                    "What a waste of time, absolutely boring.",
                    "Highly recommend this, a masterpiece.",
                    "Could not finish watching, so bad.",
                    "Amazing storyline and great characters.",
                    "Disappointing and poorly executed.",
                    "One of the best movies I've ever seen!",
                    "Not worth watching at all."
                ],
                'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
            }
            df = pd.DataFrame(sample_data)
        
        # Clean the dataset
        df = df.dropna(subset=['text', 'sentiment'])
        print(f"Dataset shape: {df.shape}")
        print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def test_basic_classification():
    """Test basic text classification with TF-IDF and Logistic Regression."""
    print("=== Testing Basic Text Classification ===")
    
    # Load data
    df = load_sentiment_data()
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(), df['sentiment'].tolist(), 
        test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Instantiate TfidfVectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    
    # Instantiate TextClassifier with the vectorizer  
    classifier = TextClassifier(vectorizer)
    
    # Train the classifier
    print("Training classifier...")
    classifier.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = classifier.predict(X_test)
    
    # Evaluate the model
    metrics = classifier.evaluate(y_test, y_pred)
    
    print("\\n=== Classification Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    
    # Test prediction probabilities
    if len(X_test) > 0:
        probabilities = classifier.predict_proba(X_test[:3])
        print("\\n=== Sample Predictions with Probabilities ===")
        for i, (text, true_label, pred_label, prob) in enumerate(zip(X_test[:3], y_test[:3], y_pred[:3], probabilities)):
            print(f"Text {i+1}: {text[:50]}...")
            print(f"  True: {true_label}, Predicted: {pred_label}")
            print(f"  Probabilities: [Negative: {prob[0]:.3f}, Positive: {prob[1]:.3f}]")
    
    return metrics

def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with noise filtering and vocabulary reduction."""
    print("\\n=== Testing Preprocessing Pipeline ===")
    
    # Sample noisy texts
    noisy_texts = [
        "Check out this link: https://example.com! It's AWESOME!!! <br>",
        "<p>This movie is great!!!</p> Visit http://test.com for more info.",
        "I absolutely LOOOVE this film!!! Best ever 😊😊😊",
        "Terrible movie... So boring 😞 #waste_of_time"
    ]
    
    print("Original texts:")
    for i, text in enumerate(noisy_texts):
        print(f"  {i+1}: {text}")
    
    # Test noise filtering
    noise_filter = NoiseFiltering()
    filtered_texts = noise_filter.preprocess_sentences(noisy_texts)
    
    print("\\nAfter noise filtering:")
    for i, text in enumerate(filtered_texts):
        print(f"  {i+1}: {text}")
    
    # Test vocabulary reduction
    vocab_reducer = VocabReduction(use_stemming=True, remove_stopwords=True)
    reduced_texts = vocab_reducer.preprocess_sentences(filtered_texts)
    
    print("\\nAfter vocabulary reduction:")
    for i, text in enumerate(reduced_texts):
        print(f"  {i+1}: {text}")
    
    # Get vocabulary statistics
    stats = vocab_reducer.get_vocabulary_stats(filtered_texts)
    print("\\n=== Vocabulary Statistics ===")
    print(f"Original vocabulary size: {stats['original_vocab_size']}")
    print(f"Processed vocabulary size: {stats['processed_vocab_size']}")
    print(f"Reduction ratio: {stats['vocabulary_reduction_ratio']:.2%}")

def test_custom_vectorizer():
    """Test our custom TF-IDF vectorizer."""
    print("\\n=== Testing Custom TF-IDF Vectorizer ===")
    
    documents = [
        "This is a sample document about machine learning.",
        "Machine learning is a subset of artificial intelligence.",
        "Text classification is an important NLP task.",
        "Natural language processing involves text analysis.",
        "Deep learning models are powerful for text classification."
    ]
    
    # Initialize and train custom vectorizer
    vectorizer = TFIDFVectorizer(max_features=50, ngram_range=(1, 2))
    vectorizer.train(documents)
    
    print(f"Vocabulary size: {vectorizer.get_vocabulary_size()}")
    print(f"Feature names (first 10): {vectorizer.get_feature_names()[:10]}")
    
    # Test vectorization
    test_sentence = "Machine learning is powerful for text analysis"
    vector = vectorizer.vectorize_sentence(test_sentence)
    print(f"\\nTest sentence: {test_sentence}")
    print(f"Vector shape: {vector.shape}")
    print(f"Non-zero elements: {np.count_nonzero(vector)}")
    
    # Test similarity
    sentence1 = "Machine learning is powerful"
    sentence2 = "Deep learning models are powerful"
    similarity = vectorizer.similarity(sentence1, sentence2)
    print(f"\\nSimilarity between:")
    print(f"  '{sentence1}'")
    print(f"  '{sentence2}'")
    print(f"Similarity score: {similarity:.4f}")

def run_all_tests():
    """Run all test functions."""
    print("🧪 Running Lab5 Test Suite")
    print("=" * 50)
    
    try:
        # Import numpy for vector operations
        global np
        import numpy as np
        
        # Test 1: Basic classification
        metrics = test_basic_classification()
        
        # Test 2: Preprocessing pipeline
        test_preprocessing_pipeline()
        
        # Test 3: Custom vectorizer
        test_custom_vectorizer()
        
        print("\\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print(f"📊 Final Classification Accuracy: {metrics['accuracy']:.1%}")
        
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()