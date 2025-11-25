"""
Lab5 Model Improvement 
This script compares different classification models and preprocessing techniques.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from time import time

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from models.text_classifier import TextClassifier
from models.naive_bayes import NaiveBayesModel
from models.neural_network import NeuralNetworkModel
from models.gbts import GradientBoostingModel

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
            # Create larger sample data if file not found
            print("⚠️ sentiments.csv not found. Creating enhanced sample data...")
            positive_samples = [
                "This movie is fantastic and I love it!",
                "The acting was superb, a truly great experience.",
                "Highly recommend this, a masterpiece.",
                "Amazing storyline and great characters.",
                "One of the best movies I've ever seen!",
                "Excellent cinematography and direction.",
                "A delightful and entertaining film.",
                "Outstanding performance by the lead actor.",
                "Beautifully crafted and emotionally moving.",
                "Incredibly engaging and well-made.",
                "Brilliant direction and screenplay.",
                "Captivating and thought-provoking content.",
                "Exceptional visual effects and sound design.",
                "Heartwarming story with great character development.",
                "Masterful storytelling and excellent pacing."
            ]
            
            negative_samples = [
                "I hate this film, it's terrible.",
                "What a waste of time, absolutely boring.",
                "Could not finish watching, so bad.",
                "Disappointing and poorly executed.",
                "Not worth watching at all.",
                "Poor acting and weak plot.",
                "Boring and predictable storyline.",
                "Terrible script and bad dialogue.",
                "Waste of money and time.",
                "Disappointing and forgettable.",
                "Poorly written and badly acted.",
                "Confusing plot and weak characters.",
                "Dull and uninteresting from start to finish.",
                "Terrible pacing and poor execution.",
                "Completely predictable and uninspiring."
            ]
            
            # Create balanced dataset
            texts = positive_samples + negative_samples
            sentiments = [1] * len(positive_samples) + [0] * len(negative_samples)
            
            df = pd.DataFrame({
                'text': texts,
                'sentiment': sentiments
            })
        
        # Clean the dataset
        df = df.dropna(subset=['text', 'sentiment'])
        print(f"Dataset shape: {df.shape}")
        print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(texts, use_noise_filtering=True, use_vocab_reduction=True):
    """Apply preprocessing pipeline to texts."""
    processed_texts = texts.copy()
    
    if use_noise_filtering:
        noise_filter = NoiseFiltering()
        processed_texts = noise_filter.preprocess_sentences(processed_texts)
    
    if use_vocab_reduction:
        vocab_reducer = VocabReduction(
            use_stemming=True,
            remove_stopwords=True
        )
        processed_texts = vocab_reducer.preprocess_sentences(processed_texts)
    
    return processed_texts

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model."""
    print(f"\\n--- Training {model_name} ---")
    
    start_time = time()
    
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = model.evaluate(y_test, y_pred)
        
        end_time = time()
        training_time = end_time - start_time
        
        print(f"✓ {model_name} completed in {training_time:.2f}s")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        # Add training time to metrics
        metrics['training_time'] = training_time
        metrics['model_name'] = model_name
        
        return metrics, y_pred
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        return None, None

def compare_preprocessing_methods(df):
    """Compare different preprocessing approaches."""
    print("\\n=== Comparing Preprocessing Methods ===")
    
    # Split data
    X = df['text'].tolist()
    y = df['sentiment'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    preprocessing_configs = {
        'No Preprocessing': (False, False),
        'Noise Filtering Only': (True, False),
        'Vocab Reduction Only': (False, True),
        'Full Preprocessing': (True, True)
    }
    
    results = {}
    
    for config_name, (use_noise, use_vocab) in preprocessing_configs.items():
        print(f"\\nTesting: {config_name}")
        
        # Preprocess data
        X_train_processed = preprocess_data(X_train, use_noise, use_vocab)
        X_test_processed = preprocess_data(X_test, use_noise, use_vocab)
        
        # Train TF-IDF vectorizer
        vectorizer = TFIDFVectorizer(max_features=1000)
        X_train_vectors = vectorizer.fit_transform(X_train_processed)
        X_test_vectors = vectorizer.transform(X_test_processed)
        
        # Train simple logistic regression model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(X_train_vectors, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_vectors)
        accuracy = np.mean(y_pred == y_test)
        
        results[config_name] = accuracy
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Show preprocessing comparison
    print(f"\\n=== Preprocessing Comparison Results ===")
    for config, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{config:<25}: {accuracy:.4f}")
    
    return results

def compare_models(df):
    """Compare different classification models."""
    print("\\n=== Comparing Classification Models ===")
    
    # Preprocess data
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()
    
    # Apply full preprocessing
    processed_texts = preprocess_data(texts, use_noise_filtering=True, use_vocab_reduction=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Create TF-IDF vectors
    vectorizer = TFIDFVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Convert sparse matrices to dense for models that require it
    X_train_dense = X_train_tfidf.toarray() if hasattr(X_train_tfidf, 'toarray') else X_train_tfidf
    X_test_dense = X_test_tfidf.toarray() if hasattr(X_test_tfidf, 'toarray') else X_test_tfidf
    
    # Define models to compare
    models = {
        'Logistic Regression': TextClassifier(vectorizer),
        'Naive Bayes': NaiveBayesModel(),
        'Neural Network': NeuralNetworkModel(hidden_layer_sizes=(50,), max_iter=300),
        'Gradient Boosting': GradientBoostingModel(n_estimators=50, max_depth=3)
    }
    
    all_results = []
    predictions = {}
    
    for model_name, model in models.items():
        if model_name == 'Logistic Regression':
            # TextClassifier handles vectorization internally
            model.vectorizer = vectorizer  # Use our trained vectorizer
            metrics, y_pred = train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test)
        else:
            # Other models work with dense feature matrices
            metrics, y_pred = train_and_evaluate_model(model, model_name, X_train_dense, X_test_dense, y_train, y_test)
        
        if metrics:
            all_results.append(metrics)
            predictions[model_name] = y_pred
    
    # Compare results
    if all_results:
        print(f"\\n=== Model Comparison Summary ===")
        print(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Time (s)':<10}")
        print("-" * 50)
        
        # Sort by F1 score
        sorted_results = sorted(all_results, key=lambda x: x['f1_score'], reverse=True)
        
        for result in sorted_results:
            print(f"{result['model_name']:<20} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {result['training_time']:<10.2f}")
        
        # Show best model details
        best_model = sorted_results[0]
        print(f"\\n🏆 Best Model: {best_model['model_name']}")
        print(f"   Accuracy: {best_model['accuracy']:.1%}")
        print(f"   F1 Score: {best_model['f1_score']:.4f}")
        
        return sorted_results
    
    return []

def analyze_feature_importance(df):
    """Analyze feature importance using Gradient Boosting model."""
    print("\\n=== Feature Importance Analysis ===")
    
    try:
        # Preprocess data
        texts = df['text'].tolist()
        labels = df['sentiment'].tolist()
        processed_texts = preprocess_data(texts, use_noise_filtering=True, use_vocab_reduction=True)
        
        # Create features
        vectorizer = TFIDFVectorizer(max_features=100, ngram_range=(1, 1))  # Smaller for interpretability
        X = vectorizer.fit_transform(processed_texts).toarray()
        
        # Train Gradient Boosting model
        gb_model = GradientBoostingModel(n_estimators=50, random_state=42)
        gb_model.fit(X, labels)
        
        # Get feature importance
        feature_names = vectorizer.get_feature_names()
        importance_dict = gb_model.get_feature_importance(feature_names, top_n=10)
        
        if importance_dict:
            print("Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(importance_dict.items(), 1):
                print(f"  {i:2d}. {feature:<15}: {importance:.4f}")
        else:
            print("Could not extract feature importance")
            
    except Exception as e:
        print(f"Feature importance analysis failed: {e}")

def run_comprehensive_analysis():
    """Run comprehensive model and preprocessing analysis."""
    print("🔬 Starting Lab5 Model Improvement Analysis")
    print("=" * 60)
    
    try:
        # Load data
        df = load_sentiment_data()
        
        if len(df) < 10:
            print("⚠️ Dataset too small for comprehensive analysis")
            return
        
        # 1. Compare preprocessing methods
        preprocessing_results = compare_preprocessing_methods(df)
        
        # 2. Compare classification models
        model_results = compare_models(df)
        
        # 3. Analyze feature importance
        analyze_feature_importance(df)
        
        print("\\n" + "=" * 60)
        print("📊 Analysis Summary:")
        
        if preprocessing_results:
            best_preprocessing = max(preprocessing_results.items(), key=lambda x: x[1])
            print(f"🔧 Best Preprocessing: {best_preprocessing[0]} ({best_preprocessing[1]:.1%})")
        
        if model_results:
            best_model = model_results[0]
            print(f"🤖 Best Model: {best_model['model_name']} (F1: {best_model['f1_score']:.4f})")
        
        print("🎉 Comprehensive analysis completed!")
        
    except Exception as e:
        print(f"💥 Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_analysis()