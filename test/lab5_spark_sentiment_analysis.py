"""
Lab5 Spark Sentiment Analysis
This script demonstrates sentiment analysis using PySpark MLlib pipeline.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def load_sentiment_data(spark):
    """Load sentiment data into Spark DataFrame."""
    try:
        # Try to find the sentiments.csv file
        possible_paths = [
            Path(__file__).parent.parent / "sentiments.csv",  # lab4/sentiments.csv  
            Path(__file__).parent.parent.parent / "lab4" / "sentiments.csv",  # ../lab4/sentiments.csv
        ]
        
        df = None
        for path in possible_paths:
            if path.exists():
                df = spark.read.csv(str(path), header=True, inferSchema=True)
                print(f"✓ Loaded dataset from: {path}")
                break
        
        if df is None:
            # Create sample data if file not found
            print("⚠️ sentiments.csv not found. Creating sample data...")
            sample_data = [
                ("This movie is fantastic and I love it!", 1),
                ("I hate this film, it's terrible.", 0),
                ("The acting was superb, a truly great experience.", 1),
                ("What a waste of time, absolutely boring.", 0),
                ("Highly recommend this, a masterpiece.", 1),
                ("Could not finish watching, so bad.", 0),
                ("Amazing storyline and great characters.", 1),
                ("Disappointing and poorly executed.", 0),
                ("One of the best movies I've ever seen!", 1),
                ("Not worth watching at all.", 0),
                ("Excellent cinematography and direction.", 1),
                ("Poor acting and weak plot.", 0),
                ("A delightful and entertaining film.", 1),
                ("Boring and predictable storyline.", 0),
                ("Outstanding performance by the lead actor.", 1),
                ("Terrible script and bad dialogue.", 0),
                ("Beautifully crafted and emotionally moving.", 1),
                ("Waste of money and time.", 0),
                ("Incredibly engaging and well-made.", 1),
                ("Disappointing and forgettable.", 0)
            ]
            
            schema = StructType([
                StructField("text", StringType(), True),
                StructField("sentiment", IntegerType(), True)
            ])
            
            df = spark.createDataFrame(sample_data, schema)
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_spark_pipeline():
    """Create the Spark ML pipeline for sentiment analysis."""
    
    # Step 1: Tokenization
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    
    # Step 2: Remove stop words
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # Step 3: Feature extraction using HashingTF
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    
    # Step 4: Apply TF-IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # Step 5: Classification using Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    
    # Create pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])
    
    return pipeline

def evaluate_model(predictions):
    """Evaluate the trained model using multiple metrics."""
    
    # Multi-class evaluator for accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    
    # Binary evaluator for AUC-ROC
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    
    # Calculate metrics
    accuracy = accuracy_evaluator.evaluate(predictions)
    auc_roc = auc_evaluator.evaluate(predictions)
    
    # Additional metrics
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )
    
    f1_score = f1_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    
    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall
    }

def display_sample_predictions(predictions, num_samples=5):
    """Display sample predictions for analysis."""
    
    print(f"\\n=== Sample Predictions (showing {num_samples}) ===")
    
    sample_predictions = predictions.select("text", "label", "prediction", "probability").limit(num_samples)
    
    for i, row in enumerate(sample_predictions.collect()):
        text = row["text"][:100] + "..." if len(row["text"]) > 100 else row["text"]
        true_label = "Positive" if row["label"] == 1 else "Negative"
        pred_label = "Positive" if row["prediction"] == 1 else "Negative"
        prob = row["probability"]
        
        print(f"\\nSample {i+1}:")
        print(f"  Text: {text}")
        print(f"  True: {true_label}, Predicted: {pred_label}")
        print(f"  Probabilities: [Negative: {prob[0]:.3f}, Positive: {prob[1]:.3f}]")

def test_different_models(training_data, test_data):
    """Test different classification models."""
    
    print("\\n=== Testing Different Models ===")
    
    # Common preprocessing steps
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    models = {
        'Logistic Regression': LogisticRegression(featuresCol="features", labelCol="label", maxIter=10),
        'Random Forest': RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20)
    }
    
    results = {}
    
    for model_name, classifier in models.items():
        print(f"\\nTraining {model_name}...")
        
        # Create pipeline for this model
        pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, classifier])
        
        # Train model
        model = pipeline.fit(training_data)
        
        # Make predictions
        predictions = model.transform(test_data)
        
        # Evaluate
        metrics = evaluate_model(predictions)
        results[model_name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return results

def main():
    """Main function to run the Spark sentiment analysis pipeline."""
    
    print("🚀 Starting Lab5 Spark Sentiment Analysis")
    print("=" * 50)
    
    # Step 1: Initialize Spark Session
    print("\\n[1/5] Initializing Spark Session...")
    
    spark = SparkSession.builder \\
        .appName("Lab5_SentimentAnalysis") \\
        .master("local[*]") \\
        .config("spark.driver.memory", "2g") \\
        .config("spark.executor.memory", "2g") \\
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    print("✓ Spark session initialized")
    
    try:
        # Step 2: Load Data
        print("\\n[2/5] Loading data...")
        df = load_sentiment_data(spark)
        
        # Normalize labels (ensure they are 0 and 1)
        df = df.withColumn("label", when(col("sentiment") > 0, 1).otherwise(0))
        
        # Remove null values
        initial_count = df.count()
        df = df.dropna(subset=["text", "sentiment"])
        final_count = df.count()
        
        print(f"Dataset statistics:")
        print(f"  - Initial rows: {initial_count}")
        print(f"  - Final rows: {final_count}")
        print(f"  - Positive samples: {df.filter(df.label == 1).count()}")
        print(f"  - Negative samples: {df.filter(df.label == 0).count()}")
        
        # Step 3: Split Data
        print("\\n[3/5] Splitting data...")
        training_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        
        train_count = training_data.count()
        test_count = test_data.count()
        print(f"  - Training set: {train_count} samples")
        print(f"  - Test set: {test_count} samples")
        
        # Step 4: Create and Train Pipeline
        print("\\n[4/5] Creating and training pipeline...")
        pipeline = create_spark_pipeline()
        
        print("Training model...")
        model = pipeline.fit(training_data)
        
        # Step 5: Make Predictions and Evaluate
        print("\\n[5/5] Making predictions and evaluating...")
        predictions = model.transform(test_data)
        
        metrics = evaluate_model(predictions)
        
        print("\\n=== Main Model Results ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        # Display sample predictions
        display_sample_predictions(predictions, num_samples=3)
        
        # Test different models
        if test_count > 10:  # Only run if we have enough data
            test_results = test_different_models(training_data, test_data)
            
            print("\\n=== Model Comparison ===")
            for model_name, model_metrics in test_results.items():
                print(f"{model_name}:")
                print(f"  - Accuracy: {model_metrics['accuracy']:.4f}")
                print(f"  - F1 Score: {model_metrics['f1_score']:.4f}")
        
        print("\\n" + "=" * 50)
        print("🎉 Spark sentiment analysis completed successfully!")
        print(f"📊 Final Accuracy: {metrics['accuracy']:.1%}")
        
    except Exception as e:
        print(f"💥 Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop Spark session
        spark.stop()
        print("\\n🔌 Spark session stopped")

if __name__ == "__main__":
    main()