"""
Lab4 Spark Word2Vec Demo
This script demonstrates Word2Vec training using Apache Spark MLlib.
"""

import sys
import os

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.types import StructType, StructField, ArrayType, StringType
from preprocessing.embedding_tokenizer import EmbeddingTokenizer

def main():
    """
    Main function to demonstrate Spark Word2Vec training.
    """
    print("=== Lab4 Spark Word2Vec Demo ===")
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Word2VecDemo") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    
    print("Spark session initialized")
    
    # Initialize tokenizer
    tokenizer = EmbeddingTokenizer()
    
    # Sample data for training
    sentences_data = [
        "I love natural language processing and machine learning",
        "Word embeddings capture semantic relationships between words",
        "Deep learning models are very powerful for text analysis",
        "Natural language understanding requires sophisticated algorithms",
        "Text mining and information retrieval use similar techniques",
        "Semantic similarity can be measured using vector representations",
        "Language models learn patterns from large text corpora",
        "Neural networks excel at learning complex language patterns"
    ]
    
    print(f"Processing {len(sentences_data)} sentences...")
    
    # Tokenize sentences
    tokenized_data = []
    for sentence in sentences_data:
        tokens = tokenizer.tokenize(sentence)
        tokenized_data.append((tokens,))
    
    # Create DataFrame
    schema = StructType([
        StructField("text", ArrayType(StringType()), True)
    ])
    
    df = spark.createDataFrame(tokenized_data, schema)
    
    print("Created Spark DataFrame:")
    df.show(5, truncate=False)
    
    # Initialize Word2Vec
    print("\n--- Training Spark Word2Vec Model ---")
    word2vec = Word2Vec(
        vectorSize=50,  # Small dimension for demo
        minCount=1,
        numPartitions=1,
        inputCol="text",
        outputCol="result"
    )
    
    # Train the model
    model = word2vec.fit(df)
    
    print("Model trained successfully!")
    print(f"Vocabulary size: {model.getVectors().count()}")
    
    # Show some word vectors
    print("\n--- Word Vectors ---")
    vectors_df = model.getVectors()
    vectors_df.show(10)
    
    # Test similarity
    print("\n--- Testing Word Similarity ---")
    try:
        # Find synonyms for a word
        synonyms = model.findSynonyms("language", 3)
        print("Words similar to 'language':")
        synonyms.show()
    except Exception as e:
        print(f"Could not find synonyms for 'language': {e}")
        # Try with another word
        try:
            synonyms = model.findSynonyms("text", 3)
            print("Words similar to 'text':")
            synonyms.show()
        except Exception as e:
            print(f"Could not find synonyms: {e}")
    
    # Save the model (optional)
    output_path = "results/spark_word2vec_model"
    print(f"\n--- Saving Model to {output_path} ---")
    try:
        # Remove existing model if it exists
        import shutil
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        model.write().overwrite().save(output_path)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    # Stop Spark session
    spark.stop()
    print("\n=== Demo Completed ===")

if __name__ == "__main__":
    main()