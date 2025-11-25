from src.config.io import TOP_K_SIMILAR, OUTPUT_FEATURES, INPUT_RAW
from src.config.output import save_similarity_results
from src.metrics.similarity import cosine_similarity

import random
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

def find_similar_documents(df: DataFrame, target_doc_index: int = None, top_k: int = TOP_K_SIMILAR):
    """
    Find and display the top K similar documents to a target document based on cosine similarity.
    Uses DataFrame row index instead of ID column.

    Args:
        df: The processed DataFrame containing document vectors.
        target_doc_index: The index (0-based) of the target document. If None, a random document is selected.
        top_k: The number of similar documents to retrieve.
    """    
    try:
        # Get total document count efficiently
        total_docs = df.count()
        print(f"Total documents in dataset: {total_docs}")
        
        # Select target document
        if target_doc_index is None or target_doc_index < 0 or target_doc_index >= total_docs:
            # Random selection
            target_doc_index = random.randint(0, total_docs - 1)
            print(f"🎲 Randomly selected target document index: {target_doc_index}")
        
        # Add row numbers to DataFrame to enable indexing
        window = Window.orderBy(F.lit(1))
        df_with_index = df.withColumn("row_index", F.row_number().over(window) - 1)
        
        # Get target document by index efficiently
        target_rows = df_with_index.filter(F.col("row_index") == target_doc_index).collect()
        
        if not target_rows:
            raise ValueError(f"No document found at index {target_doc_index}")
            
        target_row = target_rows[0]
        target_vector = target_row[OUTPUT_FEATURES]
        target_text = target_row[INPUT_RAW]
        
        print(f"\n🎯 Target Document (Index: {target_doc_index}):")
        print(f"📝 Text: {target_text[:100]}...")
        print(f"📊 Vector features: {len(target_vector.indices)} non-zero out of {target_vector.size}")

        # Instead of collecting all documents, we'll sample and process in smaller batches
        print(f"\n⚡ Computing similarities using batched processing...")
        
        # Sample documents for similarity computation to avoid memory issues
        sample_size = min(5000, total_docs)  # Limit to 5000 docs to avoid memory issues
        
        if total_docs > sample_size:
            print(f"🎲 Sampling {sample_size} documents from {total_docs} for similarity computation")
            df_sample = df_with_index.sample(False, sample_size / total_docs, seed=42)
        else:
            df_sample = df_with_index
        
        # Collect the sample
        all_docs = df_sample.collect()
        
        similarities = []
        for doc in all_docs:
            doc_index = doc['row_index']
            if doc_index != target_doc_index:  # Exclude target document
                similarity = float(cosine_similarity(target_vector, doc[OUTPUT_FEATURES]))
                similarities.append((similarity, doc_index, doc))
        
        # Sort by similarity (descending) and get top K
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_similar_docs = similarities[:top_k]

        print(f"\n🏆 Top {top_k} most similar documents:")
        print("="*50)
        
        for rank, (similarity, doc_index, doc) in enumerate(top_similar_docs, 1):
            bar_length = int(similarity * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"#{rank} [{bar}] {similarity:.4f}")
        
        print("="*50)
        
        # Prepare data for saving detailed results
        query_info = {
            'text': target_text,
            'vector_size': target_vector.size,
            'indices': list(target_vector.indices),
            'values': list(target_vector.values)
        }
        
        similarity_results = []
        for similarity, doc_index, doc in top_similar_docs:
            doc_info = {
                'text': doc[INPUT_RAW],
                'vector_size': doc[OUTPUT_FEATURES].size,
                'indices': list(doc[OUTPUT_FEATURES].indices),
                'values': list(doc[OUTPUT_FEATURES].values)
            }
            similarity_results.append((similarity, doc_info))
        
        # Save detailed results to file
        save_similarity_results(query_info, similarity_results, top_k)
        
        return target_doc_index, target_text, [(doc_index, similarity, doc[INPUT_RAW]) for similarity, doc_index, doc in top_similar_docs]
    
    except Exception as e:
        print(f"❌ Error in find_similar_documents: {e}")
        # Return a fallback result
        return None, None, []

__all__ = ['find_similar_documents']