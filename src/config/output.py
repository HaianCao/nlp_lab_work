from src.config.io import get_output_dir, get_output_path, \
                            OUTPUT_FEATURES, INPUT_RAW

import os
from pathlib import Path
from pyspark.sql import DataFrame

def save_results(spark_df: DataFrame):
    """
    Save processed results to a text file.

    Args:
        spark_df: Spark DataFrame.
    """
    # Make sure the output directory exists
    os.makedirs(get_output_dir(), exist_ok=True)

    # Get data and print to a simple file
    df_output = spark_df.select(INPUT_RAW, OUTPUT_FEATURES).collect()

    with open(get_output_path(), "w", encoding="utf-8") as f:
        for i, row in enumerate(df_output, 1):
            f.write(f"{i}. {row.text}\n")
            
            # Parse sparse vector features
            features = row.features
            if hasattr(features, 'size') and hasattr(features, 'indices') and hasattr(features, 'values'):
                # Sparse vector format: (size, [indices], [values])
                f.write(f"Vector size: {features.size}\n")
                f.write(f"Indices (HashingTF positions): {list(features.indices)}\n")
                f.write(f"Values (TF-IDF weights): {[float(v) for v in features.values]}\n")
            else:
                # Fallback for other formats
                f.write(f"Features: {features}\n")
            
            f.write("\n")

    print(f"Saved {len(df_output)} results to: {get_output_path()}")


def save_similarity_results(query_info, similarity_results, top_k=5):
    """
    Save document similarity results to a text file with detailed vector information.
    
    Args:
        query_info: Dictionary with query document information
        similarity_results: List of (similarity_score, document_info) tuples
        top_k: Number of top results to save
    """
    # Make sure the output directory exists
    os.makedirs(get_output_dir(), exist_ok=True)
    
    # Generate filename based on query
    filename = f"similarity_{top_k}_results.txt"
    filepath = get_output_dir() / filename

    with open(filepath, "w", encoding="utf-8") as f:
        # Write query information
        f.write("=== QUERY DOCUMENT ===\n")
        f.write(f"Text: {query_info[INPUT_RAW]}\n")
        f.write(f"Vector size: {query_info['vector_size']}\n")
        f.write(f"Non-zero elements: {len(query_info['indices'])}\n")
        f.write(f"Indices: {query_info['indices']}\n")
        f.write(f"Values: {[round(v, 6) for v in query_info['values']]}\n")
        f.write("\n")
        
        # Write similarity results
        f.write(f"=== TOP {top_k} SIMILAR DOCUMENTS ===\n")
        for i, (similarity_score, doc_info) in enumerate(similarity_results[:top_k], 1):
            f.write(f"\n{i}. SIMILARITY: {similarity_score:.6f}\n")
            f.write(f"Text: {doc_info[INPUT_RAW]}\n")
            f.write(f"Vector size: {doc_info['vector_size']}\n")
            f.write(f"Non-zero elements: {len(doc_info['indices'])}\n")
            f.write(f"Indices: {doc_info['indices']}\n")
            f.write(f"Values: {[round(v, 6) for v in doc_info['values']]}\n")
            
            # Calculate intersection details
            query_indices_set = set(query_info['indices'])
            doc_indices_set = set(doc_info['indices'])
            intersection = query_indices_set.intersection(doc_indices_set)
            f.write(f"Common indices with query: {sorted(intersection)} ({len(intersection)} total)\n")
    
    print(f"Saved similarity results to: {filepath}")
    return str(filepath)