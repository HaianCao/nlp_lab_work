import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two sparse vectors using numpy for optimization.

    Args:
        vec1: First sparse vector (e.g., SparseVector from PySpark).
        vec2: Second sparse vector (e.g., SparseVector from PySpark).

    Returns:
        float: Cosine similarity score between vec1 and vec2.
    """
    if vec1 is None or vec2 is None:
        return 0.0

    indices1 = np.array(vec1.indices)
    indices2 = np.array(vec2.indices)
    values1 = np.array(vec1.values)
    values2 = np.array(vec2.values)
    
    common_indices = np.intersect1d(indices1, indices2)
    
    if len(common_indices) == 0:
        return 0.0
    
    pos1 = np.searchsorted(indices1, common_indices)
    pos2 = np.searchsorted(indices2, common_indices)
    
    common_values1 = values1[pos1]
    common_values2 = values2[pos2]
    
    dot_product = np.sum(common_values1 * common_values2)
    norm1 = np.linalg.norm(values1)
    norm2 = np.linalg.norm(values2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))