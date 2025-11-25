from typing import Union, List, Optional
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from ..core.classification_interfaces import VectorizeInterface

class TFIDFVectorizer(VectorizeInterface):
    """
    TF-IDF Vectorizer for text classification tasks.
    """
    
    def __init__(self, max_features=10000, min_df=1, max_df=0.95, ngram_range=(1, 1)):
        """
        Initialize TF-IDF Vectorizer.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for a term to be considered
            max_df: Maximum document frequency for a term to be considered
            ngram_range: Range of n-grams to consider
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.is_trained = False

    def train(self, documents: List[str]):
        """
        Train TF-IDF vectorizer on the provided documents.
        
        Args:
            documents: List of documents to train on
        """
        try:
            self.vectorizer.fit(documents)
            self.is_trained = True
        except Exception as e:
            print(f"Error training TF-IDF vectorizer: {e}")
            raise

    def fit_transform(self, documents: List[str]):
        """
        Train the vectorizer and transform documents in one step.
        
        Args:
            documents: List of documents
            
        Returns:
            TF-IDF matrix
        """
        try:
            result = self.vectorizer.fit_transform(documents)
            self.is_trained = True
            return result
        except Exception as e:
            print(f"Error in fit_transform: {e}")
            raise

    def transform(self, documents: List[str]):
        """
        Transform documents using the trained vectorizer.
        
        Args:
            documents: List of documents to transform
            
        Returns:
            TF-IDF matrix
        """
        if not self.is_trained:
            raise ValueError("Vectorizer is not trained yet. Call train() or fit_transform() first.")
        
        try:
            return self.vectorizer.transform(documents)
        except Exception as e:
            print(f"Error transforming documents: {e}")
            raise

    def vectorize_word(self, word: str):
        """
        Vectorize a single word.
        
        Args:
            word: Input word
            
        Returns:
            TF-IDF vector for the word
        """
        if not self.is_trained:
            return None
        
        try:
            vector = self.vectorizer.transform([word])
            return vector.toarray()[0]
        except Exception as e:
            print(f"Error vectorizing word: {e}")
            return None

    def vectorize_words(self, words: Union[List[str], str]):
        """
        Vectorize multiple words.
        
        Args:
            words: List of words or single word string
            
        Returns:
            TF-IDF vectors for the words
        """
        if not self.is_trained:
            return None
        
        try:
            if isinstance(words, str):
                words = [words]
            
            vectors = self.vectorizer.transform(words)
            return vectors.toarray()
        except Exception as e:
            print(f"Error vectorizing words: {e}")
            return None

    def vectorize_sentence(self, sentence: str):
        """
        Vectorize a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            TF-IDF vector for the sentence
        """
        if not self.is_trained:
            return None
        
        try:
            vector = self.vectorizer.transform([sentence])
            return vector.toarray()[0]
        except Exception as e:
            print(f"Error vectorizing sentence: {e}")
            return None

    def vectorize_sentences(self, sentences: List[str]):
        """
        Vectorize multiple sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            TF-IDF matrix for the sentences
        """
        if not self.is_trained:
            return None
        
        try:
            vectors = self.vectorizer.transform(sentences)
            return vectors
        except Exception as e:
            print(f"Error vectorizing sentences: {e}")
            return None

    def similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Calculate cosine similarity between two sentences.
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            Cosine similarity score
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            vec1 = self.vectorize_sentence(sentence1)
            vec2 = self.vectorize_sentence(sentence2)
            
            if vec1 is None or vec2 is None:
                return 0.0
            
            # Reshape for sklearn
            vec1 = vec1.reshape(1, -1)
            vec2 = vec2.reshape(1, -1)
            
            sim = cosine_similarity(vec1, vec2)
            return float(sim[0][0])
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def get_feature_names(self) -> List[str]:
        """
        Get feature names (vocabulary) from the trained vectorizer.
        
        Returns:
            List of feature names
        """
        if not self.is_trained:
            return []
        
        try:
            return self.vectorizer.get_feature_names_out().tolist()
        except Exception as e:
            print(f"Error getting feature names: {e}")
            return []

    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Size of vocabulary
        """
        if not self.is_trained:
            return 0
        
        return len(self.vectorizer.vocabulary_)

    def get_top_features(self, document_index: int = 0, top_n: int = 10) -> List[tuple]:
        """
        Get top TF-IDF features for a document.
        
        Args:
            document_index: Index of the document in the last transformed batch
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, score) tuples
        """
        if not self.is_trained:
            return []
        
        try:
            # This requires storing the last transformation result
            if hasattr(self, '_last_transform_result'):
                feature_names = self.get_feature_names()
                scores = self._last_transform_result[document_index].toarray()[0]
                
                # Get indices of top scores
                top_indices = np.argsort(scores)[::-1][:top_n]
                
                return [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            else:
                return []
        except Exception as e:
            print(f"Error getting top features: {e}")
            return []