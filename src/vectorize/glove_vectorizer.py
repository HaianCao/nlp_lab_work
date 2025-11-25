import numpy as np
from typing import List, Union, Optional
import warnings

from ..core.classification_interfaces import VectorizeInterface
from ..preprocessing.sentiment_tokenizer import SentimentTokenizer

try:
    from gensim.downloader import load
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn("Warning: gensim not available. Install with: pip install gensim")

class GloveVectorizer(VectorizeInterface):
    """
    GloVe (Global Vectors for Word Representation) vectorizer using pre-trained models.
    """
    
    def __init__(self, model_name: str = "glove-wiki-gigaword-100", tokenizer=None):
        """
        Initialize GloVe Vectorizer.
        
        Args:
            model_name: Name of the pre-trained GloVe model to load
            tokenizer: Tokenizer to use for text preprocessing
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = tokenizer or SentimentTokenizer()
        self.embedding_dim = None
        self.is_trained = False
        
    def train(self, documents: List[str]):
        """
        'Train' the GloVe vectorizer (actually just loads the pre-trained model).
        
        Args:
            documents: List of documents (not used for pre-trained models)
        """
        self._load_model()
        self.is_trained = True
    
    def _load_model(self):
        """Load pre-trained GloVe model."""
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim not installed. Install with: pip install gensim")
        
        try:
            print(f"Loading {self.model_name}...")
            self.model = load(self.model_name)
            self.embedding_dim = self.model.vector_size
            print(f"✓ GloVe model loaded successfully")
            print(f"  - Vector size: {self.embedding_dim}")
            print(f"  - Vocabulary size: {len(self.model)}")
        except Exception as e:
            print(f"Error loading GloVe model: {e}")
            # Try alternative model names
            alternative_models = [
                "glove-wiki-gigaword-50",
                "glove-wiki-gigaword-100", 
                "glove-wiki-gigaword-200",
                "glove-wiki-gigaword-300"
            ]
            
            for alt_model in alternative_models:
                if alt_model != self.model_name:
                    try:
                        print(f"Trying alternative model: {alt_model}")
                        self.model = load(alt_model)
                        self.embedding_dim = self.model.vector_size
                        self.model_name = alt_model
                        print(f"✓ Alternative model loaded: {alt_model}")
                        break
                    except:
                        continue
            
            if self.model is None:
                raise RuntimeError("Could not load any GloVe model")

    def vectorize_word(self, word: str) -> np.ndarray:
        """
        Get the GloVe vector for a single word.
        
        Args:
            word: Input word
            
        Returns:
            GloVe vector for the word, or zero vector if not found
        """
        if not self.model:
            self._load_model()
        
        try:
            if word.lower() in self.model:
                return self.model[word.lower()]
            else:
                # Return zero vector for unknown words
                return np.zeros(self.embedding_dim)
        except Exception as e:
            print(f"Error vectorizing word '{word}': {e}")
            return np.zeros(self.embedding_dim)

    def vectorize_words(self, words: Union[List[str], str]) -> np.ndarray:
        """
        Vectorize multiple words.
        
        Args:
            words: List of words or single word string
            
        Returns:
            Array of GloVe vectors for the words
        """
        if isinstance(words, str):
            words = [words]
        
        vectors = []
        for word in words:
            vector = self.vectorize_word(word)
            vectors.append(vector)
        
        return np.array(vectors)

    def vectorize_sentence(self, sentence: str) -> np.ndarray:
        """
        Vectorize a sentence by averaging word vectors.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Average GloVe vector for the sentence
        """
        if not self.model:
            self._load_model()
        
        try:
            tokens = self.tokenizer.tokenize(sentence)
            if not tokens:
                return np.zeros(self.embedding_dim)
            
            vectors = []
            for token in tokens:
                vector = self.vectorize_word(token)
                # Only include non-zero vectors (words found in vocabulary)
                if not np.allclose(vector, 0):
                    vectors.append(vector)
            
            if not vectors:
                # No words found in vocabulary
                return np.zeros(self.embedding_dim)
            
            # Return average of word vectors
            return np.mean(vectors, axis=0)
            
        except Exception as e:
            print(f"Error vectorizing sentence: {e}")
            return np.zeros(self.embedding_dim)

    def vectorize_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Vectorize multiple sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Array of sentence vectors
        """
        vectors = []
        for sentence in sentences:
            vector = self.vectorize_sentence(sentence)
            vectors.append(vector)
        
        return np.array(vectors)

    def similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Calculate cosine similarity between two sentences using GloVe vectors.
        
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
            
            # Check for zero vectors
            if np.allclose(vec1, 0) or np.allclose(vec2, 0):
                return 0.0
            
            # Reshape for sklearn
            vec1 = vec1.reshape(1, -1)
            vec2 = vec2.reshape(1, -1)
            
            sim = cosine_similarity(vec1, vec2)
            return float(sim[0][0])
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def get_most_similar_words(self, word: str, top_n: int = 10) -> List[tuple]:
        """
        Get most similar words to a given word using GloVe vectors.
        
        Args:
            word: Target word
            top_n: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if not self.model:
            self._load_model()
        
        try:
            if word.lower() in self.model:
                similar_words = self.model.most_similar(word.lower(), topn=top_n)
                return similar_words
            else:
                print(f"Word '{word}' not found in vocabulary")
                return []
        except Exception as e:
            print(f"Error finding similar words: {e}")
            return []

    def get_vocabulary_coverage(self, texts: List[str]) -> dict:
        """
        Calculate vocabulary coverage for a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with coverage statistics
        """
        if not self.model:
            self._load_model()
        
        all_words = set()
        covered_words = set()
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            for token in tokens:
                word_lower = token.lower()
                all_words.add(word_lower)
                if word_lower in self.model:
                    covered_words.add(word_lower)
        
        total_words = len(all_words)
        covered_count = len(covered_words)
        coverage_ratio = covered_count / total_words if total_words > 0 else 0
        
        return {
            'total_unique_words': total_words,
            'covered_words': covered_count,
            'uncovered_words': total_words - covered_count,
            'coverage_ratio': coverage_ratio,
            'coverage_percentage': coverage_ratio * 100
        }