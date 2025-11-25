from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ModelInterface(ABC):
    """
    Abstract interface for all machine learning models in the text classification pipeline.
    """

    @abstractmethod
    def fit(self, data, labels=None):
        """
        Train the model on provided data.
        
        Args:
            data: Training data (features)
            labels: Training labels (optional for unsupervised models)
        """
        pass

    @abstractmethod
    def predict(self, input_data):
        """
        Make predictions on input data.
        
        Args:
            input_data: Data to make predictions on
            
        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def evaluate(self, test_data, test_labels):
        """
        Evaluate the model performance.
        
        Args:
            test_data: Test data features
            test_labels: True test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass

class PreprocessorInterface(ABC):
    """
    Abstract interface for text preprocessing components.
    """

    @abstractmethod
    def preprocess_sentence(self, sentence: str) -> str:
        """
        Preprocess a single sentence.
        
        Args:
            sentence: Input sentence to preprocess
            
        Returns:
            Preprocessed sentence
        """
        pass

    @abstractmethod
    def preprocess_sentences(self, sentences: List[str]) -> List[str]:
        """
        Preprocess a list of sentences.
        
        Args:
            sentences: List of input sentences
            
        Returns:
            List of preprocessed sentences
        """
        pass

class TokenizeInterface(ABC):
    """
    Abstract interface for tokenization components.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        pass

class VectorizeInterface(ABC):
    """
    Abstract interface for text vectorization components.
    """

    @abstractmethod
    def train(self, documents: List[str]):
        """
        Train the vectorizer on documents.
        
        Args:
            documents: Training documents
        """
        pass

    @abstractmethod
    def vectorize_word(self, word: str):
        """
        Vectorize a single word.
        
        Args:
            word: Input word
            
        Returns:
            Vector representation of the word
        """
        pass

    @abstractmethod
    def vectorize_words(self, words: List[str]):
        """
        Vectorize multiple words.
        
        Args:
            words: List of words to vectorize
            
        Returns:
            Vector representations
        """
        pass

    @abstractmethod
    def vectorize_sentence(self, sentence: str):
        """
        Vectorize a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Vector representation of the sentence
        """
        pass