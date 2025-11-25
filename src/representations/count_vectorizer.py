import numpy as np

from src.core.interfaces import Tokenizer

class CountVectorizer:
    """
    A simple Count Vectorizer implementation using Bag-of-Words (BoW).
    """

    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize the CountVectorizer with a tokenizer.

        Variables:
            tokenizer (Tokenizer): An instance of a tokenizer to preprocess the text.
            vocabulary (dict): A dictionary mapping tokens to feature indices.
        """
        
        self.tokenizer = tokenizer
        self.vocabulary_ = {}

    def fit(self, documents: list[str]):
        """
        Fit the CountVectorizer to the input documents.
        
        Args:
            documents (List[str]): A list of input documents to fit the vectorizer.
        """

        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                if token not in self.vocabulary_:
                    self.vocabulary_[token] = len(self.vocabulary_)

        return self.vocabulary_

    def transform(self, documents: list[str]) -> np.ndarray:
        """
        Transform the input documents into a document-term matrix.
        
        Args:
            documents (List[str]): A list of input documents to transform.

        Returns:
            np.ndarray: A 2D numpy array representing the document-term matrix.
        """

        matrix = np.zeros((len(documents), len(self.vocabulary_)), dtype=int)
        for i, doc in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    j = self.vocabulary_[token]
                    matrix[i, j] += 1

        return matrix
    
    def fit_transform(self, documents: list[str]) -> np.ndarray:
        """
        Fit the CountVectorizer to the input documents and transform them into a document-term matrix.

        Args:
            documents (List[str]): A list of input documents to fit and transform.

        Returns:
            np.ndarray: A 2D numpy array representing the document-term matrix.
        """

        self.fit(documents)

        return self.transform(documents)