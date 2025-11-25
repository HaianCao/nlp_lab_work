from abc import ABC, abstractmethod

class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.
    """
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the input text into a list of tokens.

        Args:
            text (str): The input text to tokenize.
        
        Returns:
            List[str]: A list of tokens.
        """
        pass

class Vectorizer(ABC):
    """
    Abstract base class for vectorizers.
    """
    @abstractmethod
    def fit(self, texts: list[str]):
        """
        Fit the vectorizer to the input texts.

        Args:
            texts (List[str]): A list of input texts to fit the vectorizer.
        """
        pass

    @abstractmethod
    def transform(self, texts: list[str]) -> list[list[float]]:
        """
        Transform the input texts into feature vectors.

        Args:
            texts (List[str]): A list of input texts to transform.

        Returns:
            List[List[float]]: A list of feature vectors.
        """
        pass

    @abstractmethod
    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        """
        Fit the vectorizer to the input texts and transform them into feature vectors.

        Args:
            texts (List[str]): A list of input texts to fit and transform.

        Returns:
            List[List[float]]: A list of feature vectors.
        """
        pass