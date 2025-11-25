import re

class EmbeddingTokenizer():
    """
    Tokenizer for word embeddings - specifically designed for lab3/lab4
    This is separate from the basic tokenizers in lab1 to avoid conflicts
    """
    def __init__(self):
        self.TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text: str, pattern=None) -> list[str]:
        """
        Tokenize text using regular expressions.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        if pattern is None:
            pattern = self.TOKEN_PATTERN
        
        text = text.lower()
        return pattern.findall(text)