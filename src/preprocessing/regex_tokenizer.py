import re
from src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    """
    A tokenizer that uses regular expressions to tokenize text.
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