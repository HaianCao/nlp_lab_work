from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    """
    A simple tokenizer that use basic algorithm to tokenize text.
    """

    def split_tokens(self, text: str) -> list[str]:
        """
        Split text into tokens based on whitespace and punctuation.

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: A list of tokens.
        """

        punctuations = ".,?!"
        new_text = ""
        
        for char in text:
            if char in punctuations:
                new_text += f" {char} "
            else:
                new_text += char
        tokens = new_text.split()

        return tokens

    def tokenize(self, text: str) -> list[str]:
        """
        A simple tokenizer that do these tasks:
        1. Convert text to lowercase.
        2. Separate punctuation (.,?!).
        3. Split by whitespace.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """

        text = text.lower().replace(r"\s+", " ").strip()
        tokens = self.split_tokens(text)

        return tokens