import re

from ..core.classification_interfaces import PreprocessorInterface

class NoiseFiltering(PreprocessorInterface):
    """
    Text preprocessing class for removing noise and unwanted characters.
    """
    
    def __init__(self):
        """Initialize the noise filtering preprocessor."""
        pass
    
    def lowercase(self, sentence: str) -> str:
        """Convert sentence to lowercase."""
        return sentence.lower()

    def remove_urls(self, sentence: str) -> str:
        """Remove URLs from sentence."""
        return re.sub(r'http\S+|www\S+|https\S+', '', sentence, flags=re.MULTILINE)

    def remove_html_tags(self, sentence: str) -> str:
        """Remove HTML tags from sentence."""
        return re.sub(r'<.*?>', '', sentence)

    def remove_special_characters(self, sentence: str) -> str:
        """Remove special characters from sentence."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', sentence)

    def remove_extra_whitespace(self, sentence: str) -> str:
        """Remove extra whitespace from sentence."""
        return re.sub(r'\s+', ' ', sentence).strip()

    def remove_numbers(self, sentence: str) -> str:
        """Remove standalone numbers from sentence."""
        return re.sub(r'\b\d+\b', '', sentence)

    def preprocess_sentence(self, sentence: str) -> str:
        """
        Clean sentence by removing noise and special characters.
        
        Args:
            sentence: Input sentence to clean
            
        Returns:
            Cleaned sentence
        """
        sentence = self.lowercase(sentence)
        sentence = self.remove_urls(sentence)
        sentence = self.remove_html_tags(sentence)
        sentence = self.remove_special_characters(sentence)
        sentence = self.remove_extra_whitespace(sentence)
        return sentence

    def preprocess_sentences(self, sentences: list) -> list:
        """
        Apply cleaning to all sentences.
        
        Args:
            sentences: List of input sentences
            
        Returns:
            List of cleaned sentences
        """
        return [self.preprocess_sentence(sentence) for sentence in sentences]

    def aggressive_cleaning(self, sentence: str) -> str:
        """
        More aggressive cleaning including number removal.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Aggressively cleaned sentence
        """
        sentence = self.preprocess_sentence(sentence)
        sentence = self.remove_numbers(sentence)
        sentence = self.remove_extra_whitespace(sentence)
        return sentence

    def light_cleaning(self, sentence: str) -> str:
        """
        Light cleaning that preserves more content.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Lightly cleaned sentence
        """
        sentence = self.lowercase(sentence)
        sentence = self.remove_urls(sentence)
        sentence = self.remove_html_tags(sentence)
        sentence = self.remove_extra_whitespace(sentence)
        return sentence