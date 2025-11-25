import re
from typing import List, Union

from ..core.classification_interfaces import TokenizeInterface

class SentimentTokenizer(TokenizeInterface):
    """
    Specialized tokenizer for sentiment analysis that preserves emotionally relevant features.
    """
    
    def __init__(self, pattern: str = r'\w+|[!?]+|:\)|:\(|:\||:-\)|:-\(|:-\|', preserve_case=False):
        """
        Initialize the sentiment tokenizer.
        
        Args:
            pattern: Regex pattern for tokenization (default preserves words and emoticons)
            preserve_case: Whether to preserve original case
        """
        self.pattern = pattern
        self.preserve_case = preserve_case
        self.compiled_pattern = re.compile(pattern)
        
        # Emoticon patterns
        self.emoticon_patterns = {
            'positive': re.compile(r':\)|:-\)|:D|:-D|\+1|👍|😊|😃|😄'),
            'negative': re.compile(r':\(|:-\(|:\||:-\||👎|😢|😞|😠'),
            'exclamation': re.compile(r'!+'),
            'question': re.compile(r'\?+')
        }

    def tokenize(self, text: Union[str, List[str]]) -> List[str]:
        """
        Tokenize text using the specified regex pattern with sentiment awareness.
        
        Args:
            text: Input text (string or list of strings)
            
        Returns:
            List of tokens
        """
        try:
            if isinstance(text, str):
                return self._tokenize_string(text)
            elif isinstance(text, list):
                tokens = []
                for t in text:
                    sentence_tokens = self._tokenize_string(t)
                    tokens.extend(sentence_tokens)
                return tokens
            else:
                raise ValueError("Input must be string or list of strings")
        except Exception as e:
            print(f"Error in tokenization: {e}")
            return []

    def _tokenize_string(self, text: str) -> List[str]:
        """
        Tokenize a single string.
        
        Args:
            text: Input string
            
        Returns:
            List of tokens
        """
        # Preprocess for sentiment features
        text = self._preprocess_sentiment_features(text)
        
        # Apply case handling
        if not self.preserve_case:
            text = text.lower()
        
        # Extract tokens using regex
        tokens = self.compiled_pattern.findall(text)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens

    def _preprocess_sentiment_features(self, text: str) -> str:
        """
        Preprocess text to normalize sentiment features.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text with normalized sentiment features
        """
        # Normalize repeated punctuation (e.g., "!!!" -> "!")
        text = re.sub(r'!{2,}', ' EXCLAMATION ', text)
        text = re.sub(r'\?{2,}', ' QUESTION ', text)
        
        # Normalize positive emoticons
        for emoticon in [':)', ':-)', ':D', ':-D']:
            text = text.replace(emoticon, ' POSITIVE_EMOTICON ')
        
        # Normalize negative emoticons  
        for emoticon in [':(', ':-(', ':|', ':-|']:
            text = text.replace(emoticon, ' NEGATIVE_EMOTICON ')
        
        # Normalize elongated words (e.g., "sooooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text

    def get_sentiment_features(self, text: str) -> dict:
        """
        Extract sentiment-specific features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment feature counts
        """
        features = {
            'positive_emoticons': len(self.emoticon_patterns['positive'].findall(text)),
            'negative_emoticons': len(self.emoticon_patterns['negative'].findall(text)),
            'exclamation_marks': len(self.emoticon_patterns['exclamation'].findall(text)),
            'question_marks': len(self.emoticon_patterns['question'].findall(text)),
            'caps_words': len(re.findall(r'\b[A-Z]{2,}\b', text)),
            'elongated_words': len(re.findall(r'\b\w*(.)\1{2,}\w*\b', text))
        }
        
        return features

    def tokenize_with_features(self, text: str) -> tuple:
        """
        Tokenize text and extract sentiment features.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (tokens, sentiment_features)
        """
        tokens = self.tokenize(text)
        features = self.get_sentiment_features(text)
        
        return tokens, features

    def set_pattern(self, new_pattern: str):
        """
        Update the tokenization pattern.
        
        Args:
            new_pattern: New regex pattern to use
        """
        self.pattern = new_pattern
        self.compiled_pattern = re.compile(new_pattern)