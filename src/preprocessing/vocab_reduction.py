from typing import List
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string

from ..core.classification_interfaces import PreprocessorInterface

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/averaged_perceptron_tagger')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)

class VocabReduction(PreprocessorInterface):
    """
    Vocabulary reduction preprocessing using stemming, lemmatization, and stopword removal.
    """
    
    def __init__(self, use_stemming=True, use_lemmatization=False, remove_stopwords=True):
        """
        Initialize the vocabulary reduction preprocessor.
        
        Args:
            use_stemming: Whether to apply stemming
            use_lemmatization: Whether to apply lemmatization
            remove_stopwords: Whether to remove stopwords
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords_flag = remove_stopwords
        
        if use_stemming:
            self._stemmer = PorterStemmer()
        if use_lemmatization:
            self._lemmatizer = WordNetLemmatizer()
        if remove_stopwords:
            self._stopwords = set(stopwords.words('english'))
        else:
            self._stopwords = set()

    def stem_sentence(self, sentence: str) -> str:
        """Apply stemming to the sentence using Porter Stemmer."""
        if not self.use_stemming:
            return sentence
        
        try:
            tokens = word_tokenize(sentence.lower())
            stemmed_tokens = [self._stemmer.stem(token) for token in tokens]
            return ' '.join(stemmed_tokens)
        except Exception as e:
            print(f"Error in stemming: {e}")
            return sentence

    def lemmatize_sentence(self, sentence: str) -> str:
        """Apply lemmatization to the sentence using WordNet Lemmatizer."""
        if not self.use_lemmatization:
            return sentence
        
        try:
            tokens = word_tokenize(sentence.lower())
            lemmatized_tokens = [self._lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(lemmatized_tokens)
        except Exception as e:
            print(f"Error in lemmatization: {e}")
            return sentence

    def remove_stopwords(self, sentence: str) -> str:
        """Remove stopwords from the sentence."""
        if not self.remove_stopwords_flag:
            return sentence
        
        try:
            tokens = word_tokenize(sentence.lower())
            filtered_tokens = [
                token for token in tokens 
                if token not in self._stopwords and token not in string.punctuation
            ]
            return ' '.join(filtered_tokens)
        except Exception as e:
            print(f"Error in stopword removal: {e}")
            return sentence

    def remove_punctuation(self, sentence: str) -> str:
        """Remove punctuation from the sentence."""
        try:
            tokens = word_tokenize(sentence)
            filtered_tokens = [
                token for token in tokens 
                if token not in string.punctuation
            ]
            return ' '.join(filtered_tokens)
        except Exception as e:
            print(f"Error in punctuation removal: {e}")
            return sentence

    def preprocess_sentence(self, sentence: str) -> str:
        """
        Apply vocabulary reduction pipeline to a single sentence.
        
        Args:
            sentence: Input sentence to preprocess
            
        Returns:
            Preprocessed sentence
        """
        # Step 1: Remove stopwords first (preserves more context for stemming/lemmatization)
        if self.remove_stopwords_flag:
            sentence = self.remove_stopwords(sentence)
        
        # Step 2: Remove punctuation
        sentence = self.remove_punctuation(sentence)
        
        # Step 3: Apply stemming or lemmatization (not both)
        if self.use_lemmatization:
            sentence = self.lemmatize_sentence(sentence)
        elif self.use_stemming:
            sentence = self.stem_sentence(sentence)
        
        return sentence.strip()

    def preprocess_sentences(self, sentences: List[str]) -> List[str]:
        """
        Apply vocabulary reduction to a list of sentences.
        
        Args:
            sentences: List of input sentences
            
        Returns:
            List of preprocessed sentences
        """
        return [self.preprocess_sentence(sentence) for sentence in sentences]

    def get_vocabulary_stats(self, sentences: List[str]) -> dict:
        """
        Get vocabulary statistics before and after preprocessing.
        
        Args:
            sentences: List of input sentences
            
        Returns:
            Dictionary with vocabulary statistics
        """
        # Before preprocessing
        original_words = []
        for sentence in sentences:
            original_words.extend(word_tokenize(sentence.lower()))
        
        # After preprocessing
        processed_sentences = self.preprocess_sentences(sentences)
        processed_words = []
        for sentence in processed_sentences:
            processed_words.extend(sentence.split())
        
        return {
            'original_vocab_size': len(set(original_words)),
            'original_total_words': len(original_words),
            'processed_vocab_size': len(set(processed_words)),
            'processed_total_words': len(processed_words),
            'vocabulary_reduction_ratio': 1 - (len(set(processed_words)) / max(len(set(original_words)), 1))
        }