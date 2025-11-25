import logging
import os
import subprocess
import numpy as np
import gensim
from gensim.models import Word2Vec, KeyedVectors

logging.basicConfig(level=logging.INFO)

class WordEmbedder:
    """
    A class for handling word embeddings using pre-trained GloVe models or training custom Word2Vec models.
    """

    def __init__(self):
        """
        Initialize the WordEmbedder.
        """
        self.model = None
        self.vocab = None

    def load_glove(self, file_path: str) -> None:
        """
        Load a pre-trained GloVe model from a file.

        Args:
            file_path (str): Path to the GloVe file.
        """
        try:
            # Load using gensim's built-in GloVe loader
            self.model = KeyedVectors.load_word2vec_format(file_path, binary=False, no_header=True)
            self.vocab = set(self.model.key_to_index.keys())
            logging.info(f"GloVe model loaded from {file_path}")
            logging.info(f"Vocabulary size: {len(self.vocab)}")
        except Exception as e:
            logging.error(f"Error loading GloVe model: {e}")

    def download_glove(self, dim: int = 50) -> str:
        """
        Download GloVe embeddings.

        Args:
            dim (int): Dimension of the embeddings (50, 100, 200, or 300).

        Returns:
            str: Path to the downloaded GloVe file.
        """
        if dim not in [50, 100, 200, 300]:
            raise ValueError("Dimension must be one of: 50, 100, 200, 300")
        
        # Create glove directory in data folder
        glove_dir = "data/glove"
        os.makedirs(glove_dir, exist_ok=True)
        
        filename = f"glove.6B.{dim}d.txt"
        file_path = os.path.join(glove_dir, filename)
        
        if os.path.exists(file_path):
            logging.info(f"GloVe file already exists: {file_path}")
            return file_path
        
        # Download GloVe embeddings
        url = f"https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(glove_dir, "glove.6B.zip")
        
        logging.info(f"Downloading GloVe embeddings from {url}")
        
        # Use curl or wget to download
        try:
            subprocess.run(["curl", "-o", zip_path, url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(["wget", "-O", zip_path, url], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("Unable to download GloVe embeddings. Please install curl or wget.")
        
        # Extract the file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(filename, glove_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        
        logging.info(f"GloVe embeddings downloaded and extracted to {file_path}")
        return file_path

    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get the embedding for a word.

        Args:
            word (str): The word to get the embedding for.

        Returns:
            np.ndarray: The embedding vector, or None if the word is not found.
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            return self.model[word]
        except KeyError:
            logging.warning(f"Word '{word}' not found in vocabulary.")
            return None

    def get_similar_words(self, word: str, top_n: int = 10) -> list:
        """
        Get the most similar words to a given word.

        Args:
            word (str): The target word.
            top_n (int): Number of similar words to return.

        Returns:
            list: A list of tuples (word, similarity_score).
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            return self.model.most_similar(word, topn=top_n)
        except KeyError:
            logging.warning(f"Word '{word}' not found in vocabulary.")
            return []

    def word_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate the cosine similarity between two words.

        Args:
            word1 (str): First word.
            word2 (str): Second word.

        Returns:
            float: Cosine similarity score, or None if either word is not found.
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            return self.model.similarity(word1, word2)
        except KeyError as e:
            logging.warning(f"One or both words not found in vocabulary: {e}")
            return None

    def train_word2vec(self, sentences: list, vector_size: int = 100, window: int = 5, 
                       min_count: int = 1, workers: int = 4, epochs: int = 10) -> None:
        """
        Train a Word2Vec model on the provided sentences.

        Args:
            sentences (list): List of tokenized sentences.
            vector_size (int): Dimensionality of the word vectors.
            window (int): Maximum distance between current and predicted word.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Number of worker threads.
            epochs (int): Number of training epochs.
        """
        logging.info("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs
        )
        self.vocab = set(self.model.wv.key_to_index.keys())
        logging.info(f"Word2Vec model trained. Vocabulary size: {len(self.vocab)}")

    def save_model(self, file_path: str) -> None:
        """
        Save the current model to a file.

        Args:
            file_path (str): Path where the model will be saved.
        """
        if self.model is None:
            raise ValueError("No model to save. Please load or train a model first.")
        
        if isinstance(self.model, Word2Vec):
            self.model.save(file_path)
        else:
            self.model.save_word2vec_format(file_path)
        
        logging.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """
        Load a saved Word2Vec model.

        Args:
            file_path (str): Path to the saved model.
        """
        try:
            self.model = Word2Vec.load(file_path)
            self.vocab = set(self.model.wv.key_to_index.keys())
            logging.info(f"Word2Vec model loaded from {file_path}")
            logging.info(f"Vocabulary size: {len(self.vocab)}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def document_vector(self, words: list) -> np.ndarray:
        """
        Compute document vector as the average of word vectors.

        Args:
            words (list): List of words in the document.

        Returns:
            np.ndarray: Document vector.
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        vectors = []
        for word in words:
            if word in self.vocab:
                if isinstance(self.model, Word2Vec):
                    vectors.append(self.model.wv[word])
                else:
                    vectors.append(self.model[word])
        
        if not vectors:
            logging.warning("No valid words found in document.")
            # Return zero vector with same dimension as embeddings
            if isinstance(self.model, Word2Vec):
                return np.zeros(self.model.wv.vector_size)
            else:
                return np.zeros(self.model.vector_size)
        
        return np.mean(vectors, axis=0)