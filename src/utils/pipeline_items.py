from src.interface.pipeline_interface import PipelineInterface

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, \
                                HashingTF, IDF, Normalizer
from pyspark.sql import DataFrame

class RegexTokenizerStep(PipelineInterface):
    def __init__(self,
                 inputCol: str, outputCol: str,
                 args: dict):
        self.regex_tokenizer = RegexTokenizer(inputCol=inputCol, outputCol=outputCol, **args)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply RegexTokenizer to tokenize text.

        Args:
            df: Input Spark DataFrame.

        Returns:
            DataFrame: Transformed DataFrame after tokenization.
        """
        df = self.regex_tokenizer.transform(df)
        return df

    def get_name(self) -> str:
        return "RegexTokenizer"

class StopWordsRemoverStep(PipelineInterface):
    def __init__(self,
                 inputCol: str, outputCol: str,
                 args: dict):
        self.stopwords_remover = StopWordsRemover(inputCol=inputCol, outputCol=outputCol, **args)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply StopWordsRemover to remove stop words from text.

        Args:
            df: Input Spark DataFrame.

        Returns:
            DataFrame: Transformed DataFrame after stop words removal.
        """
        df = self.stopwords_remover.transform(df)
        return df

    def get_name(self) -> str:
        return "StopWordsRemover"

class HashingTFStep(PipelineInterface):
    def __init__(self,
                 inputCol: str, outputCol: str,
                 args: dict):
        self.hashing_tf = HashingTF(inputCol=inputCol, outputCol=outputCol, **args)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply HashingTF to transform text into term frequency vectors.

        Args:
            df: Input Spark DataFrame.

        Returns:
            DataFrame: Transformed DataFrame after HashingTF.
        """
        df = self.hashing_tf.transform(df)
        return df

    def get_name(self) -> str:
        return "HashingTF"

class IDFStep(PipelineInterface):
    def __init__(self,
                 inputCol: str, outputCol: str,
                 args: dict):
        self.idf = IDF(inputCol=inputCol, outputCol=outputCol, **args)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply IDF to transform term frequency vectors into inverse document frequency vectors.

        Args:
            df: Input Spark DataFrame.

        Returns:
            DataFrame: Transformed DataFrame after IDF.
        """
        df = self.idf.fit(df).transform(df)
        return df

    def get_name(self) -> str:
        return "IDF"

class NormalizerStep(PipelineInterface):
    def __init__(self,
                 inputCol: str, outputCol: str,
                 args: dict):
        self.normalizer = Normalizer(inputCol=inputCol, outputCol=outputCol, **args)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply Normalizer to normalize feature vectors.

        Args:
            df: Input Spark DataFrame.

        Returns:
            DataFrame: Transformed DataFrame after normalization.
        """
        df = self.normalizer.transform(df)
        return df

    def get_name(self) -> str:
        return "Normalizer"

__all__ = ['RegexTokenizerStep', 
           'StopWordsRemoverStep', 
           'HashingTFStep', 
           'IDFStep', 
           'NormalizerStep']