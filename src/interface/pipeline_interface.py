from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import Tuple

class PipelineInterface(ABC):
    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Process the input DataFrame and return the transformed DataFrame.

        Args:
            df: Input Spark DataFrame.

        Returns:
            DataFrame: Transformed DataFrame after processing.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the pipeline step.

        Returns:
            str: Name of the pipeline step.
        """
        pass

__all__ = ['PipelineInterface']