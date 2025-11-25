from src.utils.timing import TimingManager
from src.interface.pipeline_interface import PipelineInterface

from pyspark.sql import DataFrame, SparkSession
from typing import Tuple, Optional

class ProcessPipeline:
    def __init__(self, steps: list[PipelineInterface], timing_manager: Optional[TimingManager] = None):
        """
        Initialize the ProcessPipeline with a list of pipeline steps and a timing manager.

        Args:
            steps: List of instantiated pipeline steps to run sequentially.
            timing_manager: TimingManager instance to track execution time of each step.
        """
        self.steps = steps
        self.timing_manager = timing_manager
        self.init_time = False

    def run_step(self, df: DataFrame, step: PipelineInterface) -> DataFrame:
        """
        Run a single step in the pipeline.

        Args:
            df: Input Spark DataFrame.
            step: The pipeline step to instantiate and run.

        Returns:
            DataFrame: Transformed Spark DataFrame.
        """
        with self.timing_manager.time_stage(step.get_name()):
            print(f"Running: {step.get_name()}")
            df = step.transform(df)
            print(f"Records after {step.get_name()}: {df.count()}")
        return df

    def run_pipeline(self, df: DataFrame) -> DataFrame:
        """
        Run the full preprocessing and vectorization pipeline.

        Args:
            df: Input Spark DataFrame.

        Returns:
            Tuple[DataFrame, str]: Transformed DataFrame and output column name after full processing.
        """
        if self.timing_manager is None:
            self.timing_manager = TimingManager()
            self.timing_manager.start_total_timing()
            self.init_time = True

        for step_idx in range(len(self.steps)):
            df = self.run_step(df, self.steps[step_idx])

        if self.init_time:
            self.timing_manager.end_total_timing()
            self.init_time = False
        return df

__ALL__ = ['ProcessPipeline']