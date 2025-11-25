from src.core.pipeline import run_pipeline

from pyspark.sql import SparkSession
    
def main():
    spark = SparkSession.builder \
        .appName("NLPLab2") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .master("local[*]") \
        .getOrCreate()

    run_pipeline(spark)
    spark.stop()

if __name__ == "__main__":
    main()