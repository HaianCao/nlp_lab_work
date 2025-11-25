from src.config.io import get_data_path, get_pipeline, get_output_log_file, LIMIT_DOCUMENTS
from src.config.output import save_results
from src.core.pipeline_process import ProcessPipeline
from src.utils.timing import TimingManager
from src.metrics.similarity import cosine_similarity
from src.core.find_similar_documents import find_similar_documents

def run_pipeline(spark):
    """
    Run the complete text processing pipeline: data loading, pipeline processing, 
    and output saving.

    Args:
        spark: SparkSession object.
    """
    try:
        # Start total timing
        timing_manager = TimingManager(log_file=get_output_log_file())
        timing_manager.start_total_timing()

        # Read data
        with timing_manager.time_stage("Data loading"):
            data_path = get_data_path()
            print(f"Loading data from: {data_path}")
            df = spark.read.json(str(data_path))
            if LIMIT_DOCUMENTS > 0 and LIMIT_DOCUMENTS < df.count():
                df = df.limit(LIMIT_DOCUMENTS)
            print("Data loaded successfully.")

            # Show basic info
            print(f"Total records: {df.count()}")
            df.printSchema()
            df.show(3, truncate=True)

        # Run pipeline
        pipeline_process = ProcessPipeline(get_pipeline(), timing_manager)
        df_processed = pipeline_process.run_pipeline(df)

        # Save results
        with timing_manager.time_stage("Output saving"):
            print("Saving results...")
            save_results(df_processed)
            print("Results saved successfully.")

        # Find similar documents
        with timing_manager.time_stage("Find similar documents"):
            find_similar_documents(df=df_processed)
            print("Similar documents found and saved successfully.")

        # Kết thúc đo thời gian và in summary
        timing_manager.end_total_timing()
        print("Pipeline completed!")
        timing_manager.print_detailed_summary()
        timing_manager.save_timing_log(
                additional_info={
                    "Total Records": df.count()
                })
    except Exception as e:
        print(f"Error: {e}")
        timing_manager._log_message(f"❌ Pipeline failed with error: {e}")
    finally:
        # Safely stop Spark session
        try:
            if spark is not None:
                spark.stop()
                print("✅ Spark session stopped successfully.")
        except Exception as stop_error:
            print(f"⚠️ Warning: Error while stopping Spark session: {stop_error}")