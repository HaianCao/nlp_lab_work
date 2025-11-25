from pathlib import Path

# Lab2 imports
from src.utils.pipeline_items import RegexTokenizerStep, StopWordsRemoverStep, \
                                HashingTFStep, IDFStep, NormalizerStep

# === Common Configuration ===
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

# === Lab1 Configuration ===
def get_base_dir() -> Path:
    """
    Get the base directory of the project.

    Returns:
        Path: The base directory of the project.
    """
    return ROOT_DIR

def get_data_dir() -> Path:
    """
    Get the data directory of the project.

    Returns:
        Path: The data directory of the project.
    """
    return DATA_DIR

def get_train_data_path() -> Path:
    """
    Get the path to the training data file.

    Returns:
        Path: The path to the training data file.
    """
    return DATA_DIR / "en_ewt-ud-train.txt"

def get_test_data_path() -> Path:
    """
    Get the path to the test data file.

    Returns:
        Path: The path to the test data file.
    """
    return DATA_DIR / "en_ewt-ud-test.txt"

def get_dev_data_path() -> Path:
    """
    Get the path to the development data file.

    Returns:
        Path: The path to the development data file.
    """
    return DATA_DIR / "en_ewt-ud-dev.txt"

# === Lab2 Configuration ===
# Basic configuration parameters
DATA_FILE = "c4-train.00000-of-01024-30K.json.gz"
OUTPUT_DIR = "results"
OUTPUT_FILE = "lab17_pipeline_output.txt"
LOG_FILE = "timing_log.txt"
LIMIT_DOCUMENTS = -1
NUM_FEATURES = 20000
TOP_K_SIMILAR = 5
MAX_DOCUMENTS_PROCESSED = 5000 # To limit memory usage during similarity search

INPUT_RAW = "text"
OUTPUT_FEATURES = "features"

def get_pipeline():
    """
    Initialize and return the list of processing steps in the pipeline.

    Returns:
        List: List of processing steps.
    """
    REGEX_TOKENIZER = RegexTokenizerStep(INPUT_RAW, "tokens", {"pattern": r"\W+"})
    STOPWORDS_REMOVER = StopWordsRemoverStep("tokens", "stopwords_removed", {})
    HASHING_TF = HashingTFStep("stopwords_removed", "hashing_tf", {"numFeatures": NUM_FEATURES})
    IDF = IDFStep("hashing_tf", "idf", {})
    NORMALIZER = NormalizerStep("idf", OUTPUT_FEATURES, {"p": 2.0})
    return [
        REGEX_TOKENIZER,
        STOPWORDS_REMOVER,
        HASHING_TF,
        IDF,
        NORMALIZER
    ]

def get_data_path() -> Path:
    """
    Get the path to the training data file.

    Returns:
        Path: The path to the training data file.
    """
    return DATA_DIR / DATA_FILE

def get_output_dir() -> Path:
    """
    Get the output directory for processed data.

    Returns:
        Path: The output directory for processed data.
    """
    return DATA_DIR / OUTPUT_DIR

def get_output_path() -> Path:
    """
    Get the output file path for processed data.

    Returns:
        Path: The output file path for processed data.
    """
    return get_output_dir() / OUTPUT_FILE

def get_output_log_file() -> Path:
    """
    Get the output log file path for timing logs.

    Returns:
        Path: The output log file path for timing logs.
    """
    return get_output_dir() / LOG_FILE