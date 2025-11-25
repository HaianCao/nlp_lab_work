from pathlib import Path

def load_raw_text_data(file_path: Path) -> list[str]:
    """
    Read documents from a text file, one document per line.

    Args:
        file_path (Path): The path to the text file.

    Returns:
        List[str]: A list of documents read from the file.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        documents = [line.strip() for line in file if line.strip()]

    return " ".join(documents)