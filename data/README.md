# Data Files Description

## en_ewt-ud-train.txt

- **Format**: Plain text (unstructured)
- **Content**: Free-form paragraphs
- **Type**: string
- **Lines**: 16,689

## en_ewt-ud-dev.txt

- **Format**: Plain text (unstructured)
- **Content**: Free-form paragraphs
- **Type**: string

## en_ewt-ud-test.txt

- **Format**: Plain text (unstructured)
- **Content**: Free-form paragraphs
- **Type**: string

## c4-train.00000-of-01024-30K.json.gz

- **Format**: Compressed JSON (gzip)
- **text**: string (document content)
- **timestamp**: string (ISO datetime)
- **url**: string (source URL)
- **Records**: ~30,000 documents

## hwu.tar.gz

- **Format**: Compressed tarball
- **Contents**: HWU64 intent classification dataset
- **train.csv**:
  - **text**: string (user utterance)
  - **category**: string (intent label)
- **val.csv**:
  - **text**: string (user utterance)
  - **category**: string (intent label)
- **test.csv**:
  - **text**: string (user utterance)
  - **category**: string (intent label)
