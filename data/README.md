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

## en_ewt-ud-train.conllu (CoNLL-U)

- **Format**: CoNLL-U (tab-separated columns), one token per row; sentences separated by a blank line.
- **Columns (10 fields)**:
  - `ID`: Integer token index in sentence, or range for multiword tokens (e.g. `1`, `1-2`).
  - `FORM`: Word form or punctuation symbol (string).
  - `LEMMA`: Lemma or `_` if not available (string).
  - `UPOS`: Universal POS tag (e.g. `NOUN`, `VERB`) (string).
  - `XPOS`: Language-specific POS tag or `_` (string).
  - `FEATS`: Morphological features (format `Key=Value|...`) or `_` if none.
  - `HEAD`: Head of the current token (integer index) or `0` for root.
  - `DEPREL`: Dependency relation to `HEAD` (string).
  - `DEPS`: Enhanced dependencies or `_`.
  - `MISC`: Miscellaneous annotations or `_`.
- **Comments & Metadata**: Lines beginning with `#` are comments and typically include sentence metadata such as `# sent_id = ...` and `# text = ...`.
- **Usage**: Standard format for POS tagging and dependency parsing. Parsers and evaluation tools expect these columns and blank-line separated sentences.
- **Example rows**:

```
# sent_id = 1
# text = The quick brown fox jumps over the lazy dog.
1	The	the	DET	DT	_	2	det	_	_
2	quick	quick	ADJ	JJ	_	4	amod	_	_
3	brown	brown	ADJ	JJ	_	4	amod	_	_
4	fox	fox	NOUN	NN	_	5	nsubj	_	_
5	jumps	jump	VERB	VBZ	_	0	root	_	_
```

## en_ewt-ud-dev.conllu (CoNLL-U)

- **Same format as train**: CoNLL-U with the same 10 columns, `#` comments, and sentence-separated-by-blank-line convention.
- **Purpose**: Development/validation set used to select hyperparameters and checkpoints.

## en_ewt-ud-test.conllu (CoNLL-U)

- **Same format as train/dev**: CoNLL-U with the same 10 columns and conventions.
- **Purpose**: Final evaluation set for reporting model performance.

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
