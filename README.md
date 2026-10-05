# Domain-Specific Text Analysis and Retrieval: Consumer Mental-Health Guidance

AS1, Natural Language Processing (DATA403), Vidyashilp University
Team: Tanmayee K M, Ahmed, Jeromi

A complete NLP and information-retrieval system over 17 public mental-health guidance documents (NICE, WHO, NIMH, NHS).
Each preprocessing step was compared experimentally on the corpus, then selected and ordered into a **Final Pipeline**:
number policy → custom tokenizer → lowercase → punctuation removal → spaCy lemmas in context → custom stop words
(negation and modal verbs kept). It is followed by a positional inverted index with keyword (TF-IDF), phrase and Boolean search,
an evaluation with relevance judgments, and a Gradio GUI.

**Report:** [Report.md](Report.md) / [Report.pdf](Report.pdf)

## Repository layout
| Path | Content |
|---|---|
| `Domain_Text_Analysis_Retrieval.ipynb` | Modules 1–6 (all exercises, experiments, decisions) |
| `app.py` | Gradio GUI |
| `src/nlp_pipeline.py` | Final Pipeline + domain NER, packaged for the GUI |
| `src/build_corpus.py` | Corpus pipeline: register → extract → clean → section → QC → build → validate |
| `results/` | All result tables as CSV (Tables A–J), `inverted_index.json`, gold sets, queries, relevance judgments |
| `data/metadata/` | Source manifest (URLs, licences, dates), QC report, cleaning and extraction logs, coverage, exclusions |
| `Report.md`, `Report.pdf` | Project report |

### Results tables
| Table | File |
|---|---|
| A. Document & preprocessing results | `results/preprocessing_results.csv`, `results/document_statistics.csv` |
| B. Tokenization comparison | `results/tokenization_comparison.csv` |
| C. Stemming vs lemmatization | `results/stem_vs_lemma.csv` |
| D. POS tagging | `results/pos_tagging_results.csv` |
| E. NER | `results/ner_results.csv` (totals), `results/ner_entities.csv` (per entity) |
| F. N-grams | `results/unigram_results.csv`, `bigram_results.csv`, `trigram_results.csv`, `ngram_results.csv` |
| G. BPE | `results/bpe_results.csv` |
| H. Pipeline comparison | `results/pipeline_comparison.csv` |
| I. Retrieval results | `results/retrieval_results.csv`, `results/inverted_index.json` |
| J. Overall performance | `results/evaluation_results.csv`, `results/evaluation_per_query.csv` |

## Source documents are not included
The guidance documents are licence-restricted (NICE: © NICE, academic use only, no redistribution), so **raw files and
extracted corpus text are not in this repository** (see `.gitignore`). `data/metadata/source_manifest.csv` lists every
document's official URL, licence and the sections used. To rebuild the corpus:

1. Download each document from its URL in the manifest and save it as `data/raw/pdf/Dnn__<name>.pdf` or
   `data/raw/html/Dnn__<name>.html` (D24 and D25 are page ranges of the WHO mhGAP guideline PDF, as given in the manifest).
2. Build the corpus:
   ```
   python src/build_corpus.py register
   python src/build_corpus.py extract
   python src/build_corpus.py clean
   python src/build_corpus.py section
   python src/build_corpus.py qc
   python src/build_corpus.py build --version 1.0
   python src/build_corpus.py validate --stage final
   ```
   This writes `data/processed/corpus.json`.

## Run
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
- **Notebook:** open `Domain_Text_Analysis_Retrieval.ipynb` and run all cells (all randomness uses SEED = 134).
  Every table is saved to `results/`.
- **GUI:** `python app.py`, then open the local URL it prints. It needs `data/processed/corpus.json` and `results/`.

## Corpus rules
- Raw files are never edited; every later stage can be regenerated.
- Cleaning never lowercases or removes stop words, numbers or punctuation (those are AS1 experiments).
- Document IDs are permanent and never reused.
