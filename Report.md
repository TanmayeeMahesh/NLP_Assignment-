# Domain-Specific Text Analysis and Retrieval System
## Consumer Mental-Health Guidance Corpus

**Course:** Natural Language Processing (DATA403), Vidyashilp University
**Assignment:** AS1: Domain-Specific Text Analysis and Retrieval System
**Team:** Tanmayee K M · Ahmed · Jeromi
**Deliverables:** `Domain_Text_Analysis_Retrieval.ipynb` (all modules), `results/` (CSV checkpoints, Tables A–J, `inverted_index.json`), `app.py` + `src/nlp_pipeline.py` (Gradio GUI), `src/build_corpus.py` (corpus pipeline)

---

## Abstract

We built a complete text-analysis and retrieval system for **public mental-health guidance**: 17 documents from NICE, WHO, NIMH and NHS (41,297 words), covering depression, anxiety, panic disorder, PTSD, self-harm and suicide, medication, sleep, loneliness, stress and substance use. Every NLP step was first compared experimentally on this corpus: tokenizers, sentence splitters, BPE, stop lists, stemmers, lemmatizers, POS taggers and NER. The step was then **selected and ordered** for the domain. The resulting **Final Pipeline** is: number policy → custom regex tokenizer → lowercase → punctuation removal → spaCy lemmatization in context (with domain rules) → custom stop-word removal that keeps negation and modal verbs. It reduces the vocabulary by 46% while keeping safety-critical meaning ("do **not** stop taking" stays *not stop take*). A positional inverted index supports keyword (TF-IDF ranked), phrase and Boolean (AND/OR/NOT, brackets) search. On 14 domain queries with relevance judgments fixed **before** retrieval, the Final Pipeline reaches macro precision 0.43, recall 0.94, F1 0.53 and P@3 0.52. That matches the best alternative pipeline and beats the stemming pipeline, while producing a readable, smaller index with far more meaningful n-grams. A Gradio GUI demonstrates every stage live.

---

## 1. Domain and motivation

People looking for mental-health information read guidance written for two audiences: **clinical guidelines** (NICE, WHO mhGAP) and **public information** (NIMH, NHS, WHO fact sheets). This text has properties that general-purpose NLP handles badly, and errors here can matter:

| Domain property | Example | Why default NLP fails |
|---|---|---|
| Negation carries the advice | "Do **not** stop taking an antidepressant suddenly" | Standard stop lists delete *not*, which reverses the advice |
| Modal strength | "should not be used", "may", "must" | Removed as stop words; the strength of a recommendation is lost |
| Hyphenated and coded terms | self-harm, 12‑week, DSM-IV, ICD-11, 1-800-662-4357 | Split into fragments by default tokenizers |
| Acronyms and plural acronyms | GAD, PTSD, CBT, SSRIs | Tagged as organisations; *SSRIs* not merged with *SSRI* |
| Drug and therapy names | sertraline, naltrexone, psychotherapy | No NER label in general models |
| Scaffolding numbers | [2004, amended 2020], 1.2.30, p-values | Noise terms in the index |
| PDF layout artefacts | running headers/footers, bullets, broken lines | Fake tokens and fake n-grams |

The system design follows the AS1 sequence: **Select → Order → Implement → Compare → Evaluate → Justify**.

---

## 2. Corpus

### 2.1 Documents

| ID | Title | Organisation | Layer | Audience | Format | Words |
|---|---|---|---|---|---|---|
| D01 | Depression in adults: treatment and management (sections) | NICE (UK) | A | professional | PDF | 4,278 |
| D02 | Generalised anxiety disorder and panic disorder in adults: management (sections) | NICE (UK) | A | professional | PDF | 3,029 |
| D03 | Self-harm: assessment, management and preventing recurrence (sections) | NICE (UK) | A | professional | PDF | 1,582 |
| D04 | mhGAP guideline, 3rd ed.: depression and anxiety modules (pp. 50–59, 90–95) | WHO | A | professional | PDF | 6,813 |
| D06 | Depressive disorder (depression) fact sheet | WHO | B | public | HTML | 1,197 |
| D07 | Depression | NIMH (US) | B | public | PDF | 2,601 |
| D08 | Generalized Anxiety Disorder: What You Need to Know | NIMH (US) | B | public | PDF | 1,961 |
| D09 | Post-Traumatic Stress Disorder | NIMH (US) | B | public | PDF | 2,162 |
| D10 | Eating Disorders: What You Need to Know | NIMH (US) | B | public | PDF | 1,497 |
| D11 | Frequently Asked Questions About Suicide | NIMH (US) | B | public | PDF | 1,827 |
| D12 | Raising low self-esteem | NHS (UK) | B | public | HTML | 967 |
| D13 | Get help with loneliness | NHS (UK) | B | public | HTML | 892 |
| D14 | Antidepressants (incl. stopping or coming off) | NHS (UK) | B | public | HTML | 1,441 |
| D16 | I'm So Stressed Out! Fact Sheet | NIMH (US) | B | public | PDF | 890 |
| D17 | Sleep problems (Every Mind Matters) | NHS (UK) | B | public | HTML | 1,154 |
| D24 | mhGAP guideline 2023: Alcohol use disorders module (pp. 44–49) | WHO | A | professional | PDF | 2,686 |
| D25 | mhGAP guideline 2023: Drug use disorders module (pp. 96–107) | WHO | A | professional | PDF | 6,320 |
| | **Total** | | | | | **41,297** |

*Layer A* = clinical guidelines; *Layer B* = public information. **Licences** are recorded per document in `data/metadata/source_manifest.csv`: NIMH is US public domain; NHS is the Open Government Licence; WHO is CC BY-NC-SA 3.0 IGO (guidelines) or WHO copyright with non-commercial reuse (fact sheet); NICE is © NICE, used for academic purposes and not redistributed.

**Gaps (recorded, not hidden):** a planned *Layer C* (Indian helpline information, Tele-MANAS, D20) is empty, because the only available file was an image-only PDF that failed quality control. Two planned documents (D05, D19) could not be obtained. D24 and D25 were taken from the larger WHO mhGAP PDF to cover substance-use topics the other documents did not. The corpus contains **no medication doses** (Exercise 6 found none), so dose handling could not be tested.

### 2.2 Corpus construction pipeline (`src/build_corpus.py`)

`register → extract → clean → section → QC → build → validate`

- **Extraction:** PyMuPDF for PDFs (page-aware), BeautifulSoup for HTML (navigation, scripts, video players and photo credits removed).
- **Cleaning (ordered rules):** remove repeated header/footer lines (the top and bottom 5 lines of each page are compared across pages), normalise bullets to one marker, unwrap PDF line breaks (never joining a bullet item to the previous line), remove boilerplate repeated on ≥ 40% of pages, normalise whitespace. Case, punctuation, stop words and numbers are **kept**, so the NLP experiments see real text.
- **Sectioning:** only the guideline sections relevant to the domain (`pages:` / `toc:` selectors in the manifest).
- **Quality control:** thresholds for length, non-alphabetic ratio, dictionary-word ratio, empty pages, URL and citation density, and near-duplicates (5-word shingles). D20 was rejected (image-only).
- **Output:** `data/processed/corpus.json` (text + metadata for every document); final validation reported 0 errors.

**Cleaning defects found through statistics and fixed (with dry runs before applying):**
1. The NICE 5-line page footer survived a 3-line header/footer zone, so the zone was widened to 5 lines.
2. Embedded video-player text and photo credits in NHS/WHO HTML were removed.
3. Bullet items were merged into run-on lines, and WHO's symbol-font bullet appeared as a stray "y". Both are now handled with a bullet marker kept until unwrapping.
4. A WHO running header on alternate pages slipped under the 50% boilerplate threshold, which was lowered to 40%.

A remaining minor defect is a 3-page print header in D16 (below the 4-page minimum for boilerplate detection).

---

## 3. Module 1: Document statistics

Per-document sentences, tokens, characters and vocabulary are in `results/document_statistics.csv`. On whitespace tokens, the corpus has **2,331 sentences, 39,961 tokens, 271,378 characters and a vocabulary of 5,421**. The two longest documents are D04 and D25, both WHO guideline excerpts.

### Table A: Document and preprocessing results (`preprocessing_results.csv`)

| Measure | Before processing | After processing (Final Pipeline) | Change |
|---|---|---|---|
| Documents | 17 | 17 | 0% |
| Tokens | 39,961 | 24,913 | −37.7% |
| Unique tokens | 5,976 | 2,919 | −51.2% |
| Vocabulary size | 5,421 | 2,919 | −46.2% |
| Average tokens per document | 2,350.6 | 1,465.5 | −37.7% |

*"Before"* = whitespace tokens of the cleaned text. *"After"* = terms produced by the Final Pipeline (Section 5). Vocabulary roughly halves, while negation, modal verbs and domain terms are kept.

---

## 4. Module 2: Exercises 1–14

Each exercise compared the methods on this corpus, measured the results, and ended in a decision. Labels: **(A)** required by AS1; **(B)** adapted to the domain; **(C)** our own design choice.

### Exercise 1: Dictionary and term dictionary

| Tokenizer | Tokens | Dictionary size | Term dictionary size | Reduction |
|---|---|---|---|---|
| `split()` | 39,961 | 5,976 | 3,804 | 36.3% |
| NLTK `word_tokenize` | 46,218 | 4,356 | 3,743 | 14.1% |
| spaCy tokenizer | 47,429 | 4,254 | 3,654 | 14.1% |

`split()` inflates the dictionary with punctuation attached to words (2,170 entries such as "depression," and "(CBT)"). The *term dictionary* (lowercased, punctuation stripped) is the meaningful size. The categories of dictionary entries (plain words, numbers, hyphenated words, URLs, case duplicates) are in `ex2_dictionary_categories.csv`.

### Exercise 2: Sentence splitting

| Splitter | Units | Avg words | Long units (> 60 words) | Fragments (< 3 words) | Time (s) |
|---|---|---|---|---|---|
| `split(".")` | 2,340 | 17.3 | 52 | 350 | 0.0 |
| NLTK `sent_tokenize` | 1,943 | 20.6 | 62 | 33 | 0.0 |
| spaCy sentencizer | 1,975 | 20.3 | 53 | 35 | 0.7 |
| spaCy `en_core_web_sm` | 1,902 | 21.0 | 60 | 19 | 18.8 |
| **LineTokenizer + sent_tokenize** | 3,337 | 12.0 | **5** | 601 | 0.1 |

Guidance text is full of headings and bullet lists without final full stops. Sentence-only splitters merge them into long run-on "sentences". **Decision (B):** split into lines first, then sentences. Run-on units fall from 62 to 5. The extra "fragments" are real headings and bullet items, which is the correct unit for n-grams, POS tagging and NER.

### Exercise 3: Tokenization (Table B)

We listed ten domain tokenization problems (P1–P10) and measured how many corpus occurrences each tokenizer keeps intact.

| Problem | Example | NLTK | spaCy | **Custom** | Hybrid |
|---|---|---|---|---|---|
| P1 hyphenated compound | self-harm | 96.2% | 0% | **98.0%** | 98.0% |
| P3 numeric range | 6–12 | 0% | 100% | **100%** | 100% |
| P5 classification code | DSM-IV, ICD-11 | 91.9% | 87.1% | **100%** | 100% |
| P6 URL | https://findtreatment.gov | 68.3% | 95.2% | **100%** | 95.2% |
| P7 helpline number | 1-800-662-4357 | 100% | 0% | **100%** | 100% |
| P9 slash term | discontinuation/withdrawal | 87.5% | 44.4% | **93.1%** | 93.1% |
| P10 abbreviation | e.g. | 0% | 97.6% | **100%** | 97.6% |
| Negation | doesn't → *does not* | n't kept (14) | n't kept (14) | **not (0 n't)** | not |
| Time (s) | | 0.34 | 0.66 | **0.18** | 0.60 |

Table B (`tokenization_comparison.csv`) shows one real corpus sentence for each problem under all four tokenizers. Examples: spaCy splits *self-harm* into *self - harm* and the helpline number into seven pieces; NLTK splits *e.g.* and the URL; the custom tokenizer gets all ten right.

**Custom tokenizer (C):** a pre-step normalises the non-breaking hyphen (U+2011) and expands negative contractions (*can't → can not*, *n't → not*). Then an NLTK `RegexpTokenizer` applies ordered rules: URLs → helpline numbers → recommendation IDs → ranges → codes → number-word compounds → hyphenated compounds → slash terms → abbreviations → numbers → words → symbols. **Hybrid** = spaCy's tokenizer with the corpus's domain terms added as special cases; it is used only where a spaCy model must tokenize (POS, NER). **Decision:** the Custom tokenizer is used for documents and queries. It is the most accurate on domain tokens and the fastest.

### Exercise 4: Byte-Pair Encoding (Table G)

BPE was trained with Hugging Face `tokenizers` (BPE model, whitespace pre-tokenizer, `BpeTrainer`).

| Target vocabulary | Learned | Corpus tokens | Tokens per word |
|---|---|---|---|
| 500 | 500 | 96,247 | 2.41 |
| 1,000 | 1,000 | 75,668 | 1.89 |
| **2,000** | 2,000 | 60,843 | **1.52** |
| 4,000 | 4,000 | 51,985 | 1.30 |
| 8,000 | 6,913 | 48,496 | 1.21 |

Table G (`bpe_results.csv`) shows how each word is split. Frequent words stay whole, but rare domain terms are split at small vocabulary sizes: *benzodiazepines* becomes 9 pieces at vocabulary 500, *psychotherapy* becomes *psych | other | apy*, *PTSD* becomes *P | T | S | D*, and *self-harm* becomes *self | - | harm* at every size. **Decision:** BPE is analysed but **not used** in the pipeline. Sub-word pieces are not meaningful search terms, and a whole-word index fits a 17-document corpus. BPE would matter for a neural model with a fixed vocabulary.

### Exercise 5: Phrase structure

spaCy noun chunks and dependency-based verb phrases show that the corpus is **instructional**: verb phrases carry the advice (*seek help*, *talk to your doctor*, *not stop taking*), and many depend on negation (*not go away with treatment*, *not smoke*). Top noun phrases per document identify each document's topic (e.g. *panic disorder*, *psychological interventions*, *ptsd*, *low self-esteem*). This motivated keeping negation and phrase information (`ex5_instruction_phrases.csv`, `ex5_top_noun_phrases.csv`).

### Exercise 6: Numbers

| Category | Occurrences | Policy |
|---|---|---|
| Duration (6 months, 12 weeks) | 72 | **keep** |
| Age, sessions/frequency | 8 + 8 | **keep** |
| Percentage | 35 | **keep** |
| Helpline number (988, 111, 999, 1-800-…) | 28 | **keep** |
| Dose | 0 | (keep; none in the corpus) |
| NICE year tag [2004, amended 2020] | 56 | drop |
| Calendar date, year | 4 + 168 | drop |
| Recommendation ID (1.2.30) | 73 | drop |
| Statistical value (0.31) | 23 | drop |

**Decision (C):** a number policy runs before tokenization. It drops scaffolding numbers (number tokens 886 → 613, vocabulary 4,367 → 4,255) and keeps every category a user might search for. A check confirmed that all kept categories are unchanged.

### Exercise 7: Stop words

| Stop list | Size | Tokens kept | Vocabulary | Negation tokens kept | Modal tokens kept |
|---|---|---|---|---|---|
| None | 0 | 39,660 | 3,662 | 327 | 828 |
| NLTK | 198 | 24,570 | 3,539 | **22** | 326 |
| spaCy | 326 | 22,968 | 3,438 | **0** | **0** |
| **Custom** | 190 | 24,913 | 3,546 | **327** | **828** |

A meaning test on real sentences (`ex7_meaning_test.csv`) shows the problem. With the standard lists, "Antidepressant medicines **should not** be used for treating depression in children" becomes *antidepressant medicines used treating depression children*, which reverses the advice. **Custom list (B):** (NLTK list − negation, modal, timing and amount words) + generic corpus words (*people, person, things, information, time, et, al*). It removes almost as many tokens as NLTK and keeps every negation and modal. Word clouds before and after are in the notebook.

### Exercise 8: Stemming

| Stemmer | Words → stems | Reduction | Problem seen |
|---|---|---|---|
| Porter | 3,546 → 2,571 | 27.5% | non-words (*activ*) |
| **Snowball** | 3,546 → 2,545 | 28.2% | non-words (*anxieti, depress*) |
| Lancaster | 3,546 → 2,358 | 33.5% | over-merging: *car/care*, *us/use* |
| Regexp (suffix rules) | 3,546 → 2,994 | 15.6% | misses irregular forms |
| Domain light stemmer (custom) | 3,546 → 3,025 | 14.7% | safe but weak |

Snowball is the best stemmer, but every stemmer produces non-words or harmful merges (*medication* and *medical* both become *medic*). Stemming became Pipeline A in Module 3.

### Exercise 9: Order of stop-word removal and stemming

| Order | Tokens | Vocabulary |
|---|---|---|
| Stop-word removal → stemming | 24,913 | 2,545 |
| Stemming → stop-word removal | 25,544 | 2,553 |

Stemming first changes surface forms, so 21 NLTK stop words no longer match the list and survive into the index (*any → ani, because → becaus, does → doe, very → veri*). **Decision:** stop words are matched on the **surface form** before or independently of normalisation.

### Exercise 10: Lemmatization (Table C)

| Method | Vocabulary | Reduction | Time (s) |
|---|---|---|---|
| None | 3,546 | n/a | 0.0 |
| Snowball stemmer | 2,545 | 28.2% | 0.3 |
| WordNet (no POS) | 3,248 | 8.4% | 0.2 |
| WordNet + POS | 2,966 | 16.4% | 2.3 |
| TextBlob (+ POS for verbs) | 2,980 | 16.0% | 2.9 |
| **spaCy (in context)** | **2,926** | **17.5%** | 13.9 |

Table C (`stem_vs_lemma.csv`) compares forms word by word: *antidepressants* → stem *antidepress* vs lemma *antidepressant*; *anxiety* → *anxieti* vs *anxiety*; *slept* → *slept* vs *sleep*; *medication* → *medic* vs *medication*. **Decision:** spaCy lemmatization, running spaCy's tagger on *our* tokens (no re-tokenization) so lemmas use sentence context. Domain rules were added through spaCy's `attribute_ruler`: plural acronyms map to the singular (SSRIs → *ssri*, RCTs → *rct*), *media* stays *media*, and *generalised/generalized* → *generalised*. Gensim was excluded because its lemmatizer was removed in version 4.

### Exercise 11: Combined preprocessing

The decisions above were combined into one function, `final_preprocess` (the Final Pipeline), and its effect measured: Table A (Section 3).

### Exercise 12: POS tagging and custom POS taggers (Table D)

**Gold standard.** 50 seeded sentences (seed 134) containing domain terms, 724 tokens. Tokens where the NLTK and spaCy taggers agree were accepted. All 119 disagreements and domain tokens were reviewed by hand using Penn Treebank conventions (proposed by the AI assistant, approved by the team). The gold set was split **by sentence**: 30 for ML training, 20 held out (303 tokens, 23 domain tokens). Every tagger is scored on the same held-out sentences. spaCy's tags were mapped to NLTK's conventions for brackets and hyphens.

**Domain terms mis-tagged by the default taggers** include imperatives at the start of a sentence (*Follow, Learn*: NLTK tags them as nouns), plural acronyms (*SSRIs, RCTs*: NLTK says NNP, gold NNS), *e.g./i.e.* (NLTK: VB/JJ/NN, gold FW), *panic* in "panic disorder" (NLTK: JJ), and *NIMH/PTSD* (spaCy: JJ/NN).

**Custom taggers**
- **Rule-based (dictionary + regex):** NLTK tags, overridden by a domain lexicon (acronyms, medicines, *e.g.* → FW) and regex rules (ranges and helpline numbers → CD, codes → NNP, plural acronyms → NNS, web addresses → NNP).
- **ML 1: n-gram backoff:** Trigram → Bigram → Unigram → Affix → Default(NN), trained on the Penn Treebank sample, with and without the 30 domain sentences.
- **ML 2: KNN:** features are the word, previous word, next word, word length, and first and last character (one-hot encoded), trained on Treebank + domain sentences.

| Tagger | Accuracy (%) | Domain-token accuracy (%) |
|---|---|---|
| Default: NLTK perceptron | 92.4 | 65.2 |
| Default: spaCy | 93.4 | 60.9 |
| **Custom: rule-based (dictionary + regex)** | **93.7** | **82.6** |
| ML: n-gram backoff (Treebank only) | 80.2 | 34.8 |
| ML: n-gram backoff (Treebank + 30 domain sentences) | 88.4 | 82.6 |
| ML: KNN, K = 3 (Treebank + domain) | 87.1 | 82.6 |

KNN by K: K=1: 86.5, **K=3: 87.1**, K=5: 85.8, K=7: 85.8. With context and word identity removed (shape features only), accuracy falls to 71.3.

**Findings.** The rule-based tagger improves domain-token accuracy by **+17.4 points** over the best default tagger, and is also best overall. Its remaining domain errors are web addresses (our rule says NNP; the gold follows the NN convention, and we did not change the rule after seeing the test data) and "ways to **self-harm**", a verb use that a context-free dictionary cannot see. For the ML taggers, **domain training data matters more than the algorithm**: adding only 30 domain sentences gives the largest improvement in the table. **Decision (C):** the rule-based tagger is our custom POS tagger; it is accurate, transparent and needs no annotated data. Table D (`pos_tagging_results.csv`) lists every held-out domain token with its default tags, custom tag, gold tag and Correct/Incorrect. *Limitation:* a small test set (23 domain tokens), and tokens where both taggers agree were not individually reviewed.

### Exercise 13: Named entity recognition (Table E)

**Applying default NER** (corpus counts in `ex13_entity_counts.csv`). In this corpus **MONEY and EVENT never occur, and PRODUCT almost never does**. DATE is mostly publication years in citations; PERSON is mostly citation authors. spaCy labels domain acronyms as ORG (GAD 63×, CBT 60×). NLTK `ne_chunk` labels capitalised ordinary words as GPE (*Key, Research, Further, Talk*). Of the domain-entity mentions found by our rules, the default spaCy model **missed 960 completely** and gave **189 the label ORG** (`ex13_domain_entities.csv`): *depression* missed 179×, *antidepressants* 76×, *988* tagged as CARDINAL.

**Custom NER (C):** a spaCy `EntityRuler` placed before the statistical NER, with domain labels **CONDITION, MEDICATION, THERAPY, HELPLINE** and organisation acronyms (NICE, NHS, WHO, NIMH, SAMHSA). Patterns are case-sensitive where the lower-case form is an ordinary word (*who, nice*). The helpline pattern is regex-based (988, 1-8xx numbers).

**Evaluation.** 40 seeded sentences, 74 gold entities (proposed by the AI assistant, approved by the team), exact span + label match. Numbers such as citation markers (CARDINAL) are out of scope.

| Scope | System | Predicted | Correct | Incorrect | Missed | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Required types | spaCy (default) | 32 | 17 | 15 | 12 | 0.53 | 0.59 | 0.56 |
| Required types | NLTK `ne_chunk` | 33 | 8 | 25 | 21 | 0.24 | 0.28 | 0.26 |
| Required types | **spaCy + EntityRuler** | 28 | 18 | 10 | 11 | **0.64** | **0.62** | **0.63** |
| Required + domain | spaCy (default) | 32 | 17 | 15 | 57 | 0.53 | 0.23 | 0.32 |
| Required + domain | NLTK `ne_chunk` | 33 | 8 | 25 | 66 | 0.24 | 0.11 | 0.15 |
| Required + domain | **spaCy + EntityRuler** | 64 | 51 | 13 | 23 | **0.80** | **0.69** | **0.74** |

Entity-level Table E (`ner_entities.csv`) lists each entity with its predicted type and Correct / Incorrect / Missed. The remaining errors of the custom system are (i) errors inherited from spaCy on citation names ("Chan et al." span, *Buchholz* tagged as GPE), and (ii) dictionary limits: unlisted drugs (fluvoxamine, amitriptyline, TCAs), plural forms (*eating disorders*), span boundaries (*Major depression*) and context (*people who self-harm* is a verb). **Decision:** spaCy + domain EntityRuler, which is also used in the GUI.

### Exercise 14: N-gram analysis (Table F)

N-grams are built **inside** sentences and lines, on content tokens with stop words kept (so phrases stay readable).

| N-gram | Total | Unique | Seen once | Top meaningful phrases (no stop word at either end) |
|---|---|---|---|---|
| Unigram | 39,660 | 3,662 | 40.6% | treatment, health, interventions, depression, symptoms |
| Bigram | 36,454 | 16,632 | 67.0% | mental health, psychological interventions, health care, panic disorder, side effects, drug use |
| Trigram | 33,501 | 23,079 | 80.2% | health care provider, certainty of evidence, drug use disorders, strength of recommendation |
| 4-gram | 30,774 | 23,852 | 85.4% | adults with anxiety disorders, side effects of medication, cognitive behavioural therapy cbt |
| 5-gram | 28,213 | 22,886 | 87.6% | potential side effects of medication, national institute of mental health |

The raw top-10 lists are dominated by function words (*of the, should be*). The check *number of n-grams = N − n + 1 per sentence* holds for every n. Frequent 4- and 5-grams are largely **repeated template text** from NICE tables ("suit people who do not like talking about their depression").

**Dictionary size vs term dictionary:** term dictionary 2,919; unigram 3,662 (1.3×); bigram 16,632 (5.7×); trigram 23,079 (7.9×); 4-gram 23,852 (8.2×); 5-gram 22,886 (7.8×), and the longer dictionaries consist almost entirely of one-off entries. **Decision (C):** do **not** index n-grams. The index stores single terms with **positions**, so phrase queries are answered by checking adjacent positions.

---

## 5. Module 3: Process selection and ordering

### 5.1 The Final Pipeline and why each step is where it is

| # | Process | Required? | Position | Depends on | If moved |
|---|---|---|---|---|---|
| 1 | Extraction | Yes: PDF/HTML are not text | first | raw files | nothing can run |
| 2 | Cleaning | Yes: layout artefacts create fake tokens | after extraction | page and line structure | after tokenization, headers are already tokens |
| 3 | Sentence splitting (lines first) | Yes, for n-grams, POS, NER and phrase boundaries | after cleaning | bullets on their own lines | before unwrapping, every PDF line is a "sentence" |
| 4 | Number policy | Yes: scaffolding numbers are noise | **before** tokenization | raw patterns ("1.2.3", "[2022]") | after tokenization the patterns are already split |
| 5 | Custom tokenization | Yes | after number policy | cleaned text | everything later depends on it |
| 6 | Lowercasing | Yes | after tokenization | tokens | before, case-based rules (acronyms) are lost |
| 7 | Punctuation removal | Yes | after lowercasing | tokens | before tokenization it breaks self-harm, e.g., URLs |
| 8 | Lemmatization (spaCy, in context, domain rules) | Yes | **before** stop-word removal | the full sentence for POS context | after stop-word removal, lemmas change (tested below) |
| 9 | Stop-word removal (custom list) | Yes | **after** lemmatization, by surface form | the list keeps negation and modals | the standard list deletes *not*; before lemmatization, see row 8 |
| — | Stemming | **No** (non-words, over-merging) | (alternative to row 8) | n/a | tested as Pipeline A |
| — | BPE | **No** (splits domain terms) | not used | n/a | n/a |
| — | POS / NER | Analysis branch (reports, GUI) | parallel, on original-case sentences | case and context | after lowercasing or stop-word removal they lose their evidence |
| — | N-grams | Analysis; positions replace them | after normalisation | sentence boundaries | across boundaries they create fake phrases |
| 10 | Positional inverted index | Yes | after the pipeline | final terms + positions | built on raw tokens, documents and queries would not match |
| 11 | Query processing | Yes: the **same** function as for documents | query time | the pipeline | a different pipeline means *antidepressants* ≠ *antidepressant* |
| 12 | Retrieval, ranking, evaluation | Yes | last | index + judgments made before retrieval | n/a |

### 5.2 Mandatory experiment: pipeline orders

All pipelines share steps 1–7 and differ only in the branch of the AS1 diagram and its order. They run per sentence on the same corpus.

| Pipeline | Steps | Tokens | Vocabulary | Readable vocabulary | Meaningful top-20 bigrams | Negation bigrams | Domain terms readable (of 14) | Time (s) |
|---|---|---|---|---|---|---|---|---|
| **A** | NLTK stop words → Snowball stemming | 24,570 | 2,535 | 56.3% | 7 | 26 | 9 | 0.6 |
| **B** | no stop-word removal → lemmatization | 39,660 | 3,011 | 89.5% | 5 | 320 | 13 | 29.2 |
| B-reordered | custom stop words → lemmatization | 24,913 | 2,940 | 90.5% | 17 | 320 | 13 | 27.4 |
| **Final** | lemmatization → custom stop words | 24,913 | 2,914 | 90.5% | **17** | 320 | 13 | 30.1 |

*"Readable"* = a WordNet word, a known domain term, or a kept number. The one "unreadable" domain term in B and Final is *should*, which is not a WordNet entry.

What each pipeline does to domain terms (`module3_domain_terms.csv`):

| Input | Pipeline A | Pipeline B | Final |
|---|---|---|---|
| generalised anxiety disorder | generalis anxieti disord | generalised anxiety disorder | generalised anxiety disorder |
| antidepressants | antidepress | antidepressant | antidepressant |
| SSRIs | ssris | ssri | ssri |
| do not stop taking | **stop take** | do not stop take | **not stop take** |
| should not be used | **use** | should not be use | **should not use** |
| talking therapies | talk therapi | talk therapy | talk therapy |

- **Pipeline A** has the smallest vocabulary, but almost half of it is not words. It turns "should not be used" into *use*, which reverses safety advice.
- **Pipeline B** keeps everything: 60% more tokens, and function-word bigrams dominate (*do not*, *there be*).
- **Order matters:** with identical steps, removing stop words *before* lemmatization changed lemmas in **231 of 3,337 sentences**. Inflections stay unmerged (*outcomes, tapering, discontinuities*), giving a larger vocabulary. Lemmatization therefore precedes stop-word removal. The experiment also exposed one spaCy error: *self-harmed* → *self-harme*. It can be fixed with a lemma rule (recorded as future work).

### Table H: Pipeline comparison (`pipeline_comparison.csv`)

| Measure | Pipeline A | Pipeline B | Final Pipeline |
|---|---|---|---|
| Token count | 24,570 | 39,660 | 24,913 |
| Vocabulary size | 2,535 | 3,011 | 2,914 |
| Meaningful n-grams (of top-20 bigrams) | 7 | 5 | **17** |
| Precision | 0.414 | 0.429 | **0.429** |
| Recall | 0.896 | 0.943 | **0.943** |
| F1-score | 0.505 | 0.534 | **0.534** |

(Precision, recall and F1 come from Module 6.)

---

## 6. Module 4: Information retrieval

**Index (C):** a **positional inverted index** (term → document → positions), saved as `results/inverted_index.json`. Positions count Final Pipeline terms, and they jump at every sentence boundary so a phrase cannot match across sentences or headings. Posting lists in the AS1 format:

```
depression  -> D01, D02, D03, D04, D06, D07, D08, D09, D10, D11, D12, D13, D14, D16, D25
ssri        -> D01, D02, D04, D06, D08, D09, D14
self-harm   -> D02, D03, D04, D06, D14
naltrexone  -> D24, D25
988         -> D07, D08, D09, D10, D11, D16
```

| Index statistic | Value |
|---|---|
| Documents | 17 |
| Vocabulary (terms) | 2,914 |
| Postings (term–document pairs) | 8,125 |
| Positions stored | 24,913 |
| Average documents per term | 2.79 |
| Terms in only one document | 51.4% |
| Terms in every document | 10 (*also, can, health, help, may, mental, more, need, not, social*) |

*The index vocabulary (2,914) is slightly smaller than Table A's (2,919) because the index lemmatizes sentence by sentence; Table A lemmatized whole documents, which gives spaCy slightly different context.*

**Search modes**
- **Keyword:** documents containing *any* query term, ranked by log-scaled TF × IDF: score(d) = Σ (1 + log tf) · log(N/df).
- **Phrase:** query terms at consecutive positions.
- **Boolean:** AND, OR, NOT in **capitals** (lower-case *not* is a search term, because negation is kept), quotes for phrases, brackets, precedence NOT > AND > OR (shunting-yard parser). Adjacent words form one operand (implicit AND), and "x NOT y" means "x AND NOT y". The parser was tested on 14 cases including malformed queries.
- Every query passes through the **same** `pipeline()` as the documents.

Examples: "do not stop" matches **all 17** documents as keywords but **1** document (D14) as a phrase. The keyword query "sleep problems" ranks the NHS sleep page (D17) first among 16 matches.

---

## 7. Module 5: Domain-specific queries (Table I)

Fourteen queries cover every type in the AS1 examples, plus natural-language queries. The queries and relevance judgments (Section 8) were written **before** any retrieval was run on them, and the Module 4 demo queries were not reused.

| ID | Query | Query type | Retrieved documents | # Results | Time (ms) |
|---|---|---|---|---|---|
| Q01 | anxiety | Unigram (keyword) | D04, D08, D16, D02, D13, D07, D10, D14, D01, D17, D03, D09, D12 | 13 | 6.6 |
| Q02 | panic disorder | Bigram (phrase) | D04, D02, D03, D07, D09, D14 | 6 | 6.8 |
| Q03 | generalised anxiety disorder | Trigram (phrase) | D08, D04, D02, D03, D07, D14 | 6 | 5.8 |
| Q04 | antidepressant side effects | Trigram (keyword) | D14, D01, D02, D07, D04, D08, … | 14 | 4.7 |
| Q05 | how to stop taking antidepressants safely | 4-gram (keyword, natural language) | D14, D01, D04, D02, … | 17 | 6.5 |
| Q06 | alcohol AND medication | Boolean AND | D01, D07, D02, D08, D09, D24, D14 | 7 | 11.8 |
| Q07 | suicide OR self-harm | Boolean OR | D03, D11, D06, D04, D07, … | 12 | 13.5 |
| Q08 | depression AND NOT anxiety | Boolean NOT | D06, D11, D25 | 3 | 12.0 |
| Q09 | trouble sleeping | Bigram (keyword) | D08, D11, D07, D09, D17, … | 12 | 6.8 |
| Q10 | PTSD OR trauma | Boolean OR | D09, D14, D25 | 3 | 13.8 |
| Q11 | crisis helpline | Bigram (keyword, natural language) | D11, D03, D07, D06, D08, D09, D10, D14, D16 | 9 | 17.0 |
| Q12 | low self-esteem | Bigram (phrase) | D12, D07 | 2 | 4.4 |
| Q13 | loneliness | Unigram (keyword) | D13 | 1 | 5.2 |
| Q14 | "drug use disorders" AND psychosocial | Combined (phrase + AND) | D25 | 1 | 14.0 |

Q03 confirms that the domain lemma rule works: the US spelling *generalized* in D08 matches the query *generalised*, and D08 is ranked first. Boolean result sets are exact; they are ordered by the TF-IDF score of the query words so that P@K is defined for every query type.

---

## 8. Module 6: Retrieval evaluation

**Relevance set.** For each query we wrote an information need and judged **every query × document pair** (14 × 17, `relevance_judgments.csv`). A document is relevant if it gives **substantive** information answering the need; passing mentions, citations, cross-references and generic advice ("talk to your doctor") are not relevant. Judgments were proposed by the AI assistant after reading the relevant passages of each document, and reviewed and approved by the team.

### Per-query results, Final Pipeline (`evaluation_per_query.csv`)

| ID | Type | Retrieved | Relevant | P | R | F1 | P@3 | R@3 | P@5 | R@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Q01 | Keyword | 13 | 4 | 0.31 | 1.00 | 0.47 | 1.00 | 0.75 | 0.80 | 1.00 |
| Q02 | Phrase | 6 | 2 | 0.33 | 1.00 | 0.50 | 0.67 | 1.00 | 0.40 | 1.00 |
| Q03 | Phrase | 6 | 3 | 0.50 | 1.00 | 0.67 | 1.00 | 1.00 | 0.60 | 1.00 |
| Q04 | Keyword | 14 | 5 | 0.36 | 1.00 | 0.53 | 1.00 | 0.60 | 0.80 | 0.80 |
| Q05 | Keyword (NL) | 17 | 3 | 0.18 | 1.00 | 0.30 | 0.67 | 0.67 | 0.60 | 1.00 |
| Q06 | Boolean AND | 7 | 1 | 0.14 | 1.00 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 |
| Q07 | Boolean OR | 12 | 2 | 0.17 | 1.00 | 0.29 | 0.67 | 1.00 | 0.40 | 1.00 |
| Q08 | Boolean NOT | 3 | 3 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.20 | 0.33 |
| Q09 | Keyword | 12 | 1 | 0.08 | 1.00 | 0.15 | 0.00 | 0.00 | 0.20 | 1.00 |
| Q10 | Boolean OR | 3 | 1 | 0.33 | 1.00 | 0.50 | 0.33 | 1.00 | 0.20 | 1.00 |
| Q11 | Keyword (NL) | 9 | 8 | 0.78 | 0.88 | 0.82 | 0.67 | 0.25 | 0.60 | 0.38 |
| Q12 | Phrase | 2 | 1 | 0.50 | 1.00 | 0.67 | 0.33 | 1.00 | 0.20 | 1.00 |
| Q13 | Keyword | 1 | 1 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 | 0.20 | 1.00 |
| Q14 | Combined | 1 | 1 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 | 0.20 | 1.00 |
| | **Macro average** | | | **0.43** | **0.94** | **0.53** | **0.52** | **0.69** | **0.39** | **0.82** |

P@K divides by K even when fewer than K documents are retrieved (standard definition), which penalises single-result queries.

### Table J: Overall performance (`evaluation_results.csv`)

Each pipeline has its **own index**; queries are processed by the same pipeline; all use the same judgments.

| Method / Pipeline | Precision | Recall | F1-score | P@3 | R@3 | P@5 | R@5 | Indexing time (s) | Avg query time (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Pipeline A (stop words → stemming) | 0.414 | 0.896 | 0.505 | 0.452 | 0.665 | 0.371 | 0.846 | 0.6 | 0.9 |
| Pipeline B (no stop-word removal → lemmatization) | 0.429 | 0.943 | 0.534 | 0.524 | 0.686 | 0.386 | 0.822 | 29.2 | 8.0 |
| **Final Pipeline (lemmatization → custom stop words)** | **0.429** | **0.943** | **0.534** | **0.524** | **0.686** | **0.386** | **0.822** | 30.1 | 7.8 |

### 8.1 Analysis

- **High recall, low precision.** Keyword search returns documents containing *any* query term, and domain words occur widely across a focused corpus. Almost every relevant document is found (recall 0.94), along with many others. **Ranking does work:** P@3 (0.52) is higher than overall precision (0.43).
- **Phrase and Boolean queries are stricter.** Q12–Q14 retrieve one or two documents with high precision.
- **Error analysis** (`module6_error_analysis.csv`):
  - *Q09 "trouble sleeping":* the sleep page D17 is ranked 5th. *Trouble* is a rare, high-IDF word, so documents with "trouble concentrating" (D08) or "legal troubles" (D11) outrank the page about sleep, which is a TF-IDF weakness on short queries.
  - *Q06 "alcohol AND medication":* the relevant alcohol module D24 is found, but ranked 6th of 7. Both words co-occur in many documents (e.g. "avoid alcohol with medication"), so a Boolean AND does not express the need *medicines for alcohol use disorder*.
  - *Q08 "depression AND NOT anxiety":* **NOT removes relevant documents** (D01 and D07 are about depression but mention anxiety) and keeps documents that merely mention depression (D11, D25). Boolean NOT is a blunt tool on documents that discuss related conditions.
  - *Q11 "crisis helpline":* *helpline* does not occur in the documents ("Lifeline", "crisis line"). Results depend on *crisis*, and D13 (NHS 111 "urgent help") is missed: **vocabulary mismatch**.
  - *Q02/Q03 phrases:* D03 matches only because it cites another guideline's title: a correct phrase match but not a relevant document.
- **Pipeline A** is lowest on every metric: stems conflate or separate forms unpredictably and negation is lost.
- **Pipeline B and the Final Pipeline score identically.** The words B keeps (*to, how, of, the*) occur in nearly every document, so their IDF is ~0: they change neither the ranking nor the set found. With 17 documents and document-level judgments, retrieval metrics **cannot separate** these two pipelines.

---

## 9. Final selection and justification

**Selected: the Final Pipeline**
`number policy → custom tokenizer → lowercase → punctuation removal → spaCy lemmas in context (+ domain rules) → custom stop words (surface form)`

1. **Retrieval effectiveness:** equal to the best alternative (Pipeline B) and better than stemming (Pipeline A) on precision, recall, F1 and P@3 (Tables H and J).
2. **Index size:** 37% fewer tokens than Pipeline B for the same retrieval quality; vocabulary 2,914 vs 3,011.
3. **Readable terms and phrases:** 90.5% readable vocabulary (vs 56.3% for stemming) and 17 of the top-20 bigrams are meaningful (vs 7 and 5). This matters because terms and phrases are **shown to users** in the GUI and the dictionary.
4. **Safety-relevant meaning is preserved:** negation and modal strength survive ("not stop take", "should not use"). This is the main domain risk, even where document-level metrics cannot measure it.
5. **Cost:** slower than stemming (about 30 s to index instead of under 1 s), but that is paid once; a query takes about 8 ms.

---

## 10. GUI (Gradio): `python app.py`

The GUI follows the suggested layout and demonstrates the complete working pipeline, using the same code (`src/nlp_pipeline.py`), the saved index and the results CSVs.

| Tab | Content |
|---|---|
| 1 · Documents | **Load Documents**: document statistics and the document-ID → name table |
| 2 · Text Analysis | Live, on any typed text: **tokenization** (NLTK vs spaCy vs custom), **preprocessing** (stage-by-stage trace of the Final Pipeline), **POS** (default vs custom, with changed tags marked), **NER** (highlighted domain entities vs default spaCy), **n-grams** (n = 1–5), **BPE**; each with the saved table (A–G) underneath |
| 3 · Terms Dictionary | The inverted index as a table (term, documents with counts, occurrences, document frequency), sortable by term or document frequency, with a filter |
| 4 · Search & Evaluation | Query, query type (Keyword / Phrase / Boolean), **SEARCH**; retrieved documents, number of results, execution time, **Precision, Recall, F1, P@3** (live for the 14 judged queries, selectable from a list); results marked relevant / not relevant, with snippets and source links |
| 5 · Compare Pipelines | The same query on Pipeline A and the Final Pipeline side by side, each with its own metrics, plus Tables H and J |
| 6 · Results Tables | All CSV checkpoints (Tables A–J and supporting tables) |

A "not medical advice" note with crisis numbers (988 US; 999/111 UK) is shown on every page. Example demo: *Compare Pipelines → "do not stop taking antidepressants" (Phrase)*. Pipeline A, which has dropped *not*, returns "…if you want to **stop taking** an antidepressant"; the Final Pipeline returns "Do **not stop taking** an antidepressant suddenly…".

---

## 11. Limitations and future work

- **Small evaluation:** 17 documents, 14 queries, binary judgments by one team; POS and NER gold sets are small (23 held-out domain tokens; 74 entities). Gold labels were proposed by an AI assistant and approved by the team, not annotated independently by several people.
- **Document-level retrieval** cannot separate pipelines that differ only in function words. **Passage-level** retrieval (sections or sentences) would give sharper rankings and evaluation.
- **TF-IDF weaknesses:** rare-word dominance (Q09), vocabulary mismatch (Q11). Future work: BM25, query expansion with a domain synonym list (helpline ↔ lifeline ↔ crisis line), or dense embeddings.
- **Dictionary-based components** (custom POS lexicon, EntityRuler) miss unlisted terms (fluvoxamine, amitriptyline) and plural patterns; they could be extended automatically from drug lists, or replaced by a trained NER with domain data.
- **Corpus gaps:** no Indian layer (image-only Tele-MANAS PDF), no dose information; a small residual print header in D16.
- **Known lemma error:** *self-harmed* → *self-harme* (spaCy); fix with a domain lemma rule.

---

## 12. Reproducibility

- **Environment:** Python 3.14; `pip install -r requirements.txt`; `python -m spacy download en_core_web_sm`.
- **Corpus:** `python src/build_corpus.py`, which writes `data/processed/corpus.json` from `data/raw/` and `data/metadata/source_manifest.csv`.
- **Notebook:** run `Domain_Text_Analysis_Retrieval.ipynb` top to bottom; every table is saved to `results/`. All randomness uses **SEED = 134**.
- **GUI:** `python app.py`, then open the printed local URL.
- **Folder structure**

```
Domain_Text_Analysis_Retrieval/
├── Domain_Text_Analysis_Retrieval.ipynb   # Modules 1–6
├── app.py                                 # Gradio GUI
├── src/build_corpus.py                    # corpus pipeline
├── src/nlp_pipeline.py                    # Final Pipeline + domain NER (used by the GUI)
├── data/raw/ (pdf, html), data/metadata/source_manifest.csv, data/processed/corpus.json
├── results/   # Tables A–J and checkpoints
│   ├── document_statistics.csv, preprocessing_results.csv (A), tokenization_comparison.csv (B)
│   ├── stem_vs_lemma.csv (C), pos_tagging_results.csv (D), ner_results.csv + ner_entities.csv (E)
│   ├── unigram/bigram/trigram_results.csv, ngram_results.csv (F), bpe_results.csv (G)
│   ├── pipeline_comparison.csv (H), retrieval_results.csv (I), evaluation_results.csv (J)
│   ├── inverted_index.json, queries.csv, relevance_judgments.csv, evaluation_per_query.csv
│   └── pos_gold.csv, ner_gold.csv (annotated gold sets)
├── requirements.txt
└── Report.pdf
```

---

## 13. Use of AI assistance

An AI coding assistant (Claude) was used to write and test code, to propose the POS and NER gold labels and the retrieval relevance judgments, and to draft explanatory text. Every proposal was reviewed and approved by the team before use, and all reported numbers come from the notebook outputs and saved CSV files. Design decisions (domain, tokenizer, stop-word policy, pipeline selection) were made by the team on the basis of the experiments reported here.

**Team contributions:** *(to be completed by the team: who did corpus collection, which modules, the GUI, the report)*

---

## References

1. NICE. *Depression in adults: treatment and management* (NG222); *Generalised anxiety disorder and panic disorder in adults: management* (CG113); *Self-harm: assessment, management and preventing recurrence* (NG225). National Institute for Health and Care Excellence.
2. World Health Organization. *mhGAP guideline for mental, neurological and substance use disorders*, 3rd ed. (2023), ISBN 9789240084278; *Depressive disorder (depression)* fact sheet.
3. National Institute of Mental Health. *Depression* (NIH Pub. No. 24-MH-8079); *Generalized Anxiety Disorder: What You Need to Know*; *Post-Traumatic Stress Disorder* (NIH Pub. No. 23-MH-8124); *Eating Disorders: What You Need to Know* (NIH Pub. No. 24-MH-4901); *Frequently Asked Questions About Suicide* (NIH Pub. No. 23-MH-6389); *I'm So Stressed Out!* fact sheet (NIH Pub. No. 20-MH-8125).
4. NHS. *Raising low self-esteem*; *Get help with loneliness*; *Antidepressants*; *Every Mind Matters: Sleep problems*.
5. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python* (NLTK).
6. Honnibal, M., Montani, I., et al. spaCy: Industrial-strength NLP (v3.8), `en_core_web_sm`.
7. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units (BPE). *ACL*.
8. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
9. Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3).
10. Hugging Face `tokenizers`; scikit-learn; Gradio.

Full source URLs, retrieval dates and licences for every document: `data/metadata/source_manifest.csv`.
