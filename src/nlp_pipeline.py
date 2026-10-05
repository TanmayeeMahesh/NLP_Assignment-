"""Final Pipeline of the AS1 project, packaged for the GUI (app.py).

Same code and decisions as the notebook (Domain_Text_Analysis_Retrieval.ipynb):
number policy (Ex6) -> custom tokenizer (Ex3) -> lowercase -> no punctuation
-> spaCy lemmas in context + domain lemma rules (Ex10) -> custom stop words removed by surface form (Ex7, Ex9).
Also the domain NER (spaCy + EntityRuler, Ex13).
"""
import json
import re
from pathlib import Path

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import LineTokenizer, RegexpTokenizer, sent_tokenize
from spacy.symbols import ORTH
from spacy.tokens import Doc

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.json"

for resource in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(resource, quiet=True)

docs = json.load(open(CORPUS_PATH, encoding="utf-8"))["documents"]

# ---------- Sentence splitting (Ex2): lines first, then NLTK sentences ----------
line_tokenizer = LineTokenizer()

def line_then_nltk_sentences(text):
    return [s for line in line_tokenizer.tokenize(text) for s in sent_tokenize(line)]

# ---------- Custom tokenizer (Ex3) ----------
def normalise_for_tokenizing(text):
    """One hyphen character; negation spelled out as 'not'."""
    text = text.replace("‑", "-")
    text = re.sub(r"\bcan't\b", "can not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"n't\b", " not", text)
    return text

custom_pattern = r"""
    https?://\S+[^\s.,;:)]               # URLs, without trailing punctuation
  | www\.\S+[^\s.,;:)]                   # www addresses
  | [\w-]+\.(?:org|gov|com|uk|int)\b     # bare web addresses: 988lifeline.org
  | 1-8\d\d-\d{3}-\d{4}                  # toll-free helpline numbers
  | \d+(?:\.\d+){2,}                     # recommendation IDs: 1.5.1
  | \d+(?:\.\d+)?[–-]\d+(?:\.\d+)?  # numeric ranges: 18-64
  | [A-Za-z]+-?(?:\d+|[IVX]+\b)          # classification codes: DSM-5, ICD-11, ANX2
  | \d+-[a-z]+(?:-[a-z]+)*               # number + word compounds: 12-week
  | [A-Za-z]+(?:-[A-Za-z]+)+             # hyphenated compounds: self-harm
  | [A-Za-z]+/[A-Za-z]+                  # slash terms: and/or
  | (?:e\.g|i\.e|etc)\.                  # abbreviations with dots
  | \d+(?:[.,]\d+)*(?:%|st|nd|rd|th|s)?  # other numbers: 0.29, 5%, 4th
  | [A-Za-z]+(?:'[a-z]+)?                # ordinary words (possessive kept)
  | [^\w\s]                              # any other single symbol
"""
custom_regexp = RegexpTokenizer(custom_pattern, flags=re.VERBOSE)

def custom_tokenize(text):
    return custom_regexp.tokenize(normalise_for_tokenizing(text))

# Domain terms kept whole by spaCy's tokenizer (Hybrid special cases, Ex3)
SPECIAL_TERM_PATTERNS = [r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b",                    # hyphenated compounds
                         r"\b[A-Z]{2,}(?:-?\d+|-[IVX]+)\b",                  # classification codes
                         r"\b1-8\d\d-\d{3}-\d{4}\b",                         # helpline numbers
                         r"(?<![./\w])[A-Za-z]+/[A-Za-z]+\b(?!/)",           # slash terms
                         r"\b\d+-[a-z]+(?:-[a-z]+)*\b"]                      # 12-week, 24-hour
special_terms = set()
for d in docs:
    text = normalise_for_tokenizing(d["text"])
    for pattern in SPECIAL_TERM_PATTERNS:
        special_terms.update(re.findall(pattern, text))

# ---------- Number policy (Ex6): drop scaffolding numbers, keep doses / durations / helplines ----------
MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DROP_NUMBER_PATTERNS = [
    r"\[\d{4}(?:, amended \d{4})?\]",                                                         # NICE year tag
    rf"\b\d{{1,2}} {MONTHS} \d{{4}}\b|\b{MONTHS} \d{{1,2}}, \d{{4}}\b|\b\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}\b",  # date
    r"\b(?:19|20)\d{2}\b",                                                                    # year
    r"\b\d+\.\d+\.\d+\b",                                                                     # recommendation ID
    r"(?<![\w.])-?0\.\d+\b(?!\s?(?:mg|ml|g|%))",                                              # statistical value
]

def apply_number_policy(text):
    for pattern in DROP_NUMBER_PATTERNS:
        text = re.sub(pattern, " ", text)
    return text

def content_tokens(text):
    """Number policy -> custom tokenizer -> lowercase -> drop punctuation-only tokens."""
    return [t.lower() for t in custom_tokenize(apply_number_policy(text)) if any(ch.isalnum() for ch in t)]

# ---------- Custom stop words (Ex7): negation, modals, timing words and amounts are KEPT ----------
NLTK_STOPWORDS = set(stopwords.words("english"))
KEEP_WORDS = {"not", "no", "nor", "never", "without", "against", "cannot",
              "should", "must", "can", "may", "might", "could", "would",
              "before", "after", "until", "off", "down", "up", "out",
              "more", "less", "only"}
EXTRA_STOPWORDS = {"people", "person", "things", "information", "time", "et", "al"}
CUSTOM_STOPWORDS = (NLTK_STOPWORDS - KEEP_WORDS) | EXTRA_STOPWORDS

# ---------- spaCy lemmatizer with domain rules (Ex10) ----------
nlp_domain = spacy.load("en_core_web_sm")
for term in special_terms:
    nlp_domain.tokenizer.add_special_case(term, [{ORTH: term}])
attribute_ruler = nlp_domain.get_pipe("attribute_ruler")
for term in sorted({m for d in docs for m in re.findall(r"\b[A-Z]{2,}s\b", d["text"])}):   # SSRIs -> ssri
    attribute_ruler.add([[{"LOWER": term.lower()}]], {"LEMMA": term[:-1].lower()})
for word, lemma in {"media": "media", "generalised": "generalised", "generalized": "generalised"}.items():
    attribute_ruler.add([[{"LOWER": word}]], {"LEMMA": lemma})

def lemmas_spacy(tokens):
    """spaCy tagger + lemmatizer on OUR tokens (no re-tokenization)."""
    doc = Doc(nlp_domain.vocab, words=tokens)
    for _, component in nlp_domain.pipeline:
        doc = component(doc)
    return [t.lemma_.lower() for t in doc]

def pipeline(text):
    """FINAL PIPELINE, used for documents and queries."""
    tokens = content_tokens(text)
    if not tokens:
        return []
    lemmas = lemmas_spacy(tokens)
    return [lemma for token, lemma in zip(tokens, lemmas) if token not in CUSTOM_STOPWORDS]

def trace_pipeline(text):
    """[(stage, output)] for every stage of the Final Pipeline."""
    step1 = apply_number_policy(text)
    step2 = custom_tokenize(step1)
    step3 = [t.lower() for t in step2 if any(ch.isalnum() for ch in t)]
    step4 = lemmas_spacy(step3) if step3 else []
    step5 = [lemma for token, lemma in zip(step3, step4) if token not in CUSTOM_STOPWORDS]
    return [("Original", text),
            ("1. Number policy", step1),
            ("2. Custom tokenizer", " | ".join(step2)),
            ("3. Lowercase, no punctuation", " | ".join(step3)),
            ("4. spaCy lemmas (in context)", " | ".join(step4)),
            ("5. Custom stop words removed (FINAL TERMS)", " | ".join(step5))]

# ---------- Domain NER (Ex13): spaCy + EntityRuler before the statistical NER ----------
DOMAIN_ENTITIES = {
    "CONDITION": ["depression", "anxiety", "PTSD", "GAD", "OCD", "panic disorder", "bipolar disorder", "psychosis", "schizophrenia",
                  "generalised anxiety disorder", "generalized anxiety disorder", "post-traumatic stress disorder",
                  "obsessive-compulsive disorder", "social anxiety disorder", "alcohol use disorder", "eating disorder",
                  "insomnia", "self-harm", "phobia"],
    "MEDICATION": ["SSRI", "SSRIs", "SNRI", "antidepressant", "antidepressants", "benzodiazepine", "benzodiazepines",
                   "sertraline", "fluoxetine", "escitalopram", "citalopram", "paroxetine", "mirtazapine", "venlafaxine",
                   "duloxetine", "naltrexone", "acamprosate", "baclofen", "disulfiram", "diazepam"],
    "THERAPY": ["CBT", "cognitive behavioural therapy", "cognitive behavioral therapy", "talking therapies", "interpersonal therapy",
                "psychotherapy", "counselling", "counseling", "guided self-help", "psychoeducation", "exposure therapy", "mindfulness"],
    "ORG": ["NICE", "NHS", "WHO", "NIMH", "NIH", "SAMHSA", "MHRA", "FDA", "National Institute of Mental Health"],
}
CASE_SENSITIVE = {"WHO", "NICE", "GAD"}

nlp_ner = spacy.load("en_core_web_sm")
for term in special_terms:
    nlp_ner.tokenizer.add_special_case(term, [{ORTH: term}])
entity_ruler = nlp_ner.add_pipe("entity_ruler", before="ner")
_patterns = []
for label, terms in DOMAIN_ENTITIES.items():
    for term in terms:
        key = "ORTH" if term in CASE_SENSITIVE else "LOWER"
        words = [t.text if key == "ORTH" else t.text.lower() for t in nlp_ner.make_doc(term)]
        _patterns.append({"label": label, "pattern": [{key: w} for w in words]})
_patterns.append({"label": "HELPLINE", "pattern": [{"TEXT": {"REGEX": r"^(988|1-8\d\d-\d{3}-\d{4})$"}}]})
entity_ruler.add_patterns(_patterns)

def entities(text):
    """[(entity text, label, start_char, end_char)]"""
    return [(e.text, e.label_, e.start_char, e.end_char) for e in nlp_ner(text).ents]

def default_entities(text):
    """spaCy's general NER without our domain rules (for comparison)."""
    return [(e.text, e.label_) for e in nlp_domain(text).ents]


# =====================  Analysis helpers for the GUI (same methods as the notebook)  =====================
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import RegexpTagger
from nltk.tokenize import word_tokenize

nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# ---------- Tokenizer comparison (Ex3) ----------
nlp_blank = spacy.blank("en")

def nltk_tokenize(text):
    return word_tokenize(text)

def spacy_tokenize(text):
    return [t.text for t in nlp_blank(text) if not t.is_space]

# ---------- Pipeline A (Module 3): NLTK stop words -> Snowball stemming ----------
SNOWBALL = SnowballStemmer("english")

def pipeline_a(text):
    return [SNOWBALL.stem(t) for t in content_tokens(text) if t not in NLTK_STOPWORDS]

# ---------- POS tagging (Ex12): default taggers and our rule-based custom tagger ----------
SPACY_TO_NLTK = {"-LRB-": "(", "-RRB-": ")", "HYPH": ":", "NFP": ":", "ADD": "NN"}
DOMAIN_LEXICON = {"e.g.": "FW", "i.e.": "FW", "etc.": "FW", "ptsd": "NNP", "gad": "NNP", "cbt": "NNP", "ocd": "NNP",
                  "mhgap": "NNP", "nhs": "NNP", "nimh": "NNP", "samhsa": "NNP", "panic": "NN", "self-harm": "NN",
                  "sertraline": "NN", "fluoxetine": "NN", "escitalopram": "NN", "citalopram": "NN", "paroxetine": "NN",
                  "mirtazapine": "NN", "venlafaxine": "NN", "duloxetine": "NN", "naltrexone": "NN", "acamprosate": "NN",
                  "baclofen": "NN", "disulfiram": "NN", "pregabalin": "NN", "lithium": "NN"}
domain_regexp_tagger = RegexpTagger([
    (r"^\d+(\.\d+)?[–-]\d+(\.\d+)?$", "CD"),          # ranges
    (r"^\d+(\.\d+){2,}$", "CD"),                           # recommendation IDs
    (r"^1-8\d\d-\d{3}-\d{4}$", "CD"),                      # helpline numbers
    (r"^[A-Z]{2,}(-?\d+|-[IVX]+)$", "NNP"),                # codes: ICD-10, DSM-IV
    (r"^[A-Z]{2,}s$", "NNS"),                              # plural acronyms: SSRIs
    (r"^(https?://|www\.)|\.(org|gov|com|uk|int)\b", "NNP")])  # web addresses

def nltk_tags(tokens):
    return [tag for _, tag in nltk.pos_tag(tokens)]

def spacy_tags(tokens):
    doc = Doc(nlp_domain.vocab, words=tokens)
    for _, component in nlp_domain.pipeline:
        doc = component(doc)
    return [SPACY_TO_NLTK.get(t.tag_, t.tag_) for t in doc]

def rule_based_tags(tokens):
    """Custom tagger: NLTK tags, overridden by the domain dictionary, then by the regex rules."""
    tags = nltk_tags(tokens)
    for i, tok in enumerate(tokens):
        if tok.lower() in DOMAIN_LEXICON:
            tags[i] = DOMAIN_LEXICON[tok.lower()]
        else:
            rule_tag = domain_regexp_tagger.tag([tok])[0][1]
            if rule_tag:
                tags[i] = rule_tag
    return tags

# ---------- BPE (Ex4): HF tokenizers, BPE + Whitespace pre-tokenizer, vocabulary 2000 ----------
_bpe = None

def bpe_tokenizer():
    """Trained once on the corpus, on first use (a few seconds)."""
    global _bpe
    if _bpe is None:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        _bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
        _bpe.pre_tokenizer = pre_tokenizers.Whitespace()
        _bpe.train_from_iterator([d["text"] for d in docs],
                                 trainers.BpeTrainer(vocab_size=2000, special_tokens=["[UNK]"]))
    return _bpe
