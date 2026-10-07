"""Gradio GUI: Domain-Specific Text Analysis & Retrieval (mental-health guidance corpus). AS1, NLP DATA403.

Run:  python app.py      then open the local URL it prints.
Uses the Final Pipeline (src/nlp_pipeline.py), the saved index (results/inverted_index.json) and the results CSVs.
"""
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import gradio as gr
import pandas as pd
from nltk.util import ngrams

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
import nlp_pipeline as nlp                                   # noqa: E402

DOC_INFO = {d["doc_id"]: d for d in nlp.docs}
SENTENCES = {d["doc_id"]: [(s, nlp.content_tokens(s)) for s in nlp.line_then_nltk_sentences(d["text"])] for d in nlp.docs}


def read_result(file_name):
    path = RESULTS / file_name
    if not path.exists():
        return pd.DataFrame({"Message": [f"{file_name} not found: run the notebook cell that writes it."]})
    return pd.read_csv(path, keep_default_na=False)


# =====================  Retrieval engine (Module 4 logic), one per pipeline  =====================
class SearchEngine:
    PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}

    def __init__(self, index, process):
        self.index, self.process = index, process          # term -> doc -> positions; text -> terms
        self.all_docs = set(DOC_INFO)

    def idf(self, term):
        n = len(self.index.get(term, {}))
        return math.log(len(self.all_docs) / n) if n else 0.0

    def keyword(self, query):
        terms = self.process(query)
        scores = Counter()
        for term in set(terms):
            for doc_id, pos in self.index.get(term, {}).items():
                scores[doc_id] += (1 + math.log(len(pos))) * self.idf(term)
        return [(d, round(s, 3)) for d, s in scores.most_common()], terms

    def phrase_matches(self, terms):
        if not terms or any(t not in self.index for t in terms):
            return {}
        matches = {}
        for doc_id, first in self.index[terms[0]].items():
            if all(doc_id in self.index[t] for t in terms[1:]):
                later = [set(self.index[t][doc_id]) for t in terms]
                starts = [p for p in first if all(p + k in later[k] for k in range(1, len(terms)))]
                if starts:
                    matches[doc_id] = starts
        return matches

    def phrase(self, query):
        terms = self.process(query)
        ranked = sorted(self.phrase_matches(terms).items(), key=lambda x: -len(x[1]))
        return [(d, len(s)) for d, s in ranked], terms

    def _parse(self, query):
        raw = re.findall(r'"[^"]+"|\(|\)|[^\s()"]+', query)
        items = []
        ends_operand = lambda: bool(items) and (items[-1][0] == "operand" or items[-1] == ("op", ")"))
        for tok in raw:
            if tok in ("AND", "OR", ")"):
                items.append(("op", tok))
            elif tok in ("NOT", "("):
                if ends_operand():
                    items.append(("op", "AND"))
                items.append(("op", tok))
            elif not tok.startswith('"') and items and items[-1][0] == "operand" and not items[-1][1].startswith('"'):
                items[-1] = ("operand", items[-1][1] + " " + tok)
            else:
                if ends_operand():
                    items.append(("op", "AND"))
                items.append(("operand", tok))
        output, stack = [], []
        for kind, value in items:
            if kind == "operand":
                output.append((kind, value))
            elif value == "(":
                stack.append(value)
            elif value == ")":
                while stack and stack[-1] != "(":
                    output.append(("op", stack.pop()))
                if not stack:
                    raise ValueError("Unbalanced brackets")
                stack.pop()
            else:
                while stack and stack[-1] != "(" and (self.PRECEDENCE[stack[-1]] > self.PRECEDENCE[value] or
                                                      (self.PRECEDENCE[stack[-1]] == self.PRECEDENCE[value] and value != "NOT")):
                    output.append(("op", stack.pop()))
                stack.append(value)
        while stack:
            if stack[-1] == "(":
                raise ValueError("Unbalanced brackets")
            output.append(("op", stack.pop()))
        return output

    def _operand(self, text):
        if text.startswith('"'):
            return set(self.phrase_matches(self.process(text.strip('"'))))
        terms = self.process(text)
        return set.intersection(*[set(self.index.get(t, {})) for t in terms]) if terms else set()

    def boolean(self, query):
        stack = []
        try:
            for kind, value in self._parse(query):
                if kind == "operand":
                    stack.append(self._operand(value))
                elif value == "NOT":
                    stack.append(self.all_docs - stack.pop())
                else:
                    right, left = stack.pop(), stack.pop()
                    stack.append(left & right if value == "AND" else left | right)
        except IndexError:
            raise ValueError("Malformed Boolean query")
        if len(stack) != 1:
            raise ValueError("Malformed Boolean query")
        # the exact set is ordered by the TF-IDF score of the query words (as in Module 5)
        words = re.sub(r'\b(AND|OR|NOT)\b|[()"]', " ", query)
        scores = dict(self.keyword(words)[0])
        ranked = sorted(stack[0], key=lambda d: (-scores.get(d, 0), d))
        positive = re.sub(r'\b(AND|OR|NOT)\b|[()"]', " ", re.sub(r'\bNOT\s+("[^"]+"|\([^)]*\)|\S+)', " ", query))
        return [(d, round(scores.get(d, 0), 3)) for d in ranked], self.process(positive)

    def search(self, query, mode):
        if mode == "Keyword":
            return self.keyword(query)
        if mode == "Phrase":
            return self.phrase(query)
        return self.boolean(query)


def build_index(process):
    """Positional index from per-sentence terms; positions jump between sentences (as in Module 4)."""
    postings = defaultdict(lambda: defaultdict(list))
    for d in nlp.docs:
        position = 0
        for sentence in nlp.line_then_nltk_sentences(d["text"]):
            terms = process(sentence)
            for offset, term in enumerate(terms):
                postings[term][d["doc_id"]].append(position + offset)
            position += len(terms) + 1
    return {t: dict(p) for t, p in postings.items()}


saved_index = json.load(open(RESULTS / "inverted_index.json", encoding="utf-8"))["index"]
FINAL = SearchEngine(saved_index, nlp.pipeline)                 # Final Pipeline: index saved by the notebook
PIPELINE_A = SearchEngine(build_index(nlp.pipeline_a), nlp.pipeline_a)   # stemming: fast to rebuild

# Relevance judgments (Module 6) for live Precision / Recall / F1
QUERIES = read_result("queries.csv")
JUDGED = {}
if "Query" in QUERIES:
    MODE_NAMES = {"keyword": "Keyword", "phrase": "Phrase", "boolean": "Boolean"}
    for _, row in QUERIES.iterrows():
        JUDGED[(" ".join(row["Query"].split()), MODE_NAMES[row["Mode"]])] = {
            "id": row["Query ID"], "need": row["Information need"], "relevant": set(row["Relevant documents"].split(", "))}
EVAL_CHOICES = [f"{v['id']} · {q} · {m}" for (q, m), v in JUDGED.items()]


# Topic labels assigned to every document when the corpus was built (corpus.json "topics"), independent of retrieval.
# Query terms (after the pipeline, or stems for Pipeline A) that name a topic -> documents with that topic count as relevant.
TOPIC_WORDS = {
    "depression": ["depression", "depressive", "depressed", "depress"],
    "anxiety": ["anxiety", "anxious", "anxieti", "gad", "panic", "worry", "generalised", "generalis"],
    "medication": ["medication", "medic", "medicine", "medicin", "antidepressant", "antidepress", "ssri", "snri",
                   "sertraline", "fluoxetine", "citalopram", "escitalopram", "paroxetine", "mirtazapine"],
    "self_harm": ["self-harm"],
    "suicide": ["suicide", "suicid", "suicidal"],
    "trauma": ["ptsd", "trauma", "traumat", "traumatic", "post-traumatic"],
    "eating": ["eating", "eat", "anorexia", "bulimia", "binge"],
    "self_esteem": ["self-esteem", "confidence", "confid"],
    "loneliness": ["loneliness", "loneli", "lonely"],
    "stress": ["stress", "stressed"],
    "sleep": ["sleep", "insomnia", "insomnia"],
    "substance_use": ["alcohol", "drug", "substance", "naltrexone", "acamprosate", "disulfiram", "baclofen", "cocaine", "opioid"],
}
TOPIC_OF = {word: topic for topic, words in TOPIC_WORDS.items() for word in words}


def parse_doc_ids(text):
    """'d1, D07 d25' -> {'D01', 'D07', 'D25'} (only IDs that exist)."""
    ids = set()
    for item in re.split(r"[,;\s]+", text or ""):
        match = re.fullmatch(r"[Dd](\d+)", item.strip())
        if match and f"D{int(match.group(1)):02d}" in DOC_INFO:
            ids.add(f"D{int(match.group(1)):02d}")
    return ids


def relevance_for(query, mode, terms, manual=""):
    """(relevant doc set or None, description). Priority: user's list > our judgments > topic-label estimate."""
    own = parse_doc_ids(manual)
    if own:
        return own, "your relevant documents"
    judged = JUDGED.get((" ".join(query.split()), mode))
    if judged:
        return judged["relevant"], f"our relevance judgments ({judged['id']}: {judged['need']})"
    topics = sorted({TOPIC_OF[t] for t in terms if t in TOPIC_OF})
    if topics:
        docs = {d for d, info in DOC_INFO.items() if set(topics) & set(info["topics"])}
        if docs:
            return docs, (f"an ESTIMATE: documents labelled with the topic(s) '{', '.join(topics)}' in the corpus metadata "
                          "(not hand-judged for this query)")
    return None, "none"


def evaluate(ranked_docs, relevant):
    hits = len(set(ranked_docs) & relevant)
    precision = hits / len(ranked_docs) if ranked_docs else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_at_3 = len(set(ranked_docs[:3]) & relevant) / 3
    return precision, recall, f1, p_at_3


stem = nlp.SNOWBALL.stem          # snippets only: compare stems, so 'taking' matches the lemma 'take'
SENTENCE_STEMS = {doc_id: [{stem(t) for t in tokens} for _, tokens in sents] for doc_id, sents in SENTENCES.items()}


def snippet(engine, doc_id, terms, require_all=False, width=230):
    """Most informative sentence for the query (IDF-weighted), headings only as a last resort; matches in bold."""
    term_stems = {t: stem(t) for t in terms}
    best, best_key = None, None
    for position, ((sentence, tokens), stems) in enumerate(zip(SENTENCES[doc_id], SENTENCE_STEMS[doc_id])):
        found = {t for t, s in term_stems.items() if s in stems}
        if not found or (require_all and len(found) < len(term_stems)):
            continue
        key = (len(tokens) >= 5, sum(engine.idf(t) + 0.01 for t in found), -position)
        if best_key is None or key > best_key:
            best, best_key = (sentence, {term_stems[t] for t in found}), key
    if best is None:
        return ""
    sentence, found_stems = best
    text = sentence if len(sentence) <= width else sentence[:width] + "..."
    return re.sub(r"[A-Za-z0-9][\w'-]*", lambda m: f"**{m.group()}**" if stem(m.group().lower()) in found_stems else m.group(), text)


def run_engine(engine, query, mode, top_k, manual_relevant="", fixed_relevance=None):
    """Returns (summary dict, results markdown). fixed_relevance=(set, source) scores several engines on one set."""
    start = time.perf_counter()
    results, terms = engine.search(query, mode)
    elapsed = 1000 * (time.perf_counter() - start)
    ranked = [d for d, _ in results]
    relevant, source = fixed_relevance or relevance_for(query, mode, terms, manual_relevant)
    summary = {"terms": " ".join(terms) or "(none: only stop words)", "docs": "  ".join(ranked) or "(none)",
               "n": len(ranked), "ms": f"{elapsed:.1f} ms", "relevant": relevant, "source": source}
    if relevant:
        summary["metrics"] = evaluate(ranked, relevant)
    lines = []
    for rank, (doc_id, score) in enumerate(results[: int(top_k)], 1):
        info = DOC_INFO[doc_id]
        mark = ""
        if relevant:
            mark = " ✅ relevant" if doc_id in relevant else " ❌ not relevant"
        lines.append(f"**{rank}. {doc_id}: {info['title']}**{mark}  \n"
                     f"*{info['organization']} · {info['document_type'].replace('_', ' ')} · score {score}*  \n"
                     f"> {snippet(engine, doc_id, terms, mode == 'Phrase') or '(no single sentence contains the terms)'}")
    if relevant:
        missed = sorted(relevant - set(ranked))
        if missed:
            lines.append(f"*Relevant but not retrieved:* {', '.join(missed)}")
    return summary, "\n\n".join(lines) or "No document matches."


def fmt(x):
    return f"{x:.2f}"


# =====================  Tab callbacks  =====================
def load_documents():
    stats = read_result("document_statistics.csv")
    table = pd.DataFrame([{"Doc ID": d["doc_id"], "Name": d["title"], "Organisation": d["organization"],
                           "Type": d["document_type"].replace("_", " "), "Words": d["n_words"], "Source": d["source_url"]}
                          for d in nlp.docs])
    total_words = sum(d["n_words"] for d in nlp.docs)
    status = (f"✅ **{len(nlp.docs)} documents loaded** from `data/processed/corpus.json` · {total_words:,} words · "
              f"{sum(len(v) for v in SENTENCES.values()):,} sentences · index: {len(saved_index):,} terms")
    return status, stats, table


def run_tokenizers(text):
    rows = [("NLTK word_tokenize", nlp.nltk_tokenize(text)), ("spaCy (default)", nlp.spacy_tokenize(text)),
            ("Custom (ours, used in the pipeline)", nlp.custom_tokenize(text))]
    return pd.DataFrame([{"Tokenizer": n, "Tokens": len(t), "Output": " | ".join(t)} for n, t in rows])


def run_preprocessing(text):
    return pd.DataFrame(nlp.trace_pipeline(text), columns=["Stage", "Output"])


def run_pos(text):
    tokens = nlp.custom_tokenize(text)
    if not tokens:
        return pd.DataFrame()
    default, spacy_t, custom = nlp.nltk_tags(tokens), nlp.spacy_tags(tokens), nlp.rule_based_tags(tokens)
    return pd.DataFrame({"Word": tokens, "Default POS (NLTK)": default, "spaCy POS": spacy_t,
                         "Custom POS (rule-based)": custom,
                         "Changed by custom rules": ["◀ changed" if d != c else "" for d, c in zip(default, custom)]})


def run_ner(text):
    ents = nlp.entities(text)
    spans, last = [], 0
    for ent_text, label, start, end in ents:
        if start > last:
            spans.append((text[last:start], None))
        spans.append((ent_text, label))
        last = end
    if last < len(text):
        spans.append((text[last:], None))
    default = dict(nlp.default_entities(text))
    table = pd.DataFrame([{"Entity": t, "Domain NER (ours)": l, "Default spaCy": default.get(t, "missed")}
                          for t, l, _, _ in ents])
    return spans, table


def run_ngrams(text, n):
    sentences = [nlp.content_tokens(s) for s in nlp.line_then_nltk_sentences(text)]
    counts = Counter(g for s in sentences for g in ngrams(s, int(n)))
    total = sum(counts.values())
    summary = f"**{int(n)}-grams:** total {total} · unique {len(counts)}"
    return summary, pd.DataFrame([{"N-gram": " ".join(g), "Frequency": c} for g, c in counts.most_common(15)])


def run_bpe(text):
    tokenizer = nlp.bpe_tokenizer()
    rows = []
    for i, word in enumerate(dict.fromkeys(nlp.custom_tokenize(text)), 1):
        if any(ch.isalnum() for ch in word):
            pieces = tokenizer.encode(word).tokens
            rows.append({"S.No.": i, "Word": word, "BPE Tokens": " | ".join(pieces), "Pieces": len(pieces)})
    return pd.DataFrame(rows)


DICTIONARY = pd.DataFrame([{"Term": term,
                            "Documents (doc: count)": ", ".join(f"{d}: {len(p)}" for d, p in sorted(docs_pos.items())),
                            "Term occurrences": sum(len(p) for p in docs_pos.values()),
                            "Document frequency": len(docs_pos)} for term, docs_pos in saved_index.items()])


def show_dictionary(order, contains):
    table = DICTIONARY
    if contains:
        table = table[table["Term"].str.contains(contains.strip().lower(), regex=False)]
    if order == "Term frequency":
        table = table.sort_values(["Term occurrences", "Term"], ascending=[False, True])
    elif order == "Document frequency":
        table = table.sort_values(["Document frequency", "Term occurrences"], ascending=[False, False])
    else:
        table = table.sort_values("Term")
    table = table.reset_index(drop=True)
    table.insert(0, "S No.", range(1, len(table) + 1))
    return f"{len(table):,} terms", table


def pick_eval_query(choice):
    if not choice:
        return gr.update(), gr.update()
    _, query, mode = choice.split(" · ")
    return query, mode


NO_RELEVANCE_NOTE = ("No relevant-document set for this query: type the relevant document IDs in the box above "
                     "(e.g. D01, D07), pick one of the 14 judged queries, or use a query that names a topic "
                     "(depression, anxiety, medication, sleep, suicide, PTSD, alcohol, ...).")


def run_search(query, mode, top_k, manual_relevant=""):
    query = (query or "").strip()
    empty = ("", "", "", "", "-", "-", "-", "-", "")
    if not query:
        return ("Type a query.",) + empty[1:]
    try:
        s, results = run_engine(FINAL, query, mode, top_k, manual_relevant)
    except ValueError as error:
        return (f"**Query error:** {error}. Use AND / OR / NOT in capitals, quotes for phrases.",) + empty[1:]
    if s["relevant"]:
        p, r, f1, p3 = s["metrics"]
        info = (f"**Metrics based on {s['source']}** · relevant documents: {', '.join(sorted(s['relevant']))}")
        metrics = (fmt(p), fmt(r), fmt(f1), fmt(p3))
    else:
        info = NO_RELEVANCE_NOTE
        metrics = ("n/a", "n/a", "n/a", "n/a")
    return (f"**Query terms after the Final Pipeline:** `{s['terms']}`", s["docs"], str(s["n"]), s["ms"], *metrics, info + "\n\n" + results)


def compare_pipelines(query, mode, top_k, manual_relevant=""):
    query = (query or "").strip()
    if not query:
        return "Type a query.", "", ""
    # One relevant set for BOTH pipelines, decided from the Final Pipeline's query terms
    try:
        final_terms = FINAL.search(query, mode)[1]
    except ValueError as error:
        return f"**Query error:** {error}", "", ""
    shared = relevance_for(query, mode, final_terms, manual_relevant)
    outputs = []
    for name, engine in [("Pipeline A (stop words → stemming)", PIPELINE_A), ("Final Pipeline (lemmas → custom stop words)", FINAL)]:
        s, results = run_engine(engine, query, mode, top_k, fixed_relevance=shared)
        metrics = ""
        if s["relevant"]:
            p, r, f1, p3 = s["metrics"]
            metrics = f"**Precision {fmt(p)} · Recall {fmt(r)} · F1 {fmt(f1)} · P@3 {fmt(p3)}**  \n"
        outputs.append(f"### {name}\nQuery terms: `{s['terms']}`  \n"
                       f"Retrieved ({s['n']}, {s['ms']}): {s['docs']}  \n{metrics}\n{results}")
    relevant, source = shared
    note = (f"**Both pipelines scored on {source}** · relevant documents: {', '.join(sorted(relevant))}" if relevant
            else NO_RELEVANCE_NOTE)
    return note, outputs[0], outputs[1]


# =====================  Layout  =====================
TEXT_EXAMPLE = ("Don't stop taking your SSRIs suddenly. NICE recommends CBT for generalised anxiety disorder (GAD), "
                "e.g. 12-week courses. If you're struggling, call 988 or 1-800-662-4357.")

with gr.Blocks(title="Domain Text Analysis & Retrieval") as demo:
    gr.Markdown("# DOMAIN-SPECIFIC TEXT ANALYSIS & RETRIEVAL\n"
                "**Mental-health guidance corpus** (NICE · WHO · NIMH · NHS) · AS1 · NLP DATA403 · Tanmayee K M, Ahmed, Jeromi  \n"
                "Final Pipeline: number policy → custom tokenizer → lowercase → spaCy lemmas (in context) → custom stop words")
    gr.Markdown("> **Not medical advice.** A student search tool over public guidance. In a crisis call or text **988** (US), "
                "call **999** or **111** (UK), or your local emergency number.")

    # ---------- 1. Documents ----------
    with gr.Tab("1 · Documents"):
        load_button = gr.Button("Load Documents", variant="primary")
        load_status = gr.Markdown()
        gr.Markdown("#### Document statistics")
        stats_table = gr.Dataframe(wrap=True, interactive=False)
        gr.Markdown("#### Document ID → Name")
        docs_table = gr.Dataframe(wrap=True, interactive=False)
        load_button.click(load_documents, None, [load_status, stats_table, docs_table])

    # ---------- 2. Text analysis ----------
    with gr.Tab("2 · Text Analysis"):
        gr.Markdown("Type any text: every step runs **live** with the same code as the notebook. "
                    "The table under each step is the saved corpus result.")
        analysis_text = gr.Textbox(label="Input text", value=TEXT_EXAMPLE, lines=3)
        with gr.Tab("Tokenization"):
            b = gr.Button("Tokenize", variant="primary")
            out = gr.Dataframe(wrap=True, interactive=False)
            b.click(run_tokenizers, analysis_text, out)
            gr.Markdown("**Saved: Table B, tokenization comparison**")
            gr.Dataframe(read_result("tokenization_comparison.csv"), wrap=True, interactive=False)
        with gr.Tab("Preprocessing"):
            b = gr.Button("Run the Final Pipeline (stage by stage)", variant="primary")
            out = gr.Dataframe(headers=["Stage", "Output"], wrap=True, interactive=False)
            b.click(run_preprocessing, analysis_text, out)
            gr.Markdown("**Saved: Table A (before / after) and Table C (stemming vs lemmatization)**")
            gr.Dataframe(read_result("preprocessing_results.csv"), wrap=True, interactive=False)
            gr.Dataframe(read_result("stem_vs_lemma.csv"), wrap=True, interactive=False)
        with gr.Tab("POS Tagging / Custom POS"):
            b = gr.Button("Tag: default vs custom", variant="primary")
            out = gr.Dataframe(wrap=True, interactive=False)
            b.click(run_pos, analysis_text, out)
            gr.Markdown("**Saved: Table D (held-out domain tokens) and tagger accuracy**")
            gr.Dataframe(read_result("pos_tagging_results.csv"), wrap=True, interactive=False)
            gr.Dataframe(read_result("ex12_tagger_accuracy.csv"), wrap=True, interactive=False)
        with gr.Tab("NER"):
            b = gr.Button("Find entities", variant="primary")
            ner_spans = gr.HighlightedText(label="Domain NER (spaCy + EntityRuler)", combine_adjacent=False)
            ner_table = gr.Dataframe(wrap=True, interactive=False)
            b.click(run_ner, analysis_text, [ner_spans, ner_table])
            gr.Markdown("**Saved: Table E (total / correct / incorrect / missed)**")
            gr.Dataframe(read_result("ner_results.csv"), wrap=True, interactive=False)
        with gr.Tab("N-Gram Analysis"):
            with gr.Row():
                n_box = gr.Slider(1, 5, value=2, step=1, label="n")
                b = gr.Button("Count n-grams", variant="primary")
            ngram_summary = gr.Markdown()
            out = gr.Dataframe(wrap=True, interactive=False)
            b.click(run_ngrams, [analysis_text, n_box], [ngram_summary, out])
            gr.Markdown("**Saved: Table F, corpus n-grams (total, unique, top 10)**")
            gr.Dataframe(read_result("ngram_results.csv"), wrap=True, interactive=False)
        with gr.Tab("BPE Analysis"):
            b = gr.Button("Split words with BPE (vocabulary 2000)", variant="primary")
            out = gr.Dataframe(wrap=True, interactive=False)
            b.click(run_bpe, analysis_text, out)
            gr.Markdown("**Saved: Table G**")
            gr.Dataframe(read_result("bpe_results.csv"), wrap=True, interactive=False)

    # ---------- 3. Terms dictionary (inverted index) ----------
    with gr.Tab("3 · Terms Dictionary"):
        gr.Markdown("The inverted index of the Final Pipeline: each term, the documents it appears in (with counts) and its frequencies.")
        with gr.Row():
            sort_box = gr.Radio(["Term frequency", "Document frequency", "Alphabetical"], value="Term frequency", label="Sort by")
            filter_box = gr.Textbox(label="Filter terms containing", placeholder="e.g. anxi")
            dict_button = gr.Button("Create Dictionary", variant="primary")
        dict_count = gr.Markdown()
        dict_table = gr.Dataframe(wrap=True, interactive=False, max_height=600)
        for trigger in (dict_button.click, sort_box.change, filter_box.submit):
            trigger(show_dictionary, [sort_box, filter_box], [dict_count, dict_table])

    # ---------- 4. Search & evaluation ----------
    with gr.Tab("4 · Search & Evaluation"):
        eval_pick = gr.Dropdown(EVAL_CHOICES, label="Optional: pick one of the 14 judged evaluation queries (gives Precision / Recall / F1)")
        with gr.Row():
            query_box = gr.Textbox(label="Query", placeholder='coming off antidepressants  |  panic disorder  |  CBT AND depression NOT medication', scale=4)
            mode_box = gr.Dropdown(["Keyword", "Phrase", "Boolean"], value="Keyword", label="Query Type", scale=1)
            topk_box = gr.Slider(1, len(DOC_INFO), value=5, step=1, label="Show top K", scale=1)
        relevant_box = gr.Textbox(label="Relevant documents (optional): your own judgment for this query, e.g. D01, D06, D07",
                                  placeholder="Leave empty: our judgments are used for the 14 evaluation queries, "
                                              "otherwise an estimate from document topic labels")
        search_button = gr.Button("SEARCH", variant="primary")
        terms_out = gr.Markdown()
        docs_out = gr.Textbox(label="Retrieved Documents (ranked)", interactive=False)
        with gr.Row():
            n_out = gr.Textbox(label="Number of Results", interactive=False)
            time_out = gr.Textbox(label="Execution Time", interactive=False)
            p_out = gr.Textbox(label="Precision", interactive=False)
            r_out = gr.Textbox(label="Recall", interactive=False)
            f_out = gr.Textbox(label="F1-Score", interactive=False)
            p3_out = gr.Textbox(label="P@3", interactive=False)
        results_out = gr.Markdown()
        gr.Markdown("*Keyword:* any query term, ranked by TF-IDF. *Phrase:* terms in this order. "
                    "*Boolean:* **AND / OR / NOT in capitals** (lower-case 'not' is a search word), quotes = phrase, brackets allowed.")
        eval_pick.change(pick_eval_query, eval_pick, [query_box, mode_box])
        search_outputs = [terms_out, docs_out, n_out, time_out, p_out, r_out, f_out, p3_out, results_out]
        search_inputs = [query_box, mode_box, topk_box, relevant_box]
        search_button.click(run_search, search_inputs, search_outputs)
        query_box.submit(run_search, search_inputs, search_outputs)
        relevant_box.submit(run_search, search_inputs, search_outputs)

    # ---------- 5. Compare pipelines ----------
    with gr.Tab("5 · Compare Pipelines"):
        gr.Markdown("The same query on **Pipeline A** (NLTK stop words → stemming) and on the **Final Pipeline**: each with its "
                    "own index, and the query processed by the same pipeline as the documents.")
        cmp_pick = gr.Dropdown(EVAL_CHOICES, label="Optional: pick a judged evaluation query")
        with gr.Row():
            cmp_query = gr.Textbox(label="Query", value="do not stop taking antidepressants", scale=4)
            cmp_mode = gr.Dropdown(["Keyword", "Phrase", "Boolean"], value="Phrase", label="Query Type", scale=1)
            cmp_k = gr.Slider(1, len(DOC_INFO), value=3, step=1, label="Show top K", scale=1)
        cmp_relevant = gr.Textbox(label="Relevant documents (optional), e.g. D14", placeholder="Same rules as the Search tab")
        cmp_button = gr.Button("Compare Pipelines", variant="primary")
        cmp_note = gr.Markdown()
        with gr.Row():
            cmp_a = gr.Markdown()
            cmp_final = gr.Markdown()
        cmp_pick.change(pick_eval_query, cmp_pick, [cmp_query, cmp_mode])
        cmp_button.click(compare_pipelines, [cmp_query, cmp_mode, cmp_k, cmp_relevant], [cmp_note, cmp_a, cmp_final])
        gr.Markdown("#### Table H: Pipeline comparison (all 14 queries)")
        gr.Dataframe(read_result("pipeline_comparison.csv"), wrap=True, interactive=False)
        gr.Markdown("#### Table J: Overall performance")
        gr.Dataframe(read_result("evaluation_results.csv"), wrap=True, interactive=False)

    # ---------- 6. All results tables ----------
    with gr.Tab("6 · Results Tables"):
        RESULT_FILES = {"A. Preprocessing": "preprocessing_results.csv", "B. Tokenization": "tokenization_comparison.csv",
                        "C. Stemming vs lemmatization": "stem_vs_lemma.csv", "D. POS tagging": "pos_tagging_results.csv",
                        "E. NER totals": "ner_results.csv", "E. NER entities": "ner_entities.csv",
                        "F. N-grams": "ngram_results.csv", "F. Unigrams": "unigram_results.csv",
                        "F. Bigrams": "bigram_results.csv", "F. Trigrams": "trigram_results.csv", "G. BPE": "bpe_results.csv",
                        "H. Pipeline comparison": "pipeline_comparison.csv", "I. Retrieval results": "retrieval_results.csv",
                        "J. Overall performance": "evaluation_results.csv", "Evaluation per query": "evaluation_per_query.csv",
                        "Error analysis": "module6_error_analysis.csv", "Queries & relevance": "queries.csv",
                        "Document statistics": "document_statistics.csv"}
        table_pick = gr.Dropdown(list(RESULT_FILES), value="I. Retrieval results", label="Table")
        table_view = gr.Dataframe(wrap=True, interactive=False)
        table_pick.change(lambda name: read_result(RESULT_FILES[name]), table_pick, table_view)

    demo.load(load_documents, None, [load_status, stats_table, docs_table])
    demo.load(lambda: read_result("retrieval_results.csv"), None, table_view)
    demo.load(show_dictionary, [sort_box, filter_box], [dict_count, dict_table])

if __name__ == "__main__":
    demo.launch()
