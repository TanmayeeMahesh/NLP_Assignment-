"""
Corpus builder: raw official documents -> data/processed/corpus.json

    python src/build_corpus.py register
    python src/build_corpus.py extract
    python src/build_corpus.py clean
    python src/build_corpus.py section
    python src/build_corpus.py qc
    python src/build_corpus.py build --version 1.0

Helpers:
    python src/build_corpus.py validate --stage plan|final   # check the manifest
    python src/build_corpus.py outline D04                   # show a PDF's bookmarks + page count

Only documents with status=active in data/metadata/source_manifest.csv are processed.
Raw files are found by ID prefix: any file named  D01__<anything>.<pdf|html|htm|docx|csv|txt>
anywhere under data/raw/.

Corpus cleaning is deliberately conservative. It NEVER lowercases, removes stop words,
stems, lemmatizes, or removes punctuation or numbers: those are AS1 experiments.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# =========================================================================== paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTRACTED = DATA / "processed" / "extracted"
CLEAN = DATA / "processed" / "clean"
UNITS = DATA / "processed" / "units"
META = DATA / "metadata"
MANIFEST = META / "source_manifest.csv"
SCHEMA = META / "metadata.schema.json"
REGISTRY = META / "registry.json"
EXTRACTION_LOG = META / "extraction_log.json"
CLEANING_LOG = META / "cleaning_log.json"
QC_REPORT = META / "qc_report.csv"
QC_DUPLICATES = META / "qc_duplicates.csv"
CORPUS = DATA / "processed" / "corpus.json"

for _d in (EXTRACTED, CLEAN, UNITS, META):
    _d.mkdir(parents=True, exist_ok=True)

RAW_EXTS = {".pdf": "pdf", ".html": "html", ".htm": "html", ".docx": "docx", ".csv": "csv", ".txt": "txt"}
PAGE_MARKER = "[[PAGE {n}]]"
PAGE_RE = re.compile(r"^\[\[PAGE (\d+)\]\]$")
HEADING_PREFIX = "[[HEADING]] "

# =========================================================================== QC thresholds
# Starting values; adjust after inspecting your data and justify changes in the report.
QC = {
    "min_words_flag": 300,
    "min_words_reject": 100,
    "near_dup_flag": 0.80,            # Jaccard or containment of word 5-gram shingles
    "shingle_size": 5,
    "max_non_alpha_ratio": 0.30,      # non-letter share of non-space characters
    "min_dictionary_ratio": 0.80,     # share of words found in an English word list
    "max_single_char_ratio": 0.10,    # 'd e p r e s s i o n' style broken spacing
    "min_english_stopword_share": 0.15,
    "max_empty_page_share": 0.20,
    "max_urls_per_1000_words": 10,
    "max_citations_per_1000_words": 5,
    "boilerplate_page_share": 0.40,   # header/footer lines repeated on >= this share of pages
                                      # (0.40, not 0.50: WHO running headers appear on alternate pages only)
    "boilerplate_min_pages": 4,
    "references_min_position": 0.60,  # only cut a References section in the last 40% of a doc
}


# =========================================================================== small utils
def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def is_marker(line: str) -> bool:
    s = line.strip()
    return bool(PAGE_RE.match(s)) or s.startswith(HEADING_PREFIX.strip())


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


# =========================================================================== manifest
def load_schema() -> dict:
    return json.loads(SCHEMA.read_text(encoding="utf-8"))


def load_manifest(active_only: bool = True) -> list[dict]:
    schema = load_schema()
    sep = schema["list_separator"]
    list_fields = [k for k, v in schema["fields"].items() if v.get("list") and v.get("source") == "manual"]
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list_fields:
            if k in r and isinstance(r[k], str):
                r[k] = [x.strip() for x in r[k].split(sep) if x.strip()]
    if active_only:
        rows = [r for r in rows if r.get("status", "active") == "active"]
    return rows


def validate_manifest(stage: str = "plan") -> list[str]:
    schema = load_schema()
    vocab = schema["vocabularies"]
    registry = read_json(REGISTRY, {})
    problems, seen = [], set()
    for r in load_manifest(active_only=False):
        did = r.get("doc_id", "?")
        if did in seen:
            problems.append(f"ERROR {did}: duplicate doc_id")
        seen.add(did)
        if r.get("status") != "active":
            continue
        for field, spec in schema["fields"].items():
            if spec.get("source") != "manual":
                continue
            val = r.get(field)
            if val in (None, "", []):
                if field == "retrieval_date" and registry.get(did, {}).get("file_date"):
                    problems.append(f"WARN {did}: retrieval_date empty; will use file date "
                                    f"{registry[did]['file_date']}")
                elif field in ("retrieval_date", "source_url") and stage == "plan":
                    problems.append(f"PENDING {did}: '{field}' not filled yet")
                elif field == "source_url" and r.get("landing_page_url"):
                    problems.append(f"WARN {did}: source_url empty; landing_page_url will be cited")
                elif spec.get("required"):
                    problems.append(f"ERROR {did}: missing required field '{field}'")
                continue
            values = val if isinstance(val, list) else [val]
            if "vocab" in spec:
                bad = [v for v in values if v not in vocab[spec["vocab"]]]
                if bad:
                    problems.append(f"ERROR {did}: '{field}' values outside vocabulary: {bad}")
            if "pattern" in spec and not all(re.match(spec["pattern"], v) for v in values):
                problems.append(f"ERROR {did}: '{field}'={val!r} does not match {spec['pattern']}")
        if stage == "final":
            todo = [k for k, v in r.items() if isinstance(v, str) and v.strip().upper().startswith("TODO")]
            if todo:
                problems.append(f"ERROR {did}: TODO placeholders left in {todo}")
            if r.get("verification_status") != "verified":
                problems.append(f"ERROR {did}: verification_status is '{r.get('verification_status')}'")
    return problems


# =========================================================================== register
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_raw(doc_id: str) -> list[Path]:
    return sorted(p for p in RAW.rglob(f"{doc_id}__*")
                  if p.is_file() and p.suffix.lower() in RAW_EXTS)


def cmd_register() -> None:
    previous = read_json(REGISTRY, {})
    registry, missing, problems = {}, [], []
    active = load_manifest()
    for r in active:
        did = r["doc_id"]
        found = discover_raw(did)
        if not found:
            missing.append(did)
            continue
        if len(found) > 1:
            problems.append(f"{did}: several raw files {[p.name for p in found]}; using {found[0].name}")
        p = found[0]
        fmt = RAW_EXTS[p.suffix.lower()]
        sha = sha256_file(p)
        entry = {
            "raw_path": str(p.relative_to(ROOT)),
            "format": fmt,
            "sha256": sha,
            "size_bytes": p.stat().st_size,
            "file_date": dt.date.fromtimestamp(p.stat().st_mtime).isoformat(),
            "registered_at": now(),
        }
        if fmt != r.get("format"):
            problems.append(f"{did}: manifest format '{r.get('format')}' but file is '{fmt}' (file wins)")
        if did in previous and previous[did].get("sha256") not in (None, sha):
            problems.append(f"{did}: raw file CHANGED since last register (rerun all later steps)")
        if p.stat().st_size < 2048:
            problems.append(f"{did}: file is only {p.stat().st_size} bytes; check the download")
        registry[did] = entry
        print(f"  {did}  {fmt:4}  {entry['size_bytes']:>10,} B  sha256 {sha[:12]}  {p.name}")

    known = {r["doc_id"] for r in load_manifest(active_only=False)}
    strays = [p for p in RAW.rglob("*") if p.is_file() and p.name != ".gitkeep"
              and not any(p.name.startswith(k + "__") for k in known)
              and "_files" not in str(p.parent)]
    for p in strays:
        problems.append(f"unmatched raw file (name must start with 'Dnn__'): {p.relative_to(ROOT)}")

    write_json(REGISTRY, registry)
    print(f"\nRegistered {len(registry)}/{len(active)} active documents.")
    if missing:
        print(f"Missing raw files: {', '.join(missing)}")
    for msg in problems:
        print("WARN", msg)


# =========================================================================== extract
def _import_fitz():
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:
        try:
            import fitz  # older PyMuPDF
        except ImportError as e:
            raise SystemExit("PyMuPDF missing: pip install pymupdf") from e
    return fitz


def extract_pdf(path: Path) -> tuple[str, dict]:
    fitz = _import_fitz()
    doc = fitz.open(path)
    parts, empty, image_heavy = [], [], []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        n = len(text.strip())
        if n < 20:
            empty.append(i)
        if n < 150 and page.get_images(full=True):
            image_heavy.append(i)  # mostly picture: text may be inside the image (needs OCR)
        parts.append(f"{PAGE_MARKER.format(n=i)}\n{text.strip()}\n")
    info = {
        "tool": f"PyMuPDF {getattr(fitz, '__version__', getattr(fitz, 'VersionBind', '?'))}",
        "n_pages": doc.page_count,
        "empty_pages": empty,
        "image_heavy_pages": image_heavy,
        "outline": [[lvl, title.strip(), page] for lvl, title, page in doc.get_toc(simple=True)],
        "embedded_title": (doc.metadata or {}).get("title") or None,
    }
    doc.close()
    return "\n".join(parts), info


HTML_DROP_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "button", "template",
                  "link", "meta", "input", "select", "textarea", "canvas", "video", "audio", "picture"]
HTML_STRUCTURAL_DROP = ["nav", "header", "footer", "aside"]
HTML_DROP_ROLES = ["navigation", "banner", "contentinfo", "search", "complementary", "dialog"]
HTML_DROP_ATTR_TOKENS = {"cookie", "cookies", "consent", "breadcrumb", "breadcrumbs", "skip", "share",
                         "sharing", "social", "feedback", "newsletter", "print", "subscribe", "modal",
                         # embedded video/audio players (NHS Brightcove / video.js) and photo credits (WHO)
                         "vjs", "brightcove", "credit"}
HTML_MAIN_SELECTORS = ["main", "[role=main]", "article", "#maincontent", "#main-content", "#main",
                       "#content", ".main-content", ".content"]
HTML_BLOCK_TAGS = ["p", "li", "dt", "dd", "blockquote", "figcaption", "pre", "tr", "table",
                   "ul", "ol", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6"]


def _attr_tokens(tag) -> set:
    vals = [tag.get("id") or ""] + list(tag.get("class") or [])
    return {t for v in vals for t in re.split(r"[-_\s]+", str(v).lower()) if t}


def extract_html(path: Path) -> tuple[str, dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise SystemExit("BeautifulSoup missing: pip install beautifulsoup4 lxml") from e
    try:
        import lxml  # noqa: F401
        parser = "lxml"
    except ImportError:
        parser = "html.parser"
    soup = BeautifulSoup(path.read_bytes(), parser)
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    for t in soup(HTML_DROP_TAGS):
        t.decompose()

    candidates = []
    for sel in HTML_MAIN_SELECTORS:
        candidates += soup.select(sel)
    candidates = [c for c in candidates if len(c.get_text(strip=True)) > 0]
    container = max(candidates, key=lambda c: len(c.get_text(" ", strip=True))) if candidates \
        else (soup.body or soup)
    selector_used = container.name if candidates else "body(fallback)"

    targets = list(container.find_all(HTML_STRUCTURAL_DROP))
    targets += container.find_all(attrs={"role": HTML_DROP_ROLES})
    targets += [t for t in container.find_all(True) if _attr_tokens(t) & HTML_DROP_ATTR_TOKENS]
    removed = 0
    for t in targets:
        try:
            if t.parent is not None:
                t.decompose()
                removed += 1
        except Exception:
            pass

    for br in container.find_all("br"):
        br.replace_with("\n")
    for h in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        h.insert(0, "\n" + HEADING_PREFIX)
    for cell in container.find_all(["td", "th"]):
        cell.append("; ")
    for tag in container.find_all(HTML_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    lines = []
    for line in container.get_text().split("\n"):
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if line.endswith(";"):
            line = line.rstrip("; ").strip()
        if line and line != HEADING_PREFIX.strip():
            lines.append(line)
    info = {"tool": f"BeautifulSoup ({parser})", "html_title": title,
            "container": selector_used, "removed_elements": removed, "n_pages": 0}
    return "\n".join(lines) + "\n", info


def extract_docx(path: Path) -> tuple[str, dict]:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:
        raise SystemExit("python-docx missing: pip install python-docx") from e
    d = docx.Document(str(path))
    lines = []
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            txt = p.text.strip()
            if txt:
                style = (p.style.name if p.style is not None else "") or ""
                lines.append((HEADING_PREFIX if style.lower().startswith("heading") else "") + txt)
        elif tag == "tbl":
            for row in Table(child, d).rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    lines.append("; ".join(dict.fromkeys(cells)))
    return "\n".join(lines) + "\n", {"tool": "python-docx", "n_pages": 0}


def extract_csv(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(raw[:4096])
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(raw.splitlines(), dialect=dialect))
    lines = []
    for row in rows:
        parts = [f"{k.strip()}: {str(v).strip()}" for k, v in row.items() if k and v and str(v).strip()]
        if parts:
            lines.append("; ".join(parts) + ".")
    return "\n".join(lines) + "\n", {"tool": "csv", "n_rows": len(rows), "n_pages": 0}


def extract_txt(path: Path) -> tuple[str, dict]:
    return path.read_text(encoding="utf-8", errors="replace"), {"tool": "plain text", "n_pages": 0}


EXTRACTORS = {"pdf": extract_pdf, "html": extract_html, "docx": extract_docx,
              "csv": extract_csv, "txt": extract_txt}


def cmd_extract() -> None:
    registry = read_json(REGISTRY, {})
    if not registry:
        raise SystemExit("Run 'register' first.")
    log = read_json(EXTRACTION_LOG, {})
    for did, entry in sorted(registry.items()):
        path = ROOT / entry["raw_path"]
        try:
            text, info = EXTRACTORS[entry["format"]](path)
        except SystemExit:
            raise
        except Exception as e:  # keep going; report the failure
            print(f"  {did}  FAILED: {type(e).__name__}: {e}")
            log[did] = {"error": f"{type(e).__name__}: {e}", "extracted_at": now()}
            continue
        (EXTRACTED / f"{did}.txt").write_text(text, encoding="utf-8")
        info.update({"n_chars": len(text), "n_words": word_count(text), "extracted_at": now()})
        log[did] = info
        extra = ""
        if entry["format"] == "pdf":
            extra = (f"  pages {info['n_pages']}, empty {len(info['empty_pages'])}, "
                     f"image-heavy {len(info['image_heavy_pages'])}, outline entries {len(info['outline'])}")
        print(f"  {did}  {entry['format']:4}  {info['n_words']:>8,} words{extra}")
    write_json(EXTRACTION_LOG, log)
    print(f"\nExtracted text written to {EXTRACTED.relative_to(ROOT)}/")


# =========================================================================== clean
LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
             "\ufb05": "ft", "\ufb06": "st"}
QUOTES = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
          "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"'}
SPACES = {"\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
          "\u3000": " "}
INVISIBLE = {"\u00ad": "", "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "", "\ufeff": ""}
BULLET_RE = re.compile(r"^[ \t]*[\u2022\u2023\u2043\u2219\u25aa\u25ab\u25cf\u25cb\u25a0\u25a1\u25e6"
                       r"\u27a2\u25ba\u25b6\u2713\u2714\u00b7\uf0b7\uf0a7\uf076\uf0d8\uf0fc][ \t]*"
                       r"|^[ \t]*y[ \t]+(?=\S)", re.M)   # WHO PDFs: symbol-font bullet extracts as a lone "y"
# Bullets are standardised to this marker (not deleted) so unwrap_lines never joins a bullet item
# onto the previous line; unwrap_lines removes the marker afterwards.
BULLET_MARK = "\u2022 "
BULLET_MARK_RE = re.compile(r"(?m)^[ \t]*\u2022 ?")
PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
TOC_LINE_RE = re.compile(r"(\.\s?){4,}\s*\d+\s*$|\u2026{2,}\s*\d+\s*$|^\s*(table of\s+)?contents\s*$", re.I)
REFERENCES_HEADING_RE = re.compile(
    r"^\s*(\d+(\.\d+)*\.?\s+)?(references|bibliography|reference list|works cited|literature cited)\s*:?\s*$",
    re.I)
HTML_BOILERPLATE_RES = [re.compile(p, re.I) for p in [
    r"^skip to (main )?content$", r"^back to top$", r"^(share|print)( this page)?$",
    r"^.*\bcookies?\b.*$", r"^accept( all)?( cookies)?$", r"^next review due:.*$",
    r"^was this (page|information) (useful|helpful)\??$", r"^report a problem.*$",
    r"^(follow us|sign up).*$", r"^.*\(opens in (a )?new (tab|window)\)$",
]]
PAGE_LAST_REVIEWED_RE = re.compile(r"^page last reviewed:\s*(.+)$", re.I)
KEEP_HYPHEN_PREFIXES = {"self", "non", "well", "long", "short", "first", "second", "third", "follow",
                        "high", "low", "one", "two", "three", "peer", "age", "evidence", "person",
                        "family", "short-term", "long-term", "post", "pre", "co", "re", "ex"}


def rule_normalise_characters(text: str, fmt: str) -> tuple[str, dict]:
    stats = Counter()
    text = unicodedata.normalize("NFC", text)
    for table, name in ((LIGATURES, "ligatures"), (QUOTES, "curly_quotes"),
                        (SPACES, "special_spaces"), (INVISIBLE, "invisible_chars")):
        for k, v in table.items():
            n = text.count(k)
            if n:
                stats[name] += n
                text = text.replace(k, v)
    text, n = BULLET_RE.subn(BULLET_MARK, text); stats["bullet_glyphs"] += n
    text, n = PRIVATE_USE_RE.subn("", text); stats["private_use_glyphs"] += n
    text, n = CONTROL_RE.subn("", text); stats["control_chars"] += n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text, dict(stats)


def rule_html_boilerplate(text: str, fmt: str) -> tuple[str, dict]:
    if fmt != "html":
        return text, {"skipped": "not html"}
    out, removed, reviewed = [], 0, None
    for line in text.split("\n"):
        s = line.strip()
        bare = s[len(HEADING_PREFIX):] if s.startswith(HEADING_PREFIX) else s
        m = PAGE_LAST_REVIEWED_RE.match(bare)
        if m:
            reviewed = m.group(1).strip()
            removed += 1
            continue
        if any(r.match(bare) for r in HTML_BOILERPLATE_RES):
            removed += 1
            continue
        out.append(line)
    stats = {"lines_removed": removed}
    if reviewed:
        stats["page_last_reviewed"] = reviewed
    return "\n".join(out), stats


def _split_pages(text: str) -> list[tuple[str | None, list[str]]]:
    """[(marker_line or None, [lines])] in order."""
    pages, current = [], (None, [])
    for line in text.split("\n"):
        if PAGE_RE.match(line.strip()):
            if current[0] is not None or current[1]:
                pages.append(current)
            current = (line.strip(), [])
        else:
            current[1].append(line)
    pages.append(current)
    return pages


def _join_pages(pages) -> str:
    out = []
    for marker, lines in pages:
        if marker:
            out.append(marker)
        out.extend(lines)
    return "\n".join(out)


def rule_headers_footers(text: str, fmt: str) -> tuple[str, dict]:
    if fmt != "pdf":
        return text, {"skipped": "not pdf"}
    pages = _split_pages(text)
    real = [p for p in pages if p[0]]
    key = lambda s: re.sub(r"\s+", " ", re.sub(r"\d+", "#", s.strip().lower()))

    def zone(lines):
        # top/bottom 5 non-empty lines: NICE footers are 5 lines long
        # (title, copyright, URL tail, "Page n of", total); a 3-line zone missed the first two
        idx = [i for i, l in enumerate(lines) if l.strip()]
        return set(idx[:5] + idx[-5:])

    counts = Counter()
    for _, lines in real:
        counts.update({key(lines[i]) for i in zone(lines)})
    threshold = max(2, math.ceil(QC["boilerplate_page_share"] * len(real)))
    repeated = {k for k, c in counts.items()
                if c >= threshold and len(real) >= QC["boilerplate_min_pages"] and k}
    removed_repeated = removed_numbers = 0
    examples = Counter()
    new_pages = []
    for marker, lines in pages:
        z = zone(lines) if marker else set()
        kept = []
        for i, l in enumerate(lines):
            if i in z and key(l) in repeated:
                removed_repeated += 1
                examples[l.strip()[:80]] += 1
                continue
            if i in z and PAGE_NUMBER_RE.match(l):
                removed_numbers += 1
                continue
            kept.append(l)
        new_pages.append((marker, kept))
    return _join_pages(new_pages), {"repeated_lines_removed": removed_repeated,
                                    "page_number_lines_removed": removed_numbers,
                                    "examples": [e for e, _ in examples.most_common(5)]}


def rule_toc_lines(text: str, fmt: str) -> tuple[str, dict]:
    out, removed = [], 0
    for line in text.split("\n"):
        if not is_marker(line) and TOC_LINE_RE.search(line):
            removed += 1
            continue
        out.append(line)
    return "\n".join(out), {"toc_lines_removed": removed}


def rule_reference_section(text: str, fmt: str) -> tuple[str, dict]:
    lines = text.split("\n")
    total = max(len(text), 1)
    pos = 0
    for i, line in enumerate(lines):
        bare = line.strip()
        bare = bare[len(HEADING_PREFIX):] if bare.startswith(HEADING_PREFIX) else bare
        if pos / total >= QC["references_min_position"] and REFERENCES_HEADING_RE.match(bare):
            cut = "\n".join(lines[i:])
            return "\n".join(lines[:i]), {"cut_at_line": i, "heading": bare,
                                          "chars_removed": len(cut)}
        pos += len(line) + 1
    return text, {"cut_at_line": None}


HYPHEN_BREAK_RE = re.compile(r"\b([A-Za-z]+)-\n([ \t]*)([a-z][A-Za-z]*)")


def rule_hyphenated_linebreaks(text: str, fmt: str) -> tuple[str, dict]:
    if fmt != "pdf":
        return text, {"skipped": "not pdf"}
    vocab = Counter(w.lower() for w in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", text))
    stats = Counter()

    def fix(m):
        a, b = m.group(1), m.group(3)
        hyph, joined = f"{a}-{b}".lower(), f"{a}{b}".lower()
        if a.lower() in KEEP_HYPHEN_PREFIXES or vocab[hyph] > vocab[joined]:
            stats["kept_hyphen"] += 1
            return f"{a}-{b}\n"
        stats["joined"] += 1
        return f"{a}{b}\n"

    return HYPHEN_BREAK_RE.sub(fix, text), dict(stats)


def rule_unwrap_lines(text: str, fmt: str) -> tuple[str, dict]:
    if fmt != "pdf":
        return BULLET_MARK_RE.sub("", text), {"skipped": "not pdf"}
    out, joins = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if out and s and not s.startswith(BULLET_MARK.strip()) and not is_marker(s) and out[-1].strip() \
                and not is_marker(out[-1]) and not re.search(r"[.!?:;]\s*$", out[-1]) and re.match(r"[a-z(]", s):
            out[-1] = out[-1].rstrip() + " " + s
            joins += 1
        else:
            out.append(line)
    return BULLET_MARK_RE.sub("", "\n".join(out)), {"lines_joined": joins}


def rule_whitespace(text: str, fmt: str) -> tuple[str, dict]:
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"
    return text, {}


CLEANING_RULES = [  # order matters; justify it in the report
    ("normalise_characters", rule_normalise_characters),
    ("html_boilerplate", rule_html_boilerplate),
    ("headers_footers", rule_headers_footers),
    ("toc_lines", rule_toc_lines),
    ("reference_section", rule_reference_section),
    ("hyphenated_linebreaks", rule_hyphenated_linebreaks),
    ("unwrap_lines", rule_unwrap_lines),
    ("whitespace", rule_whitespace),
]


def cmd_clean() -> None:
    registry = read_json(REGISTRY, {})
    log = {}
    for did, entry in sorted(registry.items()):
        src = EXTRACTED / f"{did}.txt"
        if not src.exists():
            print(f"  {did}  skipped: not extracted")
            continue
        text = src.read_text(encoding="utf-8")
        before = len(text)
        steps = []
        for name, rule in CLEANING_RULES:
            n0 = len(text)
            text, stats = rule(text, entry["format"])
            steps.append({"rule": name, "chars_delta": len(text) - n0, **stats})
        (CLEAN / f"{did}.txt").write_text(text, encoding="utf-8")
        log[did] = {"chars_before": before, "chars_after": len(text), "steps": steps, "cleaned_at": now()}
        pct = 100 * (before - len(text)) / max(before, 1)
        print(f"  {did}  {before:>9,} -> {len(text):>9,} chars  ({pct:4.1f}% removed)")
    write_json(CLEANING_LOG, log)
    print(f"\nPer-rule details in {CLEANING_LOG.relative_to(ROOT)}")


# =========================================================================== section
def _page_texts(text: str) -> dict[int, str]:
    pages = {}
    for marker, lines in _split_pages(text):
        if marker:
            pages[int(PAGE_RE.match(marker).group(1))] = "\n".join(lines)
    return pages


def _strip_markers(text: str) -> tuple[str, list[str]]:
    headings, out = [], []
    for line in text.split("\n"):
        s = line.strip()
        if PAGE_RE.match(s):
            continue
        if s.startswith(HEADING_PREFIX):
            s = s[len(HEADING_PREFIX):].strip()
            headings.append(s)
        out.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip(), headings


def _parse_pages(spec: str, n_pages: int) -> list[int]:
    pages = set()
    for part in re.split(r"[,;]", spec):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            pages.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            pages.add(int(part))
        else:
            raise ValueError(f"bad page spec '{part}'")
    return sorted(p for p in pages if 1 <= p <= n_pages)


def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _toc_pages(patterns: list[str], outline: list, n_pages: int):
    pages, matched, unmatched = set(), [], []
    for pat in patterns:
        np_ = _norm_title(pat)
        hit = next((i for i, (_, title, pg) in enumerate(outline)
                    if pg > 0 and np_ and np_ in _norm_title(title)), None)
        if hit is None:
            unmatched.append(pat)
            continue
        lvl, title, start = outline[hit]
        end = n_pages
        for lvl2, _, pg2 in outline[hit + 1:]:
            if lvl2 <= lvl and pg2 > 0:
                end = pg2 if pg2 > start else start  # include the page where the next section starts
                break
        pages.update(range(start, end + 1))
        matched.append({"pattern": pat, "outline_title": title, "pages": f"{start}-{end}"})
    return sorted(pages), matched, unmatched


def _runs(pages: list[int]) -> list[tuple[int, int]]:
    runs = []
    for p in pages:
        if runs and p == runs[-1][1] + 1:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    return [tuple(r) for r in runs]


def select_units(doc_id: str, fmt: str, text: str, spec: str, extraction: dict) -> dict:
    spec = (spec or "").strip()
    flags, matched = [], []
    if fmt != "pdf":
        if spec and spec.lower() != "all":
            flags.append("sections_ignored_for_non_pdf")
        body, headings = _strip_markers(text)
        units = [{"unit_id": f"{doc_id}-U01", "parent_id": doc_id, "pages": None,
                  "headings": headings[:30], "text": body, "n_words": word_count(body)}]
        return {"mode": "all", "spec": spec, "flags": flags, "matched": matched, "units": units}

    page_text = _page_texts(text)
    n_pages = max(page_text) if page_text else 0
    mode = "all"
    if not spec or spec.upper().startswith("TODO"):
        pages, mode = list(range(1, n_pages + 1)), "all(unspecified)"
        flags.append("sections_unspecified")
    elif spec.lower() == "all":
        pages = list(range(1, n_pages + 1))
    elif spec.lower().startswith("pages:"):
        pages, mode = _parse_pages(spec.split(":", 1)[1], n_pages), "pages"
    elif spec.lower().startswith("toc:"):
        patterns = [p.strip() for p in spec.split(":", 1)[1].split(";") if p.strip()]
        outline = extraction.get("outline") or []
        if not outline:
            pages, mode = list(range(1, n_pages + 1)), "all(no_outline)"
            flags.append("toc_requested_but_pdf_has_no_outline")
        else:
            pages, matched, unmatched = _toc_pages(patterns, outline, n_pages)
            mode = "toc"
            if unmatched:
                flags.append("toc_unmatched:" + " | ".join(unmatched))
            if not pages:
                pages, mode = list(range(1, n_pages + 1)), "all(toc_no_match)"
    else:
        raise ValueError(f"{doc_id}: unrecognised sections_included '{spec}'")

    units = []
    for k, (a, b) in enumerate(_runs(pages), start=1):
        raw = "\n".join(f"{PAGE_MARKER.format(n=p)}\n{page_text.get(p, '')}" for p in range(a, b + 1))
        body, _ = _strip_markers(raw)
        if not body.strip():
            continue
        titles = [m["outline_title"] for m in matched
                  if a <= int(m["pages"].split("-")[0]) <= b]
        units.append({"unit_id": f"{doc_id}-U{k:02d}", "parent_id": doc_id, "pages": f"{a}-{b}",
                      "headings": titles, "text": body, "n_words": word_count(body)})
    return {"mode": mode, "spec": spec, "flags": flags, "matched": matched,
            "pages_selected": len(pages), "pages_total": n_pages, "units": units}


def cmd_section() -> None:
    registry = read_json(REGISTRY, {})
    ext_log = read_json(EXTRACTION_LOG, {})
    manifest = {r["doc_id"]: r for r in load_manifest()}
    for did, entry in sorted(registry.items()):
        src = CLEAN / f"{did}.txt"
        if not src.exists():
            print(f"  {did}  skipped: not cleaned")
            continue
        try:
            sel = select_units(did, entry["format"], src.read_text(encoding="utf-8"),
                               manifest[did].get("sections_included", ""), ext_log.get(did, {}))
        except ValueError as e:
            print(f"  {did}  ERROR {e}")
            continue
        write_json(UNITS / f"{did}.json", {"doc_id": did, "sectioned_at": now(), **sel})
        words = sum(u["n_words"] for u in sel["units"])
        pages = f"pages {sel.get('pages_selected', '-')}/{sel.get('pages_total', '-')}" \
            if entry["format"] == "pdf" else "whole page"
        print(f"  {did}  {sel['mode']:18} {pages:16} units {len(sel['units']):>2}  {words:>8,} words"
              + (f"  FLAGS {sel['flags']}" if sel["flags"] else ""))


def cmd_outline(doc_id: str) -> None:
    ext = read_json(EXTRACTION_LOG, {}).get(doc_id)
    if not ext:
        raise SystemExit(f"No extraction info for {doc_id}; run register + extract first.")
    print(f"{doc_id}: {ext.get('n_pages')} pages; empty {ext.get('empty_pages')}; "
          f"image-heavy {ext.get('image_heavy_pages')}")
    outline = ext.get("outline") or []
    if not outline:
        print("No PDF outline (bookmarks). Use 'pages: a-b' in sections_included "
              "(open the PDF and read page numbers from its own page counter, not printed numbers).")
    for lvl, title, page in outline:
        print(f"  p.{page:>4}  {'  ' * (lvl - 1)}{title}")


# =========================================================================== QC
TOPIC_SEEDS = {
    "depression": ["depress", "low mood"], "anxiety": ["anxi", "panic", "worry"],
    "stress": ["stress"], "sleep": ["sleep", "insomnia"],
    "self_esteem": ["self-esteem", "self esteem", "confidence"], "loneliness": ["lonel", "isolat"],
    "grief": ["grief", "bereave"], "trauma": ["trauma", "ptsd", "post-traumatic"],
    "eating": ["eating", "anorexia", "bulimia", "binge"],
    "substance_use": ["alcohol", "drug", "substance"],
    "medication": ["medic", "antidepressant", "ssri", "dose", "prescri"],
    "self_harm": ["self-harm", "self harm", "harm yourself", "self-injur"], "suicide": ["suicid"],
    "crisis_services": ["crisis", "helpline", "988", "14416", "emergency"],
    "relationships": ["relationship", "partner"], "family": ["family", "parent"],
}
QC_STOPWORDS = set("""a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each few for from
further had has have having he her here hers him his how i if in into is it its itself just may me
might more most must my no nor not now of off on once only or other our out over own same she should
so some such than that the their them then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your yours""".split())
ENGLISH_CHECK = {"the", "and", "of", "to", "a", "in", "is", "for", "you", "or", "that", "with", "are",
                 "be", "it", "as", "your", "can", "on", "this", "if", "not", "may", "have"}
BOILERPLATE_LEFTOVERS = ["skip to main content", "all rights reserved", "cookie", "subject to notice of rights",
                         "javascript", "back to top"]


def _english_words():
    try:
        from nltk.corpus import words
        try:
            return {w.lower() for w in words.words()}
        except LookupError:
            import nltk
            nltk.download("words", quiet=True)
            return {w.lower() for w in words.words()}
    except Exception:
        return None


def _in_dictionary(tok: str, wordset: set) -> bool:
    if tok in wordset:
        return True
    for suf in ("s", "es", "ed", "ing", "ly", "al", "er", "ers", "ment", "ness", "ion", "ions"):
        if tok.endswith(suf) and tok[: -len(suf)] in wordset:
            return True
    return False


def _shingles(text: str, k: int) -> set:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(len(w) - k + 1, 0))}


def qc_document(did: str, doc_text: str, units_doc: dict, manifest_row: dict, ext: dict,
                wordset) -> tuple[list[str], list[str]]:
    """Returns (flags affecting status, informational notes)."""
    flags, notes = [], []
    n_words = word_count(doc_text)
    if n_words < QC["min_words_reject"]:
        flags.append(f"REJECT:too_short({n_words}w)")
    elif n_words < QC["min_words_flag"]:
        flags.append(f"short({n_words}w)")

    # extraction errors
    if "\ufffd" in doc_text:
        flags.append(f"replacement_chars({doc_text.count(chr(0xfffd))})")
    lig = sum(doc_text.count(k) for k in LIGATURES)
    if lig:
        flags.append(f"ligatures_left({lig})")
    nonspace = re.sub(r"\s", "", doc_text)
    if nonspace:
        non_alpha = sum(not c.isalpha() for c in nonspace) / len(nonspace)
        if non_alpha > QC["max_non_alpha_ratio"]:
            flags.append(f"non_alpha_ratio({non_alpha:.2f})")
    toks = re.findall(r"[A-Za-z]+", doc_text)
    if toks:
        single = sum(len(t) == 1 and t.lower() not in ("a", "i") for t in toks) / len(toks)
        if single > QC["max_single_char_ratio"]:
            flags.append(f"broken_spacing({single:.2f})")

    # PDF page problems, restricted to the selected pages
    if manifest_row and ext.get("n_pages"):
        sel_pages = set()
        for u in units_doc.get("units", []):
            if u.get("pages"):
                a, b = map(int, u["pages"].split("-"))
                sel_pages.update(range(a, b + 1))
        empty = [p for p in ext.get("empty_pages", []) if p in sel_pages]
        if sel_pages and len(empty) / len(sel_pages) > QC["max_empty_page_share"]:
            flags.append(f"many_empty_pages({len(empty)}/{len(sel_pages)})")
        img = [p for p in ext.get("image_heavy_pages", []) if p in sel_pages]
        if img:
            flags.append(f"image_heavy_pages_text_may_be_missing({img[:10]})")
    for f in units_doc.get("flags", []):
        flags.append(f)

    # OCR / dictionary quality
    long_toks = [t.lower() for t in toks if len(t) >= 4]
    if wordset is None:
        notes.append("dictionary_check_skipped(nltk words unavailable)")
    elif long_toks:
        ratio = sum(_in_dictionary(t, wordset) for t in long_toks) / len(long_toks)
        notes.append(f"dictionary_ratio={ratio:.2f}")
        if ratio < QC["min_dictionary_ratio"]:
            flags.append(f"low_dictionary_ratio({ratio:.2f})")

    # language
    lower = [t.lower() for t in toks]
    if lower:
        share = sum(t in ENGLISH_CHECK for t in lower) / len(lower)
        if share < QC["min_english_stopword_share"]:
            flags.append(f"possibly_not_english_prose({share:.2f})")

    # contamination
    per_k = 1000 / max(n_words, 1)
    toc_left = sum(bool(TOC_LINE_RE.search(l)) for l in doc_text.split("\n"))
    if toc_left:
        flags.append(f"toc_lines_left({toc_left})")
    urls = len(re.findall(r"https?://|www\.", doc_text))
    if urls * per_k > QC["max_urls_per_1000_words"]:
        flags.append(f"url_dense({urls})")
    cites = len(re.findall(r"\([A-Z][A-Za-z'\-]+ (et al\.?|and [A-Z][A-Za-z'\-]+),? \d{4}[a-z]?\)"
                           r"|\[\d+(?:[,\u2013-]\s*\d+)*\]", doc_text))
    if cites * per_k > QC["max_citations_per_1000_words"]:
        flags.append(f"citation_dense({cites})")
    low = doc_text.lower()
    left = [b for b in BOILERPLATE_LEFTOVERS if b in low]
    if left:
        flags.append(f"boilerplate_left({left})")

    # topic alignment
    for topic in manifest_row.get("topics", []):
        hits = sum(low.count(s) for s in TOPIC_SEEDS.get(topic, []))
        rate = hits * per_k
        if hits == 0:
            flags.append(f"topic_not_found({topic})")
        else:
            notes.append(f"{topic}={rate:.1f}/1k")
    return flags, notes


def _top_terms(texts: dict[str, str], k: int = 10) -> dict[str, str]:
    """Diagnostic only (lowercased, stopwords skipped HERE, not in the corpus)."""
    tfs = {d: Counter(t for t in re.findall(r"[a-z][a-z\-]{2,}", x.lower()) if t not in QC_STOPWORDS)
           for d, x in texts.items()}
    df = Counter(t for c in tfs.values() for t in c)
    n = len(texts)
    out = {}
    for d, c in tfs.items():
        total = sum(c.values()) or 1
        scored = sorted(c, key=lambda t: -(c[t] / total) * math.log((n + 1) / (df[t] + 0.5)))
        out[d] = ", ".join(scored[:k])
    return out


def cmd_qc() -> None:
    manifest = {r["doc_id"]: r for r in load_manifest()}
    ext_log = read_json(EXTRACTION_LOG, {})
    wordset = _english_words()
    texts, units_by_doc = {}, {}
    for did in manifest:
        f = UNITS / f"{did}.json"
        if f.exists():
            units_by_doc[did] = read_json(f, {})
            texts[did] = "\n\n".join(u["text"] for u in units_by_doc[did].get("units", []))

    # duplicates
    exact = {}
    for did, t in texts.items():
        key = hashlib.sha256(re.sub(r"[^a-z0-9]+", " ", t.lower()).strip().encode()).hexdigest()
        exact.setdefault(key, []).append(did)
    sh = {d: _shingles(t, QC["shingle_size"]) for d, t in texts.items()}
    dup_rows, dup_flags = [], {d: [] for d in texts}
    ids = sorted(texts)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            A, B = sh[a], sh[b]
            if not A or not B:
                continue
            inter = len(A & B)
            jac = inter / len(A | B)
            cont = inter / min(len(A), len(B))
            if jac >= 0.2 or cont >= 0.3:
                dup_rows.append({"doc_a": a, "doc_b": b, "jaccard": f"{jac:.3f}", "containment": f"{cont:.3f}"})
            if jac >= QC["near_dup_flag"] or cont >= QC["near_dup_flag"]:
                dup_flags[a].append(f"near_duplicate_of({b})")
                dup_flags[b].append(f"near_duplicate_of({a})")
    for group in exact.values():
        if len(group) > 1:
            for d in group:
                dup_flags[d].append(f"exact_duplicate_of({[g for g in group if g != d]})")

    top = _top_terms(texts) if texts else {}
    rows = []
    for did, row in manifest.items():
        if did not in texts:
            rows.append({"doc_id": did, "qc_status": "missing", "qc_flags": "not processed (raw file missing?)"})
            continue
        flags, notes = qc_document(did, texts[did], units_by_doc[did], row, ext_log.get(did, {}), wordset)
        flags += dup_flags.get(did, [])
        status = "reject" if any(f.startswith("REJECT") for f in flags) else ("flag" if flags else "pass")
        rows.append({"doc_id": did, "format": ext_log.get(did, {}).get("tool", ""),
                     "n_units": len(units_by_doc[did].get("units", [])), "n_words": word_count(texts[did]),
                     "qc_status": status, "qc_flags": "; ".join(flags), "qc_notes": "; ".join(notes),
                     "top_terms": top.get(did, "")})
    cols = ["doc_id", "format", "n_units", "n_words", "qc_status", "qc_flags", "qc_notes", "top_terms"]
    with QC_REPORT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with QC_DUPLICATES.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_a", "doc_b", "jaccard", "containment"])
        w.writeheader()
        w.writerows(dup_rows)
    for r in rows:
        print(f"  {r['doc_id']}  {r['qc_status']:7} {str(r.get('n_words', '')):>8}  {r['qc_flags'][:110]}")
    c = Counter(r["qc_status"] for r in rows)
    print(f"\nQC: {dict(c)}. Details: {QC_REPORT.relative_to(ROOT)}, {QC_DUPLICATES.relative_to(ROOT)}")


# =========================================================================== build
def citation_string(r: dict, retrieval_date: str | None) -> str:
    title = r["title"] + (f" ({r['reference_number']})" if r.get("reference_number") else "")
    parts = [f"{r['organization']}.", f"{title}."]
    if r.get("publication_date"):
        upd = f", updated {r['last_updated_date']}" if r.get("last_updated_date") else ""
        parts.append(f"{r['publication_date'][:4]}{upd}.")
    url = r.get("source_url") or r.get("landing_page_url")
    if url:
        parts.append(f"Retrieved {retrieval_date or 'n.d.'} from {url}.")
    return " ".join(parts)


def cmd_build(version: str) -> None:
    manifest = load_manifest()
    registry = read_json(REGISTRY, {})
    ext_log = read_json(EXTRACTION_LOG, {})
    clean_log = read_json(CLEANING_LOG, {})
    qc = {}
    if QC_REPORT.exists():
        with QC_REPORT.open(encoding="utf-8") as f:
            qc = {row["doc_id"]: row for row in csv.DictReader(f)}
    else:
        print("WARN: no qc_report.csv; run 'qc' first")
    docs, skipped = [], []
    for r in manifest:
        did = r["doc_id"]
        uf = UNITS / f"{did}.json"
        if not uf.exists():
            skipped.append((did, "no units"))
            continue
        q = qc.get(did, {})
        if q.get("qc_status") == "reject":
            skipped.append((did, "QC reject"))
            continue
        sel = read_json(uf, {})
        units = sel.get("units", [])
        text = "\n\n".join(u["text"] for u in units)
        reg = registry.get(did, {})
        retrieval = r.get("retrieval_date") or reg.get("file_date")
        ext = {k: v for k, v in ext_log.get(did, {}).items() if k != "outline"}
        steps = clean_log.get(did, {}).get("steps", [])
        reviewed = next((s.get("page_last_reviewed") for s in steps if s.get("page_last_reviewed")), None)
        docs.append({
            **{k: v for k, v in r.items()},
            "retrieval_date": retrieval,
            "retrieval_date_source": "manifest" if r.get("retrieval_date") else "file_timestamp",
            "format": reg.get("format", r.get("format")),
            "raw_path": reg.get("raw_path"),
            "sha256": reg.get("sha256"),
            "page_last_reviewed": reviewed,
            "extraction": ext,
            "cleaning": {"chars_before": clean_log.get(did, {}).get("chars_before"),
                         "chars_after": clean_log.get(did, {}).get("chars_after")},
            "selection": {k: v for k, v in sel.items() if k not in ("units", "doc_id")},
            "qc_status": q.get("qc_status"),
            "qc_flags": [x for x in (q.get("qc_flags") or "").split("; ") if x],
            "citation": citation_string(r, retrieval),
            "n_units": len(units),
            "n_chars": len(text),
            "n_words": word_count(text),
            "units": units,
            "text": text,
        })
    corpus = {
        "corpus_name": "Consumer mental-health support guidance (AS1 evidence corpus)",
        "version": version,
        "built_at": now(),
        "n_documents": len(docs),
        "n_words_total": sum(d["n_words"] for d in docs),
        "documents_skipped": [{"doc_id": d, "reason": why} for d, why in skipped],
        "qc_thresholds": QC,
        "notes": "Text is conservatively cleaned only: case, stop words, punctuation and numbers are "
                 "preserved for AS1 experiments. Page/heading markers are removed from 'text'.",
        "documents": docs,
    }
    write_json(CORPUS, corpus)
    flagged = [d["doc_id"] for d in docs if d["qc_status"] == "flag"]
    print(f"Wrote {CORPUS.relative_to(ROOT)}  version {version}")
    print(f"  documents: {len(docs)}  words: {corpus['n_words_total']:,}")
    print(f"  QC-flagged (included, review them): {flagged or 'none'}")
    print(f"  skipped: {skipped or 'none'}")


# =========================================================================== CLI
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["validate", "register", "extract", "clean", "section", "qc",
                                     "build", "outline"])
    ap.add_argument("doc_id", nargs="?", help="for 'outline', e.g. D04")
    ap.add_argument("--stage", choices=["plan", "final"], default="plan")
    ap.add_argument("--version", default="1.0")
    a = ap.parse_args(argv)
    if a.step == "validate":
        probs = validate_manifest(a.stage)
        for p in probs:
            print(p)
        errors = sum(p.startswith("ERROR") for p in probs)
        print(f"\n{errors} errors, {len(probs) - errors} warnings/pending")
        return 1 if errors else 0
    if a.step == "outline":
        if not a.doc_id:
            raise SystemExit("usage: build_corpus.py outline D04")
        cmd_outline(a.doc_id)
        return 0
    {"register": cmd_register, "extract": cmd_extract, "clean": cmd_clean,
     "section": cmd_section, "qc": cmd_qc}.get(a.step, lambda: cmd_build(a.version))()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
