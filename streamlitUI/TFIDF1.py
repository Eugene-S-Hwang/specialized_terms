import os
import re
import math
import numpy as np
from rank_bm25 import BM25Okapi
from collections import Counter
import streamlit as st
import spacy

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "i", "we", "you", "he", "she", "they",
    "not", "as", "up", "if", "so", "also", "than", "into", "about",
    "which", "who", "what", "when", "where", "how", "all", "more", "no",
    "their", "there", "then", "s", "t", "re", "ve", "ll", "d", "m", "approac", "ligenc", 
    "erimen", "ecause", "orks", "ectiv", "curren", "hiev", "ortan", "enden", "hniques", "ailable", 
    "ision", "whi", "orresp", "https", "preprint", "github"
}

labels = {
    "A": "1993-1999",
    "B": "2000-2006",
    "C": "2007-2013",
    "D": "2014-2020",
}

# text_windows = {
#     "A": {},
#     "B": {},
#     "C": {},
#     "D": {},
# }

TOKEN_RE = re.compile(r"[a-zA-Z]+")

def extract_year(filename: str):
    return filename[0:2]


def tokenize(text: list[str]) -> list[str]:
    # tokens = re.findall(r"[a-zA-Z]+", text.lower())
    blacklist = []
    if "blacklist" in st.session_state:
        blacklist = st.session_state["blacklist"]
    return [t for t in text if t not in STOP_WORDS and t not in blacklist and len(t) > 2]


def build_corpus(folder: str):
    """
    Read all .txt files, assign to one of four periods based on two-digit year prefix.
      A: 93-99  (1993-1999)
      B: 00-06  (2000-2006)
      C: 07-13  (2007-2013)
      D: 14-20  (2014-2020)
    Returns (via yield):
        progress strings, then a final dict with docs and skipped
    """
    docs: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}
    skipped: list[str] = []
    prev_year = None

    def year_to_period(year: str):
        """Return which period key(s) a two-digit year belongs to."""
        y = int(year)
        # Files from 1993-1999 have prefix 93-99
        # Files from 2000-2020 have prefix 00-20
        if (93 <= y <= 99):
            return "A"
        elif 0 <= y <= 6:
            return "B"
        elif 7 <= y <= 13:
            return "C"
        elif 14 <= y <= 20:
            return "D"

    def display_year(year: str) -> str:
        y = int(year)
        return f"19{year}" if y >= 93 else f"20{year:0>2}"

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".txt"):
            continue
        year = extract_year(fname)
        if year is None:
            skipped.append(fname)
            continue

        try:
            y = int(year)
        except ValueError:
            skipped.append(fname)
            continue

        period = year_to_period(year)
        if not period:
            skipped.append(fname)
            continue

        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read().lower()

        if prev_year is not None and year != prev_year:
            yield f"Finished processing year {display_year(prev_year)}"

        prev_year = year

        tokens = TOKEN_RE.findall(text)
        docs[period].append(tokens)

    if prev_year is not None:
        yield f"Finished processing year {display_year(prev_year)}"

    # yield {"docs": {k: " ".join(v) for k, v in docs.items()}, "skipped": skipped}
    yield {"docs": docs, "skipped": skipped}

def get_context_windows(docs: dict[str, list[str]], targets: set[str], period: str):
    windows: dict[str, list[str]] = {}
    # tokens = TOKEN_RE.findall((" ".join(docs[period])).lower())
    for doc in docs[period]:
        for i, t in enumerate(doc):
            if t in targets:
                window = doc[max(0, i - 5): i + 6]
                windows.setdefault(t, []).append(" ".join(window))
    
    return windows


#blacklist with unjoined docs
@st.cache_resource
def build_entity_blacklist(corpus: dict[str, list[list[str]]]) -> set[str]:
    """
    Run NER on a sample of the corpus and return a set of entity words to exclude.
    sample_size limits how many characters to process per period to keep it fast.
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(enable=["ner"])   # disable parser/tagger for speed

    ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"}
    blacklist: set[str] = set()

    for period, docs in corpus.items():

        # 1. sample only a subset of documents
        for doc in docs[:50]:

            # 2. truncate tokens early (VERY important)
            sample = doc[:2000]

            # 3. convert small sample only (not full corpus)
            text = " ".join(sample)

            # 4. spaCy on small chunk
            spacy_doc = nlp(text)

            for ent in spacy_doc.ents:
                if ent.label_ in ENTITY_TYPES:
                    blacklist.update(
                        token.text.lower() for token in ent
                    )

    # docs = {k: " ".join(v) for k, v in corpus.items()}
    # for period, text in docs.items():
    #     # Sample from the middle to avoid header/footer noise
    #     sample = text[:sample_size]
    #     doc = nlp(sample)
    #     for ent in doc.ents:
    #         if ent.label_ in ENTITY_TYPES:
    #             # Add each token of the entity individually
    #             for token in ent:
    #                 blacklist.add(token.text.lower())

    return blacklist


# ── TF-IDF ───────────────────────────────────────────────────────────────────

def tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency (raw count / total tokens)."""
    counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counts.items()}

#tfidf with unjoined docs
def tfidf_scores(docs: dict[str, list[list[str]]]) -> dict[str, dict[str, float]]:
    """
    Compute TF-IDF for each period, multiplied by within-period document frequency.
    docs: dict of period -> list of raw text strings from build_corpus

    TF - term frequency of a word in a period (over joined text)
    IDF = log((1 + N) / (1 + df))  [sklearn-style smooth IDF]
        N  - number of periods
        df - number of periods containing the word
    within_period_df - fraction of documents in the period containing the word
    """
    # Join each period's docs into one string for TF computation
    joined    = {k: [t for sublist in v for t in sublist] for k, v in docs.items()}
    tokenized = {k: tokenize(v) for k, v in joined.items()}
    N         = len(tokenized)

    counts  = {}
    lengths = {}
    for label, tokens in tokenized.items():
        counts[label]  = Counter(tokens)
        lengths[label] = len(tokens)

    # IDF over periods
    df: Counter = Counter()
    for tokens in tokenized.values():
        for word in set(tokens):
            df[word] += 1

    # Within-period document frequency: fraction of docs in period containing the word
    period_doc_counts: dict[str, Counter] = {k: Counter() for k in docs}
    period_num_docs:   dict[str, int]     = {}
    for label, text_list in docs.items():
        period_num_docs[label] = len(text_list)
        for text in text_list:
            for word in set(tokenize(text)):
                period_doc_counts[label][word] += 1

    scores: dict[str, dict[str, float]] = {}
    for label, token_counts in counts.items():
        scores[label] = {}
        for word, tf_val in token_counts.items():
            tf               = tf_val / lengths[label]
            idf              = math.log((1 + N) / (1 + df[word]))
            within_period_df = period_doc_counts[label][word] / period_num_docs[label]
            scores[label][word] = tf * idf * math.log(within_period_df + 1)

    return scores

# ── BM-25 ───────────────────────────────────────────────────────────────────

# def bm_25_scores(docs: dict[str, str]) -> dict[str, dict[str, float]]:
#     tokenized = {k: tokenize(v) for k, v in docs.items()}
#     N = len(tokenized)

#     # print(N)

#     bm25 = BM25Okapi(list(tokenized.values()))
#     labels = list(tokenized.keys())

#     vocab = list({word for doc in tokenized.values() for word in doc})

#     score_matrix = np.array([bm25.get_scores([term]) for term in vocab])
#     # shape: (vocab_size, num_docs)

#     best_period_per_word = np.argmax(score_matrix, axis=1)

#     scores: dict[str, dict[str, float]] = {k: {} for k in labels}
#     for word_idx, period_idx in enumerate(best_period_per_word):
#         winner = labels[period_idx]
#         word   = vocab[word_idx]
#         scores[winner][word] = float(score_matrix[word_idx, period_idx])

#     return scores

def bm_25_scores(docs: dict[str, list[list[str]]]) -> dict[str, dict[str, float]]:
    # Tokenize each individual document
    tokenized_periods: dict[str, list[list[str]]] = {
        k: [tokenize(text) for text in text_list]
        for k, text_list in docs.items()
    }

    # Flatten all documents, tracking which period each belongs to
    period_labels = []
    all_tokenized = []
    for period, doc_list in tokenized_periods.items():
        for doc in doc_list:
            all_tokenized.append(doc)
            period_labels.append(period)

    bm25   = BM25Okapi(all_tokenized)
    labels = list(tokenized_periods.keys())
    vocab  = list({word for doc in all_tokenized for word in doc})

    # Score matrix over all individual documents
    score_matrix = np.array([bm25.get_scores([term]) for term in vocab])
    # shape: (vocab_size, total_num_docs)

    # Average BM25 scores per period
    period_indices = {
        period: [i for i, p in enumerate(period_labels) if p == period]
        for period in labels
    }

    avg_score_matrix = np.zeros((len(vocab), len(labels)))
    for j, period in enumerate(labels):
        indices = period_indices[period]
        avg_score_matrix[:, j] = score_matrix[:, indices].mean(axis=1)
    # shape: (vocab_size, num_periods)

    # Keep only words where this period has the highest average score
    best_period_per_word = np.argmax(avg_score_matrix, axis=1)

    scores: dict[str, dict[str, float]] = {k: {} for k in labels}
    for word_idx, period_idx in enumerate(best_period_per_word):
        winner = labels[period_idx]
        word   = vocab[word_idx]
        scores[winner][word] = float(avg_score_matrix[word_idx, period_idx])

    return scores


# ── Helper: find candidate folders ───────────────────────────────────────────

#Not currently used
def find_folders_with_txt(root: str, max_depth: int = 3) -> list[str]:
    """Walk up to max_depth levels and return folders that contain .txt files."""
    results = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth >= max_depth:
            dirnames.clear()
            continue
        if any(f.lower().endswith(".txt") for f in filenames):
            results.append(dirpath)
    return sorted(results)


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide")
st.title("TF-IDF/BM-25 Explorer")
st.caption("Compare word importance across four time periods (1993-1999 · 2000-2006 · 2007-2013 · 2014-2020)")

st.divider()

# --- Folder selection ---
st.subheader("1. Select a folder")

if "folder_list" not in st.session_state:
    st.session_state.folder_list = ['/data/datasets/arXiv', '/data/datasets/arXiv/astro-ph', '/data/datasets/arXiv/astro-ph/CO', '/data/datasets/arXiv/astro-ph/EP', '/data/datasets/arXiv/astro-ph/GA', '/data/datasets/arXiv/astro-ph/HE', '/data/datasets/arXiv/astro-ph/IM', '/data/datasets/arXiv/astro-ph/SR', '/data/datasets/arXiv/cond-mat', '/data/datasets/arXiv/cond-mat/dis-nn', '/data/datasets/arXiv/cond-mat/mes-hall', '/data/datasets/arXiv/cond-mat/mtrl-sci', '/data/datasets/arXiv/cond-mat/other', '/data/datasets/arXiv/cond-mat/quant-gas', '/data/datasets/arXiv/cond-mat/soft', '/data/datasets/arXiv/cond-mat/stat-mech', '/data/datasets/arXiv/cond-mat/str-el', '/data/datasets/arXiv/cond-mat/supr-con', '/data/datasets/arXiv/cs/AI', '/data/datasets/arXiv/cs/AR', '/data/datasets/arXiv/cs/CC', '/data/datasets/arXiv/cs/CE', '/data/datasets/arXiv/cs/CG', '/data/datasets/arXiv/cs/CL', '/data/datasets/arXiv/cs/CR', '/data/datasets/arXiv/cs/CV', '/data/datasets/arXiv/cs/CY', '/data/datasets/arXiv/cs/DB', '/data/datasets/arXiv/cs/DC', '/data/datasets/arXiv/cs/DL', '/data/datasets/arXiv/cs/DM', '/data/datasets/arXiv/cs/DS', '/data/datasets/arXiv/cs/ET', '/data/datasets/arXiv/cs/FL', '/data/datasets/arXiv/cs/GL', '/data/datasets/arXiv/cs/GR', '/data/datasets/arXiv/cs/GT', '/data/datasets/arXiv/cs/HC', '/data/datasets/arXiv/cs/IR', '/data/datasets/arXiv/cs/IT', '/data/datasets/arXiv/cs/LG', '/data/datasets/arXiv/cs/LO', '/data/datasets/arXiv/cs/MA', '/data/datasets/arXiv/cs/MM', '/data/datasets/arXiv/cs/MS', '/data/datasets/arXiv/cs/NA', '/data/datasets/arXiv/cs/NE', '/data/datasets/arXiv/cs/NI', '/data/datasets/arXiv/cs/OH', '/data/datasets/arXiv/cs/OS', '/data/datasets/arXiv/cs/PF', '/data/datasets/arXiv/cs/PL', '/data/datasets/arXiv/cs/RO', '/data/datasets/arXiv/cs/SC', '/data/datasets/arXiv/cs/SD', '/data/datasets/arXiv/cs/SE', '/data/datasets/arXiv/cs/SI', '/data/datasets/arXiv/cs/SY', '/data/datasets/arXiv/econ/EM', '/data/datasets/arXiv/econ/GN', '/data/datasets/arXiv/econ/TH', '/data/datasets/arXiv/eess/AS', '/data/datasets/arXiv/eess/IV', '/data/datasets/arXiv/eess/SP', '/data/datasets/arXiv/eess/SY', '/data/datasets/arXiv/gr-qc', '/data/datasets/arXiv/hep-ex', '/data/datasets/arXiv/hep-lat', '/data/datasets/arXiv/hep-ph', '/data/datasets/arXiv/hep-th', '/data/datasets/arXiv/math-ph', '/data/datasets/arXiv/math/AC', '/data/datasets/arXiv/math/AG', '/data/datasets/arXiv/math/AP', '/data/datasets/arXiv/math/AT', '/data/datasets/arXiv/math/CA', '/data/datasets/arXiv/math/CO', '/data/datasets/arXiv/math/CT', '/data/datasets/arXiv/math/CV', '/data/datasets/arXiv/math/DG', '/data/datasets/arXiv/math/DS', '/data/datasets/arXiv/math/FA', '/data/datasets/arXiv/math/GM', '/data/datasets/arXiv/math/GN', '/data/datasets/arXiv/math/GR', '/data/datasets/arXiv/math/GT', '/data/datasets/arXiv/math/HO', '/data/datasets/arXiv/math/KT', '/data/datasets/arXiv/math/LO', '/data/datasets/arXiv/math/MG', '/data/datasets/arXiv/math/NA', '/data/datasets/arXiv/math/NT', '/data/datasets/arXiv/math/OA', '/data/datasets/arXiv/math/OC', '/data/datasets/arXiv/math/PR', '/data/datasets/arXiv/math/QA', '/data/datasets/arXiv/math/RA', '/data/datasets/arXiv/math/RT', '/data/datasets/arXiv/math/SG', '/data/datasets/arXiv/math/SP', '/data/datasets/arXiv/math/ST', '/data/datasets/arXiv/nlin/AO', '/data/datasets/arXiv/nlin/CD', '/data/datasets/arXiv/nlin/CG', '/data/datasets/arXiv/nlin/PS', '/data/datasets/arXiv/nlin/SI', '/data/datasets/arXiv/nucl-ex', '/data/datasets/arXiv/nucl-th', '/data/datasets/arXiv/physics/acc-ph', '/data/datasets/arXiv/physics/ao-ph', '/data/datasets/arXiv/physics/app-ph', '/data/datasets/arXiv/physics/atm-clus', '/data/datasets/arXiv/physics/atom-ph', '/data/datasets/arXiv/physics/bio-ph', '/data/datasets/arXiv/physics/chem-ph', '/data/datasets/arXiv/physics/class-ph', '/data/datasets/arXiv/physics/comp-ph', '/data/datasets/arXiv/physics/data-an', '/data/datasets/arXiv/physics/ed-ph', '/data/datasets/arXiv/physics/flu-dyn', '/data/datasets/arXiv/physics/gen-ph', '/data/datasets/arXiv/physics/geo-ph', '/data/datasets/arXiv/physics/hist-ph', '/data/datasets/arXiv/physics/ins-det', '/data/datasets/arXiv/physics/med-ph', '/data/datasets/arXiv/physics/optics', '/data/datasets/arXiv/physics/plasm-ph', '/data/datasets/arXiv/physics/pop-ph', '/data/datasets/arXiv/physics/soc-ph', '/data/datasets/arXiv/physics/space-ph', '/data/datasets/arXiv/q-bio/BM', '/data/datasets/arXiv/q-bio/CB', '/data/datasets/arXiv/q-bio/GN', '/data/datasets/arXiv/q-bio/MN', '/data/datasets/arXiv/q-bio/NC', '/data/datasets/arXiv/q-bio/OT', '/data/datasets/arXiv/q-bio/PE', '/data/datasets/arXiv/q-bio/QM', '/data/datasets/arXiv/q-bio/SC', '/data/datasets/arXiv/q-bio/TO', '/data/datasets/arXiv/q-fin/CP', '/data/datasets/arXiv/q-fin/EC', '/data/datasets/arXiv/q-fin/GN', '/data/datasets/arXiv/q-fin/MF', '/data/datasets/arXiv/q-fin/PM', '/data/datasets/arXiv/q-fin/PR', '/data/datasets/arXiv/q-fin/RM', '/data/datasets/arXiv/q-fin/ST', '/data/datasets/arXiv/q-fin/TR', '/data/datasets/arXiv/quant-ph', '/data/datasets/arXiv/stat/AP', '/data/datasets/arXiv/stat/CO', '/data/datasets/arXiv/stat/ME', '/data/datasets/arXiv/stat/ML', '/data/datasets/arXiv/stat/OT']

# Subfolder dropdown
root = "/data/datasets/arXiv"
selected_folder = None
if st.session_state.folder_list:
    selected_folder = st.selectbox(
        "Choose a folder",
        options=st.session_state.folder_list,
        format_func=lambda p: f"📁 {os.path.relpath(p, root)}",
    )
else:
    manual = st.text_input(
        "Or enter a folder path directly",
        placeholder="/path/to/txt/files",
    )
    if manual:
        selected_folder = manual

corpus_btn = st.button("Select Folder", width="content")
skipped = None, []
if corpus_btn:

    st.write("**Processing folder:**")
    log_container = st.empty()
    log_lines: list[str] = []

    for item in build_corpus(selected_folder):
        if isinstance(item, str):
            log_lines.append(item)
            log_container.markdown("\n".join(log_lines))
        else:
            st.session_state["docs"] = item["docs"]
            skipped = item["skipped"]

    if skipped:
        st.info(
            f"Skipped {len(skipped)} file(s) with unrecognised year prefix: "
            + ", ".join(skipped[:10])
            + (" …" if len(skipped) > 10 else "")
        )
    
    if "docs" in st.session_state:
        st.session_state["blacklist"] = build_entity_blacklist(st.session_state["docs"])
    

# Settings
st.subheader("2. Settings")

top_n = st.slider("Number of top words to show", min_value=5, max_value=50, value=15, step=5)

# Run the TFIDF stuff
st.subheader("3. Run analysis")

tfidf_btn = st.button("▶ Run TF-IDF", type="primary", width="content")
bm25_btn = st.button("▶ Run BM-25", type="primary", width="content") 

if tfidf_btn:
    if not selected_folder or not os.path.isdir(selected_folder):
        st.error("Please select a valid folder first.")
    else:
        # Reset text_windows for a fresh run
        docs = st.session_state["docs"]

        if not any(docs[k] for k in labels):
            st.error(
                "No matching files found. Make sure filenames start with a two-digit year "
                "(e.g. 97, 03, 09, 15)."
            )
        else:
            st.divider()
            st.subheader("Results")

            scores = tfidf_scores(docs)

            # Token counts — one metric per period
            meta_cols = st.columns(4)
            for i, (key, period) in enumerate(labels.items()):
                count = len([t for doc in docs[key] for t in doc]) if docs[key] else 0
                meta_cols[i].metric(
                    label=f"Period {key} ({period})",
                    value=f"{count:,} words",
                )

            st.write("")

            # Side-by-side results — 4 columns
            res_cols = st.columns(4)
            for i, (key, period) in enumerate(labels.items()):
                with res_cols[i]:
                    st.markdown(f"#### Period {key} &nbsp; `{period}`")
                    if not scores.get(key):
                        st.warning("No documents found for this period.")
                        continue

                    top_words = sorted(
                        scores[key].items(), key=lambda x: x[1], reverse=True
                    )[:top_n]

                    rows = []
                    for rank, (word, score) in enumerate(top_words, 1):
                        rows.append({
                            "Rank": rank,
                            "Word": word,
                            "TF-IDF Score": round(score, 6),
                        })

                    st.dataframe(
                        rows,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn(width="small"),
                            "Word": st.column_config.TextColumn(width="medium"),
                            "TF-IDF Score": st.column_config.ProgressColumn(
                                format="%.6f",
                                min_value=0,
                                max_value=top_words[0][1] if top_words else 1,
                            ),
                        },
                    )

                    st.markdown("##### Context Windows (top 10 per word)")
                    targets = set(word for word, _ in top_words)
                    word_windows = get_context_windows(docs, targets, key)
                    for word, _ in top_words:
                        if not word_windows:
                            continue
                        with st.expander(
                            f"**{word}** — {len(word_windows[word])} occurrences"
                        ):
                            ctx_rows = [{"#": i + 1, "Context": w} for i, w in enumerate(word_windows[word][:10])]
                            st.dataframe(
                                ctx_rows,
                                width="stretch",
                                hide_index=True,
                                column_config={
                                    "#": st.column_config.NumberColumn(width="small"),
                                    "Context": st.column_config.TextColumn(width="large"),
                                },
                            )

if bm25_btn:
    if not selected_folder or not os.path.isdir(selected_folder):
        st.error("Please select a valid folder first.")
    else:
        # Reset text_windows for a fresh run
        docs = st.session_state["docs"]

        if not any(docs[k] for k in labels):
            st.error(
                "No matching files found. Make sure filenames start with a two-digit year "
                "(e.g. 97, 03, 09, 15)."
            )
        else:
            st.divider()
            st.subheader("Results")

            scores = bm_25_scores(docs)

            # Token counts — one metric per period
            meta_cols = st.columns(4)
            for i, (key, period) in enumerate(labels.items()):
                count = len([t for doc in docs[key] for t in doc]) if docs[key] else 0
                meta_cols[i].metric(
                    label=f"Period {key} ({period})",
                    value=f"{count:,} words",
                )

            st.write("")

            # Side-by-side results — 4 columns
            res_cols = st.columns(4)
            for i, (key, period) in enumerate(labels.items()):
                with res_cols[i]:
                    st.markdown(f"#### Period {key} &nbsp; `{period}`")
                    if not scores.get(key):
                        st.warning("No documents found for this period.")
                        continue

                    top_words = sorted(
                        scores[key].items(), key=lambda x: x[1], reverse=True
                    )[:top_n]

                    rows = []
                    for rank, (word, score) in enumerate(top_words, 1):
                        rows.append({
                            "Rank": rank,
                            "Word": word,
                            "TF-IDF Score": round(score, 6),
                        })

                    st.dataframe(
                        rows,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn(width="small"),
                            "Word": st.column_config.TextColumn(width="medium"),
                            "TF-IDF Score": st.column_config.ProgressColumn(
                                format="%.6f",
                                min_value=0,
                                max_value=top_words[0][1] if top_words else 1,
                            ),
                        },
                    )

                    st.markdown("##### Context Windows (top 10 per word)")
                    targets = set(word for word, _ in top_words)
                    word_windows = get_context_windows(docs, targets, key)
                    for word, _ in top_words:
                        if not word_windows:
                            continue
                        with st.expander(
                            f"**{word}** — {len(word_windows[word])} occurrences"
                        ):
                            ctx_rows = [{"#": i + 1, "Context": w} for i, w in enumerate(word_windows[word][:10])]
                            st.dataframe(
                                ctx_rows,
                                width="stretch",
                                hide_index=True,
                                column_config={
                                    "#": st.column_config.NumberColumn(width="small"),
                                    "Context": st.column_config.TextColumn(width="large"),
                                },
                            )