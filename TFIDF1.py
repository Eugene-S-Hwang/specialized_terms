import os
import re
import sys
import math
import argparse
from collections import Counter
import streamlit as st


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "i", "we", "you", "he", "she", "they",
    "not", "as", "up", "if", "so", "also", "than", "into", "about",
    "which", "who", "what", "when", "where", "how", "all", "more", "no",
    "their", "there", "then", "s", "t", "re", "ve", "ll", "d", "m",
}

labels = {
    "A": "2007-2013",
    "B": "2014-2020",
}

text_windows = {
    "A": {},
    "B": {}
}

def extract_year(filename: str):
    return filename[0:2]


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def build_corpus(folder: str):
    """
    Read all .txt files, assign to period A or B (or both if year == 2011).
    Returns:
        docs  - {"A": combined_text, "B": combined_text}
        skipped - list of filenames that had no recognisable year
    """
    docs: dict[str, list[str]] = {"A": [], "B": []}
    skipped: list[str] = []
    prev_year = None

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".txt"):
            continue
        year = extract_year(fname)
        # print(fname)
        if year is None or int(year) < 7 or int(year) > 20:
            skipped.append(fname)
            continue

        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        
        if(prev_year is not None and year != prev_year):
            yield f"Finished processing year 20{prev_year}"
        
        prev_year = year

        if "07" <= year <= "13":
            docs["A"].append(text)
        if "14" <= year <= "20":
            docs["B"].append(text)
    if prev_year is not None:
        yield f"Finished processing year 20{prev_year}"

    for k, texts in docs.items():
        for text in texts:
            tokens = re.findall(r"[a-zA-Z]+", text.lower())
            for i, t in enumerate(tokens):
                if t not in STOP_WORDS and len(t) > 2:
                    window = " ".join(tokens[max(0, i-5) : i+6])
                    text_windows[k].setdefault(t, []).append(window)

    yield {"docs": {k: " ".join(v) for k, v in docs.items()}, "skipped": skipped}


# ── TF-IDF ───────────────────────────────────────────────────────────────────

def tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency (raw count / total tokens)."""
    counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counts.items()}


def tfidf_scores(
    docs: dict[str, str]
) -> dict[str, dict[str, float]]:
    """
    Compute TF-IDF for each document.
    IDF = log( (1 + N) / (1 + df) ) + 1   [sklearn-style smooth IDF]
    """
    tokenized = {k: tokenize(v) for k, v in docs.items()}
    N = len(tokenized)

    print(N)

    # document frequency
    df: Counter = Counter()
    for tokens in tokenized.values():
        for word in set(tokens):
            df[word] += 1

    scores: dict[str, dict[str, float]] = {}
    for label, tokens in tokenized.items():
        tf_vals = tf(tokens)
        scores[label] = {}
        for word, tf_val in tf_vals.items():
            idf = math.log((1 + N) / (1 + df[word]))
            scores[label][word] = tf_val * idf

    return scores

# ── Helper: find candidate folders ───────────────────────────────────────────
 
def find_folders_with_txt(root: str, max_depth: int = 3) -> list[str]:
    """Walk up to max_depth levels and return folders that contain .txt files."""
    results = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # Limit depth
        depth = dirpath[len(root):].count(os.sep)
        if depth >= max_depth:
            dirnames.clear()
            continue
        if any(f.lower().endswith(".txt") for f in filenames):
            results.append(dirpath)
    return sorted(results)
 
# ── UI ────────────────────────────────────────────────────────────────────────
 
st.title("📊 TF-IDF Explorer")
st.caption("Compare word importance across two time periods (2007–2013 vs 2014–2020)")
 
st.divider()
 
# --- Folder selection ---
st.subheader("1. Select a folder")
 
col1, col2 = st.columns([3, 1])
 
with col1:
    scan_root = st.text_input(
        "Root directory to scan for folders containing .txt files",
        value=os.getcwd(),
        placeholder="/path/to/your/data",
    )
 
with col2:
    st.write("")
    st.write("")
    scan_btn = st.button("🔍 Scan", width="stretch")
 
# Find folders
if "folder_list" not in st.session_state:
    st.session_state.folder_list = []
 
if scan_btn:
    if os.path.isdir(scan_root):
        with st.spinner("Scanning for folders with .txt files..."):
            st.session_state.folder_list = find_folders_with_txt(scan_root)
        if not st.session_state.folder_list:
            st.warning("No folders with .txt files found under that path.")
    else:
        st.error(f"'{scan_root}' is not a valid directory.")
 
# Subfolder dropdown
selected_folder = None
if st.session_state.folder_list:
    selected_folder = st.selectbox(
        "Choose a folder",
        options=st.session_state.folder_list,
        format_func=lambda p: f"📁 {os.path.relpath(p, scan_root)}",
    )
else:
    manual = st.text_input(
        "Or enter a folder path directly",
        placeholder="/path/to/txt/files",
    )
    if manual:
        selected_folder = manual
 
# Settings
st.subheader("2. Settings")
 
top_n = st.slider("Number of top words to show", min_value=5, max_value=50, value=15, step=5)
 
# Run the TFIDF stuff
st.subheader("3. Run analysis")
 
run_btn = st.button("▶ Run TF-IDF", type="primary", width="content")
 
if run_btn:
    if not selected_folder or not os.path.isdir(selected_folder):
        st.error("Please select a valid folder first.")
    else:
        st.write("**📅 Processing log:**")
        log_container = st.empty()
        log_lines: list[str] = []
        docs, skipped = None, []
 
        for item in build_corpus(selected_folder):
            if isinstance(item, str):
                log_lines.append(item)
                log_container.markdown("\n".join(log_lines))
            else:
                docs = item["docs"]
                skipped = item["skipped"]
 
        if skipped:
            st.info(f"Skipped {len(skipped)} file(s) with unrecognised year prefix: {', '.join(skipped[:10])}" +
                    (" …" if len(skipped) > 10 else ""))
 
        if not docs["A"] and not docs["B"]:
            st.error("No matching files found. Make sure filenames start with a two-digit year (e.g. 07, 14).")
        else:
            scores = tfidf_scores(docs)
 
            st.divider()
            st.subheader("Results")
 
            # Token counts
            meta_cols = st.columns(2)
            for i, (key, period) in enumerate(labels.items()):
                count = len(tokenize(docs[key])) if docs[key] else 0
                meta_cols[i].metric(label=f"Period {key} ({period}) — tokens", value=f"{count:,}")
 
            st.write("")
 
            # Side-by-side results
            res_cols = st.columns(2)
            for i, (key, period) in enumerate(labels.items()):
                with res_cols[i]:
                    st.markdown(f"#### Period {key} &nbsp; `{period}`")
                    if not scores.get(key):
                        st.warning("No documents found for this period.")
                        continue
 
                    top_words = sorted(
                        scores[key].items(), key=lambda x: x[1], reverse=True
                    )[:top_n]
 
                    # Display as a styled table
                    rows = []
                    for rank, (word, score) in enumerate(top_words, 1):
                        bar_pct = int((score / top_words[0][1]) * 100)
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
                    for word, _ in top_words:
                        word_windows = text_windows.get(key, {}).get(word, [])[:10]
                        if not word_windows:
                            continue
                        with st.expander(f"**{word}** — {len(text_windows.get(key, {}).get(word, []))} occurrences"):
                            ctx_rows = [{"#": i + 1, "Context": w} for i, w in enumerate(word_windows)]
                            st.dataframe(
                                ctx_rows,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    "#": st.column_config.NumberColumn(width="small"),
                                    "Context": st.column_config.TextColumn(width="large"),
                                },
                            )

# # ── main ─────────────────────────────────────────────────────────────────────

# """
# How to run:

# Input this command in the terminal: python TFIDF.py [folder] --top [number]
# """



# def main():
#     parser = argparse.ArgumentParser(description="TF-IDF by time period.")
#     parser.add_argument("folder", help="Path to folder containing .txt files")
#     parser.add_argument(
#         "--top", type=int, default=15, help="Number of top words to show (default 15)"
#     )
#     args = parser.parse_args()

#     folder = args.folder
#     if not os.path.isdir(folder):
#         sys.exit(f"Error: '{folder}' is not a valid directory.")

#     print(f"\nScanning folder: {os.path.abspath(folder)}\n")
#     docs, skipped = build_corpus(folder)

#     if skipped:
#         print(f"Skipped (no year in filename): {', '.join(skipped)}\n")

#     labels = {
#         "A": "2007-2013",
#         "B": "2014-2020",
#     }

#     for key, period in labels.items():
#         word_count = len(tokenize(docs[key])) if docs[key] else 0
#         print(f"Period {key} ({period}): {word_count:,} tokens")

#     if not docs["A"] and not docs["B"]:
#         sys.exit("No matching files found. Check filenames contain a year like '2009'.")

#     scores = tfidf_scores(docs)

#     for key, period in labels.items():
#         print(f"\n{'='*50}")
#         print(f"  Top {args.top} words — Period {key} ({period})")
#         print(f"{'='*50}")
#         if not scores.get(key):
#             print("  (no documents found for this period)")
#             continue
#         top_words = sorted(scores[key].items(), key=lambda x: x[1], reverse=True)[: args.top]
#         for rank, (word, score) in enumerate(top_words, 1):
#             print(f"  {rank:>2}. {word:<25} {score:.6f}")

#     print()


# if __name__ == "__main__":
#     main()