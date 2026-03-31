import os
import re
import sys
import math
import argparse
from collections import Counter


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


def extract_year(filename: str):
    return filename[2:4]


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def build_corpus(folder: str) -> tuple[dict[str, str], list[str]]:
    """
    Read all .txt files, assign to period A or B (or both if year == 2011).
    Returns:
        docs  - {"A": combined_text, "B": combined_text}
        skipped - list of filenames that had no recognisable year
    """
    docs: dict[str, list[str]] = {"A": [], "B": []}
    skipped: list[str] = []

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".txt"):
            continue
        year = extract_year(fname)
        if year is None:
            skipped.append(fname)
            continue

        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()

        if "08" <= year <= "11":
            docs["A"].append(text)
        if "12" <= year <= "15":
            docs["B"].append(text)

    return {k: " ".join(v) for k, v in docs.items()}, skipped


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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TF-IDF by time period.")
    parser.add_argument("folder", help="Path to folder containing .txt files")
    parser.add_argument(
        "--top", type=int, default=15, help="Number of top words to show (default 15)"
    )
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        sys.exit(f"Error: '{folder}' is not a valid directory.")

    print(f"\nScanning folder: {os.path.abspath(folder)}\n")
    docs, skipped = build_corpus(folder)

    if skipped:
        print(f"Skipped (no year in filename): {', '.join(skipped)}\n")

    labels = {
        "A": "2008-2011",
        "B": "2012-2015",
    }

    for key, period in labels.items():
        word_count = len(tokenize(docs[key])) if docs[key] else 0
        print(f"Period {key} ({period}): {word_count:,} tokens")

    if not docs["A"] and not docs["B"]:
        sys.exit("No matching files found. Check filenames contain a year like '2009'.")

    scores = tfidf_scores(docs)

    for key, period in labels.items():
        print(f"\n{'='*50}")
        print(f"  Top {args.top} words — Period {key} ({period})")
        print(f"{'='*50}")
        if not scores.get(key):
            print("  (no documents found for this period)")
            continue
        top_words = sorted(scores[key].items(), key=lambda x: x[1], reverse=True)[: args.top]
        for rank, (word, score) in enumerate(top_words, 1):
            print(f"  {rank:>2}. {word:<25} {score:.6f}")

    print()


if __name__ == "__main__":
    main()