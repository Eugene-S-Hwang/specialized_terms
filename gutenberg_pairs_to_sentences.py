"""
Extract 17th- and 19th-century sentences from GutenbergPairsSid
------------------------------------------------------------------
Reads every pair from the "GutenbergPairsSid" table, pulls out the
individual sentences whose period is 17th century (1600-1699) or
19th century (1800-1899), and inserts them into "GutenbergSentences".

A pair can contribute 0, 1, or 2 rows to the target table, depending
on whether one, both, or neither of its two sentences fall in the
target centuries (e.g. a 17th-vs-20th-century pair only contributes
its 17th-century sentence).

IMPORTANT: The column names written into GutenbergSentences below
(sentence, period, book_title, author, gutenberg_id, source_pair_id)
are a best guess based on the fields available in GutenbergPairsSid.
If your actual GutenbergSentences table uses different column names,
edit the dict keys in extract_target_sentences() to match exactly --
mismatched keys will cause the insert to fail.

Setup:
    pip install supabase

    Set these environment variables before running:
        SUPABASE_URL   -> your Supabase project URL
        SUPABASE_KEY   -> your Supabase key. If GutenbergPairsSid has
                           Row Level Security enabled (the default for
                           new Supabase tables), use the service_role
                           key here rather than the anon/publishable
                           key, or reads will silently return nothing.

Run:
    python extract_century_sentences.py
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

SOURCE_TABLE = "GutenbergPairsSid"
TARGET_TABLE = "GutenbergSentences"

# Period labels as they appear in period_1 / period_2. Edit these strings
# if your actual stored labels differ (e.g. "17th Century" with a capital C).
TARGET_PERIODS = {"1600-1699", "1800-1899"}
LABEL_PERIODS = {"1600-1699":"17th Century", "1800-1899":"19th Century"}

PAGE_SIZE = 1000  # Supabase's default per-request row cap; paginate past it

sentence_set = set()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Step 1: Fetch every pair, paginating past Supabase's row limit
# ---------------------------------------------------------------------------

def fetch_all_pairs():
    all_rows = []
    start = 0
    while True:
        end = start + PAGE_SIZE - 1
        response = (
            supabase.table(SOURCE_TABLE)
            .select(
                "id, pair_id, "
                "sentence_1, period_1, title_1, author_1, gutenberg_id_1, "
                "sentence_2, period_2, title_2, author_2, gutenberg_id_2"
            )
            .range(start, end)
            .execute()
        )
        batch = response.data
        all_rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return all_rows


# ---------------------------------------------------------------------------
# Step 2: Pull out sentences matching the target centuries
# ---------------------------------------------------------------------------

def extract_target_sentences(pairs):
    rows = []

    for pair in pairs:
        if pair.get("sentence_1") and pair.get("period_1") in TARGET_PERIODS and pair["sentence_1"] not in sentence_set:
            rows.append({
                "chunk": pair["sentence_1"],
                "true_period": LABEL_PERIODS[pair["period_1"]],
                "book_title": pair["title_1"],
                "author": pair["author_1"],
                "gutenberg_id": pair["gutenberg_id_1"],
            })
            sentence_set.add(pair["sentence_1"])

        if pair.get("sentence_2") and pair.get("period_2") in TARGET_PERIODS and pair["sentence_2"] not in sentence_set:
            rows.append({
                "chunk": pair["sentence_2"],
                "true_period": LABEL_PERIODS[pair["period_2"]],
                "book_title": pair["title_2"],
                "author": pair["author_2"],
                "gutenberg_id": pair["gutenberg_id_2"],
            })
            sentence_set.add(pair["sentence_2"])

    return rows


# ---------------------------------------------------------------------------
# Step 3: Insert into GutenbergSentences
# ---------------------------------------------------------------------------

def insert_sentences(rows):
    batch_size = 200
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        supabase.table(TARGET_TABLE).insert(batch).execute()


if __name__ == "__main__":
    print(f"Fetching pairs from {SOURCE_TABLE}...")
    pairs = fetch_all_pairs()
    print(f"  Fetched {len(pairs)} pairs.")

    print("Filtering for 17th- and 19th-century sentences...")
    rows = extract_target_sentences(pairs)
    print(f"  Found {len(rows)} matching sentences.")

    print(f"Inserting into {TARGET_TABLE}...")
    insert_sentences(rows)
    print("Done.")