"""
Streamlit Sentence-Pair Labeling App
--------------------------------------
Shows an annotator one unlabeled pair at a time from the Supabase table
"GutenbergPairsSid", collects three judgments, and writes them back:
 
    human_older              -> which sentence they think is older
                                 ("sentence_1" or "sentence_2")
    human_sentence_1_period  -> what period they think sentence_1 is from
    human_sentence_2_period  -> what period they think sentence_2 is from
 
Setup:
    pip install streamlit supabase
 
    Set these environment variables before running:
        SUPABASE_URL   -> your Supabase project URL
        SUPABASE_KEY   -> your Supabase service_role or anon key
 
Run:
    streamlit run streamlit_labeling_app.py
 
Note on concurrency: this app does NOT do atomic row-claiming, so if two
people run it at the same time they could occasionally be served the
same unlabeled pair. Fine for single-annotator or non-simultaneous use;
revisit with a proper claim/lock step if concurrent labeling becomes real.
"""
 
import os
import random
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TABLE_NAME = "GutenbergPairsSid"
 
# Edit these to match whatever period labels you actually used
# (e.g. the true_period_label values from your harvester script).
PERIOD_OPTIONS = ["1600-1699", "1700-1799", "1800-1899", "1900-1999"]
 
BATCH_SIZE = 50  # how many unlabeled rows to pull before picking one at random
 
st.set_page_config(page_title="Sentence Period Labeling", layout="centered")
 
 
@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_supabase_client()
 
def count_unlabeled() -> int:
    response = (
        supabase.table(TABLE_NAME)
        .select("id", count="exact")
        .filter("human_older", "is", "null")
        .execute()
    )
    return response.count or 0
 
 
def fetch_unlabeled_batch(batch_size: int = BATCH_SIZE):
    response = (
        supabase.table(TABLE_NAME)
        .select("id, sentence_1, sentence_2, pair_id")
        .filter("human_older", "is", "null")
        .limit(batch_size)
        .execute()
    )
    return response.data
 
 
def load_next_row():
    """Pull a batch of unlabeled rows and pick one at random, so
    annotators aren't always served rows in the same fixed order."""
    batch = fetch_unlabeled_batch()
    st.session_state.current_row = random.choice(batch) if batch else None
 
 
def submit_answer(row_id: int, human_older: str, period_1: str, period_2: str):
    supabase.table(TABLE_NAME).update({
        "human_older": human_older,
        "human_sentence_1_period": period_1,
        "human_sentence_2_period": period_2,
    }).eq("id", row_id).execute()
 
 
# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

if "current_row" not in st.session_state:
    load_next_row()
 
st.title("Sentence Period Labeling")
st.caption(f"{count_unlabeled()} pairs remaining")
 
row = st.session_state.current_row
 
if row is None:
    st.success("No unlabeled pairs remaining.")
else:
    st.markdown(f"**Sentence 1:** {row['sentence_1']}")
    st.markdown(f"**Sentence 2:** {row['sentence_2']}")
 
    st.divider()
 
    older_choice = st.radio(
        "Which sentence do you think is OLDER?",
        options=["Sentence 1", "Sentence 2"],
        index=None,
        key=f"older_{row['id']}",
    )
 
    col1, col2 = st.columns(2)
    with col1:
        period_1_choice = st.selectbox(
            "What period do you think Sentence 1 is from?",
            options=PERIOD_OPTIONS,
            index=None,
            key=f"p1_{row['id']}",
        )
    with col2:
        period_2_choice = st.selectbox(
            "What period do you think Sentence 2 is from?",
            options=PERIOD_OPTIONS,
            index=None,
            key=f"p2_{row['id']}",
        )
 
    st.divider()
 
    ready = (
        older_choice is not None
        and period_1_choice is not None
        and period_2_choice is not None
    )
 
    if st.button("Submit", disabled=not ready, type="primary"):
        human_older_value = "sentence_1" if older_choice == "Sentence 1" else "sentence_2"
        submit_answer(row["id"], human_older_value, period_1_choice, period_2_choice)
        load_next_row()
        st.rerun()
 
    if not ready:
        st.caption("Answer all three questions to enable Submit.")