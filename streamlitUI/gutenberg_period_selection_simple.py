import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# -------------------------
# Supabase setup
# -------------------------

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_supabase_client()

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Gutenberg Period Labeler",
    layout="centered"
)

st.title("Gutenberg Sentence Labeler")

# -------------------------
# Get an unlabeled row
# -------------------------
response = (
    supabase
    .table("GutenbergSentences")
    .select("id, chunk, book_title, author, human_period")
    .is_("human_period", "null")
    .order("id")
    .limit(1)
    .execute()
)

rows = response.data

if not rows:
    st.success("🎉 All sentences have been labeled!")
    st.stop()

row = rows[0]

# -------------------------
# Display chunk
# -------------------------
st.subheader("Text")

st.markdown(
    f"""
    <div style="
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #444;
        font-size: 18px;
        line-height: 1.6;
    ">
        {row["chunk"]}
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# Period selection
# -------------------------
period = st.radio(
    "What time period is this text from?",
    ["17th century", "19th century"],
    horizontal=True
)

# -------------------------
# Submit
# -------------------------
if st.button("Submit", type="primary", use_container_width=True):
    (
        supabase
        .table("GutenbergSentences")
        .update({"human_period": period})
        .eq("id", row["id"])
        .execute()
    )

    st.success(f"Labeled as **{period}**!")

    # Force Streamlit to rerun and fetch the next row
    st.rerun()