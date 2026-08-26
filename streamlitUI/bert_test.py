import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlitUI.arxiv_folders as arxiv_folders

PERIODS = ["2010-2014", "2015-2019", "2020-2025"]

# ----- Loading -----------------------------------------

@st.cache_data(show_spinner="Loading embeddings…")
def load_npz_as_dict(path: str, strip_suffix: str = None) -> dict:
    """
    Loads a saved .npz (keys + vectors) into a {word: vector} dict.
    The DAPT extraction script stores keys as "word_<period>" (e.g.
    "model_2010-2014"), so strip_suffix removes that trailing tag.
    """
    data = np.load(path, allow_pickle=True)
    keys = data["keys"]
    vectors = data["vectors"]

    result = {}
    for k, v in zip(keys, vectors):
        word = k
        if strip_suffix and word.endswith("_" + strip_suffix):
            word = word[: -(len(strip_suffix) + 1)]
        if word.isalpha():
            result[word] = v
    return result

@st.cache_data
def compare_embedding_sets(vecs_a: dict, vecs_b: dict) -> pd.DataFrame:
    shared_words = sorted(set(vecs_a) & set(vecs_b))

    rows = []
    for w in shared_words:
        a = vecs_a[w]
        b = vecs_b[w]
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        rows.append((w, 1/cos_sim, 1 - cos_sim)) #Inverted Cosine Similarity over Word Prototypes (PRT) 

    df = pd.DataFrame(rows, columns=["word", "PRT", "cosine_distance"])
    return df.sort_values("cosine_distance", ascending=False)

def get_embeddings_paths(selected_folder: str, period_a: str, period_b: str):
    """Resolves the .npz paths for the two chosen periods (or wikipedia)."""
    selected_folder_parsed = selected_folder.split('/')
 
    if period_b != "wikipedia":
        embeddings_dir = f"embeddings/{'/'.join(selected_folder_parsed[4:])}/distilBERT-dapt/"
        path_a = os.path.join(embeddings_dir, f"{period_a}.npz")
        path_b = os.path.join(embeddings_dir, f"{period_b}.npz")
    else:
        embeddings_dir = f"embeddings/{'/'.join(selected_folder_parsed[4:])}/distilBERT-base/"
        path_a = os.path.join(embeddings_dir, f"{period_a}.npz")
        path_b = os.path.join("embeddings/wikipedia/distilBERT-base/", "embeddings.npz")
 
    return embeddings_dir, path_a, path_b
 
# ----- UI -----------------------------------------
 
st.set_page_config(layout="wide")
st.title("DAPT arXiv Embeddings — Diachronic Comparison")
st.caption(
    "Compares your domain-adapted (DAPT) DistilBERT embeddings across two "
    "chosen time periods, ranking words by how much their contextual "
    "representation shifted between the two."
)
 
if "folder_list" not in st.session_state:
    st.session_state.folder_list = arxiv_folders.get_arxiv_folders()
 
st.subheader("1. Select embeddings folder")
selected_folder = st.selectbox(
        "Choose a folder",
        options=st.session_state.folder_list
    )
 
st.subheader("2. Choose two periods to compare or compare with the general corpus (wikipedia)")
col1, col2 = st.columns(2)
with col1:
    period_a = st.selectbox("First period", PERIODS, index=0)
with col2:
    default_b = 1 if len(PERIODS) > 1 else 0
    period_b = st.selectbox("Second period / general corpus", PERIODS + ["wikipedia"], index=default_b)
 
topn = st.slider("Number of most-changed words to show", 5, 50, 20, step=5)
 
run_btn = st.button("Compare periods")
 
if run_btn:
    if period_a == period_b:
        st.warning("Choose two different periods to compare.")
    else:
        embeddings_dir, path_a, path_b = get_embeddings_paths(selected_folder, period_a, period_b)
 
        vecs_a = load_npz_as_dict(path_a, strip_suffix=period_a)
        vecs_b = load_npz_as_dict(path_b, strip_suffix=period_b if period_b != "wikipedia" else None)
 
        # Stash in session_state so the word-lookup section below can reuse
        # these without re-running the comparison button.
        st.session_state["vecs_a"] = vecs_a
        st.session_state["vecs_b"] = vecs_b
        st.session_state["period_a_label"] = period_a
        st.session_state["period_b_label"] = period_b
 
        st.caption(f"{len(vecs_a)} words in {period_a}, {len(vecs_b)} words in {period_b}")
 
        comparison_df = compare_embedding_sets(vecs_a, vecs_b)
        st.session_state["comparison_df"] = comparison_df
        shared_count = len(set(vecs_a) & set(vecs_b))
 
        if comparison_df.empty:
            st.warning("No shared vocabulary found between these two periods.")
        else:
            st.subheader(f"Top {topn} words that changed most: {period_a} vs {period_b}")
            st.dataframe(comparison_df.head(topn), width="stretch")
            st.caption(f"{shared_count} words shared between the two periods.")
 
            st.download_button(
                "Download full ranking as CSV",
                comparison_df.to_csv(index=False),
                file_name=f"{period_a}_vs_{period_b}_dapt_shift.csv",
                mime="text/csv",
            )
 
# ----- Word lookup: cosine similarity for a specific word between the two selected periods -----
 
st.divider()
st.subheader("Look up cosine similarity for a specific word")
st.caption(
    f"Computes the similarity of a single word's embedding between the two "
    f"periods currently selected above ({period_a} vs {period_b}). Click "
    f"'Compare periods' first if you haven't yet."
)
 
query = st.text_input("Word to inspect", placeholder="e.g. transformer")
 
if query:
    query = query.strip().lower()
 
    if "comparison_df" not in st.session_state:
        st.warning("Click 'Compare periods' above first to load embeddings for the selected periods.")
    else:
        full_df = st.session_state["comparison_df"]
        match = np.where(full_df["word"] == query)[0]
 
        if match.size == 0:
            st.warning(f"'{query}' not found in the shared vocabulary between {period_a} and {period_b}.")
        else:
            row = full_df.iloc[match[0]]
            total_words = len(full_df)
            percentile = 100 * (1 - match[0] / total_words)  # higher % = more shifted
 
            m1, m2, m3, m4 = st.columns(4)
            # m1.metric("Rank", f"{int(row['rank'])} / {total_words}")
            m2.metric("Percentile (most-shifted)", f"{percentile:.1f}%")
            m3.metric("Cosine distance", f"{row['cosine_distance']:.4f}")
            m4.metric("PRT", f"{row['PRT']:.4f}")