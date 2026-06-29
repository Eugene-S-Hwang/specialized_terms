import os
import re
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import plotly.express as px
import pandas as pd

STOP_WORDS = set(stopwords.words("english"))

def extract_year(filename: str):
    return filename[0:2]

def display_year(year: str) -> str:
    y = int(year)
    return f"19{year}" if y >= 93 else f"20{year:0>2}"

class FolderSentences:
    """
    Gensim-compatible iterable that streams tokenised sentences
    one file at a time.  Word2Vec will call __iter__ twice (one
    pass to build the vocab, one to train), so we re-open files
    each time rather than holding them in memory.
    """
 
    # Regex: split on blank lines OR hard newlines (paragraph-level)
    _SPLIT_RE = re.compile(r"\n{2,}|\r\n")
 
    def __init__(self, folder: str):
        self.folder = folder
 
    def __iter__(self):
        prev_year = None

        for fname in sorted(os.listdir(self.folder)):
            if not fname.lower().endswith(".txt"):
                continue
            
            year = int(display_year(extract_year(fname)))
            if year < 2010:
                continue

            if prev_year is not None and year != prev_year:
                print(f"Finished processing year {display_year(prev_year)}")

            prev_year = year
 
            path = os.path.join(self.folder, fname)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read().lower()
 
            for chunk in self._SPLIT_RE.split(text):
                chunk = chunk.strip()
                if chunk:
                    tokens = [token
                            for token in simple_preprocess(chunk)
                            if token not in STOP_WORDS
                        ]
                    if tokens:
                        yield tokens

        if prev_year is not None:
            print(f"Finished processing year {str(prev_year)}")

# ----- Google News Embeddings -----------------------------------------
            
@st.cache_resource
def load_google_news():
    """This uses the complete google news embedding model."""
    # return api.load("word2vec-google-news-300")

    """This uses the slimmer English-only google news embedding model."""
    model_path = "GoogleNews-vectors-negative300-SLIM.bin.gz"
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model

@st.cache_resource
def align_google_to_arxiv(_arxiv_model, folder):
    google_model = load_google_news()

    common_words = list(
        set(_arxiv_model.wv.index_to_key)
        & set(google_model.key_to_index)
    )

    if len(common_words) < 100:
        raise ValueError(
            f"Only {len(common_words)} shared words found."
        )

    X = np.array([
        google_model[word]
        for word in common_words
    ])

    Y = np.array([
        _arxiv_model.wv[word]
        for word in common_words
    ])

    # Normalize
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    R, _ = orthogonal_procrustes(X, Y)

    return google_model, R

def aligned_google_vector(word, google_model, R):

    vec = google_model[word]

    vec = vec / np.linalg.norm(vec)

    return vec @ R

@st.cache_data
def arxiv_to_google_distances(
    _arxiv_model,
    _google_model,
    _R,
    folder,
    topn=20
):
    arxiv_model = _arxiv_model
    google_model = _google_model
    R = _R
    shared_words = [w for w in arxiv_model.wv.index_to_key[:topn] if w in google_model]

    arxiv_vecs = normalize(np.array([arxiv_model.wv[w] for w in shared_words]))

    google_vecs = normalize(np.array([aligned_google_vector(w, google_model, R) for w in shared_words]))

    cosine_similarities = np.sum(arxiv_vecs * google_vecs, axis=1)
    cosine_distances = 1 - cosine_similarities

    df = pd.DataFrame({
        "word": shared_words,
        "cosine_similarity": cosine_similarities,
        "cosine_distance": cosine_distances
    })

    return df.sort_values(
        "cosine_distance",
        ascending=False
    )

# ----- arXiv Embeddings -----------------------------------------

@st.cache_resource
def train_model(folder: str):
    sentences = FolderSentences(folder)
    model = Word2Vec(
        sentences=sentences,
        window=4,
        min_count=5,
        vector_size=300,
        sg=1 #skip-gram
    )
    return model

@st.cache_data
def reduce_dimensions(_model, word_list):
    vectors = np.array([_model.wv[word] for word in word_list])
    
    tsne = TSNE(
        n_components=2, 
        perplexity=min(30, len(word_list)-1), 
        n_iter=500, 
        random_state=42)
    vectors_2d = tsne.fit_transform(vectors)
    return vectors_2d

# ----- UI -----------------------------------------

st.set_page_config(layout="wide")
st.title("Word2Vec Embedding Validation Prototype")

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

st.session_state["selected_folder"] = selected_folder
corpus_btn = st.button("Select Folder", width="content")
if corpus_btn:

    st.write("**Processing folder:**")
    
    if(selected_folder):
        arxiv_model = train_model(st.session_state["selected_folder"])
    top_words = arxiv_model.wv.index_to_key[:100]

    st.session_state["top_words"] = top_words

    st.session_state["arxiv_model"] = arxiv_model

    vectors_2d = reduce_dimensions(arxiv_model, top_words)
    st.session_state["vectors_2d"] = vectors_2d

    st.session_state["google_model"], st.session_state["R"] = align_google_to_arxiv(st.session_state["arxiv_model"], st.session_state["selected_folder"])

if "vectors_2d" in st.session_state:
    top_words = st.session_state["top_words"]
    vectors_2d = st.session_state["vectors_2d"]

     # 5. Create a DataFrame for Plotting
    df = pd.DataFrame({
        'Word': top_words,
        'X': vectors_2d[:, 0],
        'Y': vectors_2d[:, 1]
    })

    # 6. Build Interactive Plotly Chart
    fig = px.scatter(
        df, 
        x='X', 
        y='Y', 
        text='Word', 
        title="2D Map of Word Semantics",
        labels={'X': 't-SNE Dimension 1', 'Y': 't-SNE Dimension 2'}
    )

    # Customize the marker text positions so they don't overlap heavily
    fig.update_traces(textposition='top center', marker=dict(size=10, color='royalblue'))
    fig.update_layout(height=600, hovermode='closest')

    # 7. Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # 8. Raw Data Viewer Sidebar
    st.sidebar.header("Explore Vector Data")
    selected_word = st.sidebar.selectbox("Select a word to view raw weights:", top_words)
    st.sidebar.write(f"Raw 300-D Vector for **'{selected_word}'**:")
    st.sidebar.json(st.session_state["arxiv_model"].wv[selected_word].tolist())


st.subheader("Most dissimilar words compared to Google News dataset")
topn = st.number_input("Number of words to compare (from most used to least used).", placeholder=20, step=1)
if topn:
    if "arxiv_model" not in st.session_state or "google_model" not in st.session_state or "R" not in st.session_state:
        st.warning("Please select a folder before running this feature.")
    else:
        results = arxiv_to_google_distances(
            st.session_state["arxiv_model"],
            st.session_state["google_model"],
            st.session_state["R"],
            st.session_state["selected_folder"],
            topn
        )

        if results is None:
            st.warning(
                "Something went wrong when comparing with Google News vectors."
            )
        else:
            st.dataframe(results)

st.subheader("Nearest neighbours")
query = st.text_input("Enter a word to find its neighbours:", placeholder="e.g. neural")
if query:
    if("arxiv_model" not in st.session_state):
        st.warning("Please select a folder before running this task")
    else:
        if query in st.session_state["arxiv_model"].wv:
            neighbours = st.session_state["arxiv_model"].wv.most_similar(query, topn=10)
            st.text("arXiv embedding model neighbors")
            st.table(pd.DataFrame(neighbours, columns=["Word", "Cosine similarity"]))
            
            if(query in st.session_state["google_model"]):
                neighbors_google = st.session_state["google_model"].most_similar(query, topn=10)
                st.text("Google News embedding model neighbors")
                st.table(pd.DataFrame(neighbors_google, columns=["Word", "Cosine similarity"]))
            else:
                st.text("Word not in the google news embedding model")
        else:
            st.warning(f"'{query}' not in vocabulary (min_count=5 may have filtered it).")