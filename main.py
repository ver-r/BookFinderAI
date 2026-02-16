import streamlit as st
import json
import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer


from src.core.recommender import (
    recommend_by_title,
    recommend_by_author,
    recommend_by_prompt,
    recommend_from_bookshelf_image,
)
from config import DEFAULT_COVER_IMAGE, CATALOG_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("book_recommender_streamlit")

GALLERY_COLUMNS = 4
THUMBNAIL_COLUMNS = 6  

# Load full book details from catalog
def load_catalog_book(book_id: str) -> Optional[Dict[str, Any]]:
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        return catalog.get(str(book_id))
    except Exception:
        return None

# Session state setup
if "books_state" not in st.session_state:
    st.session_state.books_state = []
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

def reset_selection():
    st.session_state.selected_index = None

# Search handlers
def title_search_handler(title):
    reset_selection()
    if not title:
        st.warning("Enter a title")
        return
    res = recommend_by_title(title)
    st.session_state.books_state = res.get("recommendations", []) if res.get("status") == "ok" else []

def author_search_handler(author):
    reset_selection()
    if not author:
        st.warning("Enter an author")
        return
    res = recommend_by_author(author)
    st.session_state.books_state = res.get("books", []) if res.get("status") == "ok" else []

def prompt_search_handler(genres, moods, free_text):
    reset_selection()
    res = recommend_by_prompt(genres or [], moods or [], free_text or "")
    st.session_state.books_state = res or []

def ocr_scan_handler(f):
    reset_selection()
    if f is None:
        st.warning("Upload an image")
        return
    arr = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    res = recommend_from_bookshelf_image(img)
    st.session_state.books_state = res.get("recommendations", []) if res.get("status") == "ok" else []

# Page Layout
st.set_page_config(page_title="BookFinderAI", layout="wide")

st.markdown("""
<style>
.details-panel {
    max-height: 80vh;
    overflow-y: auto;
    background: #0f0f10;
    padding: 20px;
    border-radius: 12px;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 12px;
    font-size: 0.9em;
}
.genre-badge {
    background: #eef;
    color: #333;
}
.mood-badge {
    background: #efe;
    color: #333;
}
.tag-badge {
    background: #fee;
    color: #333;
}
/* Make all grid images same size */
.stImage img {
    height: 400px;
    object-fit: cover;
    width: 100%;
}
/* Make thumbnail grid images smaller and uniform */
.thumbnail-grid img {
    height: 80px !important;
    object-fit: cover;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## BookFinderAI")

# Tabs
tabs = st.tabs(["Search by Title", "Search by Author", "Search by Prompt", "Bookshelf OCR", "Analytics"])

# TITLE TAB
with tabs[0]:
    title_str = st.text_input("Book Title")
    if st.button("Search Title"):
        title_search_handler(title_str)

# AUTHOR TAB
with tabs[1]:
    author_str = st.text_input("Author Name")
    if st.button("Search Author"):
        author_search_handler(author_str)

# PROMPT TAB
with tabs[2]:
    # Initialize session state for selected genres and moods
    if "selected_genres" not in st.session_state:
        st.session_state.selected_genres = []
    if "selected_moods" not in st.session_state:
        st.session_state.selected_moods = []
    
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Genres**")
        genre_options = ["Fantasy", "Historical Fiction", "Action", "Romance", "Science Fiction", "Young Adult", "Thriller-Mystery", "Memoir", "Biography", "Self Help", "True Crime", "Horror"]
        genre_cols = st.columns(2)
        for i, genre in enumerate(genre_options):
            with genre_cols[i % 2]:
                is_selected = genre in st.session_state.selected_genres
                if st.button(genre, key=f"genre_{genre}", width='stretch', type="primary" if is_selected else "secondary"):
                    if is_selected:
                        st.session_state.selected_genres.remove(genre)
                    else:
                        st.session_state.selected_genres.append(genre)
                    st.rerun()
        
        st.write("")
        st.markdown("**Moods**")
        mood_options = ["Dark", "Funny", "Emotional", "Suspenseful", "Romantic", "Uplifting", "Thought-provoking", "Action-packed"]
        mood_cols = st.columns(2)
        for i, mood in enumerate(mood_options):
            with mood_cols[i % 2]:
                is_selected = mood in st.session_state.selected_moods
                if st.button(mood, key=f"mood_{mood}", width='stretch', type="primary" if is_selected else "secondary"):
                    if is_selected:
                        st.session_state.selected_moods.remove(mood)
                    else:
                        st.session_state.selected_moods.append(mood)
                    st.rerun()
    
    with right:
        txt = st.text_area("Describe what you want")
        if st.button("Search Prompt"):
            prompt_search_handler(st.session_state.selected_genres, st.session_state.selected_moods, txt)

# OCR TAB
with tabs[3]:
    f = st.file_uploader("Upload bookshelf photo", type=["jpg", "png"])
    if st.button("Scan OCR"):
        ocr_scan_handler(f)

st.write("---")

books = st.session_state.books_state
idx = st.session_state.selected_index

if not books:
    st.info("Search to see results")
    st.stop()

# ANALYTICS TAB
with tabs[4]:
    st.markdown("##  Catalog Analytics")

    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        df = pd.DataFrame.from_dict(catalog, orient="index")

        if df.empty:
            st.warning("Catalog is empty.")
        else:
            # KPI Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Books", len(df))
            col2.metric("Unique Authors", df["authors"].explode().nunique())
            col3.metric("Unique Genres", df["genres"].explode().nunique())
            st.write("---")
            st.markdown("### Data Quality Analysis")

            missing_desc_pct = (
                df["description"].isna() |
                (df["description"].astype(str).str.strip() == "")
            ).mean() * 100

            # Missing ratings: None or NaN
            missing_rating_pct = df["rating"].isna().mean() * 100

            # Missing genres: not list OR empty list
            missing_genres_pct = df["genres"].apply(
                lambda x: not isinstance(x, list) or len(x) == 0
            ).mean() * 100
            q1, q2, q3 = st.columns(3)
            q1.metric("Missing Descriptions %", f"{missing_desc_pct:.1f}%")
            q2.metric("Missing Ratings %", f"{missing_rating_pct:.1f}%")
            q3.metric("Missing Genres %", f"{missing_genres_pct:.1f}%")

            st.write("---")

            # Genre Distribution
            st.markdown("### Genre Distribution")
            genre_df = df.explode("genres")
            genre_counts = genre_df["genres"].value_counts().reset_index()
            genre_counts.columns = ["Genre", "Count"]

            fig1 = px.bar(
                genre_counts,
                x="Genre",
                y="Count"
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.write("---")

            # Top Authors
            st.markdown("### Top Authors")
            author_counts = df["authors"].explode().value_counts().head(10).reset_index()
            author_counts.columns = ["Author", "Count"]

            fig2 = px.bar(
                author_counts,
                x="Author",
                y="Count"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.write("---")

            # PCA Visualization
            st.markdown("### Semantic Clustering (PCA)")

            try:
                embeddings = np.load("E:/book_recommender/vectors/embeddings.npy")
                if len(embeddings) > 2:
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(embeddings)

                    viz_df = pd.DataFrame({
                        "x": reduced[:, 0],
                        "y": reduced[:, 1],
                        "title": df["title"].values[:len(reduced)],
                        "genre": df["genres"].apply(lambda x: x[0] if isinstance(x, list) and x else "Unknown").values[:len(reduced)]
                    })

                    fig3 = px.scatter(
                        viz_df,
                        x="x",
                        y="y",
                        color="genre",
                        hover_name="title"
                    )

                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Not enough embeddings for PCA visualization.")

            except Exception as e:
                st.warning(f"PCA visualization failed: {e}")

    except Exception as e:
        st.error(f"Analytics loading failed: {e}")

    st.write("---")
    st.markdown("### Genre Correlation Matrix (Top Genres)")

    try:
        
        cleaned_genres = df["genres"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        
        all_genres = cleaned_genres.explode()
        top_genres = all_genres.value_counts().head(8).index.tolist()

        filtered = cleaned_genres.apply(
            lambda x: [g for g in x if g in top_genres]
        )

        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(filtered)

        if genre_matrix.shape[1] > 1:
            genre_df = pd.DataFrame(
                genre_matrix,
                columns=mlb.classes_
            )

            corr = genre_df.corr()

            fig, ax = plt.subplots(figsize=(12, 10))

            sns.heatmap(
                corr,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                ax=ax
            )

            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)

            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Not enough genre variety for correlation analysis.")

    except Exception as e:
        st.warning(f"Genre correlation failed: {e}")



# IF NO BOOK SELECTED – SHOW FULL GRID
if idx is None:
    cols = st.columns(GALLERY_COLUMNS)
    for i, book in enumerate(books):
        with cols[i % GALLERY_COLUMNS]:
            st.image(book.get("image_url") or DEFAULT_COVER_IMAGE, width='stretch')
            st.write(f"**{book.get('title', '')}**")
            if st.button("View", key=f"view_{i}"):
                st.session_state.selected_index = i
                st.rerun()
    st.stop()

left_col, right_col = st.columns([2, 3])

with left_col:
    st.markdown("#### All Books")
    
    st.markdown('<div class="thumbnail-grid">', unsafe_allow_html=True)
    thumb_cols = st.columns(THUMBNAIL_COLUMNS)
    for i, book in enumerate(books):
        with thumb_cols[i % THUMBNAIL_COLUMNS]:
            st.image(book.get("image_url") or DEFAULT_COVER_IMAGE, width='stretch')
            if st.button("View", key=f"thumb_{i}", help=book.get('title', ''), width='stretch'):
                st.session_state.selected_index = i
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("---")
    
    # Current book image
    selected = books[idx]
    st.markdown("#### Currently Viewing")
    st.image(selected.get("image_url") or DEFAULT_COVER_IMAGE, width=250)

with right_col:
    st.markdown("### Book Details")

    # Prev/Next/Close buttons
    pcol, ccol, ncol = st.columns([1, 1, 1])
    with pcol:
        if st.button("Previous", disabled=(idx == 0), width='stretch'):
            st.session_state.selected_index = max(0, idx - 1)
            st.rerun()
    with ccol:
        if st.button("Close", width='stretch'):
            st.session_state.selected_index = None
            st.rerun()
    with ncol:
        if st.button("Next", disabled=(idx == len(books) - 1), width='stretch'):
            st.session_state.selected_index = min(len(books) - 1, idx + 1)
            st.rerun()

    st.write("")
    
    # Load full metadata
    selected = books[idx]
    full = load_catalog_book(selected.get("book_id")) or selected
    
    # Display details 
    with st.container():
        # Title and Author
        st.markdown(f"## {full.get('title', 'Unknown Title')}")
        authors = ", ".join(full.get("authors", [])) or "Unknown Author"
        st.markdown(f"#### by {authors}")
        
        st.write("---")
        
        # Rating and Pages
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rating", full.get("rating", "N/A"))
        with col2:
            st.metric("Pages", full.get("pages", "Unknown"))
        
        st.write("---")
        
        # Description
        st.markdown("**Description**")
        desc_text = full.get("description") or "No description available"
        st.markdown(f"<p style='text-align: justify;'>{desc_text}</p>", unsafe_allow_html=True)
        
        st.write("---")
        
        # Genres
        genres = full.get("genres", []) or []
        if genres:
            st.markdown("**Genres**")
            genre_html = " ".join([f"<span class='badge genre-badge'>{g}</span>" for g in genres])
            st.markdown(genre_html, unsafe_allow_html=True)
            st.write("")
        
        # Moods
        moods = full.get("moods", []) or []
        if moods:
            st.markdown("**Moods**")
            mood_html = " ".join([f"<span class='badge mood-badge'>{m}</span>" for m in moods])
            st.markdown(mood_html, unsafe_allow_html=True)
            st.write("")
        
        # Tags
        tags = full.get("tags", []) or []
        if tags:
            st.markdown("**Tags**")
            tag_html = " ".join([f"<span class='badge tag-badge'>{t}</span>" for t in tags])
            st.markdown(tag_html, unsafe_allow_html=True)