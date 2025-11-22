import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model ONCE globally
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

#Build text input for embeddings
def build_embedding_text(book):
   
    title = book.get("title", "")
    authors = ", ".join(book.get("authors", []))
    description = book.get("description", "")
    genres = ", ".join(book.get("genres", []))
    tags = ", ".join(book.get("tags", []))
    moods = ", ".join(book.get("moods", []))

    return (
        f"Title: {title}. "
        f"Author: {authors}. "
        f"Genres: {genres}. "
        f"Moods: {moods}. "
        f"Tags: {tags}. "
        f"Description: {description}"
    )

# Generate embedding for BOOK
def embed_book(book_dict):
    
    text = build_embedding_text(book_dict)
    embedding = model.encode(text)
    return embedding.astype("float32")

def embed_query(query_text):
    embedding = model.encode(query_text)
    return embedding.astype("float32")

def embed_multiple_books(book_list):
    texts = [build_embedding_text(b) for b in book_list]
    vectors = model.encode(texts)
    return vectors.astype("float32")
