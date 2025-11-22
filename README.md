# BookFinderAI  
An Intelligent Book Recommendation System with Semantic Search, OCR, and Dynamic Catalog Expansion

---

## Overview

BookFinderAI is an end-to-end intelligent book recommendation system that combines:

- Transformer-based semantic embeddings (MiniLM)
- FAISS vector similarity search
- OCR-based bookshelf text extraction
- Dynamic catalog expansion using the Hardcover GraphQL API
- An interactive Streamlit user interface

The system supports multiple input modes:

- Search by book title  
- Search by author  
- Search by genres, moods, and free-text prompts  
- Upload a bookshelf image, extract text using OCR, and generate recommendations

---

## Features

### 1. Semantic Book Recommendations
- Uses **MiniLM-L6-v2** embeddings to generate dense semantic vectors  
- Performs **cosine similarity search** through FAISS  
- Allows open-ended free-text queries (e.g., “dark emotional historical fiction”)  
- Returns semantically similar books from the vector database

### 2. Bookshelf Image Scanning (OCR)
- Uses **PaddleOCR** to detect and extract text from book spines  
- Extracted text is converted into a semantic embedding  
- Recommendations are generated purely through vector similarity  
- No strict title matching; all OCR-detected text is treated semantically  

### 3. Dynamic Catalog Expansion
If a queried title or author is not found in the local catalog:

1. The system fetches metadata from the **Hardcover GraphQL API**  
2. The book is added to `catalog.json`  
3. An embedding is generated and appended to `embeddings.npy`  
4. FAISS index and ID map are updated  
5. Future searches automatically include the new book  

### 4. Interactive Streamlit User Interface
Provides a clean interface with:

- Title search  
- Author search  
- Semantic prompt search  
- OCR upload  
- Book gallery view  
- Detailed book viewer with metadata  

---

## Search Logic

### A. Title Search
1. Fuzzy title matching using RapidFuzz  
2. If found locally → generate embedding → FAISS similarity search  
3. If not found → fetch from Hardcover API → add to system → search again  
4. Returns top similar books

### B. Author Search
1. Fuzzy author matching in local catalog  
2. If few books exist locally → fetch author’s books from API  
3. Add missing books to catalog + FAISS  
4. Return all books by the matched author

### C. Semantic Prompt Search
1. User provides genres, moods, tags, or free text  
2. A combined prompt string is generated  
3. Embedding is computed using MiniLM  
4. FAISS retrieves the most semantically similar books

### D. OCR-Based Bookshelf Search
1. OCR extracts text from a bookshelf image  
2. All extracted text is cleaned and combined  
3. A semantic query embedding is created  
4. FAISS retrieves thematically relevant books  
5. No API calls or strict title matching are involved

---


