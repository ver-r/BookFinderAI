import logging
import json
from typing import List, Dict

# Import dependencies
from src.api.hardcover_api import search_book_api
from src.utils.embeddings import embed_book, embed_query
from src.utils.faiss_manager import FaissManager
from src.utils.title_search import find_best_title_match
from src.utils.author_search import search_author
from src.utils.ocr_reader import recommend_from_bookshelf

# Local catalog
CATALOG_PATH = r"E:\book_recommender\data\catalog.json"
DEFAULT_COVER = "https://via.placeholder.com/150x220?text=No+Cover"
DEFAULT_K = 5
MIN_TITLE_SCORE = 95

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

# Load/save catalog
def load_catalog():
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed loading catalog.json: {e}")
        return {}

def save_catalog(catalog):
    try:
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed saving catalog.json: {e}")

# Formatting function
def format_result(book_id: str, book: dict, distance: float = None) -> dict:
    return {
        "book_id": book_id,
        "title": book.get("title"),
        "authors": book.get("authors", []),
        "description": book.get("description", ""),
        "image_url": book.get("image_url") or DEFAULT_COVER,
        "distance": distance
    }

# Add new book to system 
def add_book_to_system(book: dict):
    """
    Add a book to:
      - catalog.json
      - embeddings.npy
      - FAISS index
      - id_map.json
    """
    try:
        catalog = load_catalog()
        book_id = str(book["book_id"])

        print(f" DEBUG: Checking if book {book_id} exists in catalog...")
        
        if book_id in catalog:
            print(f" DEBUG: Book {book_id} already exists in catalog")
            return False

        print(f" DEBUG: Adding new book to system:")
        print(f"   Title: {book.get('title')}")
        print(f"   ID: {book_id}")
        print(f"   Authors: {book.get('authors', [])}")

        # Save to catalog
        catalog[book_id] = book
        save_catalog(catalog)
        print(f" DEBUG: Saved to catalog.json")

        # Embed & register in FAISS
        print(f" DEBUG: Generating embedding...")
        vector = embed_book(book)
        print(f" DEBUG: Embedding generated, shape: {vector.shape}")
        faiss = FaissManager()
        print(f" DEBUG: Adding to FAISS...")
        faiss.add_new_embedding(book_id, vector)
        print(f" DEBUG: Added to FAISS successfully")

        logging.info(f"Added new book: {book['title']} ({book_id})")
        return True

    except Exception as e:
        logging.error(f"Failed to add book to system: {e}")
        print(f" DEBUG: Error adding book: {e}")
        return False

# INTERNAL: FAISS search using vector
def _search_faiss(vector, k=DEFAULT_K):
    try:
        faiss = FaissManager()
        catalog = load_catalog()
        print(f"DEBUG: FAISS search with vector shape: {vector.shape}")
        
        raw = faiss.search(vector, k=k)
        print(f"DEBUG: FAISS returned {len(raw)} results")

        results = []
        for r in raw:
            bid = str(r["book_id"])
            b = catalog.get(bid)
            if not b:
                print(f"DEBUG: Book {bid} not found in catalog but exists in FAISS")
                continue
            results.append(format_result(bid, b, r["distance"]))

        print(f" DEBUG: Formatted {len(results)} results")
        return results

    except Exception as e:
        logging.error(f"FAISS search failed: {e}")
        print(f" DEBUG: FAISS search error: {e}")
        return []
 
def recommend_by_title(title: str, k=DEFAULT_K) -> dict:
    try:
        print(f"\n DEBUG: Starting title search for: '{title}'")
        
        # First attempt fuzzy search locally
        print(f"DEBUG: Attempting local title match...")
        match = find_best_title_match(title, min_score=MIN_TITLE_SCORE)

        if match:
            print(f" DEBUG: Found local match: {match['title']} (score: {match['score']})")
            seed_id = match["book_id"]
            catalog = load_catalog()
            seed_book = catalog.get(seed_id)
            if not seed_book:
                print(f" DEBUG: Book exists in index but not in catalog - corruption!")
                return {"status": "error", "message": "Book exists but catalog is corrupted"}

            print(f"DEBUG: Generating embedding for local book...")
            vector = embed_book(seed_book)
            recs = _search_faiss(vector, k=k)

            return {
                "status": "ok",
                "source": "catalog",
                "matched_title": match["title"],
                "book_id": seed_id,
                "recommendations": recs
            }

        # Not found, fetch using API
        print(f" DEBUG: No local match found, calling API...")
        api_book = search_book_api(title)
        
        if not api_book:
            print(f" DEBUG: API returned no results")
            return {
                "status": "not_found",
                "matched_title": None,
                "recommendations": []
            }

        print(f" DEBUG: API returned book: {api_book.get('title')}")
        print(f"   ID: {api_book.get('book_id')}")

        # Add to system
        print(f" DEBUG: Adding API book to system...")
        success = add_book_to_system(api_book)
        
        if not success:
            print(f"DEBUG: Failed to add book to system")
            return {
                "status": "error", 
                "message": "Failed to add book to system",
                "recommendations": []
            }

        print(f"DEBUG: Generating embedding for newly added book...")
        vector = embed_book(api_book)
        recs = _search_faiss(vector, k=k)

        print(f"DEBUG: Search complete, returning {len(recs)} recommendations")
        return {
            "status": "ok",
            "source": "api",
            "matched_title": api_book["title"],
            "book_id": api_book["book_id"],
            "recommendations": recs
        }

    except Exception as e:
        logging.error(f"recommend_by_title() crashed: {e}")
        print(f"DEBUG: recommend_by_title error: {e}")
        return {"status": "error", "recommendations": []}

# 2) Recommend by PROMPT (genre, mood, typed text)
from src.utils.prompt_builder import build_prompt

def recommend_by_prompt(genres=None, moods=None, tags=None, free_text=None, k=DEFAULT_K):
    """
    Full semantic recommendation using genres/moods/tags/text.
    """
    try:
        #  Build semantic string
        prompt = build_prompt(
            genres=genres,
            moods=moods,
            tags=tags,
            free_text=free_text
        )

        if not prompt:
            return []

        #Convert prompt into embedding vector
        qvec = embed_query(prompt)

        #FAISS similarity search
        return _search_faiss(qvec, k=k)

    except Exception as e:
        logging.error(f"recommend_by_prompt() crashed: {e}")
        return []

# 3) SEARCH BY AUTHOR — integrate author_search
def recommend_by_author(author_name: str) -> dict:
    """
    Returns:
    {
      "status": "ok",
      "author": "matched_author",
      "books": [ { book_id, title, ... }, ... ]
    }
    """
    try:
        author_books = search_author(author_name)

        # Format output consistently
        formatted = [
            {
                "book_id": book_id,
                "title": book.get("title"),
                "authors": book.get("authors", []),
                "description": book.get("description", ""),
                "image_url": book.get("image_url") or DEFAULT_COVER
            }
            for book_id, book in author_books
        ]

        if not formatted:
            return {
                "status": "not_found",
                "author": author_name,
                "books": []
            }

        return {
            "status": "ok",
            "author": author_name,
            "books": formatted
        }

    except Exception as e:
        logging.error(f"recommend_by_author() crashed: {e}")
        return {"status": "error", "books": []}

def recommend_from_bookshelf_image(image, k=DEFAULT_K):
    """
    Process bookshelf image and return recommendations based on detected books.
    
    Args:
        image: numpy array or image file path
        k: number of recommendations
    
    Returns:
        dict with detected books and recommendations
    """
    try:
        # If image is a file path, convert to numpy array
        if isinstance(image, str):
            import cv2
            image = cv2.imread(image)
            if image is None:
                return {
                    "status": "error", 
                    "message": "Could not load image from path",
                    "detected_books": [],
                    "recommendations": []
                }
            # Convert BGR to RGB for consistency
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use the OCR reader function
        result = recommend_from_bookshelf(image, k=k)
        
        return result
        
    except Exception as e:
        logging.error(f"Bookshelf recommendation failed: {e}")
        return {
            "status": "error",
            "message": f"Processing failed: {str(e)}",
            "detected_books": [],
            "recommendations": []
        }

if __name__ == "__main__":
    print(recommend_by_title("The Nightingale"))
    print(recommend_by_prompt("historical fiction emotional"))
    print(recommend_by_author("Kristin Hannah"))
