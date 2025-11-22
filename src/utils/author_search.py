import json
import os
from rapidfuzz import fuzz, process

from src.api.hardcover_api import search_book_api, API_URL, API_TOKEN, HEADERS
from src.utils.embeddings import embed_book
from src.utils.faiss_manager import FaissManager

CATALOG_PATH = r"E:\book_recommender\data\catalog.json"

print("DEBUG AUTHOR: Script started")

#load/save catalog
def load_catalog():
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f" DEBUG AUTHOR: Failed to load catalog: {e}")
        return {}

def save_catalog(catalog):
    try:
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=4)
    except Exception as e:
        print(f" DEBUG AUTHOR: Failed to save catalog: {e}")

#fuzzy match in author search
def fuzzy_find_author(author_name, min_score=70):
    print(f"DEBUG AUTHOR: Fuzzy finding author: '{author_name}'")
    catalog = load_catalog()

    all_authors = []
    for book_id, book in catalog.items():
        for a in book.get("authors", []):
            all_authors.append(a.lower())

    print(f"DEBUG AUTHOR: Found {len(all_authors)} total authors in catalog")

    if not all_authors:
        return None

    match, score, _ = process.extractOne(
        author_name.lower(), 
        all_authors,
        scorer=fuzz.WRatio
    )

    print(f" DEBUG AUTHOR: Best match: '{match}' with score: {score}")

    if score < min_score:
        print(f" DEBUG AUTHOR: Score too low ({score} < {min_score})")
        return None

    return match

#get all books by author in the catalog
def get_author_books_from_catalog(author):
    print(f" DEBUG AUTHOR: Getting books for author: '{author}'")
    catalog = load_catalog()

    results = []
    for book_id, book in catalog.items():
        for a in book.get("authors", []):
            if a.lower() == author.lower():
                results.append((book_id, book))

    print(f" DEBUG AUTHOR: Found {len(results)} books in catalog")
    return results

#get author books from api
def fetch_books_by_author_api(author_name, limit=5):
    
    print(f" DEBUG AUTHOR: Fetching from API for author: '{author_name}'")
    
    AUTHOR_QUERY = """
    query SearchAuthor($query: String!) {
      search(query: $query, query_type: "Author", per_page: 1) {
        ids
        results
      }
    }
    """

    payload = {
        "query": AUTHOR_QUERY,
        "variables": {"query": author_name}
    }

    try:
        import requests
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        print(f" DEBUG AUTHOR: API response status: {response.status_code}")
        
        data = response.json()
        print(f" DEBUG AUTHOR: API response keys: {data.keys()}")

        if "data" not in data or data["data"]["search"] is None:
            print(" DEBUG AUTHOR: API error - data missing")
            return []

        result_block = data["data"]["search"]
        print(f" DEBUG AUTHOR: Search block: {result_block}")

        #### now hardcover results are dictionary
        results_json = result_block["results"]
        print(f" DEBUG AUTHOR: Results type: {type(results_json)}")

        hits = results_json.get("hits", [])
        print(f" DEBUG AUTHOR: Found {len(hits)} hits")

        if not hits:
            return []

        author_doc = hits[0]["document"]
        print(f" DEBUG AUTHOR: Found author: {author_doc.get('name')}")

        # Get book titles from author document
        raw_book_titles = author_doc.get("books", [])[:limit]
        print(f" DEBUG AUTHOR: Raw book titles: {raw_book_titles}")

        # Fetch full metadata for each book title
        fetched_books = []
        for title in raw_book_titles:
            print(f" DEBUG AUTHOR: Fetching book: '{title}'")
            book_data = search_book_api(title)
            if book_data:
                fetched_books.append(book_data)
                print(f" DEBUG AUTHOR: Successfully fetched: '{book_data.get('title')}'")
            else:
                print(f" DEBUG AUTHOR: Failed to fetch: '{title}'")

        print(f" DEBUG AUTHOR: Total books fetched: {len(fetched_books)}")
        return fetched_books

    except Exception as e:
        print(f" DEBUG AUTHOR: Exception while fetching author: {e}")
        return []

#add books to catalog and faiss
def add_books_to_system(book_list):
    """
    Add new books to system
    """
    print(f" DEBUG AUTHOR: Adding {len(book_list)} books to system")
    catalog = load_catalog()
    faiss = FaissManager()

    added_ids = []

    for book in book_list:
        book_id = book["book_id"]

        # skip duplicates
        if book_id in catalog:
            print(f" DEBUG AUTHOR: Book {book_id} already in catalog")
            continue

        # Add to catalog
        catalog[book_id] = book

        # Generate embedding
        vector = embed_book(book)

        # Add to faiss
        faiss.add_new_embedding(book_id, vector)

        added_ids.append(book_id)
        print(f" DEBUG AUTHOR: Added book {book_id}: {book.get('title')}")

    save_catalog(catalog)
    print(f"DEBUG AUTHOR: Saved catalog with {len(added_ids)} new books")
    return added_ids

#search author pipeline
def search_author(author_query):
    print(f"\n DEBUG AUTHOR: Starting author search for: '{author_query}'")

    # Step 1: Fuzzy match author in catalog
    matched_author = fuzzy_find_author(author_query)
    matched_author = matched_author or author_query  

    print(f" DEBUG AUTHOR: Using author name: '{matched_author}'")

    # Step 2: get catalog books
    catalog_books = get_author_books_from_catalog(matched_author)

    # Step 3: if catalog has fewer than 2 books, fetch via API
    if len(catalog_books) < 2:
        print(f" DEBUG AUTHOR: Only {len(catalog_books)} books in catalog, fetching from API...")
        api_books = fetch_books_by_author_api(matched_author)

        # Add new books
        added = add_books_to_system(api_books)
        print(f" DEBUG AUTHOR: Added {len(added)} books from API")

        # reload updated list
        catalog_books = get_author_books_from_catalog(matched_author)

    print(f" DEBUG AUTHOR: Returning {len(catalog_books)} books")
    return catalog_books

if __name__ == "__main__":
    '''print(" DEBUG AUTHOR: Running test...")
    results = search_author("Kristin Hannah")
    for book_id, book in results:
        print(f" {book_id} → {book['title']}")'''