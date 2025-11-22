import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
load_dotenv(dotenv_path=r"E:\book_recommender\.env")

API_URL = os.getenv("HARDCOVER_API_URL")
API_TOKEN = os.getenv("HARDCOVER_API_TOKEN")

HEADERS = {
    "content-type": "application/json",
    "authorization": API_TOKEN
}
SEARCH_QUERY = """
query SearchBook($query: String!) {
  search(query: $query, query_type: "Book", per_page: 1) {
    ids
    results
  }
}
"""

def clean_document(doc):
    return {
        "book_id": doc.get("id"),
        "title": doc.get("title"),
        "authors": doc.get("author_names", []),
        "description": doc.get("description", ""),
        "genres": doc.get("genres", []),
        "tags": doc.get("tags", []),
        "moods": doc.get("moods", []),
        "rating": doc.get("rating"),
        "pages": doc.get("pages"),
        "published_year": doc.get("release_year"),
        "image_url": doc.get("image", {}).get("url") if doc.get("image") else None,
        "isbn_list": doc.get("isbns", []),
        "embedding_vector_path": None,
        "added_source": "api",
        "last_updated": datetime.now().strftime("%Y-%m-%d")
    }

# search book api
def search_book_api(title):
    payload = {
        "query": SEARCH_QUERY,
        "variables": {"query": title}
    }

    try:
        print(f"DEBUG API: Searching for '{title}'")
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        print(f"DEBUG API: Response status: {response.status_code}")
        data = response.json()
        if "data" not in data or data["data"]["search"] is None:
            print("Error:", data)
            return None

        search_block = data["data"]["search"]

        if not search_block["ids"]:
            return None

        results_json = search_block["results"]

        hits = results_json.get("hits", [])
        if not hits:
            return None

        doc = hits[0]["document"]

        return clean_document(doc)

    except Exception as e:
        print("Exception while calling API:", e)
        return None


if __name__ == "__main__":
    print(search_book_api("The Nightingale"))
