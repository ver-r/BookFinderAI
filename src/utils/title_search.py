import json
import re
from rapidfuzz import fuzz, process

CATALOG_PATH = r"E:\book_recommender\data\catalog.json"

# Load Catalog Once
with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    CATALOG = json.load(f)

# Create: title to book_id lookup map
TITLE_MAP = { book["title"].lower(): book_id for book_id, book in CATALOG.items() }

# Normalize titles (remove punctuation, lowercase)
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Fuzzy title search
def find_best_title_match(user_input, min_score=70):
    user_input_norm = normalize(user_input)

    # Get list of all catalog titles
    titles = list(TITLE_MAP.keys())

    # Use RapidFuzz to get best match
    match, score, _ = process.extractOne(
        user_input_norm,
        titles,
        scorer=fuzz.WRatio
    )

    if score < min_score:
        return None

    best_book_id = TITLE_MAP[match]
    
    return {
        "book_id": best_book_id,
        "title": match,
        "score": score
    }

# Fuzzy multi-result (top K) search
def find_top_matches(user_input, limit=5):
    user_input_norm = normalize(user_input)

    titles = list(TITLE_MAP.keys())

    results = process.extract(
        user_input_norm,
        titles,
        scorer=fuzz.WRatio,
        limit=limit
    )

    output = []
    for match, score, _ in results:
        book_id = TITLE_MAP[match]
        output.append({
            "book_id": book_id,
            "title": match,
            "score": score
        })

    return output

#print(find_best_title_match("the nightngale"))
#print(find_best_title_match("nightingale"))
#print(find_best_title_match("kahd hosseni"))

