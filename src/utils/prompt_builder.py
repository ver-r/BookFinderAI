import logging
import re

# Logging Setup
logger = logging.getLogger("prompt_builder")
logger.setLevel(logging.INFO)

# 1. Clean text helper
def normalize_text(text: str) -> str:
    """
    Lowercase + remove symbols + compress spaces.
    Safe for user input like: "dark, emotional! fantasy??"
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 2. Build semantic prompt from components
def build_prompt(genres=None, moods=None, tags=None, free_text=None):
    """
    Takes any combination of:
    - genres:     list[str]
    - moods:      list[str]
    - tags:       list[str]
    - free_text:  str (like “historical fiction about sisters & war”)

    Returns a single optimized string for embeddings.
    """
    try:
        genres = genres or []
        moods = moods or []
        tags = tags or []
        free_text = free_text or ""

        parts = []

        if genres:
            clean = [normalize_text(g) for g in genres]
            parts.append("Genres: " + ", ".join(clean))

        if moods:
            clean = [normalize_text(m) for m in moods]
            parts.append("Moods: " + ", ".join(clean))

        if tags:
            clean = [normalize_text(t) for t in tags]
            parts.append("Tags: " + ", ".join(clean))

        if free_text:
            clean = normalize_text(free_text)
            parts.append("Query: " + clean)

        # combine everything
        final_prompt = ". ".join(parts).strip()

        logger.info(f"Generated prompt → {final_prompt}")
        return final_prompt

    except Exception as e:
        logger.error(f"Error building prompt: {e}")
        return ""

# 3. Helper: build pure free-text prompt
def build_free_text_prompt(text: str) -> str:
    """
    Simple wrapper used for plain search input.
    """
    try:
        clean = normalize_text(text)
        return f"Query: {clean}"
    except Exception as e:
        logger.error(f"Error in free-text prompt: {e}")
        return "query"
        
# Test
if __name__ == "__main__":
    '''print(
        build_prompt(
            genres=["Historical Fiction", "War"], 
            moods=["dark", "emotional"], 
            tags=["family", "sisters"],
            free_text="books like The Nightingale"
        )
    )'''
