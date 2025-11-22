import os
import logging
import json
import cv2
import numpy as np
from paddleocr import PaddleOCR
from rapidfuzz import fuzz
from typing import List, Tuple, Dict, Any

from src.utils.title_search import find_best_title_match, normalize
from src.api.hardcover_api import search_book_api
from src.utils.embeddings import embed_book
from src.utils.faiss_manager import FaissManager

# Config / paths
CATALOG_PATH = r"E:\book_recommender\data\catalog.json"
DEFAULT_K = 5
TITLE_MATCH_MIN_SCORE = 60  
OCR_MIN_CHAR_LEN = 3
MAX_OCR_CANDIDATES = 40

# Logging 
logger = logging.getLogger("ocr_reader")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


OCR_MODEL = PaddleOCR(lang="en")

#Catalog load/save
def load_catalog() -> dict:
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")
        return {}

def save_catalog(cat: dict):
    try:
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(cat, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save catalog: {e}")

# Preprocessing 
def preprocess_image(img_np: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    try:
        if img_np is None:
            raise ValueError("No image provided")

        img = img_np.copy()

        # If image is float (0..1), convert to 0..255
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype("uint8")

        # Ensure 3 channels
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        #convert to BGR for OpenCV
        try:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image by scale {scale:.3f}")

        # Simple contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return img

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return img_np

# OCR extraction 
def extract_raw_texts_from_image(img_bgr: np.ndarray) -> List[str]:
    """
    Run PaddleOCR and return a list of raw text strings (ordered).
    """
    try:
        print("OCR DEBUG START ")
        print(f"Input image shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
        
        # PaddleOCR accepts BGR numpy arrays
        result = OCR_MODEL.ocr(img_bgr)
        texts = []

        print(f"OCR result type: {type(result)}")
        if result is not None:
            print(f"OCR result length: {len(result)}")
        else:
            print("OCR returned None")
            return []

        for line_idx, ocr_result in enumerate(result):
            print(f"Line {line_idx}: {type(ocr_result)}")
            
            if ocr_result is None:
                continue
                
            if hasattr(ocr_result, 'rec_texts'):
                
                rec_texts = ocr_result.rec_texts
                rec_scores = getattr(ocr_result, 'rec_scores', [])
                
                print(f"  Found {len(rec_texts)} text items in OCRResult")
                for i, text in enumerate(rec_texts):
                    confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                    texts.append(text)
                    print(f"  Detected: '{text}' (confidence: {confidence:.2f})")
                    
            elif isinstance(ocr_result, dict) and 'rec_texts' in ocr_result:
                # Fallback for dictionary structure
                rec_texts = ocr_result['rec_texts']
                rec_scores = ocr_result.get('rec_scores', [])
                
                print(f"  Found {len(rec_texts)} text items in dict")
                for i, text in enumerate(rec_texts):
                    confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                    texts.append(text)
                    print(f"  Detected: '{text}' (confidence: {confidence:.2f})")
                    
            else:
                print(f"  Unexpected OCR result format: {ocr_result}")

        logger.info(f"OCR extracted {len(texts)} raw text items")
        print(f"=== OCR DEBUG END - Found {len(texts)} texts ===")
        return texts

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        print(f"OCR exception: {e}")
        return []


# Clean OCR text candidates
def clean_ocr_candidate(s: str) -> str:
    """
    Basic cleaning: trim, remove control chars, strip, filter short/numeric strings.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # remove weird unicode control characters
    s = "".join(ch for ch in s if ord(ch) >= 32)
    # normalize spaces
    s = " ".join(s.split())
    # remove purely numeric or too short
    if len(s) < OCR_MIN_CHAR_LEN:
        return ""

    return s


#Semantic search
def resolve_ocr_candidates(candidates: List[str]) -> List[Dict[str, Any]]:
    """
    Pure semantic search based on all detected OCR text.
    """
    matched_books = []
    catalog = load_catalog()
    faiss = FaissManager()
    seen_book_ids = set()

    print(f"=== PURE SEMANTIC SEARCH FOR {len(candidates)} CANDIDATES ===")
    print(f"All detected text: {candidates}")
    
    if not candidates:
        return []

    # Combine all meaningful text into a semantic query
    semantic_query = " ".join([
        clean_ocr_candidate(candidate) for candidate in candidates 
        if clean_ocr_candidate(candidate) and len(clean_ocr_candidate(candidate)) >= 2
    ])
    
    print(f"Semantic query: '{semantic_query}'")
    
    if not semantic_query or len(semantic_query.strip()) < 3:
        print(" Query too short for semantic search")
        return []

    try:
        from src.utils.prompt_builder import build_free_text_prompt
        from src.utils.embeddings import embed_query
        
        # Create embedding from combined text
        prompt = build_free_text_prompt(semantic_query)
        query_vector = embed_query(prompt)
        
        # Search for similar books - limit to 5 results
        semantic_results = faiss.search(query_vector, k=5)
        
        print(f"Found {len(semantic_results)} semantic matches")
        
        for result in semantic_results:
            bid = str(result["book_id"])
            if bid in catalog and bid not in seen_book_ids:
                book = catalog[bid]
                matched_books.append((bid, book, "semantic", result["distance"], semantic_query))
                seen_book_ids.add(bid)
                print(f"  📚 {book['title']} (distance: {result['distance']:.3f})")
                
    except Exception as e:
        print(f"Semantic search error: {e}")

    print(f"=== FINAL: {len(matched_books)} books found ===")
    return matched_books

# Main pipeline: from image numpy array to unified recommendations
def recommend_from_bookshelf(image_np: np.ndarray, k: int = DEFAULT_K) -> Dict[str, Any]:
    """
    Input: image as numpy array (H,W,3) in RGB (Gradio default)
    Returns:
      {
        "status": "ok" / "error" / "not_found",
        "detected_books": [],  # Always empty - we don't detect specific books
        "recommendations": [ {book_id, title, authors, description, image_url, distance}, ... ]
      }
    """
    try:
        print("=== STARTING BOOKSHELF RECOMMENDATION ===")
        if image_np is None:
            return {"status": "error", "message": "No image provided", "detected_books": [], "recommendations": []}

        # 1) preprocess
        img_pre = preprocess_image(image_np)
        print(f"Preprocessed image: {img_pre.shape}")

        # 2) OCR extract
        raw_texts = extract_raw_texts_from_image(img_pre)
        if not raw_texts:
            print("❌ No text detected in image")
            return {"status": "not_found", "message": "No text detected", "detected_books": [], "recommendations": []}

        # 3) clean & dedupe candidates
        candidates = []
        for t in raw_texts:
            c = clean_ocr_candidate(t)
            if not c:
                continue
            # simple dedupe by lowercase
            if c.lower() not in [x.lower() for x in candidates]:
                candidates.append(c)

        logger.info(f"{len(candidates)} OCR candidates after cleaning/dedupe")
        print(f"Cleaned candidates: {candidates}")

        if not candidates:
            return {"status": "not_found", "message": "No valid text candidates", "detected_books": [], "recommendations": []}

        # 4) Get semantic matches 
        matched = resolve_ocr_candidates(candidates)

        if not matched:
            return {"status": "not_found", "message": "No books found matching the text", "detected_books": [], "recommendations": []}

        # Convert matched books to the recommendation format
        catalog = load_catalog()
        recommendations = []
        for bid, book, source, distance, ocr_raw in matched:
            recommendations.append({
                "book_id": bid,
                "title": book.get("title"),
                "authors": book.get("authors", []),
                "description": book.get("description", ""),
                "image_url": book.get("image_url") or "https://via.placeholder.com/150x220?text=No+Cover",
                "distance": float(distance) if distance else 0.0
            })

        print(f" SUCCESS: Found {len(recommendations)} books based on text")
        return {
            "status": "ok",
            "detected_books": [],  # Empty - we're not detecting specific books
            "recommendations": recommendations
        }

    except Exception as e:
        logger.exception(f"recommend_from_bookshelf failed: {e}")
        print(f" ERROR: {e}")
        return {"status": "error", "message": str(e), "detected_books": [], "recommendations": []}
def save_uploaded_image(user_id: str, image_data, filename: str = None):
    """Save uploaded bookshelf image to user directory."""
    import cv2
    from datetime import datetime
    
    user_upload_dir = os.path.join("ocr_uploads", user_id)
    os.makedirs(user_upload_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bookshelf_{timestamp}.jpg"
    
    filepath = os.path.join(user_upload_dir, filename)
    cv2.imwrite(filepath, image_data)
    
    return filepath

if __name__ == "__main__":
    # example usage: python ocr_reader.py <path-to-image>
    import sys
    if len(sys.argv) < 2:
        logger.error("Provide an image path for quick test.")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        logger.error("Path does not exist.")
        sys.exit(1)

    img = cv2.imread(path)  # BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = recommend_from_bookshelf(img_rgb, k=5)
    print(json.dumps(out, indent=2, ensure_ascii=False))