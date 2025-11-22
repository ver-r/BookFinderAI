import os
from dotenv import load_dotenv

load_dotenv()

# Base directory
BASE_DIR = r"E:\book_recommender"

# API Configuration
HARDCOVER_API_URL = os.getenv("HARDCOVER_API_URL")
HARDCOVER_API_TOKEN = os.getenv("HARDCOVER_API_TOKEN")

# File Paths
CATALOG_PATH = os.path.join(BASE_DIR, "data", "catalog.json")
INITIAL_DATASET_PATH = os.path.join(BASE_DIR, "data", "initial_dataset.json")

# Vector Database Paths
VECTORS_DIR = os.path.join(BASE_DIR, "vectors")
EMBEDDINGS_PATH = os.path.join(VECTORS_DIR, "embeddings.npy")
FAISS_INDEX_PATH = os.path.join(VECTORS_DIR, "embeddings.faiss")
ID_MAP_PATH = os.path.join(VECTORS_DIR, "id_map.json")

# User Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_PROFILES_DIR = os.path.join(DATA_DIR, "user_profiles")
OCR_UPLOADS_DIR = os.path.join(DATA_DIR, "ocr_uploads")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Search Configuration
DEFAULT_RECOMMENDATION_COUNT = 5
MIN_TITLE_MATCH_SCORE = 95
MIN_AUTHOR_MATCH_SCORE = 70

# OCR Configuration
OCR_MIN_CHAR_LENGTH = 3
OCR_MAX_CANDIDATES = 40
OCR_MATCH_THRESHOLD = 60

# UI Configuration
DEFAULT_COVER_IMAGE = "https://via.placeholder.com/150x220?text=No+Cover"
MAX_DISPLAY_BOOKS = 20

# Create directories if they don't exist
for directory in [USER_PROFILES_DIR, OCR_UPLOADS_DIR, LOGS_DIR, VECTORS_DIR]:
    os.makedirs(directory, exist_ok=True)