import os
import json
import numpy as np
import faiss

BASE_VECTOR_DIR = r"E:\book_recommender\vectors"

EMBED_PATH = os.path.join(BASE_VECTOR_DIR, "embeddings.npy")
FAISS_PATH = os.path.join(BASE_VECTOR_DIR, "embeddings.faiss")
ID_MAP_PATH = os.path.join(BASE_VECTOR_DIR, "id_map.json")

# FAISS Manager Class
class FaissManager:

    def __init__(self):
        self.embeddings = None
        self.index = None
        self.id_map = None
        self.dimension = None

        self.load_all()

    # Load embeddings and index
    def load_all(self):
        # Load embeddings
        if os.path.exists(EMBED_PATH):
            self.embeddings = np.load(EMBED_PATH).astype("float32")
        else:
            self.embeddings = np.zeros((0, 384), dtype="float32")

        # Determine dimension
        self.dimension = 384 if self.embeddings.shape[0] == 0 else self.embeddings.shape[1]

        # Load ID map
        if os.path.exists(ID_MAP_PATH):
            with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
        else:
            self.id_map = {}

        # Load or create FAISS index
        if os.path.exists(FAISS_PATH):
            self.index = faiss.read_index(FAISS_PATH)
        else:
            # Cosine similarity = IndexFlatIP
            self.index = faiss.IndexFlatIP(self.dimension)

            # If embeddings exist, add them
            if self.embeddings.shape[0] > 0:
                self.index.add(self.embeddings)

        print("FAISS loaded. Current vectors:", self.index.ntotal)

    # Save updated FAISS/ID map/embeddings
    def save_all(self):
        faiss.write_index(self.index, FAISS_PATH)
        np.save(EMBED_PATH, self.embeddings)

        with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, indent=4)

    # Search FAISS (Top-k)
    def search(self, query_vector, k=5):
        query = np.array([query_vector], dtype="float32")
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            book_id = self.id_map.get(str(idx))
            if book_id:
                results.append({
                    "book_id": book_id,
                    "distance": float(score)  # cosine similarity
                })

        return results

    # Add a NEW embedding (for new book)
    def add_new_embedding(self, book_id, embedding_vector):
        embedding_vector = embedding_vector.astype("float32")

        # Add to FAISS
        self.index.add(np.array([embedding_vector]))

        # Add to embedding matrix
        if self.embeddings.shape[0] == 0:
            self.embeddings = np.array([embedding_vector])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding_vector])

        # Update ID map
        new_index = self.index.ntotal - 1
        self.id_map[str(new_index)] = book_id

        # Save everything
        self.save_all()

        print(f"Added new embedding → Book ID: {book_id}, FAISS Index: {new_index}")

