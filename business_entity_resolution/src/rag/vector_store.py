"""
Vector store and similarity search for EntityIQ RAG.
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=5000
        )
        self.documents: List[Dict[str, Any]] = []
        self.matrix = None

    def build_index(self, docs: List[Dict[str, Any]]):
        """Fit vectorizer and build document matrix."""
        self.documents = docs
        texts = [f"{d['title']}\n{d['text']}" for d in docs]
        if not texts:
            raise ValueError("No documents provided to build vector index.")
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Tuple[Dict[str, Any], float]]:
        """Search top-k most relevant document chunks."""
        if self.matrix is None or len(self.documents) == 0:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices[:top_k]:
            score = float(scores[idx])
            if score >= min_score:
                results.append((self.documents[idx], score))

        return results

    def save(self, filepath: str):
        """Persist index to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "documents": self.documents, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, filepath: str) -> "SimpleVectorStore":
        """Load index from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        store = cls()
        store.vectorizer = data["vectorizer"]
        store.documents = data["documents"]
        store.matrix = data["matrix"]
        return store
