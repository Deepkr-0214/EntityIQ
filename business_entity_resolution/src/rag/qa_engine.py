"""
RAG QA Engine for EntityIQ Knowledge Assistant.

Retrieves verified project context and synthesizes grounded answers with explicit source attribution.
"""
import os
from typing import Dict, Any, List, Optional
from src.rag.document_indexer import load_documents
from src.rag.vector_store import SimpleVectorStore
from src.config import BASE_DIR

INDEX_PATH = os.path.join(BASE_DIR, "models", "rag_index.pkl")


class RAGAssistant:
    def __init__(self, base_dir: str = BASE_DIR, auto_build: bool = True):
        self.base_dir = base_dir
        self.store = SimpleVectorStore()

        if os.path.exists(INDEX_PATH):
            try:
                self.store = SimpleVectorStore.load(INDEX_PATH)
            except Exception:
                if auto_build:
                    self.reindex()
        elif auto_build:
            self.reindex()

    def reindex(self):
        """Ingest all approved project documents and build index."""
        docs = load_documents(self.base_dir)
        self.store.build_index(docs)
        self.store.save(INDEX_PATH)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant context chunks with metadata."""
        matches = self.store.search(query, top_k=top_k)
        results = []
        for doc, score in matches:
            results.append({
                "source": doc["source"],
                "title": doc["title"],
                "text": doc["text"],
                "relevance_score": round(score, 4),
            })
        return results

    def answer_question(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a user question grounded exclusively on retrieved project context.
        """
        retrieved_chunks = self.retrieve(query, top_k=top_k)

        if not retrieved_chunks or all(c["relevance_score"] < 0.08 for c in retrieved_chunks):
            return {
                "query": query,
                "answer": "The retrieved project documentation does not contain sufficient information to answer this question. Please refer to official project reports or verify the query.",
                "sources": [],
                "confidence": "Low / Unanswered",
                "retrieved_context": []
            }

        # Build answer synthesis from grounded sources
        sources = list(set([f"{c['source']} ({c['title']})" for c in retrieved_chunks]))
        
        # Formulate structured answer summary
        top_context = retrieved_chunks[0]["text"]
        summary_lines = [line.strip() for line in top_context.split("\n") if line.strip() and not line.strip().startswith("#")]
        key_points = summary_lines[:4]

        answer_text = "\n".join(key_points)

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "confidence": "High" if retrieved_chunks[0]["relevance_score"] > 0.25 else "Medium",
            "retrieved_context": retrieved_chunks
        }
