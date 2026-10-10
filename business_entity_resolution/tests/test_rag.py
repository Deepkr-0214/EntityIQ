"""
Tests for RAG document ingestion, vector retrieval, and grounded QA.
"""
import pytest
import os
from src.rag.document_indexer import load_documents, chunk_text
from src.rag.vector_store import SimpleVectorStore
from src.rag.qa_engine import RAGAssistant
from src.config import BASE_DIR


def test_document_ingestion():
    docs = load_documents(BASE_DIR)
    assert len(docs) > 0
    assert any("README.md" in d["source"] for d in docs)
    assert all("text" in d and "title" in d for d in docs)


def test_chunk_text():
    sample_text = " ".join([f"word_{i}" for i in range(500)])
    chunks = chunk_text(sample_text, max_words=200, overlap=20)
    assert len(chunks) >= 3


def test_vector_store_search():
    docs = load_documents(BASE_DIR)
    store = SimpleVectorStore()
    store.build_index(docs)

    results = store.search("blocking candidate generation recall", top_k=2)
    assert len(results) > 0
    top_doc, score = results[0]
    assert score > 0.05
    assert "source" in top_doc


def test_rag_assistant_grounded_qa():
    assistant = RAGAssistant(base_dir=BASE_DIR, auto_build=True)
    
    # Domain specific query
    resp = assistant.answer_question("What is the macro F0.5 score and blocking recall?")
    assert len(resp["sources"]) > 0
    assert resp["confidence"] in ["High", "Medium"]
    assert len(resp["retrieved_context"]) > 0


def test_rag_assistant_unsupported_query():
    assistant = RAGAssistant(base_dir=BASE_DIR, auto_build=True)
    
    # Completely unrelated query
    resp = assistant.answer_question("What is the recipe for chocolate chip cookies on Mars?")
    assert "does not contain sufficient information" in resp["answer"]
    assert resp["confidence"] == "Low / Unanswered"
