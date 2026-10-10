"""
RAG Document Ingestion and Chunking Module for EntityIQ.

Extracts text, preserves document structure/metadata, and builds searchable chunks.
"""
import os
import re
from typing import List, Dict, Any

APPROVED_DOC_PATHS = [
    "README.md",
    "PROJECT_STATUS.md",
    "reports/dataset_profile.md",
    "reports/ml_evaluation_report.md",
    "Documentation_template.md",
]


def load_documents(base_dir: str) -> List[Dict[str, Any]]:
    """Load approved markdown documents and extract sections."""
    documents = []
    
    for rel_path in APPROVED_DOC_PATHS:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            continue
            
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split document into chunks based on markdown headers (H1, H2, H3)
        sections = re.split(r'\n(?=#{1,3}\s+)', content)
        doc_title = os.path.basename(rel_path)

        for idx, section in enumerate(sections):
            clean_section = section.strip()
            if not clean_section:
                continue

            # Extract header if present
            header_match = re.match(r'^#{1,3}\s+(.+)$', clean_section, re.MULTILINE)
            section_title = header_match.group(1).strip() if header_match else f"Section {idx+1}"

            documents.append({
                "doc_id": f"{doc_title}::section_{idx+1}",
                "source": rel_path,
                "title": f"{doc_title} - {section_title}",
                "text": clean_section,
                "char_length": len(clean_section),
            })

    return documents


def chunk_text(text: str, max_words: int = 250, overlap: int = 30) -> List[str]:
    """Split long text into overlapping chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += (max_words - overlap)

    return chunks
