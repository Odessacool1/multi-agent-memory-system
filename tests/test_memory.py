"""Unit tests for SemanticMemoryStore."""
import pytest
from src.memory import SemanticMemoryStore


def test_memory_add_and_search():
    store = SemanticMemoryStore()
    store.add_memory("doc1", "FastEmbed is a lightweight Python library for ONNX embeddings.")
    store.add_memory("doc2", "PostgreSQL is a powerful open-source relational database.")

    results = store.search("Tell me about vector embeddings with ONNX", top_k=1)
    assert len(results) >= 1
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] > 0.4


def test_memory_empty_search():
    store = SemanticMemoryStore()
    results = store.search("anything", top_k=5)
    assert results == []
