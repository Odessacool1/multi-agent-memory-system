"""
Semantic and Episodic Vector Memory Engine using FastEmbed and NumPy Cosine Indexing.
"""

from typing import List, Dict, Any, Optional
import time
import hashlib
import numpy as np
from pydantic import BaseModel, Field

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False
    TextEmbedding = None


class MemoryEntry(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    embedding: Optional[List[float]] = None


class SemanticMemoryStore:
    """High-throughput local semantic vector memory store backed by ONNX FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", mock_mode: bool = False):
        self.model_name = model_name
        self.mock_mode = mock_mode or (not HAS_FASTEMBED)
        self.embedding_model = None
        
        if not self.mock_mode and HAS_FASTEMBED:
            try:
                self.embedding_model = TextEmbedding(model_name=model_name)
            except Exception:
                self.mock_mode = True
                
        self.entries: List[MemoryEntry] = []
        self.vectors: Optional[np.ndarray] = None

    def _get_embedding(self, text: str) -> List[float]:
        """Generates embedding using FastEmbed ONNX or deterministic fallback vectorizer."""
        if self.embedding_model is not None and not self.mock_mode:
            try:
                emb_generator = self.embedding_model.embed([text])
                return list(next(emb_generator))
            except Exception:
                pass
                
        # Deterministic 384-dim normalized pseudo-embedding fallback for testing/offline mode
        dim = 384
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            val = ((h >> 8) % 100) / 100.0 - 0.5
            vec[idx] += float(val)
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def add_memory(self, memory_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """Embeds text and appends it to the in-memory vector index."""
        vector = self._get_embedding(text)
        
        entry = MemoryEntry(
            id=memory_id,
            content=text,
            metadata=metadata or {},
            timestamp=time.time(),
            embedding=vector
        )
        self.entries.append(entry)
        
        vec_np = np.array(vector, dtype=np.float32).reshape(1, -1)
        if self.vectors is None:
            self.vectors = vec_np
        else:
            self.vectors = np.vstack([self.vectors, vec_np])
            
        return entry

    def search(self, query: str, top_k: int = 3, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Performs normalized cosine similarity vector search against stored records."""
        if not self.entries or self.vectors is None:
            return []

        query_vector = np.array(self._get_embedding(query), dtype=np.float32)
        
        norm_vectors = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norm_query = np.linalg.norm(query_vector)
        
        normalized_db = self.vectors / np.maximum(norm_vectors, 1e-10)
        normalized_q = query_vector / max(norm_query, 1e-10)
        
        scores = np.dot(normalized_db, normalized_q)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        
        for idx in ranked_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append({
                    "id": self.entries[idx].id,
                    "content": self.entries[idx].content,
                    "metadata": self.entries[idx].metadata,
                    "score": round(score, 4)
                })
                
        return results

    def clear(self) -> None:
        """Clears all stored entries."""
        self.entries.clear()
        self.vectors = None
