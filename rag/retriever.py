"""
Retriever: embeds a set of documents and finds the most relevant chunks for a query.

Now data-driven: documents are passed IN at construction time rather than imported
from a hardcoded module. This lets the same retriever serve any project's corpus --
the key change that turns a fixed demo into a multi-project product.

Each document is a dict: {"id": str, "title": str, "text": str}.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model once at module level (expensive; reused across projects).
_model = SentenceTransformer("all-MiniLM-L6-v2")


class Retriever:
    def __init__(self, documents: list[dict]):
        if not documents:
            raise ValueError("Retriever requires at least one document.")
        self.model = _model
        self.docs = documents
        texts = [doc["text"] for doc in self.docs]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, k: int = 2):
        """Return the top-k most relevant documents, with similarity scores."""
        k = min(k, len(self.docs))  # never request more docs than exist
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_emb
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            {
                "id": self.docs[i]["id"],
                "title": self.docs[i]["title"],
                "text": self.docs[i]["text"],
                "score": float(scores[i]),
            }
            for i in top_idx
        ]