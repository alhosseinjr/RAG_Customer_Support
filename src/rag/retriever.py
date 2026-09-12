"""Module 4b: Retrieval over the FAISS knowledge-base index."""
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np

from src import config


@dataclass
class RetrievedChunk:
    instruction: str
    response: str
    category: str
    score: float


@lru_cache(maxsize=1)
def _load_index():
    import faiss  # lazy: keeps `import src.pipeline` cheap when only routing logic is needed

    if not config.FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No FAISS index at {config.FAISS_INDEX_PATH}. "
            "Run `python -m src.rag.build_index` first."
        )
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    metadata = joblib.load(config.FAISS_METADATA_PATH)
    return index, metadata


@lru_cache(maxsize=1)
def _load_embedder():
    from sentence_transformers import SentenceTransformer  # lazy, see _load_index note

    return SentenceTransformer(config.EMBEDDING_MODEL)


def retrieve(query: str, top_k: int = config.RAG_TOP_K) -> list[RetrievedChunk]:
    index, metadata = _load_index()
    embedder = _load_embedder()

    query_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_vec, top_k)

    chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = metadata.iloc[idx]
        chunks.append(
            RetrievedChunk(
                instruction=row["instruction"],
                response=row["response"],
                category=row["category"],
                score=float(score),
            )
        )
    return chunks


if __name__ == "__main__":
    for chunk in retrieve("How do I cancel my order?"):
        print(f"[{chunk.score:.3f}] ({chunk.category}) {chunk.instruction}")
        print(f"    -> {chunk.response[:100]}...")
