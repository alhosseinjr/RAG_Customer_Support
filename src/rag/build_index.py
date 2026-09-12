"""
Module 4a: Build the RAG knowledge base index.

Embeds the `instruction` column of the Bitext dataset (the customer
question side) with sentence-transformers/all-MiniLM-L6-v2 and indexes it
in a local FAISS index. The paired `response` column is kept as metadata
and injected into the LLM prompt as grounding context at query time --
we retrieve on questions (semantically closest to what a live customer
will type) and generate from answers (the actual grounded knowledge).

FAISS chosen over Qdrant Cloud: no external account/API-key dependency,
which matters for a project that has to run locally in front of an
assessor. IndexFlatIP + L2-normalized embeddings gives exact cosine
similarity search, which is plenty fast for a knowledge base this size
(~27k rows) without needing an approximate index.

Run:
    python -m src.rag.build_index
"""
import logging

import faiss
import joblib
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("Loading %s ...", config.INTENT_DATASET)
    ds = load_dataset(config.INTENT_DATASET)
    df = ds["train"].to_pandas()[["instruction", "response", "category", "intent"]]
    df = df.drop_duplicates(subset="instruction").reset_index(drop=True)
    log.info("Indexing %d unique instruction/response pairs", len(df))

    log.info("Loading embedding model %s ...", config.EMBEDDING_MODEL)
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    embeddings = embedder.encode(
        df["instruction"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    config.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    joblib.dump(df, config.FAISS_METADATA_PATH)

    log.info("Saved FAISS index -> %s", config.FAISS_INDEX_PATH)
    log.info("Saved metadata -> %s", config.FAISS_METADATA_PATH)


if __name__ == "__main__":
    main()
