"""
Trains all three classifiers and builds the RAG index, in order, with a
single command. Intended for the initial local setup or a clean re-run.

    python scripts/run_full_pipeline.py

Each stage is idempotent and independent -- if one fails (e.g. no internet
to Hugging Face), fix that and re-run the whole script; earlier artifacts
are simply overwritten, nothing is corrupted by a partial run.
"""
import logging
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

STAGES = [
    ("Language detection", "src.language_detection.train"),
    ("Sentiment classifier", "src.sentiment.train"),
    ("Intent classifier", "src.intent.train"),
    ("RAG index build", "src.rag.build_index"),
]


def run_stage(name: str, module: str):
    import importlib

    log.info("=" * 60)
    log.info("STAGE: %s (%s)", name, module)
    log.info("=" * 60)
    t0 = time.time()
    mod = importlib.import_module(module)
    mod.main()
    log.info("%s finished in %.1fs", name, time.time() - t0)


def main():
    for name, module in STAGES:
        try:
            run_stage(name, module)
        except Exception:
            log.exception("Stage '%s' failed -- fix the error above and re-run.", name)
            sys.exit(1)
    log.info("All stages complete. Start the API with:")
    log.info("  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
