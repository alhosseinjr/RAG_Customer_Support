"""Inference helpers for the language detection module."""
from functools import lru_cache

import joblib

from src import config


@lru_cache(maxsize=1)
def _load_pipeline():
    if not config.LANG_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained language model at {config.LANG_MODEL_PATH}. "
            "Run `python -m src.language_detection.train` first."
        )
    return joblib.load(config.LANG_MODEL_PATH)


def detect_language(text: str) -> tuple[str, float]:
    """Return (ISO-639-1 language code, confidence)."""
    pipe = _load_pipeline()
    proba = pipe.predict_proba([text])[0]
    idx = proba.argmax()
    label = pipe.classes_[idx]
    return label, float(proba[idx])


if __name__ == "__main__":
    for sample in ["Where is my order?", "¿Dónde está mi pedido?", "أين طلبي؟"]:
        lang, conf = detect_language(sample)
        print(f"{sample!r} -> {lang} ({conf:.2f})")
