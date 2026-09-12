"""Inference helpers for the intent classifier module."""
from functools import lru_cache

import joblib

from src import config


@lru_cache(maxsize=1)
def _load_pipeline():
    if not config.INTENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained intent model at {config.INTENT_MODEL_PATH}. "
            "Run `python -m src.intent.train` first."
        )
    return joblib.load(config.INTENT_MODEL_PATH)


def predict_intent(text: str) -> str:
    """Return one of src.config.COARSE_INTENTS."""
    pipe = _load_pipeline()
    return pipe.predict([text])[0]


if __name__ == "__main__":
    for sample in [
        "Hi there!",
        "Where is my order #4021?",
        "I want a refund, this is unacceptable.",
        "How do I reset my password?",
        "What's the weather like today?",
    ]:
        print(f"{sample!r} -> {predict_intent(sample)}")
