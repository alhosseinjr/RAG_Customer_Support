"""Inference helpers for the sentiment / emotion module."""
from functools import lru_cache

import joblib

from src import config


@lru_cache(maxsize=1)
def _load_model():
    from tensorflow import keras  # lazy: keeps `import src.pipeline` cheap for routing-only tests

    if not config.SENTIMENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained sentiment model at {config.SENTIMENT_MODEL_PATH}. "
            "Run `python -m src.sentiment.train` first."
        )
    model = keras.models.load_model(config.SENTIMENT_MODEL_PATH)
    tokenizer = joblib.load(config.SENTIMENT_TOKENIZER_PATH)
    return model, tokenizer


def predict_sentiment(text: str) -> tuple[str, float]:
    """Return (bucket label, confidence) where bucket in negative/neutral/positive."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer = _load_model()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=config.SENTIMENT_MAX_LEN, padding="post", truncating="post")
    proba = model.predict(padded, verbose=0)[0]
    idx = proba.argmax()
    return config.SENTIMENT_LABELS[idx], float(proba[idx])


if __name__ == "__main__":
    for sample in [
        "This is the third time my order hasn't arrived, I'm furious.",
        "Just checking on my delivery date.",
        "Thanks so much, that fixed it!",
    ]:
        label, conf = predict_sentiment(sample)
        print(f"{sample!r} -> {label} ({conf:.2f})")
