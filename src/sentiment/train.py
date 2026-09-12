"""
Module 2: Sentiment / Emotion Classifier.

BiLSTM over dair-ai/emotion, collapsed from 6 fine emotions into 3 routing
buckets (negative/neutral/positive) per src.config.EMOTION_TO_BUCKET.

Why BiLSTM over a Transformer here: the dataset is short, informal Twitter
text (~20k rows) and the model only needs to answer a 3-way routing
question, not do nuanced emotion understanding. A BiLSTM trains in minutes
on CPU/MPS, is trivial to explain end-to-end in an assessment, and a
fine-tuned transformer would be overkill for the actual decision this
model has to make. Documented trade-off: a transformer would likely gain
a few points of accuracy at a large increase in training/inference cost.

Known limitation (documented, not hidden): dair-ai/emotion is Twitter
text, not customer-support text, so there's a domain shift. The task
brief flags this too; see README for the mitigation (small hand-labeled
qualitative check set).

Run:
    python -m src.sentiment.train
"""
import json
import logging

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

EMOTION_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL_TO_IDX = {name: i for i, name in enumerate(config.SENTIMENT_LABELS)}


def to_bucket(fine_label_idx: int) -> int:
    fine_name = EMOTION_NAMES[fine_label_idx]
    bucket_name = config.EMOTION_TO_BUCKET[fine_name]
    return LABEL_TO_IDX[bucket_name]


def load_data():
    log.info("Loading %s ...", config.SENTIMENT_DATASET)
    ds = load_dataset(config.SENTIMENT_DATASET)
    return ds["train"], ds["validation"], ds["test"]


def build_model(vocab_size: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(config.SENTIMENT_MAX_LEN,)),
            layers.Embedding(vocab_size, 128, mask_zero=True),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(len(config.SENTIMENT_LABELS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    train, val, test = load_data()

    y_train = np.array([to_bucket(l) for l in train["label"]])
    y_val = np.array([to_bucket(l) for l in val["label"]])
    y_test = np.array([to_bucket(l) for l in test["label"]])

    tokenizer = Tokenizer(num_words=config.SENTIMENT_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train["text"])

    def vectorize(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=config.SENTIMENT_MAX_LEN, padding="post", truncating="post")

    X_train = vectorize(train["text"])
    X_val = vectorize(val["text"])
    X_test = vectorize(test["text"])

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    log.info("Class weights (handles bucket imbalance): %s", class_weight_dict)

    model = build_model(vocab_size=config.SENTIMENT_VOCAB_SIZE)
    model.summary(print_fn=log.info)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    test_preds = model.predict(X_test).argmax(axis=1)
    test_acc = accuracy_score(y_test, test_preds)
    report = classification_report(
        y_test, test_preds, target_names=config.SENTIMENT_LABELS, output_dict=True
    )
    log.info("Test accuracy: %.4f", test_acc)

    model.save(config.SENTIMENT_MODEL_PATH)
    joblib.dump(tokenizer, config.SENTIMENT_TOKENIZER_PATH)

    metrics_path = config.MODELS_DIR / "sentiment_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"test_accuracy": test_acc, "report": report}, f, indent=2)

    log.info("Saved model -> %s", config.SENTIMENT_MODEL_PATH)
    log.info("Saved tokenizer -> %s", config.SENTIMENT_TOKENIZER_PATH)
    log.info("Saved metrics -> %s", metrics_path)


if __name__ == "__main__":
    main()
