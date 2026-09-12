"""
Module 3: Intent Classifier.

Traditional supervised ML (TF-IDF + LinearSVC) on the gold `intent` column
of the Bitext customer-support dataset, collapsed from 27 fine intents into
7 coarse routing categories (src.config.FINE_TO_COARSE_INTENT). Chosen over
zero/few-shot LLM prompting because the dataset already provides gold
labels for every message -- there's no reason to pay LLM latency/cost for
a call a lightweight classifier handles in microseconds, and a supervised
model is far easier to unit-test and defend in the assessment than a
prompted one.

`out_of_scope` has no natural examples in this dataset (every row is an
in-scope support message), so it's trained as a residual class using a
small set of clearly-unrelated synthetic examples appended below. This is
a deliberate, documented choice -- see README "Design decisions".

Run:
    python -m src.intent.train
"""
import json
import logging

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# Small synthetic out-of-scope set: generic small talk / unrelated topics
# that a retail support bot will realistically see but that aren't covered
# by any of the 27 Bitext intents.
OUT_OF_SCOPE_EXAMPLES = [
    "What's the weather like today?",
    "Can you recommend a good restaurant nearby?",
    "Do you know who won the football match last night?",
    "Tell me a joke.",
    "What's the capital of France?",
    "Can you help me with my homework?",
    "Are you a real person?",
    "What's your favorite movie?",
    "How do I bake a chocolate cake?",
    "What time is it in Tokyo?",
]


def load_data() -> pd.DataFrame:
    log.info("Loading %s ...", config.INTENT_DATASET)
    ds = load_dataset(config.INTENT_DATASET)
    df = ds["train"].to_pandas()[["instruction", "intent"]].rename(
        columns={"instruction": "text"}
    )
    df["coarse_intent"] = df["intent"].map(config.FINE_TO_COARSE_INTENT)

    unmapped = df[df["coarse_intent"].isna()]["intent"].unique()
    if len(unmapped):
        log.warning("Unmapped fine intents (dropped): %s", unmapped)
    df = df.dropna(subset=["coarse_intent"])

    oos_df = pd.DataFrame(
        {"text": OUT_OF_SCOPE_EXAMPLES, "intent": "out_of_scope", "coarse_intent": "out_of_scope"}
    )
    return pd.concat([df, oos_df], ignore_index=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), max_features=30_000, min_df=2, sublinear_tf=True
                ),
            ),
            ("clf", LinearSVC(C=1.0, class_weight="balanced")),
        ]
    )


def main():
    df = load_data()
    log.info("Coarse intent distribution:\n%s", df["coarse_intent"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["coarse_intent"], test_size=0.15, stratify=df["coarse_intent"], random_state=42
    )

    pipe = build_pipeline()
    log.info("Fitting on %d samples ...", len(X_train))
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    log.info("Test accuracy: %.4f", acc)

    joblib.dump(pipe, config.INTENT_MODEL_PATH)
    metrics_path = config.MODELS_DIR / "intent_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"test_accuracy": acc, "report": report}, f, indent=2)

    log.info("Saved model -> %s", config.INTENT_MODEL_PATH)
    log.info("Saved metrics -> %s", metrics_path)


if __name__ == "__main__":
    main()
