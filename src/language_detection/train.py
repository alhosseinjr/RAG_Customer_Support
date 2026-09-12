"""
Module 1: Language Detection.

Multi-class classifier over papluca/language-identification (20 languages,
90k samples, pre-split). Uses character n-gram TF-IDF, which is the
standard, well-understood choice for language ID: languages differ more
in short character sequences ("the", "sch", "ción") than in whole-word
vocabulary, so char n-grams generalize far better than word n-grams here,
and they're robust to short, informal customer messages.

Run:
    python -m src.language_detection.train
"""
import json
import logging
import time

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def load_data():
    log.info("Loading %s ...", config.LANG_DATASET)
    ds = load_dataset(config.LANG_DATASET)
    return ds["train"], ds["validation"], ds["test"]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(1, 3),
                    max_features=50_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=5.0,
                    n_jobs=-1,
                    multi_class="multinomial",
                ),
            ),
        ]
    )


def main():
    train, val, test = load_data()
    X_train, y_train = train["text"], train["labels"]
    X_val, y_val = val["text"], val["labels"]
    X_test, y_test = test["text"], test["labels"]

    pipe = build_pipeline()

    t0 = time.time()
    log.info("Fitting on %d samples ...", len(X_train))
    pipe.fit(X_train, y_train)
    log.info("Trained in %.1fs", time.time() - t0)

    val_preds = pipe.predict(X_val)
    log.info("Validation accuracy: %.4f", accuracy_score(y_val, val_preds))

    test_preds = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    report = classification_report(y_test, test_preds, output_dict=True)
    log.info("Test accuracy: %.4f", test_acc)

    joblib.dump(pipe, config.LANG_MODEL_PATH)
    metrics_path = config.MODELS_DIR / "language_detector_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"test_accuracy": test_acc, "report": report}, f, indent=2)

    log.info("Saved model -> %s", config.LANG_MODEL_PATH)
    log.info("Saved metrics -> %s", metrics_path)


if __name__ == "__main__":
    main()
