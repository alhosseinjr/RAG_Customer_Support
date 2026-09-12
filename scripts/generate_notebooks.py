"""
Generates notebooks/01..04 by wrapping the src/ modules with exploratory
cells (EDA, evaluation plots, spot-checks). Kept as a script rather than
hand-edited notebooks so the notebooks and src/ code can never drift
apart -- re-run this after changing src/ and the notebooks stay current.

    python scripts/generate_notebooks.py
"""
import nbformat as nbf

OUT_DIR = "notebooks"


def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = cells
    n["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return n


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------- 01
nb1 = nb([
    md("# Module 1 — Language Detection\n\n"
       "TF-IDF (character n-grams) + Logistic Regression over "
       "`papluca/language-identification` (90k samples, 20 languages, pre-split).\n\n"
       "**Why char n-grams, not word n-grams**: language ID is about *how words are "
       "spelled*, not vocabulary — short character sequences separate languages "
       "cleanly even on short, informal text like a customer message."),
    code("import sys\n"
         "sys.path.append('..')\n"
         "from datasets import load_dataset\n"
         "import pandas as pd\n"
         "from src.language_detection.train import build_pipeline\n"
         "from src import config"),
    md("## Load & inspect the data"),
    code("ds = load_dataset(config.LANG_DATASET)\n"
         "train, val, test = ds['train'], ds['validation'], ds['test']\n"
         "print(train)\n"
         "pd.Series(train['labels']).value_counts()"),
    md("## Train"),
    code("pipe = build_pipeline()\n"
         "pipe.fit(train['text'], train['labels'])"),
    md("## Evaluate"),
    code("from sklearn.metrics import classification_report, ConfusionMatrixDisplay\n"
         "import matplotlib.pyplot as plt\n"
         "\n"
         "preds = pipe.predict(test['text'])\n"
         "print(classification_report(test['labels'], preds))"),
    code("fig, ax = plt.subplots(figsize=(10, 10))\n"
         "ConfusionMatrixDisplay.from_predictions(test['labels'], preds, ax=ax, xticks_rotation=90)\n"
         "plt.title('Language ID — confusion matrix')\n"
         "plt.tight_layout()\n"
         "plt.show()"),
    md("## Save the model\n\nRun the equivalent standalone script instead if you just want the artifact:\n"
       "```bash\npython -m src.language_detection.train\n```"),
    code("import joblib\n"
         "joblib.dump(pipe, config.LANG_MODEL_PATH)\n"
         "print('Saved to', config.LANG_MODEL_PATH)"),
    md("## Spot-check"),
    code("from src.language_detection.predict import detect_language\n"
         "for s in ['Where is my order?', '¿Dónde está mi pedido?', 'Où est ma commande?']:\n"
         "    print(s, '->', detect_language(s))"),
])

# ---------------------------------------------------------------- 02
nb2 = nb([
    md("# Module 2 — Sentiment / Emotion Classifier\n\n"
       "BiLSTM over `dair-ai/emotion`, collapsed from 6 fine emotions into 3 routing "
       "buckets (negative / neutral / positive). See `src/config.EMOTION_TO_BUCKET` "
       "for the exact mapping and rationale.\n\n"
       "**Domain-shift caveat** (documented, not hidden): this dataset is Twitter "
       "text, not customer-support text. Treat absolute accuracy numbers here as a "
       "routing-quality proxy, not a guarantee of production behavior — the README "
       "covers the mitigation."),
    code("import sys\n"
         "sys.path.append('..')\n"
         "from datasets import load_dataset\n"
         "import pandas as pd\n"
         "from src import config\n"
         "from src.sentiment.train import EMOTION_NAMES, to_bucket, build_model\n"
         "from tensorflow.keras.preprocessing.text import Tokenizer\n"
         "from tensorflow.keras.preprocessing.sequence import pad_sequences\n"
         "import numpy as np"),
    md("## Load & inspect"),
    code("ds = load_dataset(config.SENTIMENT_DATASET)\n"
         "train, val, test = ds['train'], ds['validation'], ds['test']\n"
         "pd.Series([EMOTION_NAMES[l] for l in train['label']]).value_counts()"),
    md("## Bucket mapping check"),
    code("bucket_counts = pd.Series([config.SENTIMENT_LABELS[to_bucket(l)] for l in train['label']])\n"
         "bucket_counts.value_counts()"),
    md("## Tokenize"),
    code("tokenizer = Tokenizer(num_words=config.SENTIMENT_VOCAB_SIZE, oov_token='<OOV>')\n"
         "tokenizer.fit_on_texts(train['text'])\n"
         "\n"
         "def vectorize(texts):\n"
         "    seqs = tokenizer.texts_to_sequences(texts)\n"
         "    return pad_sequences(seqs, maxlen=config.SENTIMENT_MAX_LEN, padding='post', truncating='post')\n"
         "\n"
         "X_train, y_train = vectorize(train['text']), np.array([to_bucket(l) for l in train['label']])\n"
         "X_val, y_val = vectorize(val['text']), np.array([to_bucket(l) for l in val['label']])\n"
         "X_test, y_test = vectorize(test['text']), np.array([to_bucket(l) for l in test['label']])"),
    md("## Train"),
    code("from sklearn.utils.class_weight import compute_class_weight\n"
         "from tensorflow import keras\n"
         "\n"
         "class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)\n"
         "class_weight_dict = dict(enumerate(class_weights))\n"
         "\n"
         "model = build_model(vocab_size=config.SENTIMENT_VOCAB_SIZE)\n"
         "history = model.fit(\n"
         "    X_train, y_train, validation_data=(X_val, y_val),\n"
         "    epochs=15, batch_size=64, class_weight=class_weight_dict,\n"
         "    callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)],\n"
         "    verbose=2,\n"
         ")"),
    md("## Training curves"),
    code("import matplotlib.pyplot as plt\n"
         "\n"
         "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
         "axes[0].plot(history.history['accuracy'], label='train')\n"
         "axes[0].plot(history.history['val_accuracy'], label='val')\n"
         "axes[0].set_title('Accuracy'); axes[0].legend()\n"
         "axes[1].plot(history.history['loss'], label='train')\n"
         "axes[1].plot(history.history['val_loss'], label='val')\n"
         "axes[1].set_title('Loss'); axes[1].legend()\n"
         "plt.tight_layout(); plt.show()"),
    md("## Evaluate"),
    code("from sklearn.metrics import classification_report, ConfusionMatrixDisplay\n"
         "\n"
         "test_preds = model.predict(X_test).argmax(axis=1)\n"
         "print(classification_report(y_test, test_preds, target_names=config.SENTIMENT_LABELS))\n"
         "ConfusionMatrixDisplay.from_predictions(y_test, test_preds, display_labels=config.SENTIMENT_LABELS)\n"
         "plt.title('Sentiment — confusion matrix')\n"
         "plt.show()"),
    md("## Save"),
    code("import joblib\n"
         "model.save(config.SENTIMENT_MODEL_PATH)\n"
         "joblib.dump(tokenizer, config.SENTIMENT_TOKENIZER_PATH)\n"
         "print('Saved.')"),
    md("## Spot-check"),
    code("from src.sentiment.predict import predict_sentiment\n"
         "for s in [\"This is the third time my order hasn't arrived, I'm furious.\",\n"
         "          'Just checking on my delivery date.',\n"
         "          'Thanks so much, that fixed it!']:\n"
         "    print(s, '->', predict_sentiment(s))"),
])

# ---------------------------------------------------------------- 03
nb3 = nb([
    md("# Module 3 — Intent Classifier\n\n"
       "TF-IDF + LinearSVC on the gold `intent` column of the Bitext dataset, "
       "collapsed from 27 fine intents into 7 coarse routing categories. Supervised "
       "on labeled data (not zero/few-shot) since the labels already exist — see "
       "`src/intent/train.py` docstring for the full rationale."),
    code("import sys\n"
         "sys.path.append('..')\n"
         "from src.intent.train import load_data, build_pipeline\n"
         "from src import config\n"
         "import pandas as pd"),
    md("## Load & inspect"),
    code("df = load_data()\n"
         "df['coarse_intent'].value_counts()"),
    code("df[['text', 'intent', 'coarse_intent']].sample(10, random_state=0)"),
    md("## Train / test split & fit"),
    code("from sklearn.model_selection import train_test_split\n"
         "\n"
         "X_train, X_test, y_train, y_test = train_test_split(\n"
         "    df['text'], df['coarse_intent'], test_size=0.15, stratify=df['coarse_intent'], random_state=42\n"
         ")\n"
         "pipe = build_pipeline()\n"
         "pipe.fit(X_train, y_train)"),
    md("## Evaluate"),
    code("from sklearn.metrics import classification_report, ConfusionMatrixDisplay\n"
         "import matplotlib.pyplot as plt\n"
         "\n"
         "preds = pipe.predict(X_test)\n"
         "print(classification_report(y_test, preds))"),
    code("fig, ax = plt.subplots(figsize=(8, 8))\n"
         "ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, xticks_rotation=45)\n"
         "plt.title('Intent — confusion matrix')\n"
         "plt.tight_layout()\n"
         "plt.show()"),
    md("## Save"),
    code("import joblib\n"
         "joblib.dump(pipe, config.INTENT_MODEL_PATH)\n"
         "print('Saved to', config.INTENT_MODEL_PATH)"),
    md("## Spot-check"),
    code("from src.intent.predict import predict_intent\n"
         "for s in ['Hi there!', 'Where is my order #4021?', 'I want a refund, this is unacceptable.',\n"
         "          'How do I reset my password?', \"What's the weather like today?\"]:\n"
         "    print(s, '->', predict_intent(s))"),
])

# ---------------------------------------------------------------- 04
nb4 = nb([
    md("# Module 4 — Q&A RAG Pipeline\n\n"
       "Embed the Bitext `instruction` column with `all-MiniLM-L6-v2`, index with "
       "FAISS (`IndexFlatIP` on normalized vectors = exact cosine similarity), "
       "retrieve top-k, and generate a grounded answer with Groq (`gpt-oss-120b`) "
       "using the prompt template from the task brief.\n\n"
       "**Requires** a `GROQ_API_KEY` in `.env` (see `.env.example`) to run the "
       "generation cells — retrieval works without it."),
    code("import sys\n"
         "sys.path.append('..')\n"
         "from src import config\n"
         "from src.rag.build_index import main as build_index\n"
         "from src.rag.retriever import retrieve\n"
         "from src.rag.generator import generate_answer"),
    md("## Build the index\n\nOne-time step; skip if `models/faiss_index/` already exists."),
    code("build_index()"),
    md("## Retrieval spot-check"),
    code("for chunk in retrieve('How do I cancel my order?'):\n"
         "    print(f'[{chunk.score:.3f}] ({chunk.category}) {chunk.instruction}')\n"
         "    print('   ->', chunk.response[:120], '...')\n"
         "    print()"),
    md("## End-to-end generation\n\nRequires `GROQ_API_KEY`."),
    code("query = 'How do I cancel my order?'\n"
         "chunks = retrieve(query)\n"
         "answer = generate_answer(query, chunks, detected_sentiment='neutral')\n"
         "print(answer)"),
    md("## Full pipeline (all 4 modules + routing)\n\n"
       "Requires all four modules to already be trained (`python scripts/run_full_pipeline.py`)."),
    code("from src.pipeline import run_pipeline\n"
         "\n"
         "for msg in ['Hi there!',\n"
         "            'This is the third time my package is late, where is it??',\n"
         "            'How do I reset my password?',\n"
         "            \"What's the weather today?\"]:\n"
         "    result = run_pipeline(msg)\n"
         "    print('>', msg)\n"
         "    print(result)\n"
         "    print()"),
])

FILES = {
    "01_language_detection.ipynb": nb1,
    "02_sentiment_classifier.ipynb": nb2,
    "03_intent_classifier.ipynb": nb3,
    "04_rag_pipeline.ipynb": nb4,
}

if __name__ == "__main__":
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    for filename, notebook in FILES.items():
        path = os.path.join(OUT_DIR, filename)
        nbf.write(notebook, path)
        print("Wrote", path)
