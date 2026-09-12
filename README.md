# E-commerce Customer Support Chatbot (RAG)

**NLP Final Project — ITI Training, 2026**

A production-structured, RAG-based support chatbot for e-commerce. Every
customer message is routed through four NLP stages before a response is
produced: **language detection → sentiment/emotion → intent classification →
retrieval-augmented generation.**

```
Customer message
      │
      ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ Language         │────▶│ Sentiment /       │────▶│ Intent             │
│ Detection        │     │ Emotion           │     │ Classification     │
│ (TF-IDF + LogReg)│     │ (BiLSTM)          │     │ (TF-IDF + LinearSVC)│
└─────────────────┘     └──────────────────┘     └─────────┬──────────┘
                                                             │
                          ┌──────────────────────────────────┼───────────────┐
                          ▼                                  ▼                ▼
                 greeting/out_of_scope              order/billing/account   complaint
                          │                                  │                │
                    canned reply                    FAISS retrieval        FAISS retrieval
                    (no LLM call)                    + Groq generation     + Groq generation
                                                             │           + apology prefix
                                                             ▼           + escalate=True
                                                    grounded response
```

## Why this design (read this before the assessment)

| Decision | Choice | Why |
|---|---|---|
| Language ID | TF-IDF (char n-grams) + Logistic Regression | Language identity lives in spelling patterns, not vocabulary — char n-grams generalize far better than word n-grams on short, informal messages. Traditional ML also satisfies the brief's explicit requirement for this module. |
| Sentiment | BiLSTM (Keras), not a Transformer | The model only needs a 3-way *routing* signal, not deep emotional nuance. A BiLSTM trains in minutes on a laptop, is fully explainable end-to-end, and the brief explicitly allows RNN or Transformer — a fine-tuned transformer would cost much more to train/serve for a marginal accuracy gain on this decision. |
| Intent | TF-IDF + LinearSVC on gold labels, not LLM few-shot | The dataset already has gold `intent` labels for every row — using them is strictly better than paying LLM latency/cost for a call a lightweight classifier answers in microseconds, and it's trivial to unit test. |
| Vector store | Local FAISS, not Qdrant Cloud | Zero external account/API-key dependency. The project must run and be demoed locally in front of an assessor — a network dependency on a third-party cloud service is a real availability risk on assessment day. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small, fast, strong general-purpose retrieval embedding — standard choice for this scale of KB (~27k short Q&A pairs). |
| LLM | Groq `openai/gpt-oss-120b` | Free tier, fast inference, already proven in other project work. |
| Complaint routing | Still generate a grounded answer, prefix with apology, set `escalate: true` | Hard-blocking auto-response on every complaint means an already-frustrated customer gets *nothing* while waiting for a human. Giving them a grounded, empathetic answer **and** flagging for human follow-up serves the customer better than silence, without pretending the bot fully resolved it. |
| Negative sentiment (non-complaint) | Same apology-prefix + escalate treatment | Sentiment and intent are independent signals. A frustrated customer asking a plain order-status question still deserves the softened tone and a human safety net — you don't need the "complaint" label to justify it. |
| Non-English messages | Translate to English for retrieval/generation, translate the response back | The knowledge base (Bitext) is English-only. Translating both directions via the LLM avoids needing a multilingual KB, at the cost of one extra LLM round-trip and one more point of failure — documented trade-off, not hidden. |
| `out_of_scope` training data | Small hand-written synthetic set (10 examples) | The Bitext dataset has zero true negative/unrelated examples — every row is an in-scope support message. Without *some* negative signal the classifier can never predict `out_of_scope` at all, so a small synthetic set was added and is called out explicitly (see `src/intent/train.py`). |
| Emotion → 3-bucket mapping | joy/love → positive, anger/fear/sadness → negative, surprise → neutral | Documented in `src/config.EMOTION_TO_BUCKET`. Surprise has no clear valence on its own in a support context, so it's treated as neutral rather than guessed either way. |

## Known limitations (documented, not hidden)

- **Domain shift on sentiment**: `dair-ai/emotion` is Twitter text, not customer-support text. Accuracy numbers from that dataset are a proxy for routing quality, not a guarantee of production behavior on real support messages. Mitigation: spot-check with real support-style phrases (see the "Spot-check" cell in `notebooks/02_sentiment_classifier.ipynb`) rather than trusting the held-out test accuracy alone.
- **`out_of_scope` intent** is trained on a small synthetic set, not real data — it will catch obviously unrelated chit-chat but won't be as robust as the other six categories, which have thousands of real examples each.
- **Non-English support quality** depends entirely on the LLM's translation quality; there's no dedicated evaluation set for this path yet.
- **No conversation memory** — each message is handled independently. A production version would carry session context (e.g. so a customer doesn't have to repeat their order number).

## Project structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── notebooks/                  # exploratory training + evaluation, one per module
│   ├── 01_language_detection.ipynb
│   ├── 02_sentiment_classifier.ipynb
│   ├── 03_intent_classifier.ipynb
│   └── 04_rag_pipeline.ipynb
├── src/
│   ├── config.py                # single source of truth for paths, labels, mappings
│   ├── language_detection/      # train.py / predict.py
│   ├── sentiment/                # train.py / predict.py
│   ├── intent/                   # train.py / predict.py
│   ├── rag/                       # build_index.py / retriever.py / generator.py
│   └── pipeline.py               # orchestrates all four stages + routing
├── app/
│   ├── main.py                    # FastAPI app (/chat, /health)
│   └── schemas.py
├── scripts/
│   ├── run_full_pipeline.py      # trains everything + builds the index, in order
│   └── generate_notebooks.py     # regenerates notebooks/ from src/ (keeps them in sync)
├── tests/
│   └── test_pipeline.py          # routing-logic unit tests, mocked — no trained models needed
└── models/                        # trained artifacts land here (gitignored)
```

`src/` and `notebooks/` are not two separate implementations — the notebooks
call directly into `src/` functions, so there's exactly one place the actual
logic lives. This means the notebooks stay demo/exploration surfaces while
`src/` stays the thing the API and tests import.

## Setup

```bash
git clone <this-repo>
cd customer-support-rag-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your GROQ_API_KEY
```

Get a free Groq API key at https://console.groq.com/keys.

## Train everything + build the index

```bash
python scripts/run_full_pipeline.py
```

This runs, in order: language detection training → sentiment training →
intent training → FAISS index build. It downloads three Hugging Face
datasets on first run (`papluca/language-identification`,
`dair-ai/emotion`, `bitext/Bitext-customer-support-llm-chatbot-training-dataset`),
so it needs internet access and will take a few minutes (the sentiment
BiLSTM is the slowest step).

Each stage can also be run individually:

```bash
python -m src.language_detection.train
python -m src.sentiment.train
python -m src.intent.train
python -m src.rag.build_index
```

Metrics for each module are written to `models/*_metrics.json`.

## Run the notebooks

```bash
jupyter notebook notebooks/
```

Or regenerate them from `src/` after making code changes:

```bash
python scripts/generate_notebooks.py
```

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order?"}'
```

```json
{
  "response": "You can track your order status from the 'My Orders' section...",
  "detected_language": "en",
  "language_confidence": 0.99,
  "sentiment": "neutral",
  "sentiment_confidence": 0.87,
  "intent": "order_status",
  "escalate": false,
  "used_rag": true,
  "retrieved_categories": ["ORDER", "ORDER", "DELIVERY"]
}
```

`GET /health` reports whether all four models loaded successfully at startup.

## Tests

```bash
pytest tests/ -v
```

These mock out the model-backed functions and test the **routing logic
only** (which path a given intent/sentiment combination takes, whether RAG
is skipped, whether escalation and translation trigger correctly). They
run without any trained model artifacts, so they're a fast sanity check
before or after training.

## Datasets

- Language: [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification)
- Sentiment: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- Intent + RAG knowledge base: [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

## Author

Al-Hossein Mahmoud — 4th-year AI & ML Engineering, Sadat Academy for
Management Sciences. Built as the final project for the ITI NLP training
track.
