"""
Central configuration for the RAG customer support chatbot.

All paths and tunables live here so training scripts, the FastAPI app,
and the notebooks stay in sync instead of hard-coding values in three
different places.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Language detection --------------------------------------------------
LANG_MODEL_PATH = MODELS_DIR / "language_detector.joblib"
LANG_VECTORIZER_PATH = MODELS_DIR / "language_vectorizer.joblib"
LANG_DATASET = "papluca/language-identification"

# Below this confidence, the language prediction is treated as unreliable
# and the pipeline defaults to English rather than trusting it. Short,
# generic phrases ("How can you help me?", "ok", "thanks") often don't
# have enough distinctive character n-grams for the classifier to be
# confident, and a wrong non-English guess triggers a costly, garbled
# failure mode: translating an English message to a wrong language and
# back through the LLM twice. English is the safe default for this KB
# and customer base; tune if your actual traffic skews non-English.
LANG_MIN_CONFIDENCE = 0.5

# --- Sentiment ------------------------------------------------------------
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_bilstm.keras"
SENTIMENT_TOKENIZER_PATH = MODELS_DIR / "sentiment_tokenizer.joblib"
SENTIMENT_DATASET = "dair-ai/emotion"
SENTIMENT_MAX_LEN = 40
SENTIMENT_VOCAB_SIZE = 20000
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# dair-ai/emotion fine labels -> 3-bucket routing signal.
# Rationale (documented in README): joy/love read as satisfied customers;
# anger/fear/sadness read as frustrated customers needing an empathetic,
# priority-flagged response; surprise is treated as neutral since it can
# go either way in a support context and there's no majority-negative
# evidence for it in this dataset.
EMOTION_TO_BUCKET = {
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "joy": "positive",
    "love": "positive",
    "surprise": "neutral",
}

# --- Intent -----------------------------------------------------------
INTENT_MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"
INTENT_VECTORIZER_PATH = MODELS_DIR / "intent_vectorizer.joblib"
INTENT_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Fine-grained (27) -> coarse routing categories, per task brief.
FINE_TO_COARSE_INTENT = {
    "greet": "greeting_goodbye_gratitude",
    "goodbye": "greeting_goodbye_gratitude",
    "thank_you": "greeting_goodbye_gratitude",
    "track_order": "order_status",
    "delivery_options": "order_status",
    "delivery_period": "order_status",
    "cancel_order": "order_management",
    "change_order": "order_management",
    "change_shipping_address": "order_management",
    "place_order": "order_management",
    "check_invoice": "billing_and_refunds",
    "get_refund": "billing_and_refunds",
    "track_refund": "billing_and_refunds",
    "payment_issue": "billing_and_refunds",
    "get_invoice": "billing_and_refunds",
    "create_account": "account_management",
    "edit_account": "account_management",
    "delete_account": "account_management",
    "switch_account": "account_management",
    "recover_password": "account_management",
    "registration_problems": "account_management",
    "feedback": "feedback",
    "compliment": "feedback",
    "complaint": "complaint",
    "review": "complaint",
    "contact_customer_service": "complaint",
    "contact_human_agent": "complaint",
    "newsletter_subscription": "out_of_scope",
    "set_up_shipping_address": "order_management",
}
COARSE_INTENTS = sorted(set[str](FINE_TO_COARSE_INTENT.values()) | {"out_of_scope", "feedback"})

# --- RAG ----------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index" / "kb.index"
FAISS_METADATA_PATH = MODELS_DIR / "faiss_index" / "kb_metadata.joblib"
RAG_TOP_K = 3

# Below this cosine similarity, the top retrieved chunk is treated as an
# unreliable match rather than a real answer -- e.g. a genuine RAG hit on
# "how to place an order" scoring against an "order status" question,
# which is topically close but not actually responsive. Escalates to a
# human even when intent/sentiment look fine. Calibrated loosely against
# all-MiniLM-L6-v2 cosine scores on this KB; tune after watching real
# queries -- if false-escalations are common, raise it; if genuinely bad
# answers are getting through, lower it.
RAG_MIN_CONFIDENCE = 0.45

# --- LLM (Groq) -----------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

RAG_SYSTEM_PROMPT = """You are a helpful, professional customer support assistant \
for an online retailer. Answer the customer's question using ONLY the information \
in the retrieved support responses below. If the customer sounds frustrated \
({detected_sentiment}), acknowledge that before answering. If the retrieved \
context does not actually answer the customer's specific question -- even if it \
looks topically related -- say so honestly and offer to escalate to a human agent \
rather than guessing or stretching an unrelated answer to fit. In that case, end \
your reply on its own new line with exactly the token [ESCALATE_TO_HUMAN] and \
nothing else on that line. Only include this token when the context genuinely \
doesn't answer the question -- do not include it when you were able to help."""
