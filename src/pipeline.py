"""
End-to-end orchestration: every customer message flows through
language detection -> sentiment -> intent -> routing -> (RAG or canned
response). This is the module the FastAPI app and the notebooks both
call, so the routing logic is defined exactly once.

Routing decisions (documented, per task brief "Guidelines"):

  greeting_goodbye_gratitude -> canned reply, no RAG, no LLM call at all.
  out_of_scope               -> canned decline, no RAG.
  complaint                  -> RAG-grounded answer IS still generated
                                 (the customer still gets help), but it is
                                 prefixed with an apology/acknowledgment
                                 and the response carries escalate=True so
                                 a human agent is looped in regardless of
                                 how good the generated answer is. This was
                                 chosen over hard-blocking auto-response
                                 because withholding an answer to an
                                 already-frustrated customer while they
                                 wait for a human tends to make things
                                 worse, not better.
  everything else             -> standard RAG-grounded answer.

Negative sentiment on a *non-complaint* intent (e.g. a frustrated
order-status question) also sets escalate=True and gets the same
apology prefix, even though the intent itself isn't "complaint" --
sentiment and intent are independent signals and either can trigger it.
"""
from dataclasses import dataclass, field

from src.intent.predict import predict_intent
from src.language_detection.predict import detect_language
from src.rag.generator import generate_answer, translate
from src.rag.retriever import retrieve
from src.sentiment.predict import predict_sentiment

CANNED_RESPONSES = {
    "greeting_goodbye_gratitude": "Hi! Thanks for reaching out — how can I help with your order today?",
    "out_of_scope": (
        "I'm your order support assistant, so I'm not able to help with that, "
        "but I'm happy to help with anything about your orders, account, or billing."
    ),
}

APOLOGY_PREFIX = (
    "I'm sorry for the trouble this has caused — I understand the frustration, "
    "and I've flagged this for a human agent to follow up as well. "
)


@dataclass
class PipelineResult:
    response: str
    detected_language: str
    language_confidence: float
    sentiment: str
    sentiment_confidence: float
    intent: str
    escalate: bool
    used_rag: bool
    retrieved_categories: list[str] = field(default_factory=list)


def run_pipeline(user_message: str) -> PipelineResult:
    # Stage 1: language
    lang, lang_conf = detect_language(user_message)

    # Stage 2: sentiment (always run on the original text)
    sentiment, sent_conf = predict_sentiment(user_message)

    # Translate to English for intent/RAG if needed (KB + intent model are English-only)
    message_en = user_message if lang == "en" else translate(user_message, "en")

    # Stage 3: intent
    intent = predict_intent(message_en)

    escalate = intent == "complaint" or sentiment == "negative"

    # Stage 4: routing
    if intent in CANNED_RESPONSES:
        response_en = CANNED_RESPONSES[intent]
        used_rag = False
        categories: list[str] = []
    else:
        chunks = retrieve(message_en)
        response_en = generate_answer(message_en, chunks, detected_sentiment=sentiment)
        used_rag = True
        categories = [c.category for c in chunks]
        if escalate:
            response_en = APOLOGY_PREFIX + response_en

    final_response = response_en if lang == "en" else translate(response_en, lang)

    return PipelineResult(
        response=final_response,
        detected_language=lang,
        language_confidence=lang_conf,
        sentiment=sentiment,
        sentiment_confidence=sent_conf,
        intent=intent,
        escalate=escalate,
        used_rag=used_rag,
        retrieved_categories=categories,
    )


if __name__ == "__main__":
    for msg in [
        "Hi there!",
        "This is the third time my package is late, where is it??",
        "How do I reset my password?",
        "What's the weather today?",
    ]:
        result = run_pipeline(msg)
        print(f"\n> {msg}")
        print(result)
