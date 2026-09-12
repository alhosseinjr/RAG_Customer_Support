"""
Module 4c: Answer generation via Groq LLM, grounded on retrieved chunks.

Also hosts the translation helpers used for non-English customers: the
knowledge base (Bitext) is English-only, so a non-English message is
translated to English before retrieval/generation, and the final answer
is translated back to the customer's detected language. This is a
deliberate v1 design choice, not a limitation we're hiding -- see README
"Design decisions" for the trade-offs (extra latency + one more point of
failure, in exchange for not needing a multilingual KB).
"""
from functools import lru_cache
from typing import TYPE_CHECKING

from src import config
from src.rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    from groq import Groq


@lru_cache(maxsize=1)
def _client() -> "Groq":
    from groq import Groq  # lazy: keeps `import src.pipeline` cheap for routing-only tests

    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )
    return Groq(api_key=config.GROQ_API_KEY)


def _chat(system: str, user: str, temperature: float = 0.3) -> str:
    resp = _client().chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def translate(text: str, target_language: str) -> str:
    """Translate `text` into `target_language` (e.g. 'en', 'es', 'ar')."""
    if target_language == "en":
        return text
    system = (
        "You are a precise translator. Translate the user's message into the "
        f"language with ISO code '{target_language}'. Return ONLY the translation, "
        "no notes, no quotes."
    )
    return _chat(system, text, temperature=0.0)


def generate_answer(
    user_message_en: str,
    chunks: list[RetrievedChunk],
    detected_sentiment: str,
) -> str:
    """Generate a grounded answer in English from retrieved KB chunks."""
    if not chunks:
        return (
            "I don't have information on that in our support knowledge base, so "
            "I don't want to guess. I'm escalating this to a human agent who can help."
        )

    context = "\n\n".join(
        f"Retrieved support response {i+1} (category: {c.category}):\n{c.response}"
        for i, c in enumerate(chunks)
    )
    system = config.RAG_SYSTEM_PROMPT.format(detected_sentiment=detected_sentiment)
    user_prompt = (
        f"Context (retrieved past support responses):\n\n{context}\n\n"
        f'Customer question: "{user_message_en}"'
    )
    return _chat(system, user_prompt)


if __name__ == "__main__":
    from src.rag.retriever import retrieve

    q = "How do I cancel my order?"
    answer = generate_answer(q, retrieve(q), detected_sentiment="neutral")
    print(answer)
