"""
Unit tests for the routing logic in src.pipeline.

These mock out the four model-backed functions (detect_language,
predict_sentiment, predict_intent, retrieve/generate_answer) so the
*routing decisions* can be verified without needing trained model
artifacts on disk -- useful for CI or a quick sanity check before you've
run the full training pipeline locally.

Run:
    pytest tests/
"""
from unittest.mock import patch

from src import config
from src.pipeline import run_pipeline
from src.rag.retriever import RetrievedChunk


def _mock_chunk():
    return RetrievedChunk(
        instruction="Where is my order?",
        response="You can track your order in the 'My Orders' section.",
        category="ORDER",
        score=0.92,
    )


@patch("src.pipeline.translate", side_effect=lambda text, lang: text)
@patch("src.pipeline.generate_answer", return_value=("You can track your order in the app.", False))
@patch("src.pipeline.retrieve", return_value=[_mock_chunk()])
@patch("src.pipeline.predict_intent", return_value="order_status")
@patch("src.pipeline.predict_sentiment", return_value=("neutral", 0.9))
@patch("src.pipeline.detect_language", return_value=("en", 0.99))
def test_neutral_order_status_uses_rag_no_escalation(*_mocks):
    result = run_pipeline("Where is my order?")
    assert result.used_rag is True
    assert result.escalate is False
    assert result.intent == "order_status"


@patch("src.pipeline.translate", side_effect=lambda text, lang: text)
@patch("src.pipeline.generate_answer", return_value=("You can track your order in the app.", False))
@patch("src.pipeline.retrieve", return_value=[_mock_chunk()])
@patch("src.pipeline.predict_intent", return_value="complaint")
@patch("src.pipeline.predict_sentiment", return_value=("negative", 0.95))
@patch("src.pipeline.detect_language", return_value=("en", 0.99))
def test_complaint_escalates_and_prefixes_apology(*_mocks):
    result = run_pipeline("This is unacceptable, I want a refund now.")
    assert result.escalate is True
    assert result.response.startswith("I'm sorry")


@patch("src.pipeline.translate", side_effect=lambda text, lang: text)
@patch("src.pipeline.predict_intent", return_value="greeting_goodbye_gratitude")
@patch("src.pipeline.predict_sentiment", return_value=("positive", 0.8))
@patch("src.pipeline.detect_language", return_value=("en", 0.99))
def test_greeting_skips_rag_entirely(*_mocks):
    with patch("src.pipeline.retrieve") as mock_retrieve:
        result = run_pipeline("Hi there!")
        mock_retrieve.assert_not_called()
    assert result.used_rag is False
    assert result.escalate is False


@patch("src.pipeline.translate", side_effect=lambda text, lang: text)
@patch("src.pipeline.predict_intent", return_value="out_of_scope")
@patch("src.pipeline.predict_sentiment", return_value=("neutral", 0.7))
@patch("src.pipeline.detect_language", return_value=("en", 0.99))
def test_out_of_scope_skips_rag(*_mocks):
    with patch("src.pipeline.retrieve") as mock_retrieve:
        result = run_pipeline("What's the weather today?")
        mock_retrieve.assert_not_called()
    assert result.used_rag is False


@patch("src.pipeline.translate", side_effect=lambda text, lang: f"[{lang}] {text}")
@patch("src.pipeline.generate_answer", return_value=("You can track your order in the app.", False))
@patch("src.pipeline.retrieve", return_value=[_mock_chunk()])
@patch("src.pipeline.predict_intent", return_value="order_status")
@patch("src.pipeline.predict_sentiment", return_value=("neutral", 0.9))
@patch("src.pipeline.detect_language", return_value=("es", 0.95))
def test_non_english_message_gets_translated_response(*_mocks):
    result = run_pipeline("¿Dónde está mi pedido?")
    assert result.detected_language == "es"
    assert result.response.startswith("[es]")


@patch("src.pipeline.translate", side_effect=lambda text, lang: text)
@patch("src.pipeline.generate_answer", return_value=("Here's how to place a new order...", False))
@patch("src.pipeline.retrieve", return_value=[RetrievedChunk(
    instruction="How do I place an order?", response="...", category="ORDER", score=0.30,
)])
@patch("src.pipeline.predict_intent", return_value="order_status")
@patch("src.pipeline.predict_sentiment", return_value=("positive", 0.8))
@patch("src.pipeline.detect_language", return_value=("en", 0.99))
def test_weak_similarity_score_escalates_without_apology_prefix(*_mocks):
    """A low cosine-similarity match should escalate, but shouldn't get
    the complaint-style apology prefix since the customer isn't upset --
    only the bot's answer confidence is low."""
    result = run_pipeline("Where is my order?")
    assert result.low_confidence_retrieval is True
    assert result.escalate is True
    assert not result.response.startswith("I'm sorry for the trouble")


@patch("src.pipeline.translate", side_effect=lambda text, lang: text)
@patch(
    "src.pipeline.generate_answer",
    return_value=(
        "I don't have details on tracking an existing order, only on placing "
        "new ones. Let me connect you with a human agent.",
        True,  # model itself flagged that the context didn't answer the question
    ),
)
@patch("src.pipeline.retrieve", return_value=[RetrievedChunk(
    # High similarity score (same general "order" topic) but NOT actually
    # responsive to the question -- this is the exact case a raw cosine
    # threshold can't catch, and why we also trust the model's own signal.
    instruction="How do I place an order?", response="...", category="ORDER", score=0.68,
)])
@patch("src.pipeline.predict_intent", return_value="order_status")
@patch("src.pipeline.predict_sentiment", return_value=("positive", 0.8))
@patch("src.pipeline.detect_language", return_value=("en", 0.99))
def test_model_self_flagged_escalation_overrides_high_similarity_score(*_mocks):
    """Even when the embedding similarity score is high, the model's own
    [ESCALATE_TO_HUMAN] signal should still trigger escalation -- this is
    the topically-close-but-not-responsive failure mode."""
    result = run_pipeline("Where is my order?")
    assert result.low_confidence_retrieval is True
    assert result.escalate is True


def test_coarse_intent_mapping_has_no_orphan_targets():
    """Every mapped coarse intent must be one of the 7 documented categories."""
    expected = {
        "greeting_goodbye_gratitude",
        "order_status",
        "order_management",
        "billing_and_refunds",
        "account_management",
        "complaint",
        "out_of_scope",
    }
    mapped_targets = set(config.FINE_TO_COARSE_INTENT.values())
    assert mapped_targets.issubset(expected)
    assert set(config.COARSE_INTENTS) == expected
