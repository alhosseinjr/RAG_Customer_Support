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
@patch("src.pipeline.generate_answer", return_value="You can track your order in the app.")
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
@patch("src.pipeline.generate_answer", return_value="You can track your order in the app.")
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
@patch("src.pipeline.generate_answer", return_value="You can track your order in the app.")
@patch("src.pipeline.retrieve", return_value=[_mock_chunk()])
@patch("src.pipeline.predict_intent", return_value="order_status")
@patch("src.pipeline.predict_sentiment", return_value=("neutral", 0.9))
@patch("src.pipeline.detect_language", return_value=("es", 0.95))
def test_non_english_message_gets_translated_response(*_mocks):
    result = run_pipeline("¿Dónde está mi pedido?")
    assert result.detected_language == "es"
    assert result.response.startswith("[es]")


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
