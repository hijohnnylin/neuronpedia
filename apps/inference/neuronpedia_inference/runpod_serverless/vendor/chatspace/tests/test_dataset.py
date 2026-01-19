"""Tests for chatspace.hf_embed.dataset module."""

import pytest

from chatspace.hf_embed.config import SentenceTransformerConfig
from chatspace.hf_embed.dataset import (
    _extract_first_assistant_response,
    _rows_from_dataset,
)


def test_extract_first_assistant_response_success():
    """Test extracting first assistant response."""
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm good!"},
    ]

    result = _extract_first_assistant_response(conversation)
    assert result == "Hi there!"


def test_extract_first_assistant_response_model_role():
    """Test extracting with 'model' role."""
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "model", "content": "Hi from model!"},
    ]

    result = _extract_first_assistant_response(conversation)
    assert result == "Hi from model!"


def test_extract_first_assistant_response_case_insensitive():
    """Test case-insensitive role matching."""
    conversation = [
        {"role": "USER", "content": "Hello"},
        {"role": "ASSISTANT", "content": "Response"},
    ]

    result = _extract_first_assistant_response(conversation)
    assert result == "Response"


def test_extract_first_assistant_response_no_assistant():
    """Test when no assistant response exists."""
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Anyone there?"},
    ]

    result = _extract_first_assistant_response(conversation)
    assert result is None


def test_extract_first_assistant_response_empty_content():
    """Test when assistant content is empty."""
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "Actual response"},
    ]

    # Empty content is skipped, returns the next one
    result = _extract_first_assistant_response(conversation)
    assert result == "Actual response"


def test_extract_first_assistant_response_invalid_input():
    """Test with invalid input types."""
    import pytest

    # Non-list inputs raise TypeError
    with pytest.raises(TypeError):
        _extract_first_assistant_response(None)
    with pytest.raises(TypeError):
        _extract_first_assistant_response("not a list")

    # Valid list inputs return None when no assistant response found
    assert _extract_first_assistant_response([]) is None
    assert _extract_first_assistant_response([{"invalid": "structure"}]) is None


def test_rows_from_dataset_basic():
    """Test basic row iteration."""
    mock_dataset = [
        {"text": "row 1"},
        {"text": "row 2"},
        {"text": "row 3"},
    ]

    cfg = SentenceTransformerConfig(dataset="test")
    rows = list(_rows_from_dataset(iter(mock_dataset), cfg))

    assert len(rows) == 3
    assert rows[0]["text"] == "row 1"
    assert rows[2]["text"] == "row 3"


def test_rows_from_dataset_max_rows():
    """Test max_rows limit."""
    mock_dataset = [{"text": f"row {i}"} for i in range(10)]

    cfg = SentenceTransformerConfig(dataset="test", max_rows=5)
    rows = list(_rows_from_dataset(iter(mock_dataset), cfg))

    assert len(rows) == 5


def test_rows_from_dataset_extract_assistant():
    """Test assistant extraction mode."""
    mock_dataset = [
        {
            "conversation": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "Question 3"},
                # No assistant response
            ]
        },
    ]

    cfg = SentenceTransformerConfig(
        dataset="test",
        text_field="conversation",
        extract_first_assistant=True,
    )
    rows = list(_rows_from_dataset(iter(mock_dataset), cfg))

    # Should have 2 rows (third is skipped)
    assert len(rows) == 2
    assert rows[0]["conversation"] == "Answer 1"
    assert rows[1]["conversation"] == "Answer 2"

    # Original should be preserved
    assert "_original_conversation" in rows[0]
    assert isinstance(rows[0]["_original_conversation"], list)


def test_rows_from_dataset_no_extraction():
    """Test that normal text fields work without extraction."""
    mock_dataset = [
        {"text": "normal text 1"},
        {"text": "normal text 2"},
    ]

    cfg = SentenceTransformerConfig(
        dataset="test",
        extract_first_assistant=False,
    )
    rows = list(_rows_from_dataset(iter(mock_dataset), cfg))

    assert len(rows) == 2
    assert rows[0]["text"] == "normal text 1"
    assert "_original_text" not in rows[0]
