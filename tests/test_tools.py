"""
tests/test_tools.py

Pytest suite for the three FitFindr tools in tools.py.

The LLM-backed tools (suggest_outfit, create_fit_card) are tested with the
Groq call stubbed out via monkeypatch, so the suite is deterministic and does
not require a network connection or a GROQ_API_KEY. The search tool is pure and
is tested directly against the mock dataset.

Run with:  .venv/bin/python -m pytest tests/ -v
"""

import pytest

import tools
from tools import search_listings, suggest_outfit, create_fit_card
from utils.data_loader import get_empty_wardrobe, get_example_wardrobe


# ── search_listings ─────────────────────────────────────────────────────────


def test_search_returns_results():
    """A valid query returns a non-empty list of matching listing dicts."""
    results = search_listings("vintage", max_price=50)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(item, dict) for item in results)


def test_search_empty_results():
    """An impossible query returns [] rather than raising."""
    results = search_listings("zzzznevermatchqwerty")

    assert results == []


def test_search_price_filter():
    """Every returned listing respects the max_price ceiling (inclusive)."""
    max_price = 40.0
    results = search_listings("jeans", max_price=max_price)

    assert len(results) > 0  # sanity: the query should match something
    assert all(item["price"] <= max_price for item in results)


# ── suggest_outfit ──────────────────────────────────────────────────────────


def test_suggest_outfit_empty_wardrobe(monkeypatch):
    """An empty wardrobe yields a non-empty string and never crashes."""
    # Stub the LLM so the test is deterministic and offline.
    monkeypatch.setattr(tools, "_chat", lambda messages, temperature=0.5: "styling advice")

    new_item = search_listings("vintage")[0]
    result = suggest_outfit(new_item, get_empty_wardrobe())

    assert isinstance(result, str)
    assert result.strip() != ""


def test_suggest_outfit_populated_wardrobe(monkeypatch):
    """A populated wardrobe also returns a non-empty string."""
    monkeypatch.setattr(tools, "_chat", lambda messages, temperature=0.5: "outfit ideas")

    new_item = search_listings("vintage")[0]
    result = suggest_outfit(new_item, get_example_wardrobe())

    assert isinstance(result, str)
    assert result.strip() != ""


def test_suggest_outfit_api_error_fallback(monkeypatch):
    """If the LLM call raises, suggest_outfit returns a safe fallback string."""
    def boom(*args, **kwargs):
        raise RuntimeError("API down")

    monkeypatch.setattr(tools, "_chat", boom)

    new_item = search_listings("vintage")[0]
    result = suggest_outfit(new_item, get_empty_wardrobe())

    assert isinstance(result, str)
    assert result.strip() != ""


# ── create_fit_card ─────────────────────────────────────────────────────────


def test_create_fit_card_empty_outfit(monkeypatch):
    """An empty outfit string still returns a non-empty caption string."""
    # Echo the prompt back so we can assert the item (not an outfit) drives it.
    monkeypatch.setattr(
        tools,
        "_chat",
        lambda messages, temperature=0.5: messages[-1]["content"],
    )

    new_item = search_listings("vintage")[0]
    result = create_fit_card("", new_item)

    assert isinstance(result, str)
    assert result.strip() != ""
    # The empty-outfit branch must not ask the model to describe an outfit.
    assert "The outfit:" not in result


def test_create_fit_card_api_error_fallback(monkeypatch):
    """If the LLM call raises, create_fit_card returns a safe fallback string."""
    def boom(*args, **kwargs):
        raise RuntimeError("API down")

    monkeypatch.setattr(tools, "_chat", boom)

    new_item = search_listings("vintage")[0]
    result = create_fit_card("a cool outfit", new_item)

    assert isinstance(result, str)
    assert result.strip() != ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
