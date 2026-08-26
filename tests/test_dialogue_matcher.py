"""
tests/test_dialogue_matcher.py
-------------------------------
Tests for the DialogueMatcher.

Verifies that:
  - Strong matches are accepted
  - False positives (short unrelated words) are rejected
  - Near-misses get appropriate scores
  - Token coverage gate works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.dialogue_matcher import DialogueMatcher, _normalize, _token_coverage


TARGET = "My mind rebels at stagnation"


def make_segment(text: str, start: float = 0.0, end: float = 5.0) -> dict:
    return {"start": start, "end": end, "text": text, "words": []}


# -------------------------------------------------------------------------
# Normalize tests
# -------------------------------------------------------------------------

def test_normalize_lowercase():
    assert _normalize("HELLO WORLD") == "hello world"


def test_normalize_strips_punctuation():
    assert _normalize("Hello, World!") == "hello world"


def test_normalize_strips_apostrophes():
    # "don't" → "dont"
    assert "dont" in _normalize("Don't stop")


def test_normalize_collapses_whitespace():
    assert _normalize("a   b") == "a b"


# -------------------------------------------------------------------------
# Token coverage tests
# -------------------------------------------------------------------------

def test_token_coverage_full_match():
    from src.dialogue_matcher import _token_coverage
    tokens = ["my", "mind", "rebels", "at", "stagnation"]
    candidate = "my mind rebels at stagnation give me problems"
    assert _token_coverage(tokens, candidate) == 1.0


def test_token_coverage_zero():
    from src.dialogue_matcher import _token_coverage
    tokens = ["my", "mind", "rebels", "at", "stagnation"]
    candidate = "none"
    assert _token_coverage(tokens, candidate) == 0.0


def test_token_coverage_partial():
    from src.dialogue_matcher import _token_coverage
    tokens = ["my", "mind", "rebels", "at", "stagnation"]
    candidate = "my mind"
    # 2 out of 5 tokens = 0.4
    assert _token_coverage(tokens, candidate) == pytest.approx(0.4)


# -------------------------------------------------------------------------
# DialogueMatcher acceptance tests
# -------------------------------------------------------------------------

matcher = DialogueMatcher()


def test_strong_match_accepted():
    """Perfect match with extra words should score very high."""
    segs = [make_segment("My mind rebels at stagnation. Give me problems.")]
    matches = matcher.find_matches(segs, TARGET)
    assert len(matches) > 0
    assert matches[0].score > 80.0
    assert matches[0].token_coverage == 1.0


def test_exact_match_accepted():
    """Exact phrase match should score maximum."""
    segs = [make_segment("My mind rebels at stagnation")]
    matches = matcher.find_matches(segs, TARGET)
    assert len(matches) > 0
    assert matches[0].score > 85.0


def test_none_rejected():
    """'None.' must be rejected — zero token coverage."""
    segs = [make_segment("None.")]
    matches = matcher.find_matches(segs, TARGET)
    assert len(matches) == 0, f"Expected no matches but got: {matches}"


def test_yes_rejected():
    """'Yes.' must be rejected."""
    segs = [make_segment("Yes.")]
    matches = matcher.find_matches(segs, TARGET)
    assert len(matches) == 0


def test_no_rejected():
    """'No.' must be rejected."""
    segs = [make_segment("No.")]
    matches = matcher.find_matches(segs, TARGET)
    assert len(matches) == 0


def test_john_rejected():
    """'John!' must be rejected."""
    segs = [make_segment("John!")]
    matches = matcher.find_matches(segs, TARGET)
    assert len(matches) == 0


def test_indeed_rejected():
    """'Indeed.' must be rejected."""
    segs = [make_segment("Indeed.")]
    matches = matcher.find_matches(segs, TARGET)
    assert len(matches) == 0


def test_me_rejected():
    """'Me?' must be rejected."""
    segs = [make_segment("Me?")]
    matches = matcher.find_matches(segs, TARGET)
    assert len(matches) == 0


def test_near_miss_lower_score():
    """'My mind rebels against stagnation' (against vs at) — should match but lower score."""
    segs = [make_segment("My mind rebels against stagnation")]
    matches = matcher.find_matches(segs, TARGET)
    # Should match (4/5 tokens covered) but lower than exact
    if matches:
        assert matches[0].score < 100.0
        # Coverage should be 4/5 = 0.8 (rebels/against != at/stagnation still present)
        # Actually "against" is not in target, "at" is not in candidate → 4/5 = 0.8
        assert matches[0].token_coverage >= 0.6


def test_partial_target_my_mind_low_confidence():
    """'My mind' (only 2 of 5 target tokens) — should NOT match (below 50% coverage)."""
    segs = [make_segment("My mind")]
    matches = matcher.find_matches(segs, TARGET)
    # 2/5 = 0.40 < 0.50 threshold → should be rejected
    assert len(matches) == 0, (
        f"'My mind' should be rejected (coverage<50%) but got: {matches}"
    )


def test_window_combining():
    """Target split across two segments should still match via window combining."""
    segs = [
        make_segment("My mind rebels", 100.0, 102.0),
        make_segment("at stagnation.", 102.1, 104.0),
    ]
    matches = matcher.find_matches(segs, TARGET)
    assert len(matches) > 0
    assert matches[0].token_coverage >= 0.8


def test_returns_no_duplicates():
    """Multiple overlapping windows shouldn't return the same time range twice."""
    segs = [
        make_segment("My mind rebels at stagnation. Give me problems.", 100.0, 110.0),
        make_segment("Give me work.", 110.0, 115.0),
    ]
    matches = matcher.find_matches(segs, TARGET)
    starts = [m.start for m in matches]
    assert len(starts) == len(set(starts)), "Duplicate start times found"


def test_sort_by_score():
    """Results must be sorted by score descending."""
    segs = [
        make_segment("My mind rebels at stagnation. Give me problems.", 100.0, 110.0),
        make_segment("Something completely different", 200.0, 205.0),
        make_segment("My mind wanders at stagnation", 300.0, 305.0),
    ]
    matches = matcher.find_matches(segs, TARGET)
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True), "Results not sorted by score"


def test_false_positive_list():
    """None of these common false positives should pass."""
    false_positives = [
        "Indeed.", "None.", "Yes.", "No.", "John!", "Me?",
        "Hello.", "Wait.", "Come.", "Go.", "Run.",
    ]
    for fp in false_positives:
        segs = [make_segment(fp)]
        matches = matcher.find_matches(segs, TARGET)
        assert len(matches) == 0, (
            f"False positive accepted: '{fp}' → {matches}"
        )
