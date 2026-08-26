"""
tests/test_timestamp_refiner.py
--------------------------------
Tests for TimestampRefiner using controlled transcript data.

Tests the key behaviors:
  - Clean word timestamps are used directly
  - Suspicious long-first-word artifact is detected and corrected
  - Fallback to coarse segment start works when no words found
  - Word normalization handles punctuation correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.timestamp_refiner import (
    TimestampRefiner,
    _find_word_timestamps,
    _normalize_word,
    LONG_FIRST_WORD_THRESHOLD,
)


TARGET = "My mind rebels at stagnation"


# -------------------------------------------------------------------------
# _normalize_word tests
# -------------------------------------------------------------------------

def test_normalize_word_lowercase():
    assert _normalize_word("Hello") == "hello"


def test_normalize_word_strips_punctuation():
    assert _normalize_word("stagnation.") == "stagnation"


def test_normalize_word_strips_question_mark():
    assert _normalize_word("rebels?") == "rebels"


def test_normalize_word_empty():
    assert _normalize_word("") == ""


# -------------------------------------------------------------------------
# _find_word_timestamps tests with clean data
# -------------------------------------------------------------------------

def make_segments_with_words(word_list):
    """Helper: build segments with specified word timestamps."""
    words = [
        {"word": w, "start": s, "end": e}
        for w, s, e in word_list
    ]
    return [{"start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(w for w, s, e in word_list), "words": words}]


CLEAN_TRANSCRIPT = make_segments_with_words([
    ("My", 100.0, 100.5),        # normal duration
    ("mind", 100.6, 101.0),
    ("rebels", 101.1, 101.5),
    ("at", 101.6, 101.8),
    ("stagnation.", 102.0, 102.8),
    ("Give", 104.0, 104.2),
    ("me", 104.2, 104.4),
    ("problems.", 104.5, 104.9),
])


def test_find_word_timestamps_clean():
    """Clean word timestamps should be found and returned correctly."""
    result = _find_word_timestamps(CLEAN_TRANSCRIPT, ["my", "mind", "rebels", "at", "stagnation"])
    assert result is not None
    phrase_start, phrase_end, first_start, first_end, suspicious = result
    assert phrase_start == 100.0
    assert phrase_end == pytest.approx(102.8)
    assert first_start == 100.0
    assert first_end == pytest.approx(100.5)
    assert suspicious is False  # 0.5s duration is not suspicious


ARTIFACT_TRANSCRIPT = make_segments_with_words([
    ("My", 321.4, 322.48),       # duration = 1.08s > LONG_FIRST_WORD_THRESHOLD
    ("mind", 325.39, 325.67),    # large gap from "My" end
    ("rebels", 325.67, 326.29),
    ("at", 326.29, 326.69),
    ("stagnation.", 326.69, 327.69),
])


def test_find_word_timestamps_detects_artifact():
    """The long 'My' timestamp from the known example should be flagged suspicious."""
    result = _find_word_timestamps(
        ARTIFACT_TRANSCRIPT,
        ["my", "mind", "rebels", "at", "stagnation"]
    )
    assert result is not None
    phrase_start, phrase_end, first_start, first_end, suspicious = result
    assert phrase_start == 321.4
    assert first_end == pytest.approx(322.48)
    # Duration of "My" = 1.08 > LONG_FIRST_WORD_THRESHOLD (1.0)
    assert suspicious is True


def test_find_word_timestamps_not_found():
    """Should return None when target is not in transcript."""
    segs = make_segments_with_words([
        ("Hello", 10.0, 10.3),
        ("world", 10.4, 10.7),
    ])
    result = _find_word_timestamps(segs, ["my", "mind", "rebels", "at", "stagnation"])
    assert result is None


def test_find_word_timestamps_punctuation_normalized():
    """'stagnation.' should match target 'stagnation' after normalization."""
    result = _find_word_timestamps(ARTIFACT_TRANSCRIPT, ["my", "mind", "rebels", "at", "stagnation"])
    assert result is not None


# -------------------------------------------------------------------------
# TimestampRefiner integration tests (no audio file needed for word ts path)
# -------------------------------------------------------------------------

def test_refiner_clean_words_no_audio():
    """Refiner should use word timestamps without needing audio file."""
    refiner = TimestampRefiner(model_size="small")
    result = refiner.refine(
        segments=CLEAN_TRANSCRIPT,
        coarse_start=100.0,
        coarse_end=105.0,
        target_dialogue=TARGET,
        audio_path=None,  # no audio needed for word timestamp path
    )
    assert result.phrase_start == pytest.approx(100.0)
    assert result.method == "word_timestamps_clean"
    assert result.confidence >= 0.70
    assert result.uncertainty_ms <= 300.0


def test_refiner_artifact_detected():
    """Refiner should detect artifact and return adjusted timestamp."""
    refiner = TimestampRefiner(model_size="small")
    result = refiner.refine(
        segments=ARTIFACT_TRANSCRIPT,
        coarse_start=321.4,
        coarse_end=328.0,
        target_dialogue=TARGET,
        audio_path=None,
    )
    # Should detect artifact and return adjusted timestamp
    assert result.method in (
        "word_timestamps_adjusted_artifact",
        "retranscribe_short_clip",
    )
    # Adjusted start should be ≥ the coarse start
    assert result.phrase_start >= 321.4
    # Uncertainty should be higher due to artifact
    assert result.uncertainty_ms >= 300.0


def test_refiner_fallback_no_words_no_audio():
    """When no word timestamps and no audio, fall back to coarse with high uncertainty."""
    segs_no_words = [{"start": 321.4, "end": 332.0, "text": "My mind rebels at stagnation.", "words": []}]
    refiner = TimestampRefiner(model_size="small")
    result = refiner.refine(
        segments=segs_no_words,
        coarse_start=321.4,
        coarse_end=332.0,
        target_dialogue=TARGET,
        audio_path=None,
    )
    assert result.method == "coarse_segment_fallback"
    assert result.phrase_start == pytest.approx(321.4)
    assert result.uncertainty_ms >= 500.0
    assert result.confidence <= 0.50


def test_refiner_result_has_all_fields():
    """RefinedTimestamp must have all required fields."""
    refiner = TimestampRefiner()
    result = refiner.refine(
        segments=CLEAN_TRANSCRIPT,
        coarse_start=100.0,
        coarse_end=105.0,
        target_dialogue=TARGET,
    )
    assert hasattr(result, "phrase_start")
    assert hasattr(result, "phrase_end")
    assert hasattr(result, "first_word_start")
    assert hasattr(result, "first_word_end")
    assert hasattr(result, "method")
    assert hasattr(result, "confidence")
    assert hasattr(result, "uncertainty_ms")
    assert 0.0 <= result.confidence <= 1.0
    assert result.uncertainty_ms > 0.0
