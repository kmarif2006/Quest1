"""
confidence.py
-------------
Evidence-based confidence scoring for the Dialogue Frame Finder pipeline.

Confidence is computed as a weighted combination of measurable signals.
Each signal is documented with its weight and rationale.

Overall confidence is in [0.0, 1.0].
Overall uncertainty_ms is estimated from individual contributor uncertainties.

Signal weights (must sum to 1.0):
  transcript_match_score  (0.35): how well the candidate matched the target
  token_coverage          (0.25): fraction of target tokens found in candidate
  alignment_confidence    (0.20): confidence from the timestamp refiner
  frame_mapping_confidence(0.20): confidence from the frame mapper

Uncertainty combines:
  - refiner uncertainty_ms (dominant)
  - frame uncertainty = (1/fps) * 1000 ms per-frame timing resolution
  - A baseline uncertainty of 100ms for Whisper's intrinsic jitter
"""

from __future__ import annotations

import math
from typing import Optional

from src.models import DialogueMatch, RefinedTimestamp, FrameMappingResult


def compute_confidence(
    match: DialogueMatch,
    refined: RefinedTimestamp,
    frame_result: FrameMappingResult,
) -> tuple[float, float]:
    """
    Compute overall confidence and timing uncertainty.

    Returns:
        (confidence: float [0..1], uncertainty_ms: float)
    """
    # ---------------------------------------------------------------
    # Signal 1: transcript_match_score (normalised from 0-100 to 0-1)
    # Weight: 0.35 — primary evidence of correct dialogue detection
    # ---------------------------------------------------------------
    match_signal = min(1.0, match.score / 100.0)

    # ---------------------------------------------------------------
    # Signal 2: token_coverage (already in [0..1])
    # Weight: 0.25 — ensures we didn't just match one word
    # ---------------------------------------------------------------
    coverage_signal = match.token_coverage

    # ---------------------------------------------------------------
    # Signal 3: alignment confidence from refiner (already in [0..1])
    # Weight: 0.20 — quality of the timestamp estimate
    # ---------------------------------------------------------------
    alignment_signal = refined.confidence

    # ---------------------------------------------------------------
    # Signal 4: frame mapping confidence (already in [0..1])
    # Weight: 0.20 — quality of the frame extraction
    # ---------------------------------------------------------------
    frame_signal = frame_result.confidence

    # Weighted sum
    confidence = (
        match_signal * 0.35
        + coverage_signal * 0.25
        + alignment_signal * 0.20
        + frame_signal * 0.20
    )

    # ---------------------------------------------------------------
    # Uncertainty calculation
    # Combines refiner uncertainty with per-frame timing resolution
    # ---------------------------------------------------------------
    # Per-frame time resolution
    frame_resolution_ms = (1.0 / frame_result.fps * 1000.0) if frame_result.fps > 0 else 40.0

    # Whisper baseline jitter ~100ms
    whisper_baseline_ms = 100.0

    # Total uncertainty: root-mean-square combination
    uncertainty_ms = math.sqrt(
        refined.uncertainty_ms ** 2
        + frame_resolution_ms ** 2
        + whisper_baseline_ms ** 2
    )

    return round(confidence, 4), round(uncertainty_ms, 1)


def describe_confidence(confidence: float) -> str:
    """Return a human-readable description of the confidence level."""
    if confidence >= 0.85:
        return "HIGH"
    elif confidence >= 0.65:
        return "MODERATE"
    elif confidence >= 0.45:
        return "LOW"
    else:
        return "VERY LOW"
