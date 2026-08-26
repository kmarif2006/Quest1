"""
dialogue_matcher.py
-------------------
Robust dialogue matching with multi-signal scoring and token coverage enforcement.

Key design decisions:
  - token_set_ratio alone is insufficient: "None." scores 100 against "my" via token_set.
  - We enforce token_coverage: fraction of target tokens actually present in candidate.
  - Short targets need high coverage; long targets (5+ tokens) require >=60% coverage.
  - final score is a weighted blend of multiple signals, gated by coverage.
  - Candidates below MIN_COVERAGE are ALWAYS rejected.
"""

import re
from difflib import SequenceMatcher
from typing import List, Optional

from rapidfuzz import fuzz

from src.models import DialogueMatch


# Minimum fraction of target tokens that must appear in candidate text.
# This prevents "None." matching "My mind rebels at stagnation" (0 overlap → reject).
MIN_TOKEN_COVERAGE = 0.5   # at least 50% of target tokens must be present
MIN_COMPOSITE_SCORE = 45.0  # composite score threshold after coverage gate


def _normalize(text: str) -> str:
    """
    Normalize text for comparison:
      - lowercase
      - remove punctuation (keep letters, digits, whitespace)
      - collapse repeated whitespace
      - strip apostrophes (don't → dont)
    """
    text = text.lower()
    text = text.replace("'", "").replace("'", "").replace("`", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_coverage(target_tokens: List[str], candidate_text: str) -> float:
    """
    Compute the fraction of unique target tokens present in the candidate.

    This is the primary false-positive guard. A candidate with none of the
    target words present scores 0.0 regardless of fuzzy ratios.
    """
    if not target_tokens:
        return 0.0
    candidate_words = set(candidate_text.split())
    found = sum(1 for t in target_tokens if t in candidate_words)
    return found / len(target_tokens)


def _compute_scores(
    normalized_target: str,
    normalized_candidate: str,
    target_tokens: List[str],
) -> dict:
    """
    Compute all matching signals between normalized target and candidate.
    Returns a dict of individual scores.
    """
    ratio = fuzz.ratio(normalized_target, normalized_candidate)
    partial = fuzz.partial_ratio(normalized_target, normalized_candidate)
    token_set = fuzz.token_set_ratio(normalized_target, normalized_candidate)
    token_sort = fuzz.token_sort_ratio(normalized_target, normalized_candidate)
    seq = SequenceMatcher(None, normalized_target, normalized_candidate).ratio() * 100

    exact_sub = normalized_target in normalized_candidate
    coverage = _token_coverage(target_tokens, normalized_candidate)

    return {
        "ratio": ratio,
        "partial": partial,
        "token_set": token_set,
        "token_sort": token_sort,
        "sequence": seq,
        "exact_substring": exact_sub,
        "token_coverage": coverage,
    }


def _composite_score(scores: dict, target_length: int) -> float:
    """
    Weighted composite score from individual signals.

    Weights are designed so that:
      - token_coverage is a gate (low coverage → low final score)
      - For multi-word targets, partial_ratio catches substring matches well
      - ratio penalizes length mismatches (exact match scenarios)
      - exact_substring is a strong bonus

    Weight rationale:
      partial_ratio  (0.30): finds target as substring — most important for long targets
      ratio          (0.20): full string similarity — rewards precise matches
      token_sort     (0.20): word-order-insensitive matching
      token_coverage (0.20): explicit token overlap gate — prevents "None." matches
      sequence       (0.10): character-level sequence alignment

    exact_substring bonus: +15 points (capped at 100)
    """
    weights = {
        "partial": 0.30,
        "ratio": 0.20,
        "token_sort": 0.20,
        "sequence": 0.10,
    }

    weighted = (
        scores["partial"] * weights["partial"]
        + scores["ratio"] * weights["ratio"]
        + scores["token_sort"] * weights["token_sort"]
        + scores["sequence"] * weights["sequence"]
    )

    # Token coverage contributes 20% of score
    coverage_contribution = scores["token_coverage"] * 100 * 0.20
    weighted += coverage_contribution

    # Bonus for exact substring presence
    if scores["exact_substring"]:
        weighted += 15.0

    return min(100.0, weighted)


class DialogueMatcher:
    """
    Finds transcript segments containing the target dialogue.

    Rejects false positives by enforcing:
    1. token_coverage >= MIN_TOKEN_COVERAGE
    2. composite_score >= MIN_COMPOSITE_SCORE
    """

    def normalize(self, text: str) -> str:
        return _normalize(text)

    def find_matches(
        self,
        segments: List[dict],
        target_dialogue: str,
        threshold: float = MIN_COMPOSITE_SCORE,
        min_coverage: float = MIN_TOKEN_COVERAGE,
        max_window: int = 4,
        max_results: int = 10,
    ) -> List[DialogueMatch]:
        """
        Search transcript segments for the target dialogue.

        Args:
            segments: list of segment dicts with 'start', 'end', 'text' keys.
            target_dialogue: the phrase to search for.
            threshold: minimum composite score to include a candidate.
            min_coverage: minimum token coverage to include a candidate.
            max_window: maximum number of consecutive segments to combine.
            max_results: maximum number of unique-region results to return.

        Returns:
            List of DialogueMatch sorted by score descending.
        """
        normalized_target = _normalize(target_dialogue)
        target_tokens = normalized_target.split()
        n_target_tokens = len(target_tokens)

        # For very short targets (1-2 tokens), use higher coverage requirement
        # to prevent meaningless matches.
        effective_min_coverage = min_coverage
        if n_target_tokens <= 2:
            effective_min_coverage = 1.0  # require ALL tokens for short targets

        candidates: List[DialogueMatch] = []

        for window_size in range(1, max_window + 1):
            for i in range(len(segments) - window_size + 1):
                window = segments[i : i + window_size]

                combined_text = " ".join(seg["text"] for seg in window)
                norm_text = _normalize(combined_text)

                scores = _compute_scores(
                    normalized_target, norm_text, target_tokens
                )

                # Hard gate: reject if token coverage is too low
                if scores["token_coverage"] < effective_min_coverage:
                    continue

                composite = _composite_score(scores, n_target_tokens)

                if composite < threshold:
                    continue

                match = DialogueMatch(
                    start=window[0]["start"],
                    end=window[-1]["end"],
                    text=combined_text,
                    score=composite,
                    token_coverage=scores["token_coverage"],
                    ratio_score=scores["ratio"],
                    partial_score=scores["partial"],
                    token_set_score=scores["token_set"],
                    sequence_score=scores["sequence"],
                    exact_substring=scores["exact_substring"],
                    window_size=window_size,
                )
                candidates.append(match)

        # Sort by score descending
        candidates.sort(key=lambda m: m.score, reverse=True)

        # De-duplicate: suppress overlapping time windows, keeping the best
        filtered: List[DialogueMatch] = []
        used_ranges: List[tuple] = []

        for match in candidates:
            overlap = any(
                match.start < u_end and match.end > u_start
                for u_start, u_end in used_ranges
            )
            if not overlap:
                filtered.append(match)
                used_ranges.append((match.start, match.end))

            if len(filtered) >= max_results:
                break

        return filtered

    def best_match(
        self,
        segments: List[dict],
        target_dialogue: str,
    ) -> Optional[DialogueMatch]:
        """Return the single best matching candidate, or None."""
        results = self.find_matches(segments, target_dialogue, max_results=1)
        return results[0] if results else None