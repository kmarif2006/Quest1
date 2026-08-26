"""
timestamp_refiner.py
--------------------
Refines coarse Whisper timestamps to obtain a more accurate phrase start time.

Strategy (in order of preference):
  1. Use word-level timestamps already present in the transcript.
     - Whisper word timestamps can have the "long first word" artifact where
       the segment start is assigned to the first word but the word is actually
       spoken later. We detect and correct this.
  2. Re-transcribe only the short audio window around the candidate with a
     potentially higher-accuracy model run on a short segment.
  3. Fall back to the segment start time with increased uncertainty.

The "My" timestamp problem:
  Faster-Whisper sometimes assigns the entire pre-speech silence interval to
  the first word of a segment. The word "My" might appear with start=321.40
  but "mind" starts at 325.39. This means:
    - Either there is genuine silence/pause between "My" and "mind"
    - Or the "My" timestamp is wrong (alignment artifact)

  Resolution: If the gap between word[0].end and word[1].start is < 2 seconds
  but word[0].end - word[0].start > 1 second, suspect alignment artifact and
  use word[1].start - 0.3s as the phrase start estimate.
"""

from __future__ import annotations

import json
import os
import subprocess
import re
import tempfile
from typing import List, Optional, Tuple

from src.models import RefinedTimestamp


# Threshold: if first word duration exceeds this (seconds), suspect artifact
LONG_FIRST_WORD_THRESHOLD = 1.0

# Padding to add before/after candidate when re-transcribing short clip
REFINEMENT_PADDING_SEC = 5.0

# If gap between end of word[0] and start of word[1] is small, treat as
# possible alignment issue
WORD_GAP_SUSPICION_THRESHOLD = 2.5


def _normalize_word(word: str) -> str:
    """Normalize a word for comparison (lowercase, strip punctuation)."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _find_word_timestamps(
    segments: List[dict],
    target_words: List[str],
) -> Optional[Tuple[float, float, float, float, bool]]:
    """
    Search all word-level timestamps for the target phrase.

    Returns:
        (phrase_start, phrase_end, first_word_start, first_word_end, suspicious)
        or None if not found.

    'suspicious' is True when the first word's duration or gap to next word
    suggests a Whisper alignment artifact.
    """
    norm_targets = [_normalize_word(w) for w in target_words if _normalize_word(w)]
    if not norm_targets:
        return None

    # Flatten all words from all segments
    all_words = []
    for seg in segments:
        for w in seg.get("words", []):
            nw = _normalize_word(w["word"])
            if nw:
                all_words.append({
                    "word": nw,
                    "start": w["start"],
                    "end": w["end"],
                    "raw": w["word"],
                })

    n = len(norm_targets)
    for i in range(len(all_words) - n + 1):
        window = all_words[i : i + n]
        if [w["word"] for w in window] == norm_targets:
            phrase_start = window[0]["start"]
            phrase_end = window[-1]["end"]
            first_start = window[0]["start"]
            first_end = window[0]["end"]

            suspicious = False

            # Detect long-first-word artifact.
            # Pattern: word[0] has unusually long duration, AND there is a
            # significant gap between word[0].end and word[1].start.
            # This is the Whisper alignment artifact where the segment start
            # timestamp is assigned to word[0], but the word is actually
            # spoken much later. The silence is absorbed into word[0]'s span.
            #
            # Example from known data:
            #   "My"  start=321.40 end=322.48  (duration=1.08s)
            #   "mind" start=325.39             (gap from "My".end = 2.91s)
            #
            # A large gap (> WORD_GAP_SUSPICION_THRESHOLD) COMBINED with a
            # long first-word duration indicates the artifact.
            first_duration = first_end - first_start
            if first_duration > LONG_FIRST_WORD_THRESHOLD and n > 1:
                gap_to_next = window[1]["start"] - first_end
                if gap_to_next > WORD_GAP_SUSPICION_THRESHOLD:
                    # Large gap after long first word = timing artifact
                    suspicious = True

            return (phrase_start, phrase_end, first_start, first_end, suspicious)

    return None


def _extract_audio_clip(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> bool:
    """Extract a short clip from the audio file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(max(0.0, start_sec)),
        "-to", str(end_sec),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    try:
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _retranscribe_clip(
    clip_path: str,
    model_size: str,
    target_dialogue: str,
    clip_offset: float,
    model=None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Re-transcribe a short audio clip with word timestamps.
    Accepts an optional pre-loaded WhisperModel to avoid loading it twice.
    Returns (phrase_start, phrase_end, first_start, first_end) in global time,
    or None on failure.
    """
    try:
        if model is None:
            from faster_whisper import WhisperModel  # type: ignore
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments_iter, _ = model.transcribe(
            clip_path,
            beam_size=3,
            word_timestamps=True,
            vad_filter=False,
        )

        clip_segments = []
        for seg in segments_iter:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": w.start + clip_offset,
                        "end": w.end + clip_offset,
                    })
            clip_segments.append({
                "start": seg.start + clip_offset,
                "end": seg.end + clip_offset,
                "text": seg.text.strip(),
                "words": words,
            })

        # Try to find the target in the re-transcribed clip
        target_words = re.sub(r"[^a-z0-9\s]", " ", target_dialogue.lower()).split()
        result = _find_word_timestamps(clip_segments, target_words)
        if result:
            phrase_start, phrase_end, first_start, first_end, _ = result
            return (phrase_start, phrase_end, first_start, first_end)

    except Exception as exc:
        print(f"[timestamp_refiner] Re-transcription failed: {exc}")

    return None


class TimestampRefiner:
    """
    Refines a coarse dialogue timestamp to the most accurate available estimate.

    Usage:
        refiner = TimestampRefiner(model_size="small")
        result = refiner.refine(segments, coarse_match, target_dialogue, audio_path)
    """

    def __init__(
        self,
        model_size: str = "small",
        whisper_model=None,
        progress_callback=None,
    ):
        self.model_size = model_size
        self._model = whisper_model          # re-use the already-loaded model
        self._log = progress_callback or print

    def refine(
        self,
        segments: List[dict],
        coarse_start: float,
        coarse_end: float,
        target_dialogue: str,
        audio_path: Optional[str] = None,
    ) -> RefinedTimestamp:
        """
        Attempt to refine the dialogue timestamp through multiple strategies.

        Strategy order:
        1. Word-level timestamps from existing transcript
           - with suspicion detection for alignment artifacts
        2. Re-transcription of short audio clip (if audio_path provided)
        3. Fallback to coarse segment start with high uncertainty
        """
        target_words = re.sub(
            r"[^a-z0-9\s]", " ", target_dialogue.lower()
        ).split()

        # ---------------------------------------------------------------
        # STRATEGY 1: Use existing word timestamps
        # ---------------------------------------------------------------
        word_result = _find_word_timestamps(segments, target_words)

        if word_result:
            phrase_start, phrase_end, first_start, first_end, suspicious = word_result

            if not suspicious:
                # Word timestamps look clean
                return RefinedTimestamp(
                    phrase_start=phrase_start,
                    phrase_end=phrase_end,
                    first_word_start=first_start,
                    first_word_end=first_end,
                    method="word_timestamps_clean",
                    confidence=0.80,
                    uncertainty_ms=150.0,
                )
            else:
                # Suspicious: long first-word artifact detected
                # Hypothesis: the real phrase start is closer to word[1].start
                # We adjust: phrase_start = first_end (end of first word token),
                # because Whisper may have assigned silence to "My".
                # Better estimate: use the second word start as the anchor.
                # Use word[1].start - 0.3s as best estimate, with higher uncertainty.
                adjusted_start = max(phrase_start, first_end - 0.5)

                self._log(
                    "[timestamp_refiner] Suspicious first-word alignment detected."
                )
                self._log(
                    f"  Word 'first' duration: {first_end - first_start:.2f}s"
                )
                self._log(
                    f"  Adjusting phrase_start: {phrase_start:.2f}s -> {adjusted_start:.2f}s"
                )

                # Try re-transcription to get better timing
                if audio_path and os.path.exists(audio_path):
                    clip_start = max(0.0, adjusted_start - REFINEMENT_PADDING_SEC)
                    clip_end = phrase_end + REFINEMENT_PADDING_SEC

                    self._log(
                        f"  Re-transcribing clip [{clip_start:.1f}s -> {clip_end:.1f}s]..."
                    )

                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp:
                        tmp_path = tmp.name

                    try:
                        extracted = _extract_audio_clip(
                            audio_path, clip_start, clip_end, tmp_path
                        )
                        if extracted:
                            retrans_result = _retranscribe_clip(
                                tmp_path,
                                self.model_size,
                                target_dialogue,
                                clip_offset=clip_start,
                                model=self._model,
                            )
                            if retrans_result:
                                r_start, r_end, r_first, r_first_end = retrans_result
                                self._log(
                                    f"  Re-transcription refined start: {r_start:.3f}s"
                                )
                                return RefinedTimestamp(
                                    phrase_start=r_start,
                                    phrase_end=r_end,
                                    first_word_start=r_first,
                                    first_word_end=r_first_end,
                                    method="retranscribe_short_clip",
                                    confidence=0.75,
                                    uncertainty_ms=200.0,
                                )
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

                # Return adjusted with elevated uncertainty
                return RefinedTimestamp(
                    phrase_start=adjusted_start,
                    phrase_end=phrase_end,
                    first_word_start=adjusted_start,
                    first_word_end=first_end,
                    method="word_timestamps_adjusted_artifact",
                    confidence=0.60,
                    uncertainty_ms=500.0,  # wider uncertainty due to artifact
                )

        # ---------------------------------------------------------------
        # STRATEGY 2: Re-transcribe short clip
        # ---------------------------------------------------------------
        if audio_path and os.path.exists(audio_path):
            clip_start = max(0.0, coarse_start - REFINEMENT_PADDING_SEC)
            clip_end = coarse_end + REFINEMENT_PADDING_SEC

            self._log(
                f"  Re-transcribing clip [{clip_start:.1f}s -> {clip_end:.1f}s]..."
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                extracted = _extract_audio_clip(
                    audio_path, clip_start, clip_end, tmp_path
                )
                if extracted:
                    retrans_result = _retranscribe_clip(
                        tmp_path,
                        self.model_size,
                        target_dialogue,
                        clip_offset=clip_start,
                        model=self._model,
                    )
                    if retrans_result:
                        r_start, r_end, r_first, r_first_end = retrans_result
                        self._log(f"  Re-transcription refined start: {r_start:.3f}s")
                        return RefinedTimestamp(
                            phrase_start=r_start,
                            phrase_end=r_end,
                            first_word_start=r_first,
                            first_word_end=r_first_end,
                            method="retranscribe_short_clip",
                            confidence=0.70,
                            uncertainty_ms=300.0,
                        )
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # ---------------------------------------------------------------
        # STRATEGY 3: Fallback — use coarse segment start
        # ---------------------------------------------------------------
        self._log(
            "[timestamp_refiner] Falling back to coarse segment start "
            "with high uncertainty."
        )
        return RefinedTimestamp(
            phrase_start=coarse_start,
            phrase_end=coarse_end,
            first_word_start=coarse_start,
            first_word_end=coarse_start + 0.5,
            method="coarse_segment_fallback",
            confidence=0.40,
            uncertainty_ms=1000.0,
        )
