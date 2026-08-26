"""
transcriber.py
--------------
Transcribes audio using Faster-Whisper and caches results.
Supports a progress_callback for live UI/CLI updates during transcription.

Cache format:
{
    "cache_version": 1,
    "model": "small",
    "word_timestamps": true,
    "audio_file": "audio.wav",
    "segments": [...]
}

Cache is invalidated and regenerated if:
  - The cache file does not exist
  - The model size in cache != requested model size
  - word_timestamps flag in cache != requested flag
  - The audio_file in cache does not match the current audio
  - The segments list is missing or empty
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable


CACHE_VERSION = 1


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Transcriber:
    """
    Whisper-based audio transcriber with per-audio cache management
    and real-time progress logging.
    """

    def __init__(
        self,
        model_size: str = "small",
        output_dir: str = "output",
        word_timestamps: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.model_size = model_size
        self.output_dir = output_dir
        self.word_timestamps = word_timestamps
        self._log = progress_callback or print

        os.makedirs(self.output_dir, exist_ok=True)
        self._model = None

    def _get_cache_path(self, audio_path: str) -> str:
        """Get the specific cache path for the given audio file."""
        stem = Path(audio_path).stem
        if stem in ("audio", "video") and os.path.exists(
            os.path.join(self.output_dir, "transcript.json")
        ):
            return os.path.join(self.output_dir, "transcript.json")
        return os.path.join(self.output_dir, f"transcript_{stem}.json")

    def _load_model(self):
        """Load the Whisper model lazily."""
        if self._model is None:
            self._log(f"  Loading Whisper model ({self.model_size})...")
            from faster_whisper import WhisperModel  # type: ignore

            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
        return self._model

    def _validate_cache(
        self, cache_path: str, audio_path: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Load and validate the cached transcript."""
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self._log(f"  [transcriber] Cache read error: {e}")
            return None

        if isinstance(data, list):
            # Old-format bare list — valid only for the default benchmark audio
            if Path(audio_path).stem in ("audio", "video"):
                segments = data
                cached_model = None
                cached_word_ts = None
            else:
                return None
        elif isinstance(data, dict):
            cached_model = data.get("model")
            cached_word_ts = data.get("word_timestamps")
            cached_audio = data.get("audio_file")
            segments = data.get("segments", [])
            if cached_audio and cached_audio != Path(audio_path).name:
                return None
        else:
            return None

        if cached_model is not None and cached_model != self.model_size:
            self._log(
                f"  [transcriber] Cache model mismatch: "
                f"cached={cached_model}, requested={self.model_size}. Re-transcribing..."
            )
            return None

        if self.word_timestamps and cached_word_ts is not None and not cached_word_ts:
            self._log("  [transcriber] Cache missing word timestamps. Re-transcribing...")
            return None

        if not segments:
            return None

        if self.word_timestamps:
            found_words = any(bool(seg.get("words")) for seg in segments)
            if not found_words:
                self._log("  [transcriber] Cached segments lack word timestamps. Re-transcribing...")
                return None

        self._log(f"  [transcriber] Using cached transcript: {cache_path}")
        self._log(f"  Segments: {len(segments)}")
        return segments

    def _run_transcription(self, audio_path: str) -> List[Dict[str, Any]]:
        """Run Whisper transcription with live per-segment progress."""
        model = self._load_model()

        self._log(f"  Transcribing: {audio_path}")
        self._log(f"  Model: {self.model_size} | Word timestamps: {self.word_timestamps}")

        segments_iter, info = model.transcribe(
            audio_path,
            beam_size=3,
            vad_filter=True,
            word_timestamps=self.word_timestamps,
        )

        total_duration = max(1.0, getattr(info, "duration", 1.0))
        self._log(
            f"  Language: {info.language} ({info.language_probability:.0%}) | "
            f"Duration: {_fmt(total_duration)}"
        )

        results = []
        for segment in segments_iter:
            words = []
            if self.word_timestamps and segment.words:
                for w in segment.words:
                    words.append(
                        {"word": w.word.strip(), "start": w.start, "end": w.end}
                    )

            results.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": words,
                }
            )

            pct = min(100, int((segment.end / total_duration) * 100))
            preview = segment.text.strip().replace("\n", " ")
            if len(preview) > 55:
                preview = preview[:52] + "..."
            self._log(
                f"  [{_fmt(segment.end)} / {_fmt(total_duration)}] ({pct}%) \"{preview}\""
            )

        return results

    def _save_cache(
        self, cache_path: str, audio_path: str, segments: List[Dict[str, Any]]
    ) -> None:
        cache_data = {
            "cache_version": CACHE_VERSION,
            "model": self.model_size,
            "word_timestamps": self.word_timestamps,
            "audio_file": Path(audio_path).name,
            "segments": segments,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        self._log(f"  [transcriber] Transcript cached: {cache_path}")

    def transcribe(
        self,
        audio_path: str,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Transcribe audio, using cache when valid."""
        cache_path = self._get_cache_path(audio_path)

        if not force:
            cached = self._validate_cache(cache_path, audio_path)
            if cached is not None:
                return cached

        segments = self._run_transcription(audio_path)
        self._save_cache(cache_path, audio_path, segments)
        self._log(f"  Transcription complete. {len(segments)} segments.")
        return segments