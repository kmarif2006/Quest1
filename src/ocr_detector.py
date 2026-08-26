"""
ocr_detector.py
---------------
OPTIONAL OCR verification for subtitle/caption detection in video frames.

This module is NOT part of the default pipeline. It is only instantiated
when --enable-ocr is passed (CLI) or the OCR checkbox is enabled (Streamlit).

Why OCR is optional:
  - The primary signal is spoken audio (Whisper transcription).
  - Many videos do not have burned-in subtitles.
  - EasyOCR is slow and heavyweight.
  - OCR cannot detect audio; it can only find on-screen text.

If OCR is enabled, it scans a limited number of frames near the candidate
timestamp for any visible text matching the target dialogue.
OCR is treated as SUPPLEMENTARY evidence only.

EasyOCR is imported lazily (inside __init__) so the module can be imported
safely without easyocr installed.
"""

from __future__ import annotations

import os
from difflib import SequenceMatcher
from typing import List, Dict, Optional

import cv2


class OCRDetector:
    """
    Optional OCR verification for on-screen subtitle/caption text.

    Args:
        max_frames: maximum number of frames to scan (prevents slow runs).
    """

    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames
        self._reader = None  # loaded lazily

    def _get_reader(self):
        """Load EasyOCR reader on first use."""
        if self._reader is None:
            try:
                import easyocr  # type: ignore
                print("\n[ocr] Loading EasyOCR model...")
                self._reader = easyocr.Reader(["en"], gpu=False)
            except ImportError:
                raise ImportError(
                    "EasyOCR is not installed. "
                    "Install it with: pip install easyocr"
                )
        return self._reader

    def _similarity(self, target: str, detected: str) -> float:
        """Compute text similarity ratio."""
        return SequenceMatcher(
            None, target.lower(), detected.lower()
        ).ratio() * 100

    def scan_frames(
        self,
        frames_dir: str,
        target_text: str,
        max_frames: Optional[int] = None,
    ) -> List[Dict]:
        """
        Scan extracted frames for visible text matching the target.

        Args:
            frames_dir: directory containing frame PNG files.
            target_text: the dialogue text to look for visually.
            max_frames: override the instance max_frames limit.

        Returns:
            List of result dicts sorted by similarity score, descending.
        """
        limit = max_frames or self.max_frames
        reader = self._get_reader()

        try:
            frame_files = sorted(
                [f for f in os.listdir(frames_dir) if f.endswith(".png")],
                key=lambda x: int(x.split("_")[1].split(".")[0])
                if x.startswith("frame_") else 0,
            )
        except Exception:
            frame_files = sorted(
                [f for f in os.listdir(frames_dir) if f.endswith(".png")]
            )

        # Limit number of frames to scan
        frame_files = frame_files[:limit]

        results = []

        for filename in frame_files:
            path = os.path.join(frames_dir, filename)
            image = cv2.imread(path)
            if image is None:
                continue

            height, width = image.shape[:2]

            # Focus on lower 45% of frame (subtitle area)
            roi = image[int(height * 0.55) :, :]

            # Enlarge for better OCR accuracy
            roi = cv2.resize(
                roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
            )

            try:
                detections = reader.readtext(roi, detail=1, paragraph=True)
            except Exception as e:
                print(f"[ocr] Error reading {filename}: {e}")
                continue

            combined_text = " ".join(d[1] for d in detections)
            score = self._similarity(target_text, combined_text)

            results.append({
                "frame_file": filename,
                "text": combined_text,
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results