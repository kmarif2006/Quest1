"""
models.py
---------
Structured result dataclasses for the Dialogue Frame Finder pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class DialogueMatch:
    """Candidate segment from transcript matching target dialogue."""
    start: float
    end: float
    text: str
    score: float               # composite [0..100]
    token_coverage: float      # fraction of target tokens found [0..1]
    ratio_score: float
    partial_score: float
    token_set_score: float
    sequence_score: float
    exact_substring: bool
    window_size: int


@dataclass
class RefinedTimestamp:
    """Result of timestamp refinement stage."""
    phrase_start: float       # seconds
    phrase_end: float         # seconds
    first_word_start: float   # seconds
    first_word_end: float     # seconds
    method: str
    confidence: float         # [0..1]
    uncertainty_ms: float     # ± ms


@dataclass
class FrameMappingResult:
    """Result of mapping refined timestamp to a video frame."""
    frame_number: int
    timestamp: float          # actual frame PTS in seconds
    method: str
    confidence: float
    fps: float
    is_cfr: bool


@dataclass
class DialogueDetectionResult:
    """Full end-to-end result from the pipeline."""
    status: str
    error_message: Optional[str]
    video_url: str
    target_dialogue: str
    detected_dialogue: str
    candidate_start_seconds: float
    candidate_end_seconds: float
    refined_start_seconds: float
    refined_end_seconds: float
    frame_number: int
    timestamp_str: str
    confidence: float
    uncertainty_ms: float
    frame_image_path: Optional[str]
    methods: Dict[str, str] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
    all_candidates: List[DialogueMatch] = field(default_factory=list)
