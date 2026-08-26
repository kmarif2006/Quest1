"""
pipeline.py
-----------
End-to-end orchestration for the Dialogue Frame Finder.

This module is the single source of truth for the pipeline logic.
Both main.py (CLI) and app.py (Streamlit frontend) call this.

Pipeline stages:
  1. Download / use cached video
  2. Analyze video metadata
  3. Extract audio (if not cached)
  4. Transcribe audio (Faster-Whisper, with caching)
  5. Match target dialogue in transcript
  6. Refine timestamp (word-level alignment, artifact detection)
  7. Map refined timestamp to video frame (PyAV sequential decode)
  8. Extract result frame as PNG
  9. Calculate confidence
  10. Generate result JSON
"""

from __future__ import annotations

import json
import os
from typing import Optional, Callable, List

from src.downloader import VideoDownloader
from src.video_analyzer import VideoAnalyzer
from src.audio_extractor import AudioExtractor
from src.transcriber import Transcriber
from src.dialogue_matcher import DialogueMatcher
from src.timestamp_refiner import TimestampRefiner
from src.frame_mapper import FrameMapper
from src.frame_extractor import FrameExtractor
from src.confidence import compute_confidence, describe_confidence
from src.models import DialogueDetectionResult, DialogueMatch


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class DialogueFramePipeline:
    """
    Orchestrates the complete dialogue-frame-finding pipeline.

    Args:
        model_size: Whisper model size.
        output_dir: directory for output files.
        enable_ocr: whether to run OCR verification (disabled by default).
        debug: whether to enable debug output/extra frame extraction.
        force_transcribe: ignore transcription cache and re-run Whisper.
        force_download: ignore cached video and re-download.
    """

    def __init__(
        self,
        model_size: str = "small",
        output_dir: str = "output",
        enable_ocr: bool = False,
        debug: bool = False,
        force_transcribe: bool = False,
        force_download: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.model_size = model_size
        self.output_dir = output_dir
        self.enable_ocr = enable_ocr
        self.debug = debug
        self.force_transcribe = force_transcribe
        self.force_download = force_download
        self.progress = progress_callback or print

        os.makedirs(output_dir, exist_ok=True)

    def _log(self, message: str):
        self.progress(message)

    def run(
        self,
        video_url: str,
        target_dialogue: str,
    ) -> DialogueDetectionResult:
        """
        Execute the full pipeline.

        Args:
            video_url: URL of the video to process.
            target_dialogue: the phrase to search for.

        Returns:
            DialogueDetectionResult with all output fields populated.
        """
        debug_info = {}

        # ------------------------------------------------------------------
        # STAGE 1: Download / locate video
        # ------------------------------------------------------------------
        self._log("\n[1/9] Locating video...")
        downloader = VideoDownloader()
        if self.force_download:
            # Remove cached files to force re-download
            for ext in ["mp4", "mkv", "webm"]:
                p = f"data/video.{ext}"
                if os.path.exists(p):
                    os.remove(p)

        try:
            video_path = downloader.download(video_url)
        except Exception as e:
            return DialogueDetectionResult(
                status="error",
                error_message=f"Video download failed: {e}",
                video_url=video_url,
                target_dialogue=target_dialogue,
                detected_dialogue="",
                candidate_start_seconds=0.0,
                candidate_end_seconds=0.0,
                refined_start_seconds=0.0,
                refined_end_seconds=0.0,
                frame_number=0,
                timestamp_str="00:00:00.000",
                confidence=0.0,
                uncertainty_ms=0.0,
                frame_image_path=None,
            )

        self._log(f"  Video: {video_path}")

        # ------------------------------------------------------------------
        # STAGE 2: Analyze video metadata
        # ------------------------------------------------------------------
        self._log("\n[2/9] Analyzing video metadata...")
        analyzer = VideoAnalyzer(video_path)
        metadata = analyzer.get_metadata()
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        self._log(
            f"  FPS: {fps:.3f} | Frames: {total_frames} | "
            f"Duration: {_format_timestamp(metadata['duration_seconds'])}"
        )
        debug_info["video_metadata"] = metadata

        # ------------------------------------------------------------------
        # STAGE 3: Extract audio
        # ------------------------------------------------------------------
        self._log("\n[3/9] Extracting audio...")
        extractor = AudioExtractor()
        audio_path = extractor.extract(video_path)
        self._log(f"  Audio: {audio_path}")

        # ------------------------------------------------------------------
        # STAGE 4: Transcribe
        # ------------------------------------------------------------------
        self._log("\n[4/9] Transcribing audio...")
        transcriber = Transcriber(
            model_size=self.model_size,
            word_timestamps=True,
            progress_callback=self._log,
        )
        segments = transcriber.transcribe(
            audio_path,
            force=self.force_transcribe,
        )
        # Keep the loaded model alive so the refiner can reuse it
        _loaded_whisper_model = transcriber._model
        self._log(f"  Total segments: {len(segments)}")

        # ------------------------------------------------------------------
        # STAGE 5: Match target dialogue
        # ------------------------------------------------------------------
        self._log("\n[5/9] Matching target dialogue...")
        matcher = DialogueMatcher()
        matches = matcher.find_matches(segments, target_dialogue)

        if not matches:
            return DialogueDetectionResult(
                status="not_found",
                error_message=(
                    f"Target dialogue not found in transcript. "
                    f"The phrase '{target_dialogue}' was not detected in the audio."
                ),
                video_url=video_url,
                target_dialogue=target_dialogue,
                detected_dialogue="",
                candidate_start_seconds=0.0,
                candidate_end_seconds=0.0,
                refined_start_seconds=0.0,
                refined_end_seconds=0.0,
                frame_number=0,
                timestamp_str="00:00:00.000",
                confidence=0.0,
                uncertainty_ms=0.0,
                frame_image_path=None,
            )

        best_match = matches[0]
        self._log(
            f"  Best match (score={best_match.score:.1f}, "
            f"coverage={best_match.token_coverage:.0%}): "
            f"\"{best_match.text[:80]}\""
        )
        self._log(
            f"  Candidate: {_format_timestamp(best_match.start)} -> "
            f"{_format_timestamp(best_match.end)}"
        )

        status = "success"
        if len(matches) > 1 and matches[1].score > best_match.score * 0.90:
            status = "ambiguous"
            self._log(
                f"  [AMBIGUOUS] Multiple high-scoring candidates found."
            )

        debug_info["all_matches"] = [
            {
                "text": m.text[:100],
                "score": m.score,
                "coverage": m.token_coverage,
                "start": m.start,
                "end": m.end,
            }
            for m in matches
        ]

        # ------------------------------------------------------------------
        # STAGE 6: Refine timestamp
        # ------------------------------------------------------------------
        self._log("\n[6/9] Refining timestamp...")
        refiner = TimestampRefiner(
            model_size=self.model_size,
            whisper_model=_loaded_whisper_model,
            progress_callback=self._log,
        )
        refined = refiner.refine(
            segments=segments,
            coarse_start=best_match.start,
            coarse_end=best_match.end,
            target_dialogue=target_dialogue,
            audio_path=audio_path,
        )

        self._log(f"  Method: {refined.method}")
        self._log(
            f"  Phrase start: {_format_timestamp(refined.phrase_start)} "
            f"(+/-{refined.uncertainty_ms:.0f}ms)"
        )
        self._log(f"  Alignment confidence: {refined.confidence:.0%}")

        debug_info["refinement"] = {
            "phrase_start": refined.phrase_start,
            "phrase_end": refined.phrase_end,
            "first_word_start": refined.first_word_start,
            "first_word_end": refined.first_word_end,
            "method": refined.method,
            "confidence": refined.confidence,
            "uncertainty_ms": refined.uncertainty_ms,
        }

        # ------------------------------------------------------------------
        # STAGE 7: Map to video frame
        # ------------------------------------------------------------------
        self._log("\n[7/9] Mapping to video frame...")
        frame_mapper = FrameMapper()
        frame_result = frame_mapper.map(
            video_path=video_path,
            target_time=refined.phrase_start,
            fps=fps,
        )

        self._log(f"  Method: {frame_result.method}")
        self._log(
            f"  Frame: {frame_result.frame_number} "
            f"at {_format_timestamp(frame_result.timestamp)} "
            f"(FPS={frame_result.fps:.3f}, CFR={frame_result.is_cfr})"
        )

        debug_info["frame_mapping"] = {
            "frame_number": frame_result.frame_number,
            "timestamp": frame_result.timestamp,
            "method": frame_result.method,
            "confidence": frame_result.confidence,
            "fps": frame_result.fps,
            "is_cfr": frame_result.is_cfr,
        }

        # ------------------------------------------------------------------
        # STAGE 8: Extract result frame
        # ------------------------------------------------------------------
        self._log("\n[8/9] Extracting result frame...")
        result_frame_path = os.path.join(self.output_dir, "result_frame.png")
        frame_extractor = FrameExtractor()

        try:
            frame_extractor.extract_single_frame(
                video_path=video_path,
                frame_number=frame_result.frame_number,
                output_path=result_frame_path,
            )
            self._log(f"  Saved: {result_frame_path}")
        except RuntimeError as e:
            self._log(f"  [WARNING] Frame extraction failed: {e}")
            result_frame_path = None

        # Optional debug: extract nearby frames
        if self.debug and result_frame_path:
            self._log("  [DEBUG] Extracting +/-1s nearby frames...")
            debug_extractor = FrameExtractor(
                output_dir=os.path.join(self.output_dir, "frames")
            )
            debug_extractor.extract_range(
                video_path=video_path,
                start_time=max(0.0, refined.phrase_start - 1.0),
                end_time=refined.phrase_start + 1.0,
                fps=fps,
                step=1,
            )

        # ------------------------------------------------------------------
        # STAGE 9: Calculate confidence and generate result
        # ------------------------------------------------------------------
        self._log("\n[9/9] Calculating confidence...")
        overall_confidence, overall_uncertainty_ms = compute_confidence(
            match=best_match,
            refined=refined,
            frame_result=frame_result,
        )

        self._log(
            f"  Overall confidence: {overall_confidence:.0%} "
            f"({describe_confidence(overall_confidence)})"
        )
        self._log(f"  Timing uncertainty: +/-{overall_uncertainty_ms:.0f}ms")

        # ------------------------------------------------------------------
        # Generate result.json
        # ------------------------------------------------------------------
        result = DialogueDetectionResult(
            status=status,
            error_message=None,
            video_url=video_url,
            target_dialogue=target_dialogue,
            detected_dialogue=best_match.text,
            candidate_start_seconds=best_match.start,
            candidate_end_seconds=best_match.end,
            refined_start_seconds=refined.phrase_start,
            refined_end_seconds=refined.phrase_end,
            frame_number=frame_result.frame_number,
            timestamp_str=_format_timestamp(refined.phrase_start),
            confidence=overall_confidence,
            uncertainty_ms=overall_uncertainty_ms,
            frame_image_path=result_frame_path,
            methods={
                "transcription": f"faster-whisper/{self.model_size}",
                "alignment": refined.method,
                "frame_mapping": frame_result.method,
            },
            debug=debug_info if self.debug else {},
            all_candidates=matches,
        )

        self._save_result_json(result)

        return result

    def _save_result_json(self, result: DialogueDetectionResult) -> None:
        """Save the result to output/result.json."""
        output_path = os.path.join(self.output_dir, "result.json")

        data = {
            "status": result.status,
            "video_url": result.video_url,
            "target_dialogue": result.target_dialogue,
            "detected_dialogue": result.detected_dialogue,
            "candidate_start_seconds": result.candidate_start_seconds,
            "candidate_end_seconds": result.candidate_end_seconds,
            "refined_start_seconds": result.refined_start_seconds,
            "refined_end_seconds": result.refined_end_seconds,
            "timestamp": result.timestamp_str,
            "frame_number": result.frame_number,
            "confidence": result.confidence,
            "uncertainty_ms": result.uncertainty_ms,
            "frame_image": result.frame_image_path,
            "methods": result.methods,
        }

        if result.error_message:
            data["error_message"] = result.error_message

        if result.debug:
            data["debug"] = result.debug

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        self._log(f"\n  Result JSON: {output_path}")
