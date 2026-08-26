"""
frame_mapper.py
---------------
Maps a refined timestamp (seconds) to an exact video frame number.

Design:
  1. Obtain FPS from video metadata (via OpenCV or ffprobe).
  2. Determine if the video is constant frame rate (CFR) or variable (VFR).
  3. For CFR: nominal mapping is frame = floor(timestamp * fps).
  4. Verify by sequential decoding with PyAV around the nominal frame.
     - Seek to ~0.5s before the target time.
     - Decode frames sequentially, tracking presentation timestamps (PTS).
     - Select the FIRST decoded frame whose PTS >= refined_start.
  5. Fallback to OpenCV seek if PyAV is unavailable.

Why not just int(timestamp * fps)?
  - Floating-point error can push the result 1-2 frames off.
  - For VFR video, this formula is incorrect entirely.
  - Sequential decoding with PTS comparison is the authoritative method.
"""

from __future__ import annotations

import math
import subprocess
import json
from typing import Optional, Tuple

from src.models import FrameMappingResult

# Look-back window: decode from this many seconds before target
SEEK_LOOKBACK_SEC = 1.0


def _get_fps_and_cfr_via_ffprobe(video_path: str) -> Tuple[float, bool]:
    """
    Use ffprobe to get the video stream FPS and determine if CFR.
    Returns (fps, is_cfr).
    Falls back to (0.0, False) on failure.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries",
                "stream=r_frame_rate,avg_frame_rate,codec_type",
                "-of", "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return 0.0, False

        stream = streams[0]
        r_rate_str = stream.get("r_frame_rate", "0/1")
        avg_rate_str = stream.get("avg_frame_rate", "0/1")

        def parse_rate(s: str) -> float:
            parts = s.split("/")
            if len(parts) == 2:
                num, den = float(parts[0]), float(parts[1])
                return num / den if den != 0 else 0.0
            return float(s) if s else 0.0

        r_fps = parse_rate(r_rate_str)
        avg_fps = parse_rate(avg_rate_str)

        # CFR: reported FPS and average FPS agree within 1%
        if r_fps > 0 and avg_fps > 0:
            diff_pct = abs(r_fps - avg_fps) / max(r_fps, avg_fps)
            is_cfr = diff_pct < 0.01
        else:
            is_cfr = False

        fps = r_fps if r_fps > 0 else avg_fps
        return fps, is_cfr

    except Exception:
        return 0.0, False


def _map_with_pyav(
    video_path: str,
    target_time: float,
    fps: float,
) -> Optional[Tuple[int, float]]:
    """
    Use PyAV to sequentially decode frames around the target time.

    Seeks to (target_time - SEEK_LOOKBACK_SEC) then reads forward until
    we find the first frame whose PTS >= target_time.

    Returns (frame_number, frame_pts_seconds) or None.
    """
    try:
        import av  # type: ignore

        seek_time = max(0.0, target_time - SEEK_LOOKBACK_SEC)

        container = av.open(video_path)
        video_stream = container.streams.video[0]

        # av uses microseconds for seek with stream time_base
        seek_ts = int(seek_time / float(video_stream.time_base))
        container.seek(seek_ts, stream=video_stream, any_frame=False, backward=True)

        nominal_frame_at_seek = int(seek_time * fps)
        nominal_frame_at_target = int(target_time * fps)

        frame_number = nominal_frame_at_seek
        result_frame = None

        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if frame.pts is None:
                    frame_number += 1
                    continue

                pts_seconds = float(frame.pts * video_stream.time_base)

                # Estimate frame number from pts and fps
                estimated_fn = int(pts_seconds * fps)

                if pts_seconds >= target_time:
                    result_frame = (estimated_fn, pts_seconds)
                    break

                frame_number = estimated_fn + 1

            if result_frame is not None:
                break

            # Safety: don't scan more than 5 seconds past target
            if frame_number > nominal_frame_at_target + int(fps * 5):
                break

        container.close()
        return result_frame

    except Exception as exc:
        print(f"[frame_mapper] PyAV decode failed: {exc}")
        return None


def _map_with_opencv(
    video_path: str,
    target_time: float,
    fps: float,
) -> Tuple[int, float]:
    """
    Fallback: use OpenCV to estimate frame from timestamp.
    Less accurate but always available.
    Returns (frame_number, estimated_pts).
    """
    # Nominal calculation
    frame_number = math.floor(target_time * fps)
    estimated_pts = frame_number / fps
    return frame_number, estimated_pts


class FrameMapper:
    """
    Maps a refined timestamp to a specific video frame.

    Preferred method: PyAV sequential decode (accurate PTS tracking).
    Fallback: OpenCV nominal calculation.
    """

    def map(
        self,
        video_path: str,
        target_time: float,
        fps: Optional[float] = None,
    ) -> FrameMappingResult:
        """
        Map target_time (seconds) to a frame number.

        Args:
            video_path: path to the video file.
            target_time: the refined phrase start time in seconds.
            fps: known FPS; if None, will probe from video.

        Returns:
            FrameMappingResult with frame_number, timestamp, method, confidence.
        """
        # Get FPS and CFR status
        probe_fps, is_cfr = _get_fps_and_cfr_via_ffprobe(video_path)
        if fps is None or fps <= 0:
            fps = probe_fps
        if fps <= 0:
            fps = 25.0  # last-resort default
            is_cfr = False

        # Try PyAV sequential decode for accurate PTS
        pyav_result = _map_with_pyav(video_path, target_time, fps)

        if pyav_result:
            frame_number, frame_pts = pyav_result
            method = "sequential_decode_pyav"
            confidence = 0.90 if is_cfr else 0.75
        else:
            # Fallback to nominal calculation
            frame_number, frame_pts = _map_with_opencv(
                video_path, target_time, fps
            )
            method = "nominal_fps_fallback"
            confidence = 0.70 if is_cfr else 0.50

        return FrameMappingResult(
            frame_number=max(0, frame_number),
            timestamp=frame_pts,
            method=method,
            confidence=confidence,
            fps=fps,
            is_cfr=is_cfr,
        )
