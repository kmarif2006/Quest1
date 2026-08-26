"""
frame_extractor.py
------------------
Extracts frames from video files using OpenCV.

Primary method: extract_single_frame
  - Seeks to the specified frame number using CAP_PROP_POS_FRAMES
  - Saves the frame as PNG to the specified path

Debug method: extract_range
  - Extracts a range of frames (only in debug/verification mode)
  - Should NOT be called in normal production pipeline

Note: CAP_PROP_POS_FRAMES seek in OpenCV is not guaranteed to be
frame-accurate for all codecs. For critical precision, use PyAV
(see frame_mapper.py). However, for extracting a known frame number
that was already derived accurately, this is sufficient.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import cv2


class FrameExtractor:
    """Extract individual frames or frame ranges from a video file."""

    def __init__(self, output_dir: str = "output/frames"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_single_frame(
        self,
        video_path: str,
        frame_number: int,
        output_path: str,
    ) -> str:
        """
        Extract a single frame by frame number and save to output_path.

        Args:
            video_path: path to the video file.
            frame_number: 0-indexed frame number to extract.
            output_path: full path (including filename) to save the PNG.

        Returns:
            The output_path if successful.

        Raises:
            RuntimeError: if the video cannot be opened or frame cannot be read.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        try:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video.read()

            if not success:
                raise RuntimeError(
                    f"Could not read frame {frame_number} from {video_path}"
                )

            ok = cv2.imwrite(output_path, frame)
            if not ok:
                raise RuntimeError(f"Could not write frame to {output_path}")

        finally:
            video.release()

        return output_path

    def extract_frame_by_number(
        self,
        video_path: str,
        frame_number: int,
        filename: Optional[str] = None,
    ) -> str:
        """
        Extract a frame and save to the output_dir.
        Kept for backward compatibility.
        """
        if filename is None:
            filename = f"frame_{frame_number}.png"

        output_path = os.path.join(self.output_dir, filename)
        return self.extract_single_frame(video_path, frame_number, output_path)

    def extract_range(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        fps: float,
        step: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Extract a range of frames between start_time and end_time.
        Used for debugging/verification only — not in the default pipeline.

        Args:
            video_path: path to video.
            start_time: start time in seconds.
            end_time: end time in seconds.
            fps: frames per second.
            step: extract every Nth frame.

        Returns:
            List of dicts with 'frame', 'path', 'timestamp' keys.
        """
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        saved_frames = []

        for frame_number in range(start_frame, end_frame + 1, step):
            filename = f"frame_{frame_number}.png"
            output_path = os.path.join(self.output_dir, filename)

            try:
                self.extract_single_frame(video_path, frame_number, output_path)
                saved_frames.append({
                    "frame": frame_number,
                    "path": output_path,
                    "timestamp": frame_number / fps,
                })
            except RuntimeError as e:
                print(f"[frame_extractor] Skipping frame {frame_number}: {e}")

        return saved_frames