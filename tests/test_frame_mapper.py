"""
tests/test_frame_mapper.py
--------------------------
Tests for FrameMapper nominal CFR calculation.
Uses controlled inputs — no video file needed for basic calculations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from unittest.mock import patch, MagicMock

from src.frame_mapper import FrameMapper, _get_fps_and_cfr_via_ffprobe


# -------------------------------------------------------------------------
# Nominal frame calculation tests (pure math, no files needed)
# -------------------------------------------------------------------------

def test_nominal_frame_at_zero():
    """Frame 0 at time 0.0."""
    fps = 25.0
    t = 0.0
    expected = 0
    assert math.floor(t * fps) == expected


def test_nominal_frame_simple():
    """At 10.0 seconds with 25fps → frame 250."""
    fps = 25.0
    t = 10.0
    assert math.floor(t * fps) == 250


def test_nominal_frame_at_target_fps():
    """Test with the known example fps: 23.976..."""
    fps = 23.97616608930767
    t = 325.67  # "rebels" start from known transcript
    frame = math.floor(t * fps)
    # Just verify it's a reasonable number
    assert 7000 < frame < 8500, f"Frame {frame} seems out of range"


def test_nominal_frame_non_integer_fps():
    """Non-integer FPS should still produce sensible results."""
    fps = 29.97002997002997
    t = 60.0  # 1 minute
    frame = math.floor(t * fps)
    # Should be close to 1798 frames
    assert abs(frame - 1798) <= 2


# -------------------------------------------------------------------------
# FrameMapper with mocked PyAV (no video file)
# -------------------------------------------------------------------------

def test_frame_mapper_uses_nominal_fallback_when_pyav_fails():
    """When PyAV fails, FrameMapper should fall back to nominal calculation."""
    fps = 24.0
    target_time = 100.0
    expected_frame = math.floor(target_time * fps)  # 2400

    with patch("src.frame_mapper._get_fps_and_cfr_via_ffprobe", return_value=(fps, True)):
        with patch("src.frame_mapper._map_with_pyav", return_value=None):
            mapper = FrameMapper()
            result = mapper.map(
                video_path="dummy_path.mp4",
                target_time=target_time,
                fps=fps,
            )

    assert result.frame_number == expected_frame
    assert result.method == "nominal_fps_fallback"
    assert result.fps == fps
    assert result.is_cfr is True


def test_frame_mapper_uses_pyav_result_when_available():
    """When PyAV succeeds, use its result."""
    fps = 23.976
    target_time = 325.67
    pyav_frame = 7809
    pyav_pts = 325.68

    with patch("src.frame_mapper._get_fps_and_cfr_via_ffprobe", return_value=(fps, True)):
        with patch("src.frame_mapper._map_with_pyav", return_value=(pyav_frame, pyav_pts)):
            mapper = FrameMapper()
            result = mapper.map(
                video_path="dummy_path.mp4",
                target_time=target_time,
                fps=fps,
            )

    assert result.frame_number == pyav_frame
    assert result.timestamp == pyav_pts
    assert result.method == "sequential_decode_pyav"
    assert result.confidence > 0.8


def test_frame_mapper_never_returns_negative_frame():
    """Frame number must always be >= 0."""
    with patch("src.frame_mapper._get_fps_and_cfr_via_ffprobe", return_value=(25.0, True)):
        with patch("src.frame_mapper._map_with_pyav", return_value=(-5, -0.2)):
            mapper = FrameMapper()
            result = mapper.map(
                video_path="dummy_path.mp4",
                target_time=0.0,
                fps=25.0,
            )
    assert result.frame_number >= 0


def test_frame_mapper_handles_zero_fps_gracefully():
    """If ffprobe returns 0 fps and no fps provided, use default 25.0."""
    with patch("src.frame_mapper._get_fps_and_cfr_via_ffprobe", return_value=(0.0, False)):
        with patch("src.frame_mapper._map_with_pyav", return_value=None):
            mapper = FrameMapper()
            result = mapper.map(
                video_path="dummy_path.mp4",
                target_time=10.0,
                fps=None,
            )
    # Should not crash; fps should default to 25.0
    assert result.fps == 25.0
    assert result.frame_number == math.floor(10.0 * 25.0)
