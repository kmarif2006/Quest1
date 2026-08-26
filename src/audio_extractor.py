"""
audio_extractor.py
------------------
Extracts 16kHz mono 16-bit PCM WAV audio from any video file using FFmpeg.
Caches audio per video file so multiple videos don't overwrite each other.
"""

import os
import subprocess
from pathlib import Path


class AudioExtractor:

    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract(self, video_path: str) -> str:
        """
        Extract audio to a WAV file derived from the video filename.
        """
        video_name = Path(video_path).stem

        # For the default data/video.mp4 benchmark, reuse data/audio.wav if present
        if video_name == "video" and os.path.exists(os.path.join(self.output_dir, "audio.wav")):
            audio_path = os.path.join(self.output_dir, "audio.wav")
        else:
            audio_path = os.path.join(self.output_dir, f"{video_name}.wav")

        # Use existing audio if available and non-empty
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            print("\nExisting audio found:")
            print(audio_path)
            return audio_path

        print(f"\nExtracting audio from video: {video_path}")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            audio_path
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as error:
            print("\nFFmpeg audio extraction failed.")
            print(error.stderr)
            raise

        print("\nAudio extraction completed:")
        print(audio_path)

        return audio_path