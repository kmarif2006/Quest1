"""
downloader.py
-------------
Downloads video from a media URL using yt-dlp or handles local video files.
Supports caching per URL so custom videos don't collide with existing media.

Robustness improvements:
  - Browser-like headers to avoid server-side connection resets (e.g. ok.ru)
  - Exponential-backoff retry loop for transient ConnectionResetError / 10054
  - HTTP-only format fallback when best+audio merge fails
  - Optional cookies file support (cookies.txt next to script)
"""

import os
import hashlib
import time
from typing import Optional
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


# Default benchmark URL for backward-compatibility with data/video.mp4
BENCHMARK_URL = "https://ok.ru/video/248244667877"


class VideoDownloader:

    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_url_hash(self, url: str) -> str:
        """Compute short deterministic hash for the URL."""
        return hashlib.md5(url.strip().encode("utf-8")).hexdigest()[:10]

    def get_existing_video(self, url: Optional[str] = None) -> Optional[str]:
        """
        Check if video for this URL or path already exists on disk.
        """
        if not url:
            return None

        # 1. If it's a local file path that exists, return it directly
        if os.path.isfile(url):
            return os.path.abspath(url)

        # 2. If it's the benchmark URL, check default data/video.* files
        if url.strip() == BENCHMARK_URL:
            for ext in ["mp4", "mkv", "webm"]:
                path = os.path.join(self.output_dir, f"video.{ext}")
                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    return path

        # 3. Check for URL-specific cached file
        url_hash = self._get_url_hash(url)
        for ext in ["mp4", "mkv", "webm"]:
            hashed_path = os.path.join(self.output_dir, f"video_{url_hash}.{ext}")
            if os.path.exists(hashed_path) and os.path.getsize(hashed_path) > 1000:
                return hashed_path

        return None

    def download(self, url: str) -> str:
        """
        Download or return existing video file for the given URL / local path.
        """
        # If user passed a local file path
        if os.path.isfile(url):
            print(f"\nUsing local video file: {url}")
            return os.path.abspath(url)

        existing_video = self.get_existing_video(url)
        if existing_video:
            print("\nExisting cached video found:")
            print(existing_video)
            return existing_video

        url_hash = self._get_url_hash(url)
        output_template = os.path.join(
            self.output_dir,
            f"video_{url_hash}.%(ext)s"
        )

        # Build a browser-like User-Agent to avoid server-side connection resets.
        # Many platforms (e.g. ok.ru, VK) aggressively close non-browser connections.
        _USER_AGENT = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )

        _BASE_HEADERS = {
            "User-Agent": _USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        # Attempt up to 3 download strategies before giving up.
        # Strategy order: best quality → http-only fallback → worst single-stream
        _format_strategies = [
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
            "best[ext=mp4]/best",
            "worst",
        ]

        # Optional cookies file located next to this project root
        _cookies_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cookies.txt",
        )
        _cookiefile_opt = _cookies_file if os.path.isfile(_cookies_file) else None

        base_options = {
            "outtmpl": output_template,
            "noplaylist": True,
            "retries": 5,
            "fragment_retries": 5,
            "socket_timeout": 45,
            "force_ipv4": True,
            "quiet": False,
            "merge_output_format": "mp4",
            "http_headers": _BASE_HEADERS,
            # Keep connections alive like a browser
            "keepvideo": False,
            # Throttle slightly so the server doesn't rate-limit us
            "sleep_interval": 1,
            "max_sleep_interval": 3,
        }
        if _cookiefile_opt:
            base_options["cookiefile"] = _cookiefile_opt
            print(f"  [downloader] Using cookies from {_cookiefile_opt}")

        print(f"\nDownloading video for URL: {url}")

        last_error: Optional[Exception] = None

        for attempt, fmt in enumerate(_format_strategies, start=1):
            options = {**base_options, "format": fmt}
            backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s between retries

            try:
                print(f"  [downloader] Attempt {attempt}/3 — format: {fmt!r}")
                with YoutubeDL(options) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)

                base_name = os.path.splitext(filename)[0]
                mp4_candidate = f"{base_name}.mp4"
                if os.path.exists(mp4_candidate):
                    return mp4_candidate
                if os.path.exists(filename):
                    return filename

                # Fallback: scan output_dir for video_{url_hash}.*
                for ext in ["mp4", "mkv", "webm"]:
                    path = os.path.join(self.output_dir, f"video_{url_hash}.{ext}")
                    if os.path.exists(path):
                        return path

                return filename

            except DownloadError as exc:
                last_error = exc
                err_str = str(exc)

                # ConnectionResetError(10054) and similar transient errors — worth retrying
                is_transient = any(
                    token in err_str
                    for token in (
                        "ConnectionResetError",
                        "Connection aborted",
                        "10054",
                        "10060",
                        "Connection refused",
                        "RemoteDisconnected",
                        "IncompleteRead",
                        "timed out",
                    )
                )

                if is_transient and attempt < len(_format_strategies):
                    print(
                        f"  [downloader] Transient network error on attempt {attempt}. "
                        f"Retrying in {backoff}s with fallback format..."
                    )
                    time.sleep(backoff)
                    continue

                # Non-transient or final attempt — surface the error
                print("\nDOWNLOAD FAILED")
                print(exc)
                print(
                    "\nTroubleshooting tips:"
                    "\n  1. Download the video manually (e.g. via a browser extension "
                    "or yt-dlp from your terminal) and pass the local .mp4 path instead."
                    "\n  2. Create a 'cookies.txt' file (Netscape format) in the project "
                    "root — export it from your browser while logged into the site."
                    "\n  3. Try updating yt-dlp:  pip install -U yt-dlp"
                    "\n  4. Some sites block all automated downloads regardless of headers."
                )
                raise

        # Should not reach here, but raise the last captured error just in case
        raise last_error  # type: ignore[misc]