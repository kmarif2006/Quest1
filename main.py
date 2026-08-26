"""
main.py
-------
CLI entry point for the Dialogue Frame Finder.

Usage:
    python main.py "<video_url>" "<target_dialogue>" [options]

Options:
    --model {tiny,base,small,medium,large-v3}  Whisper model size (default: small)
    --debug                                     Enable debug output and extra frame extraction
    --enable-ocr                                Enable OCR verification (disabled by default)
    --force-transcribe                          Ignore transcript cache and re-run Whisper
    --force-download                            Ignore cached video and re-download

Example:
    python main.py "https://ok.ru/video/248244667877" "My mind rebels at stagnation"
"""

import argparse
import sys

# Ensure Windows terminal doesn't crash on unicode
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.pipeline import DialogueFramePipeline
from src.confidence import describe_confidence


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def print_separator(char: str = "=", width: int = 56):
    print(char * width)


def print_header():
    print_separator()
    print("  DIALOGUE FRAME FINDER")
    print_separator()


def print_final_result(result):
    print()
    print_separator()
    print("  FINAL RESULT")
    print_separator()

    status_display = {
        "success": "SUCCESS",
        "ambiguous": "AMBIGUOUS (multiple candidates)",
        "not_found": "NOT FOUND",
        "error": "ERROR",
    }.get(result.status, result.status.upper())

    print(f"\nStatus               : {status_display}")

    if result.error_message:
        print(f"\nError                : {result.error_message}")
        print_separator()
        return

    print(f"\nTarget Dialogue      : \"{result.target_dialogue}\"")
    print(f"\nDetected Dialogue    : \"{result.detected_dialogue[:120]}\"")

    print(f"\nBest Estimated Dialogue Start:")
    print(f"  {result.timestamp_str}")
    print(f"  (Candidate window: {format_timestamp(result.candidate_start_seconds)} "
          f"-> {format_timestamp(result.candidate_end_seconds)})")

    print(f"\nFrame Number         : {result.frame_number}")
    print(f"  (Best estimated first frame - not guaranteed exact)")

    print(f"\nConfidence           : {result.confidence:.0%} "
          f"({describe_confidence(result.confidence)})")
    print(f"Timing Uncertainty   : +/-{result.uncertainty_ms:.0f} ms")

    print(f"\nAlignment Method     : {result.methods.get('alignment', 'unknown')}")
    print(f"Frame Map Method     : {result.methods.get('frame_mapping', 'unknown')}")
    print(f"Transcription        : {result.methods.get('transcription', 'unknown')}")

    if result.frame_image_path:
        print(f"\nFrame Image          : {result.frame_image_path}")
    else:
        print(f"\nFrame Image          : [not extracted]")

    print(f"\nResult JSON          : output/result.json")

    if result.status == "ambiguous":
        print(f"\n--- All candidates ---")
        for i, cand in enumerate(result.all_candidates[:5], 1):
            print(
                f"  #{i}: score={cand.score:.1f}, "
                f"coverage={cand.token_coverage:.0%}, "
                f"start={format_timestamp(cand.start)}, "
                f"text=\"{cand.text[:60]}\""
            )

    print_separator()


def main():
    parser = argparse.ArgumentParser(
        description="Dialogue Frame Finder — locates a spoken dialogue in a video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "video_url",
        help="URL of the video to search",
    )
    parser.add_argument(
        "target_dialogue",
        help="The spoken dialogue phrase to find",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="small",
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output and extra frame extraction",
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable OCR verification (disabled by default; not needed for speech)",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Ignore cached transcript and re-run Whisper",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached video and re-download",
    )

    args = parser.parse_args()

    print_header()
    print(f"\nVideo URL  : {args.video_url}")
    print(f"Target     : \"{args.target_dialogue}\"")
    print(f"Model      : {args.model}")
    print(f"Debug      : {args.debug}")
    print(f"OCR        : {args.enable_ocr}")
    print()
    print("Searching for dialogue...")
    print_separator("-")

    pipeline = DialogueFramePipeline(
        model_size=args.model,
        enable_ocr=args.enable_ocr,
        debug=args.debug,
        force_transcribe=args.force_transcribe,
        force_download=args.force_download,
        progress_callback=print,
    )

    try:
        result = pipeline.run(
            video_url=args.video_url,
            target_dialogue=args.target_dialogue,
        )
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    print_final_result(result)

    # Exit code
    if result.status == "error":
        sys.exit(2)
    elif result.status == "not_found":
        sys.exit(3)


if __name__ == "__main__":
    main()