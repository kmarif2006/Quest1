"""
app.py
------
Streamlit frontend for the Dialogue Frame Finder.

Run with:
    streamlit run app.py

This frontend delegates all processing to the pipeline module.
No core logic is duplicated here.
"""

import os
import sys
import time
import json
from pathlib import Path

import streamlit as st
from PIL import Image

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import DialogueFramePipeline
from src.confidence import describe_confidence


# -------------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------------

st.set_page_config(
    page_title="Dialogue Frame Finder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------------------
# Custom CSS for premium look
# -------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%);
    min-height: 100vh;
}

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%);
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 1rem 0 0.5rem 0;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}

.metric-chip {
    background: rgba(96, 165, 250, 0.1);
    border: 1px solid rgba(96, 165, 250, 0.3);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    color: #93c5fd;
    font-size: 0.85rem;
    font-weight: 500;
}

.status-success {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #4ade80;
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-weight: 600;
    display: inline-block;
}

.status-ambiguous {
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid rgba(251, 191, 36, 0.4);
    color: #fbbf24;
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-weight: 600;
    display: inline-block;
}

.status-error {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #f87171;
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-weight: 600;
    display: inline-block;
}

.timestamp-display {
    font-family: 'Courier New', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: #60a5fa;
    text-align: center;
    padding: 1rem;
}

.frame-number-display {
    font-size: 1.3rem;
    color: #a78bfa;
    text-align: center;
    font-weight: 600;
}

.confidence-bar-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-bottom: 0.2rem;
}

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

.candidate-row {
    background: rgba(255,255,255,0.02);
    border-left: 3px solid #a78bfa;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: #cbd5e1;
}

.log-output {
    background: #0a0a0a;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #a3e635;
    max-height: 300px;
    overflow-y: auto;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------

st.markdown('<div class="hero-title">🎬 Dialogue Frame Finder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Find the exact video frame where a spoken dialogue begins — '
    'powered by Whisper speech recognition.'
    '</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# Input form
# -------------------------------------------------------------------------

col_left, col_right = st.columns([2, 1])

with col_left:
    video_url = st.text_input(
        "🔗 Video URL or Local File Path",
        placeholder="https://ok.ru/video/248244667877 or data/video.mp4",
        help="Enter any publicly accessible video URL or a path to a local .mp4/.mkv file",
        key="video_url_input",
    )

    target_dialogue = st.text_input(
        "💬 Target Dialogue",
        placeholder="My mind rebels at stagnation",
        help="The spoken phrase to search for in the video",
        key="target_dialogue_input",
    )

with col_right:
    model_size = st.selectbox(
        "🧠 Whisper Model",
        options=["small", "base", "tiny", "medium", "large-v3"],
        index=0,
        help="Larger models are more accurate but slower",
    )

    force_transcribe = st.checkbox(
        "🔄 Force re-transcribe",
        value=False,
        help="Ignore cached transcript and re-transcribe the audio",
    )

    enable_ocr = st.checkbox(
        "🔍 Enable OCR verification",
        value=False,
        help="Scan frames for burned-in subtitles (usually not needed)",
    )

    debug_mode = st.checkbox(
        "🐛 Debug mode",
        value=False,
        help="Extract nearby frames and show detailed output",
    )

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Action button
# -------------------------------------------------------------------------

col_btn, col_warn = st.columns([1, 3])
with col_btn:
    run_button = st.button(
        "🔎 Find Dialogue Frame",
        type="primary",
        use_container_width=True,
        disabled=not (video_url and target_dialogue),
    )

if not video_url or not target_dialogue:
    st.info("Enter a video URL and target dialogue to begin.")

# -------------------------------------------------------------------------
# Pipeline execution
# -------------------------------------------------------------------------

if run_button and video_url and target_dialogue:

    import queue
    import threading
    import html
    import time

    log_queue = queue.Queue()
    result_holder = {}

    def update_log(msg: str):
        log_queue.put(msg)

    def run_pipeline():
        try:
            pipeline = DialogueFramePipeline(
                model_size=model_size,
                enable_ocr=enable_ocr,
                debug=debug_mode,
                force_transcribe=force_transcribe,
                force_download=False,
                progress_callback=update_log,
            )
            result_holder["result"] = pipeline.run(
                video_url=video_url,
                target_dialogue=target_dialogue,
            )
        except Exception as e:
            result_holder["error"] = str(e)

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    log_lines = []
    log_placeholder = st.empty()
    spinner_placeholder = st.empty()

    while thread.is_alive() or not log_queue.empty():
        changed = False
        try:
            while True:
                msg = log_queue.get_nowait()
                for line in msg.split("\n"):
                    stripped = line.rstrip()
                    if stripped:
                        log_lines.append(stripped)
                changed = True
        except queue.Empty:
            pass

        if changed:
            escaped = [html.escape(l) for l in log_lines[-80:]]
            log_placeholder.markdown(
                f'<div class="log-output">' + "<br>".join(escaped) + "</div>",
                unsafe_allow_html=True,
            )

        if thread.is_alive():
            spinner_placeholder.markdown("⏳ **Pipeline running — please wait...**")
            time.sleep(0.4)

    spinner_placeholder.empty()
    thread.join()

    if "error" in result_holder:
        err_msg = result_holder["error"]
        st.error(f"Pipeline failed: {err_msg}")

        # Give actionable guidance for common download/network failures
        _download_error_tokens = (
            "ConnectionResetError", "Connection aborted", "10054",
            "Unable to download webpage", "TransportError",
            "Video download failed", "DownloadError",
        )
        if any(tok in err_msg for tok in _download_error_tokens):
            st.markdown(
                """
                <div class="result-card" style="border-color: rgba(251,191,36,0.4);">
                <span style="color:#fbbf24; font-weight:700;">⚠️ Download Blocked — Workarounds</span><br><br>

                The remote server rejected the automated download request (common on ok.ru, VK, etc.).
                Try one of the following:

                <ol style="color:#cbd5e1; margin-top:0.8rem; line-height:2;">
                  <li><strong>Use a local file</strong> — download the video manually with a browser
                  extension or yt-dlp in your terminal, save it as <code>data/video.mp4</code>,
                  then paste <code>data/video.mp4</code> into the URL field above.</li>

                  <li><strong>Use cookies</strong> — while logged into the site in your browser,
                  export cookies to a <code>cookies.txt</code> (Netscape format) using the
                  <em>Get cookies.txt LOCALLY</em> extension, and place it in the project root.
                  The downloader will pick it up automatically.</li>

                  <li><strong>Update yt-dlp</strong> — run <code>pip install -U yt-dlp</code>
                  in your virtual environment. Older versions lack newer site extractors.</li>

                  <li><strong>Try again later</strong> — the server may be rate-limiting.
                  Transient errors often resolve after a few minutes.</li>
                </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.stop()

    result = result_holder.get("result")
    if result is None:
        st.error("Pipeline returned no result. Check logs above.")
        st.stop()

    # -------------------------------------------------------------------------
    # Display results
    # -------------------------------------------------------------------------

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.subheader("📊 Results")

    # Status badge
    status_class = {
        "success": "status-success",
        "ambiguous": "status-ambiguous",
    }.get(result.status, "status-error")

    status_label = {
        "success": "✅ SUCCESS",
        "ambiguous": "⚠️ AMBIGUOUS",
        "not_found": "❌ NOT FOUND",
        "error": "🔴 ERROR",
    }.get(result.status, result.status.upper())

    st.markdown(
        f'<span class="{status_class}">{status_label}</span>',
        unsafe_allow_html=True,
    )

    if result.error_message:
        st.error(result.error_message)
    else:
        # Main result columns
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("**🎯 Target Dialogue**")
            st.info(f'"{result.target_dialogue}"')

            st.markdown("**📝 Detected Dialogue**")
            st.success(f'"{result.detected_dialogue[:150]}"')

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            st.markdown("**⏱️ Best Estimated Dialogue Start**")
            st.markdown(
                f'<div class="timestamp-display">{result.timestamp_str}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="frame-number-display">Frame #{result.frame_number}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Best estimated first frame — not guaranteed frame-exact")

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Confidence
            conf_pct = result.confidence * 100
            conf_label = describe_confidence(result.confidence)
            st.markdown(f"**🎯 Confidence: {conf_pct:.0f}% ({conf_label})**")
            st.progress(result.confidence)
            st.caption(f"Timing uncertainty: ±{result.uncertainty_ms:.0f} ms")

            # Methods
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("**🔬 Methods Used**")
            for key, val in result.methods.items():
                st.markdown(f"- **{key.replace('_', ' ').title()}**: `{val}`")

        with res_col2:
            # Frame image
            if result.frame_image_path and os.path.exists(result.frame_image_path):
                st.markdown("**🖼️ Extracted Frame**")
                try:
                    img = Image.open(result.frame_image_path)
                    st.image(img, caption=f"Frame {result.frame_number} @ {result.timestamp_str}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display frame: {e}")
            else:
                st.warning("Frame image not available.")

            # Candidate details
            if result.status == "ambiguous" and result.all_candidates:
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown("**🔀 All Candidates** (ambiguous result)")
                for i, cand in enumerate(result.all_candidates[:5], 1):
                    st.markdown(
                        f'<div class="candidate-row">'
                        f'<strong>#{i}</strong> '
                        f'Score: {cand.score:.1f} | '
                        f'Coverage: {cand.token_coverage:.0%} | '
                        f'Start: {cand.start:.2f}s<br>'
                        f'<em>"{cand.text[:80]}"</em>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Result JSON
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        result_json_path = "output/result.json"
        if os.path.exists(result_json_path):
            with open(result_json_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            with st.expander("📄 View Full Result JSON"):
                st.json(result_data)

# -------------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------------

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#475569; font-size:0.8rem;">'
    'Dialogue Frame Finder • Speech-first pipeline • '
    'OCR optional • Frame timing estimated via Whisper + PyAV'
    '</div>',
    unsafe_allow_html=True,
)
