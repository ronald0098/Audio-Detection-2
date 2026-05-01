"""
app.py — AudioForensics AI
Optimized for Streamlit Community Cloud (1 GB RAM free tier)

Key RAM optimizations vs original:
  - tensorflow-cpu instead of full tensorflow          saves ~300 MB
  - faster-whisper (int8) instead of openai-whisper    saves ~350 MB (removes PyTorch entirely)
  - Lazy model loading — nothing loads until needed
  - gc.collect() after every heavy operation
  - @st.cache_resource so models load only ONCE ever
"""

import gc
import io
import os
import tempfile

import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from utilsv4 import extract_mfcc, extract_mel, plot_mel, plot_mfcc, transcribe_audio
from reportsv4 import generate_report

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AudioForensics AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# GLOBAL CSS
# ===============================
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0d0f14; color: #e0e6f0; }
[data-testid="stSidebar"] { background-color: #111520; border-right: 1px solid #1e2535; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.mono { font-family: 'Space Mono', monospace; }
.hero-block {
    background: linear-gradient(135deg,#0d1b2e 0%,#0d0f14 60%,#130d1e 100%);
    border: 1px solid #1e2a40; border-radius: 16px;
    padding: 40px 48px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero-block::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background:radial-gradient(circle,rgba(56,189,248,.08) 0%,transparent 70%);
    border-radius:50%;
}
.hero-title  { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#f0f6ff; margin:0 0 8px; }
.hero-sub    { font-size:.95rem; color:#7a90b0; font-weight:300; }
.hero-badge  {
    display:inline-block; background:rgba(56,189,248,.1); border:1px solid rgba(56,189,248,.25);
    color:#38bdf8; font-family:'Space Mono',monospace; font-size:.68rem;
    padding:3px 10px; border-radius:100px; margin-bottom:14px; letter-spacing:1px;
}
.card { background:#111520; border:1px solid #1e2535; border-radius:12px; padding:24px 28px; margin-bottom:20px; }
.card-title { font-family:'Space Mono',monospace; font-size:.72rem; letter-spacing:2px; text-transform:uppercase; color:#38bdf8; margin-bottom:14px; }
.info-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1a2030; font-size:.88rem; }
.info-row:last-child { border-bottom:none; }
.info-label { color:#5a7090; font-weight:500; }
.info-value { color:#c8d8f0; font-family:'Space Mono',monospace; font-size:.80rem; }
.result-real { background:linear-gradient(135deg,#0a2418,#0d1e14); border:1.5px solid #22c55e; border-radius:12px; padding:28px 32px; text-align:center; }
.result-fake { background:linear-gradient(135deg,#2a0a0a,#1e0d0d); border:1.5px solid #ef4444; border-radius:12px; padding:28px 32px; text-align:center; }
.result-label { font-family:'Space Mono',monospace; font-size:.70rem; letter-spacing:3px; text-transform:uppercase; margin-bottom:10px; }
.result-real .result-label { color:#4ade80; }
.result-fake .result-label { color:#f87171; }
.result-verdict { font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700; letter-spacing:4px; }
.result-real .result-verdict { color:#22c55e; }
.result-fake .result-verdict { color:#ef4444; }
.conf-bar-wrap { margin:6px 0; }
.conf-bar-label { display:flex; justify-content:space-between; font-size:.80rem; margin-bottom:4px; color:#7a90b0; font-family:'Space Mono',monospace; }
.conf-bar-bg { background:#1a2030; border-radius:100px; height:8px; overflow:hidden; }
.conf-bar-fill-real { height:8px; border-radius:100px; background:linear-gradient(90deg,#16a34a,#4ade80); }
.conf-bar-fill-fake { height:8px; border-radius:100px; background:linear-gradient(90deg,#dc2626,#f87171); }
.transcript-box { background:#0d1220; border:1px solid #1e3050; border-radius:8px; padding:18px 22px; font-size:.92rem; color:#c8d8f0; line-height:1.75; white-space:pre-wrap; word-break:break-word; }
.transcript-empty { color:#3a5070; font-style:italic; font-size:.84rem; }
.sidebar-logo    { font-family:'Space Mono',monospace; font-size:1.05rem; font-weight:700; color:#f0f6ff; margin-bottom:4px; }
.sidebar-version { font-family:'Space Mono',monospace; font-size:.68rem; color:#38bdf8; letter-spacing:1.5px; margin-bottom:20px; }
.sidebar-section { font-family:'Space Mono',monospace; font-size:.64rem; letter-spacing:2px; text-transform:uppercase; color:#38bdf8; margin:20px 0 8px; padding-bottom:6px; border-bottom:1px solid #1e2535; }
.sidebar-item    { font-size:.84rem; color:#7a90b0; padding:4px 0; line-height:1.5; }
.sidebar-item strong { color:#c8d8f0; }
.stButton>button {
    background:linear-gradient(135deg,#1d4ed8,#0ea5e9); color:white; border:none;
    border-radius:8px; font-family:'Space Mono',monospace; font-size:.78rem;
    letter-spacing:1px; padding:10px 24px; width:100%; transition:opacity .2s;
}
.stButton>button:hover { opacity:.85; }
[data-testid="stFileUploader"] { background:#0d1220; border:1.5px dashed #1e3050; border-radius:10px; padding:8px; }
[data-testid="stFileUploader"]:hover { border-color:#38bdf8; }
audio { width:100%; border-radius:8px; }
[data-testid="stMetric"] { background:#0d1220; border:1px solid #1e2535; border-radius:10px; padding:14px 18px; }
.footer { margin-top:48px; padding-top:20px; border-top:1px solid #1e2535; text-align:center; font-family:'Space Mono',monospace; font-size:.66rem; color:#3a5070; letter-spacing:1px; }
</style>
""", unsafe_allow_html=True)


# ===============================
# MODEL LOAD — cached, loads once, lives forever in memory
# Uses @st.cache_resource so it survives reruns without reloading
# ===============================
@st.cache_resource(show_spinner="🔄 Loading detection model...")
def load_my_model():
    """
    Downloads model from HuggingFace Hub on first run, then caches it.
    HF_TOKEN must be set in Streamlit Community Cloud secrets:
      Settings → Secrets → add:  HF_TOKEN = "hf_xxxxxxxxxxxx"
    """
    from huggingface_hub import hf_hub_download
    from tensorflow.keras.models import load_model

    token = os.environ.get("HF_TOKEN")
    model_path = hf_hub_download(
        repo_id  = "Ashleyyy04/Audiodetectionh5",
        filename = "deepfake_model.h5",
        token    = token,
    )
    model = load_model(model_path, compile=False)
    gc.collect()
    return model


model = load_my_model()


# ===============================
# SIDEBAR
# ===============================
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🎙️ AudioForensics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-version">V 7.0 — CNN/MFCC</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-item"><strong>Architecture:</strong> CNN (3 conv blocks)</div>
        <div class="sidebar-item"><strong>Features:</strong> MFCC — 40 coefficients</div>
        <div class="sidebar-item"><strong>Dataset:</strong> ASVspoof 2019 LA</div>
        <div class="sidebar-item"><strong>Transcription:</strong> faster-whisper tiny</div>
        """, unsafe_allow_html=True)

        # Live RAM monitor — useful to check you're under 1 GB
        try:
            import psutil
            mem = psutil.virtual_memory()
            st.markdown('<div class="sidebar-section">System</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="sidebar-item"><strong>RAM Used:</strong> {mem.used / 1024**2:.0f} MB</div>
            <div class="sidebar-item"><strong>RAM Total:</strong> {mem.total / 1024**2:.0f} MB</div>
            <div class="sidebar-item"><strong>RAM Free:</strong> {mem.available / 1024**2:.0f} MB</div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

        st.markdown("""
        <div class="footer">
            AUDIOFORENSICS AI &nbsp;·&nbsp; CNN + MFCC<br>
            ASVspoof 2019 LA &nbsp;·&nbsp; Streamlit Cloud
        </div>
        """, unsafe_allow_html=True)


# ===============================
# ANALYSIS PAGE
# ===============================
def page_analyze():
    st.markdown("""
    <div class="hero-block">
        <div class="hero-badge">⬡ FORENSIC AI SYSTEM</div>
        <div class="hero-title">EchoShield: Audio Deepfake Detection</div>
        <div class="hero-sub">Upload a voice recording to analyze authenticity using deep neural feature extraction and CNN classification.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">⬆ Upload Audio File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Upload Audio File",
        type=["wav", "flac", "mp3"],
        help="Supported: WAV, FLAC, MP3 — Recommended: 16kHz mono",
        accept_multiple_files=False,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.session_state.pop("analysis_cache", None)
        return

    file_ext   = os.path.splitext(uploaded_file.name)[-1].lower()
    file_bytes = uploaded_file.read()
    file_key   = f"{uploaded_file.name}_{len(file_bytes)}"
    cache      = st.session_state.get("analysis_cache", {})

    # Use cached result if same file uploaded again — saves RAM & time
    if cache.get("file_key") == file_key:
        duration    = cache["duration"]
        result      = cache["result"]
        real_conf   = cache["real_conf"]
        fake_conf   = cache["fake_conf"]
        transcript  = cache["transcript"]
        mel         = cache["mel"]
        mfcc        = cache["mfcc"]
        audio_bytes = cache["audio_bytes"]
    else:
        file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(file_bytes)
                file_path = tmp.name

            with st.spinner("🔬 Analyzing audio signal..."):
                y, sr    = librosa.load(file_path, sr=16000)
                duration = librosa.get_duration(y=y, sr=sr)

                audio_buffer = io.BytesIO()
                wavfile.write(audio_buffer, sr, y.astype(np.float32))
                audio_bytes = audio_buffer.getvalue()
                del audio_buffer

                mel  = extract_mel(file_path)
                mfcc = extract_mfcc(file_path)

                mfcc_input = np.expand_dims(np.expand_dims(mfcc, -1), 0)
                pred       = model.predict(mfcc_input, verbose=0)
                fake_conf  = float(pred[0][0]) * 100
                real_conf  = float(pred[0][1]) * 100
                result     = "REAL" if real_conf > fake_conf else "FAKE"

                # Free memory immediately after prediction
                del mfcc_input, pred, y
                gc.collect()

            with st.spinner("🗣 Transcribing speech..."):
                try:
                    transcript = transcribe_audio(file_path)
                except Exception as te:
                    transcript = ""
                    st.warning(f"Transcription failed: {te}")

            st.session_state["analysis_cache"] = {
                "file_key":    file_key,
                "duration":    duration,
                "result":      result,
                "real_conf":   real_conf,
                "fake_conf":   fake_conf,
                "transcript":  transcript,
                "mel":         mel,
                "mfcc":        mfcc,
                "audio_bytes": audio_bytes,
            }

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
            return
        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            gc.collect()

    # ── File Info + Player ────────────────────────────────────────────────────
    col_info, col_player = st.columns([1.4, 1])

    with col_info:
        st.markdown('<div class="card"><div class="card-title">📁 File Information</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-row"><span class="info-label">File Name</span><span class="info-value">{uploaded_file.name}</span></div>
        <div class="info-row"><span class="info-label">Duration</span><span class="info-value">{duration:.2f} sec</span></div>
        <div class="info-row"><span class="info-label">Sample Rate</span><span class="info-value">16,000 Hz</span></div>
        <div class="info-row"><span class="info-label">MFCC Coefficients</span><span class="info-value">40</span></div>
        <div class="info-row"><span class="info-label">Feature Frames</span><span class="info-value">157</span></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_player:
        st.markdown('<div class="card"><div class="card-title">▶ Playback</div>', unsafe_allow_html=True)
        st.audio(audio_bytes, format="audio/wav")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Mel Spectrogram ───────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🌊 Mel Spectrogram Analysis</div>', unsafe_allow_html=True)
    fig_mel = plot_mel(mel)
    fig_mel.patch.set_facecolor('#111520')
    st.pyplot(fig_mel, use_container_width=True)
    plt.close(fig_mel)
    del fig_mel
    st.markdown('</div>', unsafe_allow_html=True)

    # ── MFCC Feature Analysis ─────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🧩 MFCC Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.82rem;color:#5a7090;margin-bottom:14px;line-height:1.6;">
        The heatmap below shows the <strong style="color:#c8d8f0;">40 Mel-Frequency Cepstral Coefficients</strong>
        (rows) extracted across <strong style="color:#c8d8f0;">157 temporal frames</strong> (columns).
        This matrix is the exact input tensor fed to the CNN classifier.
    </div>
    """, unsafe_allow_html=True)

    fig_mfcc = plot_mfcc(mfcc)
    st.pyplot(fig_mfcc, use_container_width=True)
    plt.close(fig_mfcc)
    del fig_mfcc

    with st.expander("🔢  View raw MFCC matrix (40 × 157)", expanded=False):
        import pandas as pd
        df_mfcc = pd.DataFrame(
            mfcc,
            index=[f"C{i:02d}" for i in range(mfcc.shape[0])],
            columns=[f"F{j:03d}" for j in range(mfcc.shape[1])],
        )
        st.dataframe(
            df_mfcc.style.background_gradient(cmap="coolwarm", axis=None),
            height=400,
            use_container_width=True,
        )
        st.markdown(
            f'<div style="font-size:.75rem;color:#4a6080;margin-top:6px;">'
            f'Shape: {mfcc.shape[0]} × {mfcc.shape[1]} &nbsp;|&nbsp; '
            f'Min: {mfcc.min():.3f} &nbsp;|&nbsp; Max: {mfcc.max():.3f} &nbsp;|&nbsp; '
            f'Mean: {mfcc.mean():.3f}</div>',
            unsafe_allow_html=True,
        )
        del df_mfcc

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Result ────────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🧠 Detection Result</div>', unsafe_allow_html=True)
    col_verdict, col_conf = st.columns([1, 1.2])

    with col_verdict:
        if result == "REAL":
            st.markdown(f"""
            <div class="result-real">
                <div class="result-label">✓ Verdict</div>
                <div class="result-verdict">REAL</div>
                <div style="color:#4ade80;font-size:.78rem;margin-top:8px;font-family:'Space Mono',monospace;">Authentic Voice Signal</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-label">⚠ Verdict</div>
                <div class="result-verdict">FAKE</div>
                <div style="color:#f87171;font-size:.78rem;margin-top:8px;font-family:'Space Mono',monospace;">Synthetic / Spoofed Audio Detected</div>
            </div>""", unsafe_allow_html=True)

    with col_conf:
        st.markdown(f"""
        <div style="padding:8px 0;">
            <div class="conf-bar-wrap">
                <div class="conf-bar-label"><span>REAL CONFIDENCE</span><span>{real_conf:.1f}%</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill-real" style="width:{real_conf}%"></div></div>
            </div>
            <div style="margin-top:14px"></div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-label"><span>FAKE CONFIDENCE</span><span>{fake_conf:.1f}%</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill-fake" style="width:{fake_conf}%"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Real", f"{real_conf:.2f}%")
        st.metric("Fake", f"{fake_conf:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Transcript ────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🗣 Speech Transcription</div>', unsafe_allow_html=True)
    if transcript:
        st.markdown(f'<div class="transcript-box">{transcript}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="transcript-empty">No speech detected or audio is silent.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Report ────────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">📄 Forensic Report</div>', unsafe_allow_html=True)

    mel_img_path  = tempfile.mktemp(suffix=".png")
    mfcc_img_path = tempfile.mktemp(suffix=".png")

    fig_mel_save = plot_mel(mel)
    fig_mel_save.savefig(mel_img_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_mel_save)
    del fig_mel_save

    fig_mfcc_save = plot_mfcc(mfcc)
    fig_mfcc_save.savefig(mfcc_img_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_mfcc_save)
    del fig_mfcc_save

    gc.collect()

    if st.button("📄 Generate & Download Forensic Report"):
        with st.spinner("Generating PDF..."):
            report_path = generate_report(
                file_name     = uploaded_file.name,
                prediction    = result,
                real_conf     = real_conf,
                fake_conf     = fake_conf,
                duration      = duration,
                mel_img_path  = mel_img_path,
                mfcc_img_path = mfcc_img_path,
                mfcc          = mfcc,
                transcript    = transcript,
            )
        with open(report_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label     = "⬇ Download PDF Report",
            data      = pdf_bytes,
            file_name = f"forensic_report_{uploaded_file.name}.pdf",
            mime      = "application/pdf",
        )
        os.remove(report_path)

    st.markdown('</div>', unsafe_allow_html=True)

    # Cleanup temp image files
    for p in (mel_img_path, mfcc_img_path):
        if os.path.exists(p):
            os.remove(p)

    gc.collect()


# ===============================
# MAIN
# ===============================
def main():
    render_sidebar()
    page_analyze()


if __name__ == "__main__":
    main()
