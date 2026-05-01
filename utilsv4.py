import gc
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# =========================
# CONSTANTS
# =========================
SAMPLE_RATE = 16000
N_MFCC      = 40
N_MELS      = 128
MAX_LEN     = 157

# =========================
# MFCC (USED FOR MODEL)
# =========================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Normalization — must match training preprocessing exactly
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    # Pad or truncate to fixed length
    if mfcc.shape[1] < MAX_LEN:
        pad  = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    # Free audio array from memory immediately
    del y
    gc.collect()

    return mfcc  # shape: (40, 157)


# =========================
# MEL (USED FOR VISUALIZATION ONLY)
# =========================
def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y     = librosa.util.normalize(y)

    mel    = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    del y, mel
    gc.collect()

    return mel_db


# =========================
# PLOT MEL
# =========================
def plot_mel(mel):
    fig, ax = plt.subplots(figsize=(10, 4))

    img = librosa.display.specshow(
        mel,
        sr=SAMPLE_RATE,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        cmap='magma',
        ax=ax,
    )

    ax.set_title("Mel Spectrogram", color="#38bdf8")
    ax.set_xlabel("Time", color="#7a90b0")
    ax.set_ylabel("Frequency (Hz)", color="#7a90b0")
    ax.tick_params(axis='x', colors='#7a90b0')
    ax.tick_params(axis='y', colors='#7a90b0')

    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#7a90b0')
    plt.setp(cbar.ax.get_yticklabels(), color='#7a90b0')

    fig.patch.set_facecolor('#111520')
    ax.set_facecolor('#111520')
    plt.tight_layout()
    return fig


# =========================
# PLOT MFCC HEATMAP
# =========================
def plot_mfcc(mfcc):
    fig, ax = plt.subplots(figsize=(10, 4))

    img = ax.imshow(
        mfcc,
        aspect='auto',
        origin='lower',
        cmap='coolwarm',
        interpolation='nearest',
    )

    ax.set_title("MFCC Feature Matrix  (40 coefficients × 157 frames)", color="#38bdf8", fontsize=10)
    ax.set_xlabel("Frame Index  →  Time", color="#7a90b0", fontsize=9)
    ax.set_ylabel("MFCC Coefficient Index", color="#7a90b0", fontsize=9)
    ax.tick_params(axis='x', colors='#7a90b0')
    ax.tick_params(axis='y', colors='#7a90b0')
    ax.set_yticks(range(0, 40, 5))
    ax.set_yticklabels([f"C{i}" for i in range(0, 40, 5)], color="#7a90b0", fontsize=7)
    ax.set_xticks(range(0, 157, 20))
    ax.grid(axis='y', color='#1e2535', linewidth=0.4, linestyle='--')

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Normalized Amplitude", color="#7a90b0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#7a90b0")
    plt.setp(cbar.ax.get_yticklabels(), color="#7a90b0")

    fig.patch.set_facecolor('#111520')
    ax.set_facecolor('#0d1220')
    plt.tight_layout()
    return fig


# =========================
# TRANSCRIPTION — faster-whisper
# Replaces openai-whisper + torch (~400MB saved)
# Uses int8 quantization for minimum RAM
# =========================
_whisper_model = None

def _get_whisper():
    """Lazy-load Whisper — only loads when first transcription is needed."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # int8 = smallest possible compute type, CPU only
        _whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
        )
    return _whisper_model


def transcribe_audio(file_path):
    """
    Transcribe audio using faster-whisper (tiny, int8, CPU).
    Much lighter than openai-whisper — no PyTorch dependency.
    """
    model    = _get_whisper()
    segments, _ = model.transcribe(file_path, beam_size=1)
    text     = " ".join(seg.text for seg in segments).strip()
    gc.collect()
    return text
