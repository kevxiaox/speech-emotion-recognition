
import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# =========================================================
# 页面与项目路径
# =========================================================
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_transfer_best.pt"


# =========================================================
# 与训练阶段完全相同的参数
# =========================================================
SAMPLE_RATE = 16000
FIXED_DURATION = 4.0
FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION)

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8000
TOP_DB = 80

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

EMOTION_ICONS = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲"
}

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
).view(1, 3, 1, 1)


# =========================================================
# 加载模型
# =========================================================
@st.cache_resource
def load_model():
    device = torch.device("cpu")

    model = resnet18(weights=None)

    input_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(input_features, len(EMOTIONS))
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    # 兼容可能由 DataParallel 保存的模型
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device


# =========================================================
# 读取音频
# =========================================================
def read_audio(audio_bytes):
    audio, original_sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        always_2d=False
    )

    # 双声道或多声道转为单声道
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # 重新采样为 16 kHz
    if original_sample_rate != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=original_sample_rate,
            target_sr=SAMPLE_RATE
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    return audio


# =========================================================
# 音频预处理
# =========================================================
def preprocess_audio(audio):
    original_duration = len(audio) / SAMPLE_RATE

    # 去除开头和结尾的静音
    trimmed_audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    if len(trimmed_audio) == 0:
        trimmed_audio = audio

    # 中心裁剪或补零到固定4秒
    if len(trimmed_audio) > FIXED_SAMPLES:
        start = (
            len(trimmed_audio) - FIXED_SAMPLES
        ) // 2

        fixed_audio = trimmed_audio[
            start:start + FIXED_SAMPLES
        ]

    else:
        total_padding = (
            FIXED_SAMPLES - len(trimmed_audio)
        )

        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        fixed_audio = np.pad(
            trimmed_audio,
            (left_padding, right_padding),
            mode="constant"
        )

    # 计算 Mel 频谱
    mel_spectrogram = librosa.feature.melspectrogram(
        y=fixed_audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    # 转为 dB
    log_mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
        top_db=TOP_DB
    )

    # 从 [-80, 0] 归一化到 [0, 1]
    normalized_log_mel = (
        log_mel_db + TOP_DB
    ) / TOP_DB

    normalized_log_mel = np.clip(
        normalized_log_mel,
        0.0,
        1.0
    ).astype(np.float32)

    return (
        fixed_audio,
        log_mel_db,
        normalized_log_mel,
        original_duration
    )


# =========================================================
# 转换为 ResNet18 输入
# =========================================================
def create_model_input(normalized_log_mel, device):
    feature_tensor = torch.from_numpy(
        normalized_log_mel
    ).unsqueeze(0).unsqueeze(0)

    # 单通道复制为三通道
    feature_tensor = feature_tensor.repeat(
        1,
        3,
        1,
        1
    )

    # 调整为 ImageNet 模型输入尺寸
    feature_tensor = F.interpolate(
        feature_tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    # ImageNet 标准化
    feature_tensor = (
        feature_tensor - IMAGENET_MEAN
    ) / IMAGENET_STD

    return feature_tensor.to(device)


# =========================================================
# 模型预测
# =========================================================
def predict_emotion(
    model,
    device,
    normalized_log_mel
):
    feature_tensor = create_model_input(
        normalized_log_mel,
        device
    )

    with torch.no_grad():
        logits = model(feature_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_emotion = EMOTIONS[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_emotion,
        confidence,
        probabilities
    )


# =========================================================
# 网页界面
# =========================================================
st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Record your voice or upload an audio file. "
    "The ResNet18 transfer-learning model predicts "
    "one of eight emotions."
)

st.info(
    "For the best result, speak one short sentence "
    "clearly for approximately 2–4 seconds."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}"
    )
    st.stop()

try:
    model, device = load_model()

except Exception as error:
    st.error(
        f"Unable to load the model: {error}"
    )
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.write("**Model:** ResNet18")
    st.write("**Method:** Transfer learning")
    st.write("**Dataset:** RAVDESS")
    st.write("**Test accuracy:** 56.67%")
    st.write("**Macro F1:** 0.5457")
    st.write(f"**Device:** {device}")

input_method = st.radio(
    "Choose the audio input method:",
    [
        "Record with microphone",
        "Upload an audio file"
    ],
    horizontal=True
)

audio_file = None

if input_method == "Record with microphone":
    audio_file = st.audio_input(
        "Record your voice"
    )

else:
    audio_file = st.file_uploader(
        "Upload a WAV, FLAC or OGG file",
        type=["wav", "flac", "ogg"]
    )

if audio_file is not None:
    audio_bytes = audio_file.getvalue()

    st.subheader("Audio preview")
    st.audio(audio_bytes)

    try:
        raw_audio = read_audio(audio_bytes)

        (
            fixed_audio,
            log_mel_db,
            normalized_log_mel,
            original_duration
        ) = preprocess_audio(raw_audio)

        (
            predicted_emotion,
            confidence,
            probabilities
        ) = predict_emotion(
            model,
            device,
            normalized_log_mel
        )

        st.subheader("Prediction")

        result_column, confidence_column = st.columns(2)

        with result_column:
            st.metric(
                "Predicted emotion",
                (
                    f"{EMOTION_ICONS[predicted_emotion]} "
                    f"{predicted_emotion.capitalize()}"
                )
            )

        with confidence_column:
            st.metric(
                "Model confidence",
                f"{confidence * 100:.2f}%"
            )

        if confidence < 0.40:
            st.warning(
                "The prediction confidence is low. "
                "Try recording again with clearer speech "
                "and less background noise."
            )

        st.subheader("Log-Mel spectrogram")

        figure, axis = plt.subplots(
            figsize=(11, 4)
        )

        spectrogram_image = librosa.display.specshow(
            log_mel_db,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            fmin=FMIN,
            fmax=FMAX,
            cmap="magma",
            ax=axis
        )

        axis.set_title(
            "Processed Log-Mel Spectrogram"
        )

        figure.colorbar(
            spectrogram_image,
            ax=axis,
            format="%+2.0f dB"
        )

        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        st.subheader("Emotion probabilities")

        probability_df = pd.DataFrame({
            "Emotion": EMOTIONS,
            "Probability": probabilities
        }).sort_values(
            "Probability",
            ascending=False
        )

        probability_df["Probability (%)"] = (
            probability_df["Probability"] * 100
        )

        st.bar_chart(
            probability_df.set_index(
                "Emotion"
            )["Probability (%)"]
        )

        st.dataframe(
            probability_df[
                ["Emotion", "Probability (%)"]
            ].style.format({
                "Probability (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        with st.expander(
            "Audio preprocessing information"
        ):
            st.write(
                f"Original duration: "
                f"{original_duration:.2f} seconds"
            )
            st.write(
                f"Model duration: "
                f"{FIXED_DURATION:.2f} seconds"
            )
            st.write(
                f"Sample rate: {SAMPLE_RATE} Hz"
            )
            st.write(
                f"Log-Mel shape: "
                f"{normalized_log_mel.shape}"
            )

    except Exception as error:
        st.error(
            f"Unable to process this audio file: {error}"
        )

else:
    st.caption(
        "No audio has been provided yet."
    )
