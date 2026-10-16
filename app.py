
"""MediScan AI — Educational chest X-ray classification dashboard."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pneumonia_mobilenet.keras"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
CURVES_PATH = ROOT / "outputs" / "training_curves.png"
CONFUSION_PATH = ROOT / "outputs" / "confusion_matrix.png"

CLASS_NAMES = ("NORMAL", "PNEUMONIA")

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Medical Blue theme -------------------------

# ------------------------- Dark Medical Blue theme -------------------------

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #0b1220 !important;
        color: #e8f0fc !important;
    }

    [data-testid="stHeader"] {
        background: #0b1220 !important;
    }

    [data-testid="stSidebar"] {
        background: #101c30 !important;
        border-right: 1px solid #243750;
    }

    [data-testid="stSidebar"] * {
        color: #e4efff;
    }

    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] li,
    [data-testid="stMain"] span,
    [data-testid="stMain"] .stMarkdown {
        color: #e0eaf8;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f3f7ff !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #b5c6dc !important;
    }

    .hero {
        background: linear-gradient(120deg, #10294b, #075985);
        padding: 30px 32px;
        border: 1px solid #24547c;
        border-radius: 18px;
        color: #ffffff !important;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 2.45rem;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero p {
        color: #e0f2fe !important;
        margin: 10px 0 0;
        font-size: 1rem;
    }

    .eyebrow {
        color: #67e8f9 !important;
        font-size: 0.75rem;
        letter-spacing: 2px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .section-title {
        color: #7dd3fc !important;
        font-weight: 700;
        font-size: 1.25rem;
    }

    [data-testid="stMetric"] {
        background: #142238 !important;
        border: 1px solid #2b4260 !important;
        padding: 16px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #bfd4ee !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #ffffff !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #111e32 !important;
        border-color: #2b4260 !important;
        border-radius: 14px;
    }

    [data-testid="stFileUploader"] {
        background: #142238 !important;
        border: 1px dashed #3d638b !important;
        border-radius: 12px;
        padding: 8px;
    }

    [data-testid="stFileUploader"] * {
        color: #e5efff !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #2b4260;
        border-radius: 10px;
    }

    [data-testid="stDataFrame"] * {
        color: #e5efff;
    }

    .stSelectbox div,
    .stTextInput input,
    .stTextArea textarea {
        background-color: #142238 !important;
        color: #ffffff !important;
    }

    div.stButton > button {
        background: #172b45 !important;
        color: #f0f7ff !important;
        border: 1px solid #3a5d82 !important;
        border-radius: 9px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"] {
        background: #0879bd !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 9px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        border-color: #67e8f9 !important;
        color: #ffffff !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #293d57 !important;
    }

    a {
        color: #7dd3fc !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------- Model and data -------------------------

@st.cache_resource
def load_classifier(model_path: str):
    """Load the existing trained model once per Streamlit session."""
    return tf.keras.models.load_model(model_path)


def load_metrics() -> dict:
    """Read saved evaluation metrics when available."""
    if not METRICS_PATH.is_file():
        return {}

    try:
        with METRICS_PATH.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def predict_image(image: Image.Image) -> float:
    """Use the original 224x224 RGB inference pipeline."""
    resized = image.resize((224, 224), Image.Resampling.BILINEAR)
    batch = np.expand_dims(
        np.asarray(resized, dtype=np.float32),
        axis=0,
    )

    model = load_classifier(str(MODEL_PATH))
    output = np.asarray(model.predict(batch, verbose=0)).reshape(-1)

    if output.size != 1:
        raise ValueError(
            "The existing app expects a single sigmoid output from its model."
        )

    score = float(output[0])

    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("The model returned an invalid score.")

    return score


if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ------------------------- Sidebar -------------------------

with st.sidebar:
    st.markdown("## 🩺 MediScan AI")
    st.caption("MEDICAL IMAGE ANALYSIS")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Image Analysis", "Analysis History", "Model Information"],
        key="navigation",
    )

    st.divider()

    if MODEL_PATH.is_file():
        st.success("Model checkpoint found")
    else:
        st.warning("Model checkpoint not found")

    st.markdown("### About this demo")
    st.caption(
        "An educational chest X-ray classification project "
        "using a trained MobileNetV2 model."
    )

    st.divider()
    st.caption("Local application · Research and education only")


# ------------------------- Main header -------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-POWERED MEDICAL IMAGING</div>
        <h1>MediScan AI</h1>
        <p>
            Explore chest X-ray image classification through an
            educational AI research dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational demonstration only. This model is not a medical device "
    "and must not be used to diagnose disease or guide treatment. "
    "Do not upload identifiable or private patient images."
)


# ------------------------- Image analysis page -------------------------

if page == "Image Analysis":
    history = st.session_state.analysis_history

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Images analyzed", len(history))

    with col2:
        st.metric("Classification classes", "2")

    with col3:
        st.metric(
            "Model status",
            "Ready" if MODEL_PATH.is_file() else "Unavailable",
        )

    st.divider()
    st.subheader("Image analysis workspace")

    upload_col, result_col = st.columns([0.9, 1.1], gap="large")

    with upload_col:
        with st.container(border=True):
            st.markdown("### 📤 Upload an image")
            st.caption("Supported formats: JPG, JPEG and PNG")

            uploaded = st.file_uploader(
                "Choose a chest X-ray image",
                type=["jpg", "jpeg", "png"],
                key="xray_upload",
            )

            image = None

            if uploaded is not None:
                try:
                    image = Image.open(uploaded).convert("RGB")
                    st.image(
                        image,
                        caption="Image preview",
                        width="stretch",
                    )
                except (OSError, ValueError):
                    st.error("This image could not be opened.")
                    image = None

            if not MODEL_PATH.is_file():
                st.error(
                    "The trained model file is missing. "
                    "Check the models folder before running analysis."
                )

            analyze = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True,
                disabled=(image is None or not MODEL_PATH.is_file()),
            )

            if analyze and image is not None:
                try:
                    with st.spinner("Running the existing model..."):
                        score = predict_image(image)

                    prediction_index = int(score >= 0.5)
                    prediction = CLASS_NAMES[prediction_index]

                    result = {
                        "filename": uploaded.name,
                        "prediction": prediction,
                        "score": score,
                        "timestamp": datetime.now().strftime(
                            "%d %b %Y, %I:%M:%S %p"
                        ),
                    }

                    st.session_state.last_result = result
                    st.session_state.analysis_history.insert(0, result)

                    st.success("Analysis completed.")
                except Exception as error:
                    st.error(
                        f"Analysis failed: {error}",
                        icon="⚠️",
                    )

    with result_col:
        with st.container(border=True):
            st.markdown("### 📊 Analysis results")

            result = st.session_state.last_result

            if result is None:
                st.info(
                    "Upload an image and select Analyze Image. "
                    "The result will appear here."
                )
            else:
                st.markdown("**Latest model output**")
                st.caption(f"File: {result['filename']}")

                if result["prediction"] == "PNEUMONIA":
                    st.warning(
                        "Model label: PNEUMONIA",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        "Model label: NORMAL",
                        icon="ℹ️",
                    )

                score = result["score"]
                normal_score = 1.0 - score

                score_col1, score_col2 = st.columns(2)

                with score_col1:
                    st.metric("NORMAL score", f"{normal_score:.1%}")

                with score_col2:
                    st.metric("PNEUMONIA score", f"{score:.1%}")

                st.markdown("#### Model score comparison")

                chart_data = pd.DataFrame({
                    "Class": ["NORMAL", "PNEUMONIA"],
                    "Score": [normal_score, score],
                })

                st.bar_chart(
                    chart_data,
                    x="Class",
                    y="Score",
                    horizontal=True,
                    x_label="Class",
                    y_label="Model score",
                )

                st.caption(
                    "These are model scores derived from the existing "
                    "sigmoid output and its complement. They are not "
                    "calibrated clinical probabilities or diagnostic certainty."
                )
                st.caption(f"Analyzed: {result['timestamp']}")

    st.divider()
    st.markdown("#### How to interpret this demo")
    st.write(
        "The displayed label uses the existing model's threshold of 0.5. "
        "Model outputs can be wrong and may be affected by image quality, "
        "dataset bias and differences between training images and new images."
    )


# ------------------------- History page -------------------------

elif page == "Analysis History":
    st.subheader("🕘 Analysis history")
    st.caption(
        "Records are kept in this browser session only. They are not "
        "saved to a database or written to a history file."
    )

    history = st.session_state.analysis_history

    if not history:
        st.info("No images have been analyzed in this session yet.")
    else:
        history_data = pd.DataFrame([
            {
                "Time": item["timestamp"],
                "Filename": item["filename"],
                "Model label": item["prediction"],
                "NORMAL score": f"{1.0 - item['score']:.1%}",
                "PNEUMONIA score": f"{item['score']:.1%}",
            }
            for item in history
        ])

        st.dataframe(
            history_data,
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear session history"):
            st.session_state.analysis_history = []
            st.session_state.last_result = None
            st.rerun()


# ------------------------- Model information page -------------------------

elif page == "Model Information":
    st.subheader("🧠 Model information")
    st.write(
        "MediScan AI is a demonstration interface for the existing "
        "chest X-ray classification model."
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("#### Model configuration")
        st.write("Architecture: MobileNetV2-based classifier")
        st.write("Input size: 224 × 224 RGB")
        st.write("Classes: NORMAL and PNEUMONIA")
        st.write("Demonstration threshold: 0.5")
        st.write(
            "Model file:",
            str(MODEL_PATH.relative_to(ROOT)),
        )

    with info_col2:
        st.markdown("#### Saved evaluation metrics")
        metrics = load_metrics()

        if metrics:
            for key, label in [
                ("accuracy", "Accuracy"),
                ("sensitivity_recall_pneumonia", "Pneumonia sensitivity"),
                ("specificity_recall_normal", "Normal specificity"),
                ("roc_auc", "ROC AUC"),
            ]:
                value = metrics.get(key)

                if value is not None:
                    if key == "roc_auc":
                        st.metric(label, f"{value:.3f}")
                    else:
                        st.metric(label, f"{value:.1%}")
        else:
            st.info(
                "No saved metrics were found. Train and evaluate the "
                "model using the project's documented workflow first."
            )

    if CONFUSION_PATH.is_file():
        st.markdown("#### Confusion matrix")
        st.image(
            str(CONFUSION_PATH),
            caption="Previously saved evaluation output",
            width=650,
        )

    if CURVES_PATH.is_file():
        st.markdown("#### Training curves")
        st.image(
            str(CURVES_PATH),
            caption="Previously saved training output",
            width="stretch",
        )

    st.divider()
    st.markdown("#### Responsible use")
    st.info(
        "This project is for education and experimentation only. "
        "Its results are not validated for clinical use and must not "
        "be used to make medical decisions."
    )
