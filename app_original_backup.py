"""Streamlit web interface for the educational chest X-ray classifier."""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "images" / "thorax_ai_logo.png"
ICON_PATH = ROOT / "images" / "thorax_ai_icon.png"
MODEL_PATH = ROOT / "models" / "pneumonia_mobilenet.keras"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
CURVES_PATH = ROOT / "outputs" / "training_curves.png"
CONFUSION_PATH = ROOT / "outputs" / "confusion_matrix.png"
CLASS_NAMES = ("NORMAL", "PNEUMONIA")

st.set_page_config(
    page_title="Thorax AI | Medical Imaging Lab",
    page_icon=str(ICON_PATH),
    layout="wide",
)

st.logo(str(LOGO_PATH), size="large", icon_image=str(ICON_PATH))
st.markdown(
    """
    <style>
    .thorax-eyebrow {
        color: #73dccc;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .thorax-title {
        background: linear-gradient(105deg, #f2fbfa 15%, #8ce6d7 88%);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        font-size: clamp(2rem, 3.8vw, 3.15rem);
        font-weight: 700;
        letter-spacing: -0.055em;
        line-height: 1.02;
        margin-bottom: 0.7rem;
    }
    .thorax-subtitle {
        color: #a8bdc5;
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 0;
        max-width: 700px;
    }
    .thorax-eyebrow,
    .thorax-title,
    .thorax-subtitle {
        overflow-wrap: anywhere;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(77, 212, 192, 0.22);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption("THORAX AI · MEDICAL IMAGING LAB")
    if MODEL_PATH.is_file():
        st.badge("Model checkpoint ready", icon=":material/check_circle:", color="green")
    else:
        st.badge("Model not trained", icon=":material/pending:", color="orange")
    st.markdown("### A focused research demo")
    st.caption("MobileNetV2 · Image-classification research")
    st.markdown("**Research workflow**")
    st.markdown(
        ":material/upload_file: Upload an authorized image  \n"
        ":material/analytics: Review the model output  \n"
        ":material/school: Interpret in an educational context"
    )
    st.caption("Local processing · Uploads are not written to disk")

with st.container(border=True):
    icon_column, title_column, badge_column = st.columns(
        [0.12, 0.68, 0.20],
        vertical_alignment="center",
    )
    with icon_column:
        st.image(
            str(ICON_PATH),
            width=78,
            alt="Thorax AI medical imaging logo",
        )
    with title_column:
        st.markdown('<div class="thorax-eyebrow">THORAX AI · IMAGING LAB</div>', unsafe_allow_html=True)
        st.markdown('<div class="thorax-title">Imaging intelligence</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="thorax-subtitle">A considered space for exploring medical image '
            'classification through responsible, education-first AI.</p>',
            unsafe_allow_html=True,
        )
    with badge_column:
        st.badge("Research demo", icon=":material/science:", color="blue")

st.warning(
    "Not a medical device. Never use outputs for diagnosis or care decisions. "
    "Do not upload identifiable or private patient images.",
    title="Research and education only",
)


@st.cache_resource
def load_classifier(model_path: str):
    """Load the trained model once per Streamlit process."""
    return tf.keras.models.load_model(model_path)


def load_metrics() -> dict:
    if not METRICS_PATH.is_file():
        return {}
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


section = st.segmented_control(
    "Explore the project",
    ["Image demo", "Model snapshot", "About this project"],
    default="Image demo",
    key="project_section",
)

if section == "Image demo":
    st.subheader("Image assessment", icon=":material/image_search:")
    st.caption(
        "Add an authorized image to view a model output. Files are processed in memory "
        "by this local app and are not saved."
    )
    left, right = st.columns(
        2,
        gap="medium",
        vertical_alignment="top",
    )
    with left:
        with st.container(border=True, height=350):
            st.subheader("Image input", icon=":material/upload_file:")
            st.caption("JPG or PNG · resized to 224 × 224 for the model")
            uploaded = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"],
                help="Processed in memory by this local app. Avoid private or identifiable images.",
            )
            if not MODEL_PATH.is_file():
                st.warning(
                    "Train the model before requesting a prediction.",
                    title="Model unavailable",
                )
    with right:
        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
                with st.container(border=True):
                    st.subheader("Review & result", icon=":material/visibility:")
                    preview = image.copy()
                    preview.thumbnail((640, 640))
                    st.image(
                        preview,
                        caption="Uploaded image preview",
                        alt="Preview of the uploaded X-ray image",
                        width="stretch",
                    )

                    if MODEL_PATH.is_file():
                        resized = image.resize((224, 224), Image.Resampling.BILINEAR)
                        batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
                        with st.spinner("Running model inference..."):
                            model = load_classifier(str(MODEL_PATH))
                            score = float(model.predict(batch, verbose=0)[0][0])
                        index = int(score >= 0.5)
                        st.badge(
                            f"Model prediction · {CLASS_NAMES[index]}",
                            icon=":material/analytics:",
                            color="blue",
                        )
                        st.metric("PNEUMONIA class score", f"{score:.1%}")
                        st.progress(min(max(score, 0.0), 1.0))
                        st.caption(
                            "Uncalibrated model score using the demonstration threshold of 0.5. "
                            "Not a diagnosis or clinical probability."
                        )
            except (OSError, ValueError) as error:
                st.error(f"Could not read that image: {error}", title="Invalid image")
        else:
            with st.container(border=True, height=350, vertical_alignment="center"):
                st.subheader("Ready for an image", icon=":material/radiology:")
                st.caption(
                    "Your preview and model output will appear here after you choose a file."
                )

elif section == "Model snapshot":
    st.subheader("Model performance", icon=":material/monitoring:")
    st.caption("Held-out test snapshot · results describe this dataset and training run only")
    metrics = load_metrics()
    if metrics:
        with st.container(horizontal=True):
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}", border=True)
            st.metric(
                "Pneumonia sensitivity",
                f"{metrics['sensitivity_recall_pneumonia']:.1%}",
                border=True,
            )
            st.metric(
                "Normal specificity",
                f"{metrics['specificity_recall_normal']:.1%}",
                border=True,
            )
            auc = metrics.get("roc_auc")
            st.metric("ROC AUC", f"{auc:.3f}" if auc is not None else "N/A", border=True)
        st.caption(
            f"Recorded held-out test results at threshold {metrics.get('threshold', 0.5)}. "
            "Performance describes this dataset and run only; it does not imply clinical utility."
        )
        if metrics.get("specificity_recall_normal", 1) < 0.5:
            st.warning(
                "Normal specificity is low in this run, meaning many NORMAL examples "
                "were classified as PNEUMONIA at the demonstration threshold.",
                title="Interpret the result carefully",
            )
    else:
        st.info("No evaluation metrics found yet. Run `python main.py evaluate` after training.")

    if CONFUSION_PATH.is_file():
        st.subheader("Error breakdown", icon=":material/grid_on:")
        st.caption(
            "Read rows as the dataset labels and columns as model labels. "
            "Diagonal cells are matches; off-diagonal cells are errors."
        )
        _, matrix_column, _ = st.columns([1, 3, 1])
        with matrix_column:
            st.image(
                str(CONFUSION_PATH),
                caption="Held-out test split · image counts at threshold 0.5",
                alt="Confusion matrix for NORMAL and PNEUMONIA labels",
                width=640,
            )
    if CURVES_PATH.is_file():
        st.subheader("Training progress", icon=":material/show_chart:")
        st.caption(
            "Training and validation curves show how model fit changed by epoch. "
            "A gap between the lines can indicate a difference in fit between the splits."
        )
        _, curves_column, _ = st.columns([1, 4, 1])
        with curves_column:
            st.image(
                str(CURVES_PATH),
                caption="Accuracy and loss · validation is separate from the held-out test split",
                alt="Training and validation accuracy and loss by epoch",
                width=960,
            )

elif section == "About this project":
    st.subheader("Inside the research demo", icon=":material/biotech:")
    st.caption("A compact transfer-learning workflow built for education and experimentation.")
    st.markdown(
        """
        1. Images are resized to **224 × 224 RGB**.
        2. A MobileNetV2 feature extractor with ImageNet weights feeds a binary
           classification head.
        3. The displayed label uses a fixed score threshold of **0.5**.
        4. The model was developed for an educational dataset exercise, not a
           clinical workflow.

        The dataset is organized into existing `train`, `val`, and `test` splits.
        Training and evaluation commands, dataset setup, and project limitations
        are documented in the project README.
        """
    )
    st.info(
        "Chest X-ray data can be sensitive. This app is intended for local educational "
        "use; only upload images you are authorized to process."
    )
