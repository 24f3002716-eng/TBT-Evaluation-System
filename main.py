
import streamlit as st
import cv2
import os
import uuid
from pathlib import Path
from src.audio_extractor import extract_audio

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="TBT Evaluation System",
    page_icon="🏗️",
    layout="wide"
)

# -------------------------------
# PROJECT DIRECTORIES
# -------------------------------

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# -------------------------------
# APPLICATION HEADER
# -------------------------------

st.title("🏗️ Toolbox Talk Evaluation System")

st.subheader("AI-Powered Construction Safety Meeting Evaluation")

st.write(
    "Upload a Toolbox Talk recording to begin "
    "the evaluation process."
)

st.divider()

# -------------------------------
# VIDEO UPLOAD
# -------------------------------

st.header("1. Upload Toolbox Talk Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv"]
)

# -------------------------------
# VIDEO PROCESSING
# -------------------------------

if uploaded_file is not None:

    st.success("Video uploaded successfully!")

    # Generate a unique filename
    extension = Path(uploaded_file.name).suffix.lower()
    filename = f"{uuid.uuid4().hex}{extension}"

    video_path = UPLOAD_DIR / filename

    # Save uploaded video
    with open(video_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.header("2. Video Preview")

    st.video(str(video_path))

    # ---------------------------
    # VIDEO INFORMATION
    # ---------------------------

    cap = cv2.VideoCapture(str(video_path))

    if cap.isOpened():

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frames / fps if fps > 0 else 0

        st.header("3. Video Information")

        col1, col2, col3 = st.columns(3)

        col1.metric("Duration", f"{duration:.2f} seconds")
        col2.metric("FPS", f"{fps:.2f}")
        col3.metric("Resolution", f"{width} × {height}")

        st.success("Video validation completed!")

        cap.release()

    else:
        st.error("Unable to read this video.")

    st.divider()

    st.header("4. Audio Extraction")

if st.button("Extract Audio", type="primary"):

    try:
        with st.spinner("Extracting audio..."):

            audio_path = extract_audio(video_path)

        st.success("Audio extracted successfully!")

        st.audio(str(audio_path))

        st.write("Saved audio file:", str(audio_path))

    except Exception as e:
        st.error(f"Audio extraction failed: {e}")