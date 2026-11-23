
import os
import re
import json
import uuid
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import librosa
import cv2

from faster_whisper import WhisperModel

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib.units import inch


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
AUDIO_DIR = BASE_DIR / "data" / "audio"
REPORT_DIR = BASE_DIR / "data" / "reports"
FRAME_DIR = BASE_DIR / "data" / "frames"

for folder in [UPLOAD_DIR, AUDIO_DIR, REPORT_DIR, FRAME_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="IIT Madras | TBT Evaluation System",
    page_icon="🏗️",
    layout="wide"
)


# ==========================================================
# 2. CUSTOM STYLING
# ==========================================================

st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: 700;
}
.subtitle {
    color: #888888;
    font-size: 15px;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.2);
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. SESSION STATE
# ==========================================================

defaults = {
    "video_path": None,
    "audio_path": None,
    "video_name": None,
    "transcript": None,
    "segments": [],
    "language": None,
    "duration": 0.0,
    "audio_stats": None,
    "evaluation": None,
    "evaluation_score": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# 4. WHISPER MODEL
# ==========================================================

@st.cache_resource
def load_whisper_model(model_size):
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )


# ==========================================================
# 5. VIDEO INFORMATION
# ==========================================================

def get_video_duration(video_path):

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return float(result.stdout.strip())


# ==========================================================
# 6. AUDIO EXTRACTION
# ==========================================================

def extract_audio(video_path, audio_path):

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:])

    return audio_path


# ==========================================================
# 7. AUDIO ANALYSIS
# ==========================================================

def analyse_audio(audio_path):

    y, sr = librosa.load(
        str(audio_path),
        sr=16000,
        mono=True
    )

    duration = len(y) / sr

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    rms = librosa.feature.rms(y=y)[0]

    zero_crossing = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "duration": duration,
        "sample_rate": sr,
        "average_rms": float(np.mean(rms)),
        "maximum_rms": float(np.max(rms)),
        "average_zero_crossing": float(np.mean(zero_crossing)),
        "peak_amplitude": float(np.max(np.abs(y)))
    }


# ==========================================================
# 8. SPEECH RECOGNITION
# ==========================================================

def transcribe_audio(audio_path, model_size, language):

    model = load_whisper_model(model_size)

    language_code = None if language == "Auto Detect" else {
        "Hindi": "hi",
        "English": "en"
    }[language]

    segments, info = model.transcribe(
        str(audio_path),
        language=language_code,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False
    )

    transcript_segments = []

    for segment in segments:

        transcript_segments.append({
            "Start": round(segment.start, 2),
            "End": round(segment.end, 2),
            "Duration": round(segment.end - segment.start, 2),
            "Text": segment.text.strip()
        })

    transcript = " ".join(
        item["Text"] for item in transcript_segments
    )

    return transcript, transcript_segments, info.language


# ==========================================================
# 9. VIDEO FRAME EXTRACTION
# ==========================================================

def extract_video_frames(video_path, interval=10):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Unable to open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video frame rate.")

    video_duration = total_frames / fps
    extracted = []

    timestamps = np.arange(0, video_duration, interval)

    for timestamp in timestamps:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            float(timestamp * 1000)
        )

        success, frame = cap.read()

        if success:

            frame_name = (
                f"frame_{int(timestamp)}_"
                f"{uuid.uuid4().hex[:6]}.jpg"
            )

            frame_path = FRAME_DIR / frame_name

            cv2.imwrite(str(frame_path), frame)

            extracted.append({
                "Time (seconds)": round(float(timestamp), 2),
                "Image": str(frame_path)
            })

    cap.release()

    return extracted


# ==========================================================
# 10. MEETING EVALUATION
# ==========================================================

EVALUATION_ATTRIBUTES = {
    "Meeting preparation": (
        "Was the meeting organised and conducted with a clear purpose?"
    ),
    "Safety topic coverage": (
        "Were relevant construction safety risks and precautions discussed?"
    ),
    "Instructor communication": (
        "Was the instructor's communication clear and understandable?"
    ),
    "Worker participation": (
        "Did workers actively participate in the discussion?"
    ),
    "Two-way communication": (
        "Was there meaningful interaction between instructor and workers?"
    ),
    "Question and answer": (
        "Were worker questions invited and answered?"
    ),
    "Practical relevance": (
        "Were safety instructions connected to actual site activities?"
    ),
    "Meeting closure": (
        "Was the meeting concluded with clear instructions or key takeaways?"
    )
}


def calculate_evaluation(ratings):

    total = sum(ratings.values())
    maximum = len(ratings) * 4

    score = (total / maximum) * 100

    if score >= 85:
        grade = "Excellent"
    elif score >= 70:
        grade = "Good"
    elif score >= 50:
        grade = "Average"
    else:
        grade = "Poor"

    return round(score, 2), grade


# ==========================================================
# 11. PDF REPORT GENERATION
# ==========================================================

def generate_pdf():

    filename = (
        "TBT_Evaluation_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".pdf"
    )

    pdf_path = REPORT_DIR / filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "TOOLBOX TALK EFFECTIVENESS EVALUATION",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "IIT Madras | Civil Engineering Department",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Report generated: "
            + datetime.now().strftime("%d-%m-%Y %H:%M"),
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph("VIDEO INFORMATION", styles["Heading2"])
    )

    video_data = [
        ["Parameter", "Value"],
        ["Video", st.session_state.video_name or "N/A"],
        ["Duration (minutes)", str(round(st.session_state.duration / 60, 2))],
        ["Detected language", str(st.session_state.language)],
        ["Total transcript words", str(len((st.session_state.transcript or "").split()))]
    ]

    table = Table(video_data, colWidths=[180, 300])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph("EVALUATION RESULTS", styles["Heading2"])
    )

    evaluation = st.session_state.evaluation or {}

    score = st.session_state.evaluation_score or (0, "Not evaluated")

    story.append(
        Paragraph(
            f"Overall evaluation score: {score[0]}%",
            styles["Heading3"]
        )
    )

    story.append(
        Paragraph(
            f"Overall grade: {score[1]}",
            styles["Heading3"]
        )
    )

    evaluation_data = [
        ["Attribute", "Rating (1-4)"]
    ]

    for key, value in evaluation.items():
        evaluation_data.append([key, str(value)])

    table = Table(
        evaluation_data,
        colWidths=[350, 130]
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph("TRANSCRIPTION", styles["Heading2"])
    )

    transcript = st.session_state.transcript or "Not available"

    # Split long transcript into manageable PDF paragraphs.
    for i in range(0, len(transcript), 2500):

        paragraph_text = transcript[i:i + 2500]

        paragraph_text = (
            paragraph_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        story.append(
            Paragraph(paragraph_text, styles["BodyText"])
        )

        story.append(Spacer(1, 8))

    doc.build(story)

    return pdf_path


# ==========================================================
# 12. APPLICATION HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">'
    '🏗️ TBT Effectiveness Evaluation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted construction meeting analysis prototype'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================================
# 13. SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("Project Information")

    st.write("**Institution:** IIT Madras")

    st.write("**Department:** Civil Engineering")

    st.write("**Project:** Toolbox Talk Evaluation")

    st.write("**Version:** 1.0")

    st.divider()

    st.header("Processing Settings")

    model_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium"],
        index=2
    )

    st.caption(
        "Small is a reasonable starting point for your MacBook M3."
    )

    st.divider()

    st.caption(
        "This prototype uses local processing. "
        "Model downloads require internet access."
    )


# ==========================================================
# 14. VIDEO UPLOAD
# ==========================================================

st.header("1. Upload Construction-Site Video")

uploaded_file = st.file_uploader(
    "Choose a video recording",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file:

    if (
        st.session_state.video_name != uploaded_file.name
        or st.session_state.video_path is None
    ):

        unique_name = (
            uuid.uuid4().hex[:8]
            + "_"
            + Path(uploaded_file.name).name
        )

        video_path = UPLOAD_DIR / unique_name

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.video_path = video_path
        st.session_state.video_name = uploaded_file.name

        st.session_state.audio_path = None
        st.session_state.transcript = None
        st.session_state.segments = []
        st.session_state.language = None
        st.session_state.audio_stats = None
        st.session_state.evaluation = None
        st.session_state.evaluation_score = None

    st.video(str(st.session_state.video_path))

    col1, col2 = st.columns(2)

    col1.metric(
        "Video Size",
        f"{uploaded_file.size / (1024 * 1024):.2f} MB"
    )

    try:

        duration = get_video_duration(
            st.session_state.video_path
        )

        st.session_state.duration = duration

        col2.metric(
            "Video Duration",
            f"{duration / 60:.2f} minutes"
        )

    except Exception as e:

        st.error(f"Video information error: {e}")


# ==========================================================
# 15. AUDIO EXTRACTION
# ==========================================================

if st.session_state.video_path:

    st.divider()

    st.header("2. Extract Audio")

    if st.button(
        "Extract Audio from Video",
        type="primary"
    ):

        audio_path = (
            AUDIO_DIR
            / f"{uuid.uuid4().hex[:8]}.wav"
        )

        try:

            with st.spinner("Extracting audio..."):

                extract_audio(
                    st.session_state.video_path,
                    audio_path
                )

            st.session_state.audio_path = audio_path

            st.success("Audio extraction successful!")

        except Exception as e:

            st.error(str(e))


# ==========================================================
# 16. AUDIO ANALYSIS
# ==========================================================

if st.session_state.audio_path:

    st.divider()

    st.header("3. Audio Analysis")

    if st.button("Analyse Audio"):

        try:

            with st.spinner("Analysing audio..."):

                stats = analyse_audio(
                    st.session_state.audio_path
                )

                st.session_state.audio_stats = stats

            st.success("Audio analysis completed!")

        except Exception as e:

            st.error(f"Audio analysis failed: {e}")

    if st.session_state.audio_stats:

        stats = st.session_state.audio_stats

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Audio Duration",
            f"{stats['duration'] / 60:.2f} min"
        )

        c2.metric(
            "Average RMS",
            f"{stats['average_rms']:.4f}"
        )

        c3.metric(
            "Peak Amplitude",
            f"{stats['peak_amplitude']:.4f}"
        )

        st.caption(
            "RMS and amplitude are signal-level measurements, "
            "not direct measures of clarity or engagement."
        )


# ==========================================================
# 17. TRANSCRIPTION
# ==========================================================

if st.session_state.audio_path:

    st.divider()

    st.header("4. Speech Recognition")

    language = st.selectbox(
        "Speech language",
        ["Auto Detect", "Hindi", "English"],
        index=0
    )

    st.caption(
        "Use Auto Detect for Hindi-English mixed speech. "
        "Recognition quality will depend on audio conditions."
    )

    if st.button(
        "Start Speech Transcription",
        type="primary"
    ):

        try:

            with st.spinner(
                "Loading speech model and processing audio..."
            ):

                transcript, segments, detected_language = (
                    transcribe_audio(
                        st.session_state.audio_path,
                        model_size,
                        language
                    )
                )

                st.session_state.transcript = transcript
                st.session_state.segments = segments
                st.session_state.language = detected_language

            st.success("Transcription completed!")

        except Exception as e:

            st.error(f"Transcription failed: {e}")


# ==========================================================
# 18. TRANSCRIPTION RESULTS
# ==========================================================

if st.session_state.transcript is not None:

    st.divider()

    st.header("5. Transcription Results")

    transcript = st.session_state.transcript

    segments = st.session_state.segments

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Words",
        len(transcript.split())
    )

    c2.metric(
        "Speech Segments",
        len(segments)
    )

    c3.metric(
        "Detected Language",
        str(st.session_state.language).upper()
    )

    st.text_area(
        "Complete Meeting Transcript",
        value=transcript,
        height=250
    )

    if segments:

        df = pd.DataFrame(segments)

        st.subheader("Timestamped Speech")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Download Transcript CSV",
            data=csv_data,
            file_name="tbt_transcript.csv",
            mime="text/csv"
        )

        json_data = json.dumps(
            segments,
            ensure_ascii=False,
            indent=4
        )

        st.download_button(
            "Download Transcript JSON",
            data=json_data,
            file_name="tbt_transcript.json",
            mime="application/json"
        )


# ==========================================================
# 19. VIDEO FRAME SAMPLING
# ==========================================================

if st.session_state.video_path:

    st.divider()

    st.header("6. Video Frame Sampling")

    interval = st.slider(
        "Extract one frame every (seconds)",
        min_value=2,
        max_value=60,
        value=10
    )

    if st.button("Extract Video Frames"):

        try:

            with st.spinner("Extracting video frames..."):

                frames = extract_video_frames(
                    st.session_state.video_path,
                    interval
                )

            st.session_state.extracted_frames = frames

            st.success(
                f"Extracted {len(frames)} frames."
            )

        except Exception as e:

            st.error(str(e))

    if st.session_state.get("extracted_frames"):

        for frame in st.session_state.extracted_frames[:12]:

            st.image(
                frame["Image"],
                caption=f"At {frame['Time (seconds)']} seconds",
                width=300
            )

        st.caption(
            "Frame sampling is available. Automated worker "
            "or PPE detection is not yet implemented."
        )


# ==========================================================
# 20. HUMAN-REVIEWED MEETING EVALUATION
# ==========================================================

if st.session_state.video_path:

    st.divider()

    st.header("7. Meeting Effectiveness Evaluation")

    st.write(
        "Review the video and transcript, then rate each attribute. "
        "These are human-entered assessments, not automatic AI findings."
    )

    st.caption(
        "Rating: 1 = Poor | 2 = Below Average | "
        "3 = Good | 4 = Excellent"
    )

    ratings = {}

    for attribute, description in EVALUATION_ATTRIBUTES.items():

        ratings[attribute] = st.select_slider(
            attribute,
            options=[1, 2, 3, 4],
            value=2,
            help=description
        )

        st.caption(description)

    if st.button(
        "Calculate Meeting Evaluation",
        type="primary"
    ):

        score, grade = calculate_evaluation(ratings)

        st.session_state.evaluation = ratings

        st.session_state.evaluation_score = (
            score,
            grade
        )

        st.success("Evaluation calculated!")

    if st.session_state.evaluation_score:

        score, grade = st.session_state.evaluation_score

        st.metric(
            "Meeting Effectiveness Score",
            f"{score}%"
        )

        st.subheader(f"Overall Grade: {grade}")

        st.progress(score / 100)

        evaluation_df = pd.DataFrame([
            {
                "Attribute": key,
                "Rating": value,
                "Maximum": 4
            }
            for key, value in
            st.session_state.evaluation.items()
        ])

        st.dataframe(
            evaluation_df,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# 21. FINAL REPORT
# ==========================================================

if (
    st.session_state.transcript is not None
    and st.session_state.evaluation_score is not None
):

    st.divider()

    st.header("8. Generate Final Evaluation Report")

    if st.button(
        "Generate PDF Report",
        type="primary"
    ):

        try:

            report_path = generate_pdf()

            with open(report_path, "rb") as f:

                st.download_button(
                    "Download Final PDF Report",
                    data=f.read(),
                    file_name=report_path.name,
                    mime="application/pdf"
                )

            st.success("PDF report generated successfully!")

        except Exception as e:

            st.error(f"Report generation failed: {e}")


# ==========================================================
# 22. FOOTER
# ==========================================================

st.divider()

st.caption(
    "IIT Madras | Civil Engineering Department | "
    "AI-Assisted Toolbox Talk Evaluation Prototype"
)

st.caption(
    "Prototype V1.0 | Scores are human-reviewed and "
    "should not be treated as validated automated assessments."
)
