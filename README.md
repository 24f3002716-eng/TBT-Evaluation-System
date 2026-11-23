# TBT Evaluation System
### AI-Based Transcription and Meeting Evaluation Platform

**Project Status:** Under Development  
**Technology:** Python, Streamlit, Faster-Whisper, FFmpeg

## 1. Project Overview

The TBT Evaluation System is an AI-assisted platform designed to analyse recorded meetings and Toolbox Talks (TBTs) using audio and video processing techniques.

The system aims to convert recorded discussions into structured information and evaluate meeting effectiveness through measurable parameters.

The initial focus is on construction-site Toolbox Talks, with the architecture designed to support other types of meetings.

## 2. Project Objectives

- Extract audio from recorded video files.
- Convert speech into text using automatic speech recognition.
- Support Hindi and English transcription.
- Divide long recordings into manageable audio segments.
- Analyse meeting participation and interaction.
- Develop measurable meeting-quality evaluation parameters.
- Generate structured evaluation reports.

## 3. Proposed System Architecture

Video Input → Audio Extraction → Audio Processing → Speech-to-Text → Transcript Analysis → Meeting Evaluation → Scoring → Report Generation

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Speech Recognition | Faster-Whisper |
| Audio Extraction | FFmpeg |
| Audio Processing | Python audio-processing libraries |
| Development Environment | Visual Studio Code |
| Version Control | Git and GitHub |

## 5. Project Structure

```text
TBT_EVALUATION_SYSTEM/
│
├── data/
├── models/
├── outputs/
├── reports/
├── src/
│   ├── __init__.py
│   ├── audio_extractor.py
│   ├── chunked_transcriber.py
│   └── transcriber.py
│
├── uploads/
├── .gitignore
├── main.py
├── main_yt.py
├── requirements.txt
├── test_transcription.py
└── README.md
```

## 6. Evaluation Parameters

The planned evaluation framework will investigate:

- Meeting participation
- Worker engagement and attention
- Two-way communication
- Question-and-answer interaction
- Understanding and comprehension
- Clarity of safety instructions
- Overall meeting effectiveness

Final parameters and scoring weights will be established through research and validation.

## 7. Development Roadmap

1. Audio and video extraction.
2. Speech-to-text implementation.
3. Hindi and English transcription improvement.
4. Transcript analysis.
5. Meeting effectiveness evaluation.
6. Scoring and classification.
7. Automated report generation.
8. Streamlit dashboard integration.
9. Testing and validation using recorded meetings.

## 8. Research and Development

This project is being developed as an AI-assisted meeting evaluation prototype with an initial application in construction-site safety meetings.

The objective is to develop a systematic and measurable approach to evaluating meeting effectiveness using recorded audio and video data.

---

**Repository:** TBT-Evaluation-System  
**Version:** 0.1.0  
**Status:** Active Development
