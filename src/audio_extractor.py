
import subprocess
from pathlib import Path


def extract_audio(video_path, output_dir="outputs"):
    """
    Extract audio from a video using FFmpeg.

    Output:
        WAV file
        Mono channel
        16 kHz sampling rate
        PCM 16-bit encoding
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    audio_path = output_dir / f"{video_path.stem}.wav"

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
        raise RuntimeError(
            f"Audio extraction failed: {result.stderr}"
        )

    if not audio_path.exists():
        raise RuntimeError("Audio file was not created.")

    return audio_path
