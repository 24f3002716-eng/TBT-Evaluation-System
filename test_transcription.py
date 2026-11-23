
from src.transcriber import transcribe_audio
from pathlib import Path
import wave

audio_path = Path(
    "outputs/a0b142b2f25640e8b5c9d3ee2ba15495.wav"
)

if not audio_path.exists():
    raise FileNotFoundError(audio_path)

with wave.open(str(audio_path), "rb") as wav:
    duration = wav.getnframes() / wav.getframerate()

print("Selected audio:", audio_path)
print("Audio duration:", duration, "seconds")


result = transcribe_audio(
    audio_path,
    output_dir="outputs/hindi_test"
)

print("\nDetected language:", result["language"])
print("Language probability:", result["language_probability"])
print("Number of segments:", len(result["segments"]))

print("\nTRANSCRIPTION PREVIEW:\n")

for segment in result["segments"][:10]:
    print(
        segment["start"],
        "-",
        segment["end"],
        segment["text"]
    )

print("\nTXT:", result["txt_path"])
print("CSV:", result["csv_path"])
