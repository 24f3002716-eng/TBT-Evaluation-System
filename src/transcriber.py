
import csv
from pathlib import Path
from faster_whisper import WhisperModel


def transcribe_audio(audio_path, output_dir="outputs"):

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    print("Loading Whisper Small model...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    print("Starting transcription...")

    segments, info = model.transcribe(
        str(audio_path),
        language="hi",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False
    )

    results = []

    
    for segment in segments:

        item = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        }

        results.append(item)

        print(
            f"[{item['start']} - {item['end']}] "
            f"{item['text']}",
            flush=True
        )


    # Save transcript as TXT
    txt_path = output_dir / f"{audio_path.stem}_transcript.txt"

    with open(txt_path, "w", encoding="utf-8") as file:

        for item in results:
            file.write(
                f"[{item['start']} - {item['end']}] "
                f"{item['text']}\n"
            )

    # Save transcript as CSV
    csv_path = output_dir / f"{audio_path.stem}_transcript.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["start", "end", "text"]
        )

        writer.writeheader()
        writer.writerows(results)

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": results,
        "txt_path": txt_path,
        "csv_path": csv_path
    }
