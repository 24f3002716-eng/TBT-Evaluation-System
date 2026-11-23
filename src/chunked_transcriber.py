
import av
import numpy as np
import pandas as pd
from pathlib import Path
from faster_whisper import WhisperModel

AUDIO_PATH = Path(
    "outputs/a0b142b2f25640e8b5c9d3ee2ba15495.wav"
)

OUTPUT_DIR = Path("outputs/experiment_60s")
CHUNK_SECONDS = 60
SAMPLE_RATE = 16000


def run_chunked_transcription():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(AUDIO_PATH)

    print("Loading audio...")

    audio_parts = []

    with av.open(str(AUDIO_PATH)) as container:
        for frame in container.decode(audio=0):
            audio_parts.append(
                frame.to_ndarray().reshape(-1)
            )

    audio = np.concatenate(audio_parts).astype(np.float32) / 32768.0

    duration = len(audio) / SAMPLE_RATE

    print(f"Audio duration: {duration:.2f} seconds")

    print("Loading Whisper Small model...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    all_results = []

    total_chunks = int(np.ceil(duration / CHUNK_SECONDS))

    print(f"Total chunks: {total_chunks}")

    for i in range(total_chunks):

        start_sample = i * CHUNK_SECONDS * SAMPLE_RATE
        end_sample = min(
            (i + 1) * CHUNK_SECONDS * SAMPLE_RATE,
            len(audio)
        )

        chunk = audio[start_sample:end_sample]
        offset = start_sample / SAMPLE_RATE

        print(f"\nProcessing chunk {i + 1}/{total_chunks}")

        segments, info = model.transcribe(
            chunk,
            language="hi",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False
        )

        
        chunk_duration = len(chunk) / SAMPLE_RATE

        for segment in segments:

            # Ignore segments starting outside the audio chunk
            if segment.start >= chunk_duration:
                continue

            start_time = offset + segment.start
            end_time = min(
                offset + segment.end,
                duration
            )

            if end_time <= start_time:
                continue

            item = {
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": segment.text.strip()
            }


            all_results.append(item)

            print(
                f"[{item['start']} - {item['end']}] "
                f"{item['text']}",
                flush=True
            )

    df = pd.DataFrame(all_results)

    csv_path = OUTPUT_DIR / "chunked_transcript.csv"
    txt_path = OUTPUT_DIR / "chunked_transcript.txt"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(
                f"[{item['start']} - {item['end']}] "
                f"{item['text']}\n"
            )

    print("\nTRANSCRIPTION COMPLETE!")
    print("Total segments:", len(all_results))
    print("CSV:", csv_path)
    print("TXT:", txt_path)


if __name__ == "__main__":
    run_chunked_transcription()
