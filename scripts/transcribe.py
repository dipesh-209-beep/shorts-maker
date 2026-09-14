import argparse
from pathlib import Path
from whisper_utils import get_whisper_model


def ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    sec = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio clips to SRT using faster-whisper."
    )
    parser.add_argument("input_file", type=str, help="Path to input media file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional output SRT path (defaults to same name with .srt)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="faster-whisper model size (default: small)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".srt")

    model = get_whisper_model(args.model)
    segments, _ = model.transcribe(str(input_path), vad_filter=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{ts(s.start)} --> {ts(s.end)}\n")
            f.write(f"{s.text.strip()}\n\n")

    print(f"Saved subtitles to: {output_path}")


if __name__ == "__main__":
    main()