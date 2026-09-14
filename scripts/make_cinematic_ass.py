import argparse
from pathlib import Path
from faster_whisper import WhisperModel


def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def create_ass_subtitles(words_data, output_ass_path):
    # ASS Header: 1080x1920 canvas, Montserrat ExtraBold, 60px, 4px black outline, bottom-centered (~62% height)
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cinematic,Montserrat ExtraBold,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,50,50,720,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    # Group words into small chunks (max 5 words per chunk)
    chunk_size = 5
    chunks = [
        words_data[i : i + chunk_size]
        for i in range(0, len(words_data), chunk_size)
    ]

    for chunk in chunks:
        if not chunk:
            continue

        # Split chunk into maximum 2 lines
        mid = (len(chunk) + 1) // 2
        line1_words = chunk[:mid]
        line2_words = chunk[mid:]

        # Animate each word within the chunk duration
        for active_idx, target_word in enumerate(chunk):
            start_t = format_ass_time(target_word["start"])
            # Set frame end time to next word's start, or chunk end
            if active_idx < len(chunk) - 1:
                end_t = format_ass_time(chunk[active_idx + 1]["start"])
            else:
                end_t = format_ass_time(chunk[-1]["end"])

            # Build formatted text with ASS red color tag &H000000FF for active word
            formatted_line1 = []
            for w in line1_words:
                txt = w["word"].strip()
                if w == target_word:
                    formatted_line1.append(f"{{\\c&H0000FF&}}{txt}{{\\c&HFFFFFF&}}")
                else:
                    formatted_line1.append(txt)

            formatted_line2 = []
            for w in line2_words:
                txt = w["word"].strip()
                if w == target_word:
                    formatted_line2.append(f"{{\\c&H0000FF&}}{txt}{{\\c&HFFFFFF&}}")
                else:
                    formatted_line2.append(txt)

            text_str = " ".join(formatted_line1)
            if formatted_line2:
                text_str += "\\N" + " ".join(formatted_line2)

            events.append(
                f"Dialogue: 0,{start_t},{end_t},Cinematic,,0,0,0,,{text_str}"
            )

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input video/audio file")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_ass = input_path.with_suffix(".ass")

    model = WhisperModel("small", device="cuda", compute_type="int8_float16")
    segments, _ = model.transcribe(
        str(input_path), vad_filter=True, word_timestamps=True
    )

    words_data = []
    for segment in segments:
        for w in segment.words:
            words_data.append(
                {"word": w.word, "start": w.start, "end": w.end}
            )

    create_ass_subtitles(words_data, output_ass)
    print(f"Generated ASS file: {output_ass}")


if __name__ == "__main__":
    main()
