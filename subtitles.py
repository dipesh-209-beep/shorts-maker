from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="int8_float16"
)

segments, info = model.transcribe(
    "clips/clip01_vertical.mp4",
    vad_filter=True
)

with open("clips/clip01.srt", "w", encoding="utf-8") as f:
    for i, s in enumerate(segments, 1):
        start = s.start
        end = s.end

        def ts(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            sec = int(t % 60)
            ms = int((t % 1) * 1000)
            return f"{h:02}:{m:02}:{sec:02},{ms:03}"

        f.write(f"{i}\n")
        f.write(f"{ts(start)} --> {ts(end)}\n")
        f.write(f"{s.text.strip()}\n\n")
