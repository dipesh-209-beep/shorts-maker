# Shorts Pipeline — Full Workflow Guide

End-to-end walkthrough: picking a clip out of a source video, all the way to
a finished vertical cinematic clip with captions and ducked music.

## 0. Prerequisites

- `ffmpeg` on PATH
- Python packages: `faster-whisper`
- `make_cinematic_ass.py` present alongside `make_shorts.py`
- A source video (e.g. `source.mp4`)
- A background music file (e.g. `music/music.mp3`)

```bash
pip install faster-whisper
```

## 1. Find your clip's start/end times

Scrub the source video (VLC, `ffplay`, whatever you use) and note:
- `start` — timestamp in seconds where the clip begins
- `dur` — how many seconds long the clip should be

```bash
ffplay source.mp4
```

## 2. Run a single clip through the full pipeline

```bash
python make_shorts.py \
  --video source.mp4 \
  --start 120 --dur 45 \
  --name clip01
```

What happens, in order:
1. **Extract** — cuts 45s starting at 0:02:00 out of `source.mp4` → `clips/clip01.mp4`
2. **Blur/reframe** — builds the 1080x1920 vertical version: centered clip, blurred zoomed copy filling top/bottom → `clips/clip01_blurred.mp4`
3. **Transcribe** — runs faster-whisper (GPU if available, CPU fallback otherwise) → `clips/clip01.srt` + `clips/clip01_blurred.ass`
4. **Mix music** — auto-ducks the music under dialogue, holds at `--music-volume` in gaps → `clips/clip01_final.mp4`
5. **Burn captions** → `clips/clip01_cinematic.mp4` ← **this is your finished file**

## 3. Batch: multiple clips from one source

Create `clips.json`:
```json
[
  {"name": "clip01", "start": 120, "dur": 45},
  {"name": "clip02", "start": 900, "dur": 30},
  {"name": "clip03", "start": 1500, "dur": 60}
]
```

Run:
```bash
python make_shorts.py --video source.mp4 --clips clips.json
```

Each clip runs independently — if one fails, the rest still process, and you
get a summary at the end:
```
=== Summary: 2/3 clips succeeded ===
Failed: clip02
```

## 4. Starting from a clip you already cut elsewhere

```bash
python make_shorts.py --existing my_clip.mp4 --name clip01
```
Skips step 1, starts straight at the blur/reframe stage.

## 5. Common adjustments

**No music at all:**
```bash
python make_shorts.py --video source.mp4 --start 120 --dur 45 --name clip01 --no-music
```

**Flat music volume instead of auto-ducking:**
```bash
python make_shorts.py --video source.mp4 --start 120 --dur 45 --name clip01 --no-duck
```

**Tune how aggressively music ducks under dialogue:**
```bash
python make_shorts.py --video source.mp4 --start 120 --dur 45 --name clip01 \
  --duck-threshold 0.03 --duck-ratio 12 --duck-attack 5 --duck-release 400
```
- Lower `--duck-threshold` → music ducks on quieter speech too
- Higher `--duck-ratio` → deeper duck (music gets quieter under dialogue)
- Lower `--duck-attack` → music drops faster once speech starts
- Higher `--duck-release` → music takes longer to climb back after speech ends

**Different music starting point / volume ceiling:**
```bash
python make_shorts.py --video source.mp4 --start 120 --dur 45 --name clip01 \
  --music music/other_track.mp3 --music-start 00:00:45 --music-volume 0.15
```

**Faster or more accurate transcription:**
```bash
python make_shorts.py --video source.mp4 --start 120 --dur 45 --name clip01 \
  --whisper-model tiny      # fastest, least accurate
  # or
  --whisper-model large-v3  # slowest, most accurate
```

**Stronger/weaker background blur:**
```bash
python make_shorts.py --video source.mp4 --start 120 --dur 45 --name clip01 \
  --blur 30:8   # radius:passes — higher = blurrier
```

## 6. Re-running after a change or a failure

By default, each step is **skipped if its output file already exists** — safe
to re-run the same command after a crash; it picks up where it left off.

To force a full redo of one clip:
```bash
python make_shorts.py --existing clip01.mp4 --name clip01 --force
```

To redo only the music/burn stage (keep the existing transcript and blurred
video, e.g. after changing duck settings) without `--force` re-running
everything:
```bash
rm clips/clip01_final.mp4 clips/clip01_cinematic.mp4
python make_shorts.py --existing clip01.mp4 --name clip01
```

## 7. Flag reference

| Flag | Default | Purpose |
|---|---|---|
| `--video` | `video.mp4` | source video to cut clips from |
| `--clips` | — | JSON file of `{"name","start","dur"}` for batch runs |
| `--start` / `--dur` / `--name` | — | single-clip mode |
| `--existing` | — | use an already-cut clip instead of extracting |
| `--music` | `music/music.mp3` | background music file |
| `--music-start` | `00:02:00` | offset into the music file to start from |
| `--music-volume` | `0.12` | music ceiling volume (0–1) |
| `--no-music` | off | skip music entirely |
| `--no-duck` | off | flat music volume instead of auto-ducking |
| `--duck-threshold` | `0.05` | dialogue level that triggers ducking |
| `--duck-ratio` | `8` | how much music is compressed under dialogue |
| `--duck-attack` | `5` (ms) | how fast music ducks down |
| `--duck-release` | `300` (ms) | how fast music recovers after dialogue |
| `--whisper-model` | `small` | faster-whisper model size |
| `--blur` | `20:5` | background blur `radius:passes` |
| `--out-dir` | `clips` | output directory |
| `--force` | off | redo all steps even if outputs exist |

## 8. Output files per clip

For a clip named `clip01`, you'll end up with:
```
clips/clip01.mp4            # raw extracted segment
clips/clip01_blurred.mp4    # vertical reframed version
clips/clip01.srt            # plain subtitle file
clips/clip01_blurred.ass    # styled word-level subtitles
clips/clip01_final.mp4      # blurred + music mixed
clips/clip01_cinematic.mp4  # ← finished, captioned output
```
