# Shorts Pipeline — Full Workflow Guide

End-to-end walkthrough: picking a clip out of a source video, all the way to
a finished vertical cinematic clip with captions and ducked music.

## Project layout

```
shorts-maker/
├── video/                    # source video(s)
│   └── video.mp4
├── music/                    # background music
│   └── music.mp3
├── scripts/                  # all Python code
│   ├── make_shorts.py        # main pipeline (one command does everything)
│   ├── make_cinematic_ass.py # word-level ASS subtitle generator
│   ├── whisper_utils.py      # shared whisper model loading (GPU->CPU fallback + cache)
│   └── transcribe.py         # standalone CLI: media -> SRT only
├── clips/                    # generated output
│   ├── clip01/
│   └── clip02/
├── clips.json                # optional: batch/config definitions
├── requirements.txt          # Python dependencies
├── venv/                     # Python environment (git-ignored)
└── readme.md
```

## 0. Prerequisites

- `ffmpeg` on PATH
- A Python 3.12 venv with `faster-whisper` (a `venv/` in the project root is
  assumed by `scripts/make_shorts.py`)
- A source video in `video/` and a background music file in `music/`

```bash
venv/bin/pip install -r requirements.txt
```

## 1. Find your clip's start/end times

Scrub the source video (VLC, `ffplay`, whatever you use) and note:
- `start` — timestamp in seconds where the clip begins
- `dur` — how many seconds long the clip should be

```bash
ffplay video/video.mp4
```

## 2. Run a single clip through the full pipeline

```bash
venv/bin/python scripts/make_shorts.py \
  --start 120 --dur 45 \
  --name clip01
```

The video and music defaults come from `video/video.mp4` and
`music/music.mp3`, so for the bundled files you only need `--start`, `--dur`,
and `--name`.

What happens, in order (everything lands in `clips/<name>/`):
1. **Extract** — cuts 45s starting at 2:00 out of `video/video.mp4` → `clips/clip01/raw.mp4`
2. **Blur/reframe** — builds the 1080x1920 vertical version: centered clip, blurred zoomed copy filling top/bottom → `clips/clip01/blurred.mp4`
3. **Transcribe** — runs faster-whisper (GPU if available, CPU fallback otherwise) → `clips/clip01/subtitles.srt` + `clips/clip01/captions.ass`
4. **Mix music** — auto-ducks the music under dialogue, holds at `--music-volume` in gaps → `clips/clip01/final.mp4`
5. **Burn captions** → `clips/clip01/cinematic.mp4` ← **this is your finished file**

## 3. Batch: multiple clips from one source

Create `clips.json`. It can be a single JSON object with shared config plus a
`clips` list — every `--flag` in the reference below is available as a key, and
any key can also be set per-clip:

```json
{
  "video": "video/video.mp4",
  "music": "music/music.mp3",
  "music_volume": 0.12,
  "out_dir": "clips",
  "clips": [
    {"name": "clip01", "start": 120,  "dur": 45},
    {"name": "clip02", "start": 900,  "dur": 30, "music_volume": 0.15},
    {"name": "clip03", "start": 1500, "dur": 60, "no_music": true}
  ]
}
```

A plain list of clips also still works:
```json
[
  {"name": "clip01", "start": 120, "dur": 45}
]
```

Run:
```bash
venv/bin/python scripts/make_shorts.py --clips clips.json
```

Priority (lowest to highest): CLI defaults → `clips.json` top-level keys →
per-clip keys. Relative paths in `clips.json` are resolved against the project
root.

Each clip runs independently — if one fails, the rest still process, and you
get a summary at the end:
```
=== Summary: 2/3 clips succeeded ===
Failed: clip02
```

## 4. Starting from a clip you already cut elsewhere

```bash
venv/bin/python scripts/make_shorts.py --existing clips/clip02/raw.mp4 --name clip02
```
Skips step 1, starts straight at the blur/reframe stage. `--existing` also
works per-clip as an `"existing"` key in `clips.json`.

## 5. Common adjustments

**No music at all:**
```bash
venv/bin/python scripts/make_shorts.py --start 120 --dur 45 --name clip01 --no-music
```

**Flat music volume instead of auto-ducking:**
```bash
venv/bin/python scripts/make_shorts.py --start 120 --dur 45 --name clip01 --no-duck
```

**Tune how aggressively music ducks under dialogue:**
```bash
venv/bin/python scripts/make_shorts.py --start 120 --dur 45 --name clip01 \
  --duck-threshold 0.03 --duck-ratio 12 --duck-attack 5 --duck-release 400
```
- Lower `--duck-threshold` → music ducks on quieter speech too
- Higher `--duck-ratio` → deeper duck (music gets quieter under dialogue)
- Lower `--duck-attack` → music drops faster once speech starts
- Higher `--duck-release` → music takes longer to climb back after speech ends

**Different music starting point / volume ceiling:**
```bash
venv/bin/python scripts/make_shorts.py --start 120 --dur 45 --name clip01 \
  --music music/other_track.mp3 --music-start 00:00:45 --music-volume 0.15
```

**Different source video:**
```bash
venv/bin/python scripts/make_shorts.py --video video/source.mp4 --start 120 --dur 45 --name clip01
```

**Faster or more accurate transcription:**
```bash
venv/bin/python scripts/make_shorts.py --start 120 --dur 45 --name clip01 \
  --whisper-model tiny      # fastest, least accurate
  # or
  --whisper-model large-v3  # slowest, most accurate
```

**Stronger/weaker background blur:**
```bash
venv/bin/python scripts/make_shorts.py --start 120 --dur 45 --name clip01 \
  --blur 30:8   # radius:passes — higher = blurrier
```

## 6. Re-running after a change or a failure

By default, each step is **skipped if its output file already exists** — safe
to re-run the same command after a crash; it picks up where it left off.

**Sanity-check a long batch before running it:**
```bash
venv/bin/python scripts/make_shorts.py --clips clips.json --dry-run
```
Prints every clip's extract/blur/transcribe/mix/burn plan with resolved paths
and which steps would be skipped — without running ffmpeg or loading whisper.

**Free up disk space on batch runs:** `--cleanup` deletes `raw.mp4`,
`blurred.mp4`, and `final.mp4` for each clip once `cinematic.mp4` exists (an
`--existing` input file is never deleted):
```bash
venv/bin/python scripts/make_shorts.py --clips clips.json --cleanup
```

To force a full redo of one clip:
```bash
venv/bin/python scripts/make_shorts.py --existing clips/clip01/raw.mp4 --name clip01 --force
```

To redo only the music/burn stage (keep the existing transcript and blurred
video, e.g. after changing duck settings) without `--force` re-running
everything:
```bash
rm clips/clip01/final.mp4 clips/clip01/cinematic.mp4
venv/bin/python scripts/make_shorts.py --existing clips/clip01/raw.mp4 --name clip01
```

## 7. Flag reference

| Flag | Default | Purpose |
|---|---|---|
| `--video` | `video/video.mp4` | source video to cut clips from |
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
| `--out-dir` | `clips` | parent output directory (per-clip subfolders inside) |
| `--force` | off | redo all steps even if outputs exist |
| `--dry-run` | off | print the step plan (paths + skip/run) without running anything |
| `--cleanup` | off | delete `raw`/`blurred`/`final` after `cinematic.mp4` is produced |

Every flag maps to an equally-named key usable in `clips.json` (top-level or
per-clip), e.g. `"--no-music"` becomes `"no_music": true`.

## 8. Output files per clip

For a clip named `clip01`, everything lives in `clips/clip01/`:
```
clips/clip01/raw.mp4           # raw extracted segment
clips/clip01/blurred.mp4       # vertical reframed version
clips/clip01/subtitles.srt     # plain subtitle file
clips/clip01/captions.ass      # styled word-level subtitles
clips/clip01/final.mp4         # blurred + music mixed
clips/clip01/cinematic.mp4     # ← finished, captioned output
```