#!/usr/bin/env python3
"""One-command shorts pipeline.

For each clip in --clips (or a single --start/--dur/--name):
   1. extract the segment from the source video
   2. build a 1080x1920 blurred-background vertical version
   3. transcribe -> srt + cinematic word-level ass subtitles
   4. mix background music under the dialogue
   5. burn the subtitles in -> clips/<name>/cinematic.mp4

Changes from the original version:
   - GPU->CPU fallback for transcription instead of a hard crash
   - ffmpeg stderr is captured and printed on failure (was silently discarded)
   - .ass subtitle path is escaped for the ffmpeg filter graph (colons broke it)
   - each pipeline step is skipped if its output already exists (use --force to redo)
   - one clip failing no longer aborts the rest of a batch; failures are summarized
     at the end and the process exits non-zero if any clip failed
   - whisper model size and boxblur strength are now CLI flags instead of hardcoded
   - whisper model loading (with GPU->CPU fallback + caching) lives in whisper_utils.py
   - --dry-run prints the step plan (paths + skip/run) without running anything
   - --cleanup deletes raw/blurred/final after cinematic.mp4 is produced
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from whisper_utils import get_whisper_model
from make_cinematic_ass import create_ass_subtitles


def ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    sec = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"


def escape_ass_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg -vf ass=... filter argument.
    Colons and backslashes are filter-graph metacharacters."""
    s = str(path).replace("\\", "\\\\").replace(":", "\\:")
    return s


def run(cmd, step_name=""):
    """Run a subprocess, capturing output so failures are debuggable."""
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"--- ffmpeg/command failed during {step_name or cmd[0]} ---")
        print(result.stderr[-4000:])  # tail, in case it's huge
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)


def extract_clip(video, start, dur, out, force=False):
    if out.exists() and not force:
        print(f"  [skip] {out.name} already exists")
        return
    run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start), "-i", str(video), "-t", str(dur),
            "-c:v", "libx264", "-c:a", "aac",
            str(out),
        ],
        step_name="extract_clip",
    )


def make_blurred(raw, out, blur_strength="20:5", force=False):
    if out.exists() and not force:
        print(f"  [skip] {out.name} already exists")
        return
    run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(raw), "-filter_complex",
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,boxblur={blur_strength}[bg];"
            "[0:v]scale=1080:-1[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2",
            "-c:v", "libx264", "-c:a", "aac",
            str(out),
        ],
        step_name="make_blurred",
    )


def transcribe(media, srt_path, ass_path, model_size="small", force=False):
    srt_path = Path(srt_path)
    ass_path = Path(ass_path)
    if srt_path.exists() and ass_path.exists() and not force:
        print(f"  [skip] subtitles already exist, reusing {srt_path}")
        return

    model = get_whisper_model(model_size)

    segments, _ = model.transcribe(str(media), vad_filter=True, word_timestamps=True)

    words = []
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{ts(s.start)} --> {ts(s.end)}\n")
            f.write(f"{s.text.strip()}\n\n")
            for w in s.words:
                words.append({"word": w.word, "start": w.start, "end": w.end})

    create_ass_subtitles(words, str(ass_path))
    print(f"  saved {srt_path} and {ass_path}")


def mix_music(media, music, music_start, volume, out, force=False, duck=True,
              duck_threshold=0.05, duck_ratio=8, duck_attack=5, duck_release=300):
    """Mix background music under the dialogue track.

    When duck=True (default), music volume is auto-lowered whenever dialogue
    is present (sidechain compression keyed off the dialogue track) and rises
    back up in gaps, instead of sitting at one fixed level for the whole clip.
    `volume` becomes the ceiling level the music sits at when there's no
    dialogue; it'll duck down from there under speech.
    """
    if out.exists() and not force:
        print(f"  [skip] {out.name} already exists")
        return

    if duck:
        filter_complex = (
            f"[1:a]volume={volume}[music];"
            "[0:a]asplit=2[dia_main][dia_sc];"
            f"[music][dia_sc]sidechaincompress=threshold={duck_threshold}:"
            f"ratio={duck_ratio}:attack={duck_attack}:release={duck_release}[ducked];"
            "[dia_main][ducked]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
    else:
        filter_complex = (
            f"[1:a]volume={volume}[bg];"
            "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )

    run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(media), "-ss", str(music_start), "-i", str(music),
            "-filter_complex", filter_complex,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v", "-map", "[a]",
            str(out),
        ],
        step_name="mix_music",
    )


def burn_ass(mixed, ass, out, force=False):
    if out.exists() and not force:
        print(f"  [skip] {out.name} already exists")
        return
    run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mixed),
            "-vf", f"ass={escape_ass_path(ass)}",
            "-c:v", "libx264", "-c:a", "copy",
            str(out),
        ],
        step_name="burn_ass",
    )


def process_clip(opts, clip):
    name = clip["name"]
    start = clip.get("start")
    dur = clip.get("dur")
    out_dir = Path(opts.out_dir)
    clip_dir = out_dir / name

    raw = clip_dir / "raw.mp4"
    blurred = clip_dir / "blurred.mp4"
    final = clip_dir / "final.mp4"
    cinematic = clip_dir / "cinematic.mp4"
    srt = clip_dir / "subtitles.srt"
    ass = clip_dir / "captions.ass"

    if opts.dry_run:
        print(f"=== {name} ===  [DRY RUN]")
        if opts.existing:
            raw = Path(opts.existing)
            print(f"  input    : {raw} (existing)")
        else:
            video = Path(opts.video)
            if not video.exists():
                print(f"  !!! source video not found: {video}")
            if start is None or dur is None:
                print(f"  !!! clip needs 'start' and 'dur' (or an 'existing' path)")
            print(f"  extract  : {video} -ss {start} -t {dur} -> {raw}"
                  + ("  (skip: exists)" if raw.exists() and not opts.force else ""))
        print(f"  blur     : {raw} -> {blurred}"
              + ("  (skip: exists)" if blurred.exists() and not opts.force else ""))
        print(f"  transcribe: {blurred} -> {srt} + {ass}"
              + ("  (skip: exists)" if srt.exists() and ass.exists() and not opts.force else ""))
        print(f"  burn     : {blurred} + {ass} -> {final}"
              + ("  (skip: exists)" if final.exists() and not opts.force else ""))
        if opts.no_music or not Path(opts.music).exists():
            print(f"  copy     : {final} -> {cinematic}  (no music)")
        else:
            print(f"  mix      : {final} + {opts.music} [{opts.music_start} @ vol {opts.music_volume}] -> {cinematic}"
                  + ("  (skip: exists)" if cinematic.exists() and not opts.force else ""))
        if opts.cleanup:
            print("  cleanup  : remove raw/blurred once final and cinematic exist")
        return

    if cinematic.exists() and not opts.force:
        print(f"  [skip] '{name}' already has {cinematic.name}, nothing to do")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {name} ===")
    t0 = time.time()

    if opts.existing:
        raw = Path(opts.existing)
        if not raw.exists():
            raise FileNotFoundError(f"Input clip not found: {raw}")
        print(f"Using existing clip {raw}")
    else:
        video = Path(opts.video)
        if not video.exists():
            raise FileNotFoundError(f"Source video not found: {video}")
        if start is None or dur is None:
            raise ValueError(f"Clip '{name}' needs start/dur (or pass --existing)")
        print(f"=== {start}s + {dur}s ===")
        extract_clip(video, start, dur, raw, force=opts.force)

    make_blurred(raw, blurred, blur_strength=opts.blur, force=opts.force)
    transcribe(blurred, srt, ass, model_size=opts.whisper_model, force=opts.force)

    burn_ass(blurred, ass, final, force=opts.force)
    print(f"FINAL -> {final} ({time.time() - t0:.1f}s)")

    if opts.no_music or not Path(opts.music).exists():
        print("Skipping music mix, copying final -> cinematic")
        if not cinematic.exists() or opts.force:
            shutil.copy2(final, cinematic)
        mix_base = final
    else:
        mix_music(
            final, opts.music, opts.music_start, opts.music_volume, cinematic,
            force=opts.force, duck=not opts.no_duck,
            duck_threshold=opts.duck_threshold, duck_ratio=opts.duck_ratio,
            duck_attack=opts.duck_attack, duck_release=opts.duck_release,
        )
        mix_base = cinematic

    print(f"DONE -> {cinematic}  ({time.time() - t0:.1f}s)")

    if opts.cleanup:
        removed = [blurred]
        if not opts.existing:
            removed.insert(0, raw)
        if mix_base is cinematic:
            removed.append(final)
        for p in removed:
            if p.exists():
                p.unlink()
                print(f"  [cleanup] removed {p}")


def _resolve(path: str) -> Path:
    """Resolve a configured path; relative paths are anchored to the project root."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def effective_opts(opts, config, clip):
    """Per-clip options: CLI defaults, overridden by clips.json config keys,
    then by any per-clip keys."""
    ropts = argparse.Namespace(**vars(opts))
    for key in ("video", "music", "music_start", "music_volume", "out_dir",
                "whisper_model", "blur", "no_music", "no_duck", "existing",
                "dry_run", "cleanup",
                "duck_threshold", "duck_ratio", "duck_attack", "duck_release"):
        if config.get(key) is not None:
            setattr(ropts, key, config[key])
        if clip.get(key) is not None:
            setattr(ropts, key, clip[key])
    for key in ("video", "music", "out_dir"):
        try:
            setattr(ropts, key, str(_resolve(getattr(ropts, key))))
        except TypeError:
            pass
    return ropts


def main():
    parser = argparse.ArgumentParser(description="Automatic shorts pipeline")
    parser.add_argument(
        "--video",
        default=str(PROJECT_ROOT / "video" / "video.mp4"),
        help="source video",
    )
    parser.add_argument("--clips", type=str, help="JSON file: [{\"name\",\"start\",\"dur\"}]")
    parser.add_argument("--start", type=float, help="clip start in seconds (single clip)")
    parser.add_argument("--dur", type=float, help="clip duration in seconds")
    parser.add_argument("--name", type=str, help="clip output name (single clip)")
    parser.add_argument("--music", default=str(PROJECT_ROOT / "music" / "music.mp3"))
    parser.add_argument("--music-start", default="00:02:00", help="offset into music file")
    parser.add_argument("--music-volume", type=float, default=0.12)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "clips"))
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--no-duck", action="store_true",
                         help="disable auto-ducking; mix music at a flat --music-volume instead")
    parser.add_argument("--duck-threshold", type=float, default=0.05,
                         help="dialogue level (0-1) above which music starts ducking")
    parser.add_argument("--duck-ratio", type=float, default=8,
                         help="compression ratio applied to music under dialogue")
    parser.add_argument("--duck-attack", type=float, default=5,
                         help="ms for music to duck down once dialogue starts")
    parser.add_argument("--duck-release", type=float, default=300,
                         help="ms for music to recover once dialogue stops")
    parser.add_argument("--existing", type=str, help="start from an already-cut clip file")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size")
    parser.add_argument("--blur", default="20:5", help="boxblur strength as 'radius:passes'")
    parser.add_argument("--force", action="store_true", help="redo all steps even if outputs exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan for every clip (paths, skip/run) without executing anything")
    parser.add_argument("--cleanup", action="store_true",
                        help="delete raw/blurred/final intermediate files for a clip once cinematic.mp4 exists")
    opts = parser.parse_args()

    if opts.clips:
        with open(opts.clips, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            config = data
            clips = data.get("clips") or []
        else:
            config = {}
            clips = data
    else:
        config = {}
        if opts.existing and not opts.name:
            parser.error("--existing also requires --name")
        elif opts.existing:
            clips = [{"name": opts.name}]
        elif opts.start is not None and opts.dur is not None and opts.name:
            clips = [{"name": opts.name, "start": opts.start, "dur": opts.dur}]
        else:
            parser.error("provide --clips file, or --start/--dur/--name")

    failures = []
    for clip in clips:
        try:
            process_clip(effective_opts(opts, config, clip), clip)
        except Exception as e:
            print(f"!!! FAILED: {clip.get('name', '?')}: {e}")
            failures.append(clip.get("name", "?"))

    if len(clips) > 1:
        ok = len(clips) - len(failures)
        print(f"\n=== Summary: {ok}/{len(clips)} clips succeeded ===")
        if failures:
            print("Failed:", ", ".join(failures))

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()