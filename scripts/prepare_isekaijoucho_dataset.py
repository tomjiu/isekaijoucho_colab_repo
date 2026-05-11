import json
import math
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import imageio_ffmpeg
import webrtcvad
from faster_whisper import WhisperModel


SOURCE_DIR = Path(r"E:\Code\1\isekaijoucho")
OUTPUT_DIR = Path(r"E:\Code\ai\applio_isekaijoucho_local\datasets\isekaijoucho_autotrim")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
WINDOW_SECONDS = 10
WINDOW_FRAMES = int(WINDOW_SECONDS * 1000 / FRAME_MS)
SPEECH_RATIO_THRESHOLD = 0.55
MIN_START_PADDING_SECONDS = 2.0
WHISPER_SCAN_SECONDS = 240
WHISPER_MODEL = "base"


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffprobe_duration(path: Path) -> float | None:
    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-i",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = proc.stderr
    marker = "Duration: "
    if marker not in text:
        return None
    raw = text.split(marker, 1)[1].split(",", 1)[0].strip()
    hours, minutes, seconds = raw.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def frame_rms(frame: bytes) -> float:
    total = 0
    count = len(frame) // 2
    if count == 0:
        return 0.0
    for i in range(0, len(frame), 2):
        sample = int.from_bytes(frame[i : i + 2], "little", signed=True)
        total += sample * sample
    return math.sqrt(total / count) / 32768.0


def detect_speech_start(path: Path) -> tuple[float, dict[str, float | int | None]]:
    vad = webrtcvad.Vad(3)
    cmd = [
        ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    decisions: deque[int] = deque(maxlen=WINDOW_FRAMES)
    rms_values: deque[float] = deque(maxlen=WINDOW_FRAMES)
    frame_index = 0
    detected_at: float | None = None

    while True:
        frame = proc.stdout.read(FRAME_BYTES)
        if len(frame) < FRAME_BYTES:
            break
        is_speech = 1 if vad.is_speech(frame, SAMPLE_RATE) else 0
        decisions.append(is_speech)
        rms_values.append(frame_rms(frame))
        frame_index += 1

        if len(decisions) < WINDOW_FRAMES:
            continue

        speech_ratio = sum(decisions) / len(decisions)
        avg_rms = sum(rms_values) / len(rms_values)
        current_time = frame_index * FRAME_MS / 1000
        if speech_ratio >= SPEECH_RATIO_THRESHOLD and avg_rms > 0.006:
            detected_at = max(0.0, current_time - WINDOW_SECONDS - MIN_START_PADDING_SECONDS)
            break

    proc.kill()
    proc.communicate()

    if detected_at is None:
        detected_at = 0.0

    return detected_at, {
        "frames_scanned": frame_index,
        "scan_seconds": round(frame_index * FRAME_MS / 1000, 3),
    }


def looks_like_transcribed_speech(text: str) -> bool:
    return any("\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in text)


def export_intro(src: Path, dst: Path) -> None:
    cmd = [
        ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        str(WHISPER_SCAN_SECONDS),
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-vn",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def detect_whisper_start(
    path: Path,
    model: WhisperModel,
) -> tuple[float | None, dict[str, float | int | str | None]]:
    with tempfile.TemporaryDirectory(prefix="isekaijoucho_intro_") as tmp_dir:
        intro_path = Path(tmp_dir) / "intro.wav"
        export_intro(path, intro_path)
        segments, _ = model.transcribe(
            str(intro_path),
            language="ja",
            vad_filter=True,
            beam_size=3,
            condition_on_previous_text=False,
        )
        for segment in segments:
            text = segment.text.strip()
            if not text or not looks_like_transcribed_speech(text):
                continue
            if segment.no_speech_prob > 0.75 or segment.avg_logprob < -1.2:
                continue
            return max(0.0, segment.start - MIN_START_PADDING_SECONDS), {
                "whisper_start_seconds": round(segment.start, 3),
                "whisper_text": text[:120],
                "whisper_avg_logprob": round(segment.avg_logprob, 3),
                "whisper_no_speech_prob": round(segment.no_speech_prob, 3),
            }
    return None, {
        "whisper_start_seconds": None,
        "whisper_text": None,
        "whisper_avg_logprob": None,
        "whisper_no_speech_prob": None,
    }


def export_trimmed(src: Path, dst: Path, start_seconds: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "40000",
        "-vn",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_paths = [
        path
        for path in sorted(SOURCE_DIR.iterdir())
        if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ]
    manifest = []
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    for index, src in enumerate(audio_paths, start=1):
        duration = run_ffprobe_duration(src)
        vad_start_seconds, vad_stats = detect_speech_start(src)
        whisper_start_seconds, whisper_stats = detect_whisper_start(src, whisper_model)
        start_seconds = whisper_start_seconds if whisper_start_seconds is not None else vad_start_seconds
        if duration is not None and duration < 60 and start_seconds > duration * 0.4:
            start_seconds = 0.0
        dst = OUTPUT_DIR / f"isekaijoucho_{index:02d}.wav"
        print(f"[{index}/{len(audio_paths)}] {src.name}")
        print(
            f"  duration={duration:.2f}s trim_start={start_seconds:.2f}s -> {dst.name}"
            if duration
            else f"  trim_start={start_seconds:.2f}s -> {dst.name}"
        )
        export_trimmed(src, dst, start_seconds)
        manifest.append(
            {
                "source": str(src),
                "output": str(dst),
                "duration_seconds": duration,
                "trim_start_seconds": start_seconds,
                "vad_start_seconds": vad_start_seconds,
                **vad_stats,
                **whisper_stats,
            }
        )
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
