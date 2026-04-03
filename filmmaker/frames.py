"""Frame extraction and image utilities."""

import base64
import subprocess
from pathlib import Path

from PIL import Image


def extract_last_frame(video_path: Path) -> Path:
    """Extract the last frame of a video clip as PNG.

    Uses ffprobe to get the duration, then seeks near the end to extract
    only the final frame — avoids decoding the entire video.
    """
    frame_path = video_path.with_suffix(".last_frame.png")
    if frame_path.exists() and frame_path.stat().st_size > 0:
        print(f"    Frame exists: {frame_path.name}")
        return frame_path

    # Get duration so we can seek near the end
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        duration = float(probe.stdout.strip())
        # Seek to 0.1s before the end
        seek_pos = max(0, duration - 0.1)
    except (ValueError, TypeError):
        seek_pos = None

    cmd = ["ffmpeg", "-y"]
    if seek_pos is not None:
        cmd += ["-ss", f"{seek_pos:.3f}"]
    cmd += [
        "-i", str(video_path),
        "-update", "1",
        "-q:v", "2",
        str(frame_path),
    ]

    result = subprocess.run(cmd, capture_output=True)

    if not frame_path.exists() or frame_path.stat().st_size == 0:
        raise RuntimeError(
            f"Failed to extract frame from {video_path}: "
            f"{result.stderr.decode()[-200:]}"
        )

    print(f"    Extracted: {frame_path.name}")
    return frame_path


def encode_image_base64(image_path: Path) -> str:
    """Encode an image file as base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resize_for_video(
    image_path: Path,
    target_w: int = 1280,
    target_h: int = 720,
) -> Path:
    """Resize and center-crop an image to the target video dimensions.

    Returns path to the resized image. Skips if already exists.
    """
    resized_path = image_path.parent / f"{image_path.stem}.resized.png"
    if resized_path.exists():
        return resized_path

    img = Image.open(image_path)
    w, h = img.size
    target_ratio = target_w / target_h
    img_ratio = w / h

    if img_ratio > target_ratio:
        new_h = target_h
        new_w = int(w * (target_h / h))
    else:
        new_w = target_w
        new_h = int(h * (target_w / w))

    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    img.save(resized_path)
    print(f"    Resized {image_path.name}: {w}x{h} -> {target_w}x{target_h}")
    return resized_path
