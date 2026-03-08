"""FFmpeg assembly — concatenate clips and overlay audio."""

import subprocess
from pathlib import Path


def assemble(
    clips_dir: Path,
    audio_path: Path,
    output_path: Path,
    n_clips: int,
    slow_motion_factor: float = 1.0,
) -> Path:
    """Concatenate video clips and overlay the audio track.

    Builds an ffmpeg concat demuxer file, stitches all clips, then
    muxes the audio. Uses yuv420p pixel format for broad compatibility
    (yuv444p produces blank video on most players).

    Does NOT use -shortest flag (causes audio cutoff on last clip).

    Args:
        clips_dir: Directory containing clip_00.mp4, clip_01.mp4, etc.
        audio_path: Path to the audio file (WAV, MP3, etc.).
        output_path: Where to write the final video.
        n_clips: Number of clips to concatenate.
        slow_motion_factor: If > 1.0, slows video by this factor.

    Returns:
        Path to the assembled video.
    """
    # Build concat file
    concat_path = clips_dir / "concat.txt"
    with open(concat_path, "w") as f:
        for i in range(n_clips):
            clip = clips_dir / f"clip_{i:02d}.mp4"
            if clip.exists():
                f.write(f"file '{clip.name}'\n")
            else:
                print(f"    WARNING: {clip.name} missing, skipping")

    # Step 1: Concatenate clips (no audio)
    stitched_path = clips_dir / "stitched_noaudio.mp4"
    print(f"    Stitching {n_clips} clips...")

    filter_complex = None
    if slow_motion_factor > 1.0:
        filter_complex = f"setpts=PTS*{slow_motion_factor}"

    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
    ]
    if filter_complex:
        concat_cmd += ["-filter:v", filter_complex]
    concat_cmd += [
        "-c:v", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        str(stitched_path),
    ]

    result = subprocess.run(concat_cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed: {result.stderr.decode()[-500:]}"
        )
    print(f"    Stitched: {stitched_path.name}")

    # Step 2: Overlay audio
    print(f"    Adding audio track...")
    mux_cmd = [
        "ffmpeg", "-y",
        "-i", str(stitched_path),
        "-i", str(audio_path),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ]

    result = subprocess.run(mux_cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio mux failed: {result.stderr.decode()[-500:]}"
        )

    print(f"    Final video: {output_path.name}")
    return output_path
