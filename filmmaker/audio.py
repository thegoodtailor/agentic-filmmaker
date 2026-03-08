"""Audio analysis — detect section boundaries from a WAV file.

Uses librosa for onset detection, energy analysis, and structural
segmentation. Optionally reconciles detected boundaries with lyrics
via an LLM to produce a fully populated project config.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


def get_duration(audio_path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def detect_sections(
    audio_path: Path,
    min_section_duration: float = 10.0,
) -> list[dict]:
    """Detect structural section boundaries in an audio file.

    Uses librosa's chroma-based self-similarity matrix and agglomerative
    clustering to find major structural transitions (verse/chorus/bridge).
    Falls back to energy-based segmentation if librosa is unavailable.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.).
        min_section_duration: Minimum section length in seconds.

    Returns:
        List of dicts with keys: start_sec, end_sec, energy (relative).
    """
    if librosa is None:
        raise ImportError(
            "librosa is required for audio analysis. "
            "Install with: pip install librosa"
        )

    print(f"    Loading audio: {audio_path.name}")
    y, sr = librosa.load(str(audio_path), sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    # Compute chroma features for harmonic content
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Build recurrence (self-similarity) matrix
    rec = librosa.segment.recurrence_matrix(
        chroma,
        mode="affinity",
        sym=True,
    )

    # Detect boundaries via novelty curve on the recurrence matrix
    novelty = librosa.segment.novelty(rec)

    # Find peaks in novelty curve (= section boundaries)
    hop_length = 512  # librosa default
    times = librosa.frames_to_time(np.arange(len(novelty)), sr=sr, hop_length=hop_length)

    # Adaptive threshold: peaks above mean + 0.5 * std
    threshold = np.mean(novelty) + 0.5 * np.std(novelty)
    peak_indices = []
    for i in range(1, len(novelty) - 1):
        if novelty[i] > threshold and novelty[i] > novelty[i - 1] and novelty[i] > novelty[i + 1]:
            peak_indices.append(i)

    # Convert to timestamps, enforce minimum duration
    boundaries = [0.0]
    for idx in peak_indices:
        t = times[idx]
        if t - boundaries[-1] >= min_section_duration:
            boundaries.append(t)
    boundaries.append(duration)

    # Compute RMS energy per section for relative intensity
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    sections = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        # Average RMS in this section
        mask = (rms_times >= start) & (rms_times < end)
        avg_energy = float(np.mean(rms[mask])) if np.any(mask) else 0.0
        sections.append({
            "start_sec": round(start, 1),
            "end_sec": round(end, 1),
            "energy": round(avg_energy, 4),
        })

    print(f"    Detected {len(sections)} sections from audio analysis")
    return sections


def reconcile_with_lyrics(
    sections: list[dict],
    lyrics_text: str,
    duration: float,
    clip_duration: int,
    title: str = "",
    client=None,
    model: str = "anthropic/claude-sonnet-4",
) -> list[dict]:
    """Use an LLM to match detected audio sections with lyrics.

    Takes the raw section boundaries from audio analysis and the full
    lyrics text, and produces named sections with lyrics mapping, mood
    descriptions, clip ranges, and slow_motion flags.

    Args:
        sections: Output from detect_sections().
        lyrics_text: Full lyrics as plain text.
        duration: Total audio duration in seconds.
        clip_duration: Seconds per video clip (for computing clip ranges).
        title: Song title (for context).
        client: OpenAI-compatible client for the LLM.
        model: Model name/slug.

    Returns:
        List of section dicts ready for YAML serialization.
    """
    sections_desc = json.dumps(sections, indent=2)
    total_clips = int(duration / clip_duration)

    prompt = f"""I have a song called "{title}" that is {duration:.1f} seconds long.
It will be divided into {total_clips} video clips of {clip_duration} seconds each.

Audio analysis detected these section boundaries (with relative energy levels):
{sections_desc}

Here are the full lyrics:
---
{lyrics_text}
---

Please map the lyrics to the detected audio sections. For each section, provide:
1. A name (e.g. "Intro", "Verse 1", "Chorus", "Bridge", "Outro")
2. Which lyrics belong to it (or "(Instrumental)" if none)
3. A cinematic mood description (2-3 sentences describing the visual feel, what should be happening on screen)
4. Whether it should be slow_motion (true for instrumental/emotional sections, false for energetic ones)
5. The clip range as [start_clip, end_clip] — clips are 0-indexed, {clip_duration}s each

Respond with ONLY a JSON array. Each element:
{{
  "name": "Section Name",
  "clips": [start, end],
  "lyrics": "the lyrics for this section",
  "mood": "cinematic mood description",
  "slow_motion": false
}}"""

    print(f"    Reconciling sections with lyrics via {model}...")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a music video director mapping song structure to "
                    "visual scenes. Return only valid JSON, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
    )

    text = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    result = json.loads(text)
    print(f"    LLM mapped {len(result)} sections")
    return result
