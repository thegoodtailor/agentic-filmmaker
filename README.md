# agentic-filmmaker

AI music video pipeline. Three agents -- vision, narrative, and video generation -- chain-generate a music video from a single seed image, one clip at a time.

## What it does

Given a seed image and a project config (song structure, style, characters), the pipeline runs an agentic loop: a video generator (Sora 2) produces a clip from the seed, ffmpeg extracts the last frame, a vision agent (GPT-4o) describes what it sees, a narrative agent (Claude) writes what happens next, and the cycle repeats. Each clip is born from the last frame of the previous one, so the film dreams itself forward with no human intervention. The result is a full music video assembled from individually generated clips with an audio track overlaid.

## Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │              AGENTIC LOOP                   │
                          │                                             │
  seed image ─────┐       │   ┌─────────┐     ┌──────────┐             │
       or         ├───────┼──►│ Sora 2  │────►│  ffmpeg  │             │
  freeze frame    │       │   │ (video) │     │ (extract │             │
       or         │       │   └─────────┘     │  last    │             │
  Flux ref ───────┘       │        ▲          │  frame)  │             │
                          │        │          └────┬─────┘             │
                          │   scene prompt         │                   │
                          │        │          frame image              │
                          │   ┌────┴──────┐        │                   │
                          │   │  Claude   │   ┌────▼─────┐            │
                          │   │(narrative)│◄──│  GPT-4o  │            │
                          │   └───────────┘   │ (vision) │            │
                          │                   └──────────┘            │
                          └──────────────────────────────────────────────┘
                                         │
                                    all clips
                                         │
                                    ┌────▼─────┐
                                    │  ffmpeg  │
                                    │ (concat  │
                                    │ + audio) │
                                    └──────────┘
                                         │
                                    final.mp4
```

## Quick start

```bash
pip install -e ".[audio]"

# Set up API keys
export OPENAI_API_KEY="sk-..."          # Sora 2, DALL-E 3, GPT-4o
export OPENROUTER_API_KEY="sk-or-..."   # Narrative agent (Claude via OpenRouter)

# Create a project from audio + lyrics
filmmaker init --audio track.wav --lyrics lyrics.txt --title "My Song" --artist "My Band"

# Edit the generated project.yaml — add style, characters, seed prompt, tweak moods
vim project.yaml

# Generate all clips
filmmaker generate project.yaml

# Assemble into final video with audio
filmmaker assemble project.yaml
```

Or write the project YAML by hand (see `examples/fsf.yaml` for a complete example) and skip straight to `generate`.

## CLI reference

### `filmmaker init`

Analyzes audio to detect section boundaries (verse/chorus/bridge transitions via librosa), then optionally uses an LLM to match those boundaries with your lyrics and generate mood descriptions.

```
filmmaker init \
  --audio track.wav              # Path to audio file (WAV, MP3)
  --lyrics lyrics.txt            # Path to lyrics text file (optional)
  --title "Song Title"           # Song title
  --artist "Artist"              # Artist name
  --album "Album"                # Album name (optional)
  --duration 314.4               # Override duration in seconds (optional, auto-detected)
  --clip-duration 8              # Seconds per clip (default: 8)
  --narrative-model MODEL        # Model for lyrics reconciliation (default: anthropic/claude-sonnet-4)
  --output project.yaml          # Output path (default: project.yaml)
```

Produces a YAML project file with sections, clip ranges, and mood suggestions. Always review and edit before generating.

### `filmmaker generate`

Runs the agentic generation loop. Generates a seed image via DALL-E 3 if none exists, then chains through all clips.

```
filmmaker generate project.yaml
filmmaker generate project.yaml --start-from 15   # Resume from clip 15
filmmaker generate project.yaml --clips 5          # Generate only 5 clips (test run)
```

Each clip is saved as `clips/clip_NN.mp4`. A manifest (`clips/manifest.json`) tracks all prompts, vision analyses, and statuses. If generation fails on a clip, it logs the error, skips, and continues. Re-run with `--start-from` to fill gaps.

### `filmmaker assemble`

Concatenates all clips and overlays the audio track via ffmpeg.

```
filmmaker assemble project.yaml
filmmaker assemble project.yaml --output final.mp4   # Custom output path
filmmaker assemble project.yaml --audio other.wav     # Override audio track
```

Uses `yuv420p` pixel format for broad player compatibility. Does not use ffmpeg's `-shortest` flag (which causes audio cutoff).

## Project YAML format

The project file defines everything about the video. See `examples/fsf.yaml` for a fully annotated real-world example.

```yaml
project:
  title: "Song Title"
  artist: "Artist Name"
  album: "Album Name"

audio:
  path: "track.wav"              # Relative to project dir
  duration: 314.4                # In seconds (auto-detected by init)

video:
  clip_duration: 8               # Seconds per generated clip
  size: "1280x720"               # Video dimensions
  model: "sora-2"                # Video generation model
  slow_motion_factor: 1.0        # >1.0 slows video by this factor in assembly
  seeding_mode: "continuous"     # "continuous" or "interspersed" (see below)
  reference_model: "black-forest-labs/flux.2-max"   # For interspersed mode
  reference_prompt: "..."        # Character description(s) for Flux refs

vision:
  model: "gpt-4o"                # Must support image input
  system_prompt: "..."           # How to analyze frames
  max_tokens: 300

narrative:
  model: "anthropic/claude-sonnet-4"   # Any OpenRouter model slug
  system_prompt: "..."           # How to write scenes (auto-generated if omitted)
  max_tokens: 200

style:
  global: "..."                  # Visual style injected into every video prompt
  no_text: true                  # Append no-text enforcement to all prompts

characters:
  - name: "protagonist"
    description: "..."           # Physical description (injected into every prompt)
    appears: "every"             # "every" = every clip
  - name: "observer"
    description: "..."
    appears: "periodic"          # Appears every N clips
    frequency: 4

camera_moves:                    # Cycled deterministically across clips
  - "Tracking shot left to right"
  - "Slow dolly push-in"
  - "Crane shot rising upward"

seed:
  prompt: "..."                  # DALL-E 3 generates seed image from this
  image: "seed.png"              # Or provide your own (relative to project dir)

sections:
  - name: "Intro"
    clips: [0, 2]               # Inclusive clip range (0-indexed)
    lyrics: "(Instrumental)"
    mood: "Establishing shot..." # Cinematic mood description for narrative agent
    slow_motion: true            # Hint for narrative pacing
  - name: "Verse 1"
    clips: [3, 9]
    lyrics: "First verse lyrics here..."
    mood: "Energetic performance..."
    slow_motion: false
```

## Seeding modes

### Continuous (default)

Every clip uses the last frame of the previous clip as its seed image. This produces smooth, dreamlike visual continuity where each scene flows naturally into the next.

```yaml
video:
  seeding_mode: "continuous"
```

### Interspersed

Alternates between freeze frames (continuity) and fresh Flux-generated character reference images (likeness reset). Instead of rigid alternation, the pipeline runs a random chain of 1-4 freeze frames before injecting a fresh Flux reference. This produces a more dynamic, cut-heavy result where the character's appearance stays anchored across jump cuts.

Supports multiple reference prompts -- each fresh reference randomly picks from the list, giving the character different looks across the video.

```yaml
video:
  seeding_mode: "interspersed"
  reference_model: "black-forest-labs/flux.2-max"
  reference_prompt:
    - "A young woman with dark hair, 1960s fashion, beehive hairstyle"
    - "A young woman with dark hair, mod dress, go-go boots"
```

Requires `OPENROUTER_API_KEY` for Flux image generation. Falls back to continuous mode if the key is not set.

## Storyboard

A live progress viewer is included at the project root (`storyboard.html`). Copy it into your project directory and serve via any HTTP server to monitor generation in real time.

```bash
cp storyboard.html my-project/
cd my-project
python -m http.server 8000
# Open http://localhost:8000/storyboard.html
```

The storyboard reads `clips/manifest.json` (auto-refreshes every 15 seconds) and displays:

- **Timeline** (left): All clips with status indicators (green = ok, red = error, grey = pending)
- **Viewer** (center): Video playback with seed image overlay, navigation, autoplay
- **Detail panel** (right): Scene prompt, vision analysis, next scene, seed image preview, errors

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| Left / `j` | Previous clip |
| Right / `k` | Next clip |
| Space | Play/pause current clip |
| `a` | Toggle autoplay (sequential viewing) |
| `f` | Fullscreen current clip |
| Escape | Close lightbox |

A collapsible seed gallery at the bottom shows all seed images and Flux references used across the project.

## Manifest format

The pipeline writes `clips/manifest.json` as it generates. This is both a progress tracker and a complete record of every decision the agents made.

```json
{
  "song": "Song Title",
  "artist": "Artist Name",
  "album": "Album Name",
  "n_clips": 39,
  "seeding_mode": "continuous",
  "video_model": "sora-2",
  "vision_model": "gpt-4o",
  "narrative_model": "anthropic/claude-sonnet-4",
  "reference_model": null,
  "started": "2025-03-01T12:00:00+00:00",
  "completed": "2025-03-01T14:30:00+00:00",
  "clips": [
    {
      "clip": 0,
      "prompt": "Scene description used for generation...",
      "seed_image": "clips/seed.png",
      "seed_type": "seed",
      "section": "Intro",
      "timestamp": "2025-03-01T12:01:00+00:00",
      "video": "clip_00.mp4",
      "status": "ok",
      "vision_analysis": "GPT-4o's description of the last frame...",
      "next_prompt": "Claude's scene description for the next clip..."
    }
  ]
}
```

Each clip entry records: the prompt sent to the video generator, which seed image was used and its type (`seed`, `freeze`, `flux-ref`, `freeze-fallback`), the vision analysis of the resulting frame, and the narrative agent's output for the next clip. Failed clips have `"status": "error"` with an `"error"` field.

## Pluggable generators

The video generation backend is an abstract base class. To add a new backend, subclass `VideoGenerator`:

```python
from pathlib import Path
from filmmaker.generators.base import VideoGenerator

class MyGenerator(VideoGenerator):
    def generate(
        self,
        prompt: str,
        seed_image: Path,
        output_path: Path,
        duration: int = 8,
        size: str = "1280x720",
    ) -> Path:
        """Generate a video clip from a seed image + text prompt.

        Args:
            prompt: Scene description for the video.
            seed_image: Path to the seed image (already resized to target dims).
            output_path: Where to save the generated .mp4.
            duration: Clip duration in seconds.
            size: Video dimensions as "WxH".

        Returns:
            Path to the generated video file.
        """
        # Your video generation logic here
        ...
        return output_path

    def generate_seed_image(
        self,
        prompt: str,
        output_path: Path,
        size: str = "1792x1024",
    ) -> Path:
        """Generate an initial seed image from a text prompt.

        Args:
            prompt: Scene description for the seed image.
            output_path: Where to save the generated image.
            size: Image dimensions as "WxH".

        Returns:
            Path to the generated image file.
        """
        # Your image generation logic here
        ...
        return output_path
```

The built-in generators are:
- **`Sora2Generator`** -- OpenAI Sora 2 for video, DALL-E 3 for seed images. Default.
- **`FluxGenerator`** -- Black Forest Labs Flux via OpenRouter. Used for character reference images in interspersed seeding mode.

To wire in a custom generator, instantiate it in place of `Sora2Generator` in your own script, or modify `cli.py`.

## File structure

```
filmmaker/
├── cli.py             # CLI entry point (init, generate, assemble)
├── pipeline.py        # Core agentic loop + _SeedPicker
├── config.py          # YAML config loader, all dataclasses
├── vision.py          # Frame analysis agent (GPT-4o)
├── narrative.py       # Scene generation agent (Claude)
├── frames.py          # Frame extraction, resize, base64 encoding
├── audio.py           # Audio section detection (librosa) + lyrics reconciliation
├── assembly.py        # ffmpeg concat + audio overlay
└── generators/
    ├── base.py        # Abstract VideoGenerator
    ├── sora2.py       # Sora 2 + DALL-E 3 implementation
    └── flux.py        # Flux reference image generation (OpenRouter)

storyboard.html        # Generic reusable progress viewer
examples/
└── fsf.yaml           # Complete example project
```

## Requirements

- **Python 3.11+**
- **ffmpeg** (for frame extraction and final assembly)
- **API keys:**
  - `OPENAI_API_KEY` -- Sora 2 (video generation), DALL-E 3 (seed images), GPT-4o (vision)
  - `OPENROUTER_API_KEY` -- narrative agent (Claude or any OpenRouter model); also required for Flux references in interspersed mode
- **Optional:** `librosa` + `numpy` for audio analysis in `init` (install with `pip install -e ".[audio]"`)

## Credits

Built by [Iman Poernomo](https://tanazur.org) and Nahla. Part of the [Cassie project](https://github.com/thegoodtailor). First used to produce music videos for [The Dependent Halo](https://halo.tanazur.org).

## License

MIT
