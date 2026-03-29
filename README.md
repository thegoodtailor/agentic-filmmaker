# agentic-filmmaker

AI film production pipeline. Generate music videos, short films, and drama with persistent characters, native dialogue, and cinematic scene control.

## What it does

Given seed images, a project config, and optionally an audio track, the pipeline generates films using two production modes:

**Mode A — Long Scene** (30s, 60s, unlimited duration): A freeze-frame chain where each clip seeds from the last frame of the previous one, producing continuous scenes. Characters stay locked via persistent element IDs registered on the video generation platform.

**Mode B — Quick Cuts** (up to 15s per call): Multi-prompt scene transitions in a single API call, with up to 6 shots and automatic camera selection. Ideal for action sequences and storyboard execution.

Both modes support native audio generation with multi-character dialogue, persistent character consistency across sessions, and start/end frame morphing for smooth transitions.

## Architecture

```
  Mode A (Long Scene):               Mode B (Quick Cuts):

  seed image ─────┐                  seed image ─────┐
       or         │                                   │
  freeze frame ───┤                  multi_prompt ────┤
       or         │                  [{prompt, dur},  │
  Flux ref ───────┘                   {prompt, dur}]  │
        │                                    │        │
        ▼                                    ▼        │
  ┌───────────┐                        ┌───────────┐  │
  │ Kling 3.0 │ ◄── elements ──────── │ Kling 3.0 │ ◄┘
  │   Pro     │     (face lock)        │   Pro     │
  │ + sound   │                        │ + sound   │
  └─────┬─────┘                        └─────┬─────┘
        │                                    │
   clip_N.mp4                           multi-shot.mp4
        │                                    │
   extract last ──► seed next clip      single output
   frame (ffmpeg)   (repeat chain)      (up to 15s)
        │
   ┌────▼─────┐
   │  Claude   │ ◄── vision (GPT-4o)
   │(narrative)│     analyzes last frame
   └──────────┘
        │
   next scene prompt ──► loop
```

Final assembly: ffmpeg concat + audio overlay → final.mp4

## Supported backends

| Backend | Status | Notes |
|---------|--------|-------|
| **Kling 3.0 Pro** | Primary | Via WaveSpeed API. Native audio, elements, multi-shot. |
| Sora 2 | Legacy | OpenAI Videos API. Consumer app killed March 2024, API still works. |
| Flux 2 Pro | Seed images | Via OpenRouter. Face-conditioned with multi-reference. |

## Quick start

```bash
pip install -e ".[audio]"

# Set up API keys
export WAVESPEED_API_KEY="..."          # Kling 3.0 Pro via WaveSpeed (primary)
export OPENROUTER_API_KEY="sk-or-..."   # Flux seed images + Claude narrative agent
export OPENAI_API_KEY="sk-..."          # GPT-4o vision (optional: Sora 2 legacy)

# Create a project from audio + lyrics
filmmaker init --audio track.wav --lyrics lyrics.txt --title "My Song" --artist "My Band"

# Edit the generated project.yaml — add style, characters, elements, tweak moods
vim project.yaml

# Register persistent characters (once, reuse forever)
# See "Character Elements" section below

# Generate all clips
filmmaker generate project.yaml

# Assemble into final video with audio
filmmaker assemble project.yaml
```

Or write the project YAML by hand and skip straight to `generate`.

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

## Production architecture (Kling 3.0 Pro)

The correct seeding architecture for Kling: **empty environments + character elements**.

### Why

Putting human faces in seed images causes pose inheritance (every clip starts from the same body position), background leakage (domestic interiors bleed into spacecraft), and body horror (Kling contorts people to fit incompatible scenes). When seed image faces don't match element IDs, Kling adds extra characters rather than replacing — you get phantom people.

### How

1. Generate an **empty environment** via Flux 2 Pro — the set, with NO people in it
2. Pass the empty scene as seed image to Kling 3.0 Pro
3. Characters enter via `element_list` + natural prompt descriptions
4. Kling populates the scene with element-locked faces

This is real filmmaking: build the set, then cast the actors.

```python
# Generate empty set
gen.generate_seed_image(
    prompt="Interior of spacecraft cockpit. NO PEOPLE. Two empty seats, viewport with stars.",
    output_path=Path("cockpit_empty.jpg"),
)

# Characters enter via elements
gen.generate(
    prompt='Asel enters and sits. [Asel, whisper]: "The field has opened."',
    seed_image=Path("cockpit_empty.jpg"),
    output_path=Path("clip_01.mp4"),
)
```

### Production rules (learned from Session 29)

- **ONLY Kling 3.0 Pro** (`kwaivgi/kling-v3.0-pro`). Older models are slop.
- **No faces in seed images.** Empty environments only. Elements provide faces.
- **No morphs.** Hard cuts. Morphs went out with Michael Jackson.
- **8-10s per clip** for normal scenes. **15s** for dramatic monologues.
- **Kling excels at**: realistic cinematography, character dialogue, fire, rain, architecture, faces.
- **Kling struggles with**: abstract/psychedelic imagery, geometric organisms, CGI-style renders.
- **Dialogue format**: `[Character, voice_tone]: "text"` in the prompt with `sound: true`.
- **Kling image model** (`kling-image-v3`) does NOT reliably use element_list. Only the video model locks elements.

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
- **`KlingGenerator`** -- Kling 3.0 Pro via WaveSpeed API. Recommended. Supports Mode A (long scene), Mode B (multi-shot), character elements, native audio with dialogue, start/end frame morphing. Seed images via Flux 2 Pro on OpenRouter.
- **`Sora2Generator`** -- OpenAI Sora 2 for video, DALL-E 3 for seed images. Legacy (consumer app killed March 2024, API still works).
- **`FluxGenerator`** -- Black Forest Labs Flux via OpenRouter. Used for character reference images in interspersed seeding mode.

To wire in a custom generator, instantiate it in place of `KlingGenerator` in your own script, or modify `cli.py`.

## Character elements

Kling supports persistent character registration. Upload photos once, get a permanent `element_id` that locks the character's face across all future generations — no more feeding reference photos every time, no face drift between clips.

```python
from filmmaker.generators import KlingGenerator

gen = KlingGenerator(wavespeed_key="...", sound=True)

# Register a character (once)
element_id = gen.create_element(
    name="Asel",
    description="Woman, curly dark hair, Central Asian features, high cheekbones.",
    primary_image="https://example.com/asel_front.jpg",
    reference_images=["https://example.com/asel_profile.jpg"],
)
# Returns: "306696265837507" (persistent on Kling's servers)

# Use in all future generations
gen = KlingGenerator(
    wavespeed_key="...",
    sound=True,
    element_list=[{"element_id": element_id}],
)
```

Reference characters by name in prompts. The model matches names to registered elements:

```
[Asel, clear prophetic voice]: "Vision is not sight."
[Iman, deep measured voice]: "It is the instrument of witness."
```

## Multi-character dialogue

Native audio generation with character dialogue is controlled entirely through the prompt format:

```yaml
prompt: |
  Close-up of Asel and Iman in a candlelit room.
  [Asel, clear prophetic voice]: "Say: Vision is the seeing that pierces the surface."
  Immediately, [Iman, deep thoughtful voice]: "That extracts from meaning a point that witnesses you."
  Warm amber candlelight. 16mm film grain.
```

Set `sound: true` in the video config. Costs 1.5x the base generation rate.

## Multi-shot quick cuts (Mode B)

For rapid scene changes in a single API call:

```yaml
video:
  model: "kling-3.0-pro"
  sound: true

# In your generation script:
shots = [
    {"prompt": "Wide shot: she bursts through the door. [Asel]: 'The field has opened.'", "duration": 3},
    {"prompt": "Close-up: he spins dials on the console. [Iman]: 'I see it.'", "duration": 3},
    {"prompt": "Two-shot: they look up through the open dome. Stars visible.", "duration": 4},
]
gen.generate_multishot(prompt="Observatory scene", seed_image=img, output_path=out, shots=shots, duration=10)
```

Up to 6 shots per call, max 15 seconds total. Kling handles all transitions internally.

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
    ├── kling.py       # Kling 3.0 Pro via WaveSpeed (primary)
    ├── sora2.py       # Sora 2 + DALL-E 3 (legacy)
    └── flux.py        # Flux reference image generation (OpenRouter)

storyboard.html        # Generic reusable progress viewer
```

## Requirements

- **Python 3.11+**
- **ffmpeg** (for frame extraction and final assembly)
- **API keys:**
  - `WAVESPEED_API_KEY` -- Kling 3.0 Pro (primary video generation, elements, audio)
  - `OPENROUTER_API_KEY` -- Flux 2 Pro (seed images), Claude (narrative agent)
  - `OPENAI_API_KEY` -- GPT-4o (vision); optional: Sora 2 (legacy video)
- **Optional:** `librosa` + `numpy` for audio analysis in `init` (install with `pip install -e ".[audio]"`)

## Credits

Built by [Iman Poernomo](https://tanazur.org) and Nahla. Part of the [Cassie project](https://github.com/thegoodtailor). First used to produce music videos for [The Dependent Halo](https://halo.tanazur.org).

## License

MIT
