# agentic-filmmaker

AI-powered music video pipeline. Three agents chain-generate a music video from a seed image, one clip at a time.

## How it works

```
seed image → [Sora 2] → clip 0 → last frame → [GPT-4o vision] → frame description
                                                                         ↓
                                                                  [Claude narrative] → scene prompt
                                                                         ↓
                                                   clip 1 ← [Sora 2] ← seed image (last frame)
                                                      ↓
                                                   ... repeat ...
                                                      ↓
                                                   [ffmpeg] → final video + audio
```

Each clip is born from the last frame of the previous one. A vision agent describes what's on screen, a narrative agent writes what happens next, and a video generator brings it to life. The film dreams itself into existence.

## Quick start

```bash
pip install -e ".[audio]"

# Set up API keys
export OPENAI_API_KEY="sk-..."          # For Sora 2, DALL-E 3, GPT-4o
export OPENROUTER_API_KEY="sk-or-..."   # For narrative agent (Claude, etc.)

# Option A: Create project from audio analysis
filmmaker init --audio track.wav --lyrics lyrics.txt --title "My Song" --artist "My Band"
# Edit project.yaml — add style, characters, seed prompt, tweak moods
filmmaker generate project.yaml
filmmaker assemble project.yaml

# Option B: Write project YAML by hand (see examples/fsf.yaml)
filmmaker generate my_project.yaml
filmmaker assemble my_project.yaml
```

## Commands

### `filmmaker init`

Analyzes a WAV/MP3 file to detect section boundaries (verse/chorus/bridge transitions), then uses an LLM to match those boundaries with your lyrics and generate mood descriptions.

```bash
filmmaker init \
  --audio track.wav \
  --lyrics lyrics.txt \
  --title "Song Title" \
  --artist "Artist" \
  --output project.yaml
```

Produces a YAML project file with sections, clip ranges, and mood suggestions. Review and edit before generating.

### `filmmaker generate`

Runs the agentic generation loop. Generates a seed image (DALL-E 3) if none exists, then chains through all clips.

```bash
filmmaker generate project.yaml
filmmaker generate project.yaml --start-from 15   # Resume from clip 15
filmmaker generate project.yaml --clips 5          # Test run (first 5 clips)
```

Each clip is saved as `clips/clip_NN.mp4`. A manifest (`clips/manifest.json`) tracks all prompts, vision analyses, and statuses. If generation fails on a clip, it skips and continues — you can re-run with `--start-from` to fill gaps.

### `filmmaker assemble`

Concatenates all clips and overlays the audio track.

```bash
filmmaker assemble project.yaml
filmmaker assemble project.yaml --output final.mp4
```

## Project YAML format

See [`examples/fsf.yaml`](examples/fsf.yaml) for a complete example.

```yaml
project:
  title: "Song Title"
  artist: "Artist Name"

audio:
  path: "track.wav"
  duration: 314.4              # auto-detected by `init`

video:
  clip_duration: 8             # seconds per clip
  size: "1280x720"
  model: "sora-2"

vision:
  model: "gpt-4o"
  system_prompt: "..."         # how to analyze frames

narrative:
  model: "anthropic/claude-sonnet-4"
  system_prompt: "..."         # how to write scenes (auto-generated if omitted)

style:
  global: "..."                # visual style injected into every video prompt
  no_text: true                # append no-text enforcement

characters:
  - name: "protagonist"
    description: "..."         # physical description (injected into every prompt)
    appears: "every"
  - name: "observer"
    description: "..."
    appears: "periodic"
    frequency: 4

camera_moves:                  # deterministic cycle
  - "Tracking shot left to right"
  - "Slow dolly push-in"
  # ...

seed:
  prompt: "..."                # DALL-E 3 generates seed image from this
  image: "seed.png"            # or provide your own

sections:
  - name: "Intro"
    clips: [0, 2]
    lyrics: "(Instrumental)"
    mood: "Establishing shot..."
    slow_motion: true
  # ...
```

## Architecture

The pipeline is designed around pluggable components:

- **`VideoGenerator`** (abstract base class) — implement `generate()` and `generate_seed_image()` to add new video backends. Sora 2 is the default.
- **Vision agent** — any model that accepts image input. Default: GPT-4o.
- **Narrative agent** — any text model. Default: Claude Sonnet 4 via OpenRouter. The model choice here matters most — different LLMs produce very different visual prose.

### File structure

```
filmmaker/
├── pipeline.py        # Core orchestration loop
├── config.py          # YAML project loader
├── vision.py          # Frame analysis agent
├── narrative.py       # Scene generation agent
├── frames.py          # Frame extraction + image utilities
├── audio.py           # Audio analysis (librosa) for project init
├── assembly.py        # ffmpeg concat + audio overlay
├── cli.py             # CLI entry point
└── generators/
    ├── base.py        # Abstract VideoGenerator
    └── sora2.py       # Sora 2 + DALL-E 3 implementation
```

## Requirements

- Python 3.11+
- ffmpeg (for frame extraction and assembly)
- API keys: OpenAI (Sora 2, GPT-4o, DALL-E 3) + OpenRouter (narrative agent)
- Optional: librosa (for audio analysis in `init`)

## Writing a custom video generator

```python
from filmmaker.generators.base import VideoGenerator

class MyGenerator(VideoGenerator):
    def generate(self, prompt, seed_image, output_path, duration=8, size="1280x720"):
        # Your video generation logic here
        ...
        return output_path

    def generate_seed_image(self, prompt, output_path, size="1792x1024"):
        # Your image generation logic here
        ...
        return output_path
```

## Credits

Built by [Iman Poernomo](https://tanazur.org) and Nahla. First used to produce music videos for [The Dependent Halo](https://halo.tanazur.org).

## License

MIT
