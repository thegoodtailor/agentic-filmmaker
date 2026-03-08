"""Project configuration — YAML-based, fully typed."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    path: str
    duration: float = 0.0


@dataclass
class VideoConfig:
    clip_duration: int = 8
    size: str = "1280x720"
    model: str = "sora-2"
    slow_motion_factor: float = 1.0


@dataclass
class VisionConfig:
    model: str = "gpt-4o"
    system_prompt: str = (
        "You are analyzing a frame from a music video.\n"
        "Describe precisely:\n"
        "- Characters: appearance, pose, expression, body language\n"
        "- Environment: setting, lighting, colour palette\n"
        "- Implied motion or mood\n"
        "Be specific and visual. 3-4 sentences."
    )
    max_tokens: int = 300


@dataclass
class NarrativeConfig:
    model: str = "anthropic/claude-sonnet-4"
    system_prompt: str = ""
    max_tokens: int = 200


@dataclass
class StyleConfig:
    global_style: str = ""
    no_text: bool = True


@dataclass
class Character:
    name: str
    description: str
    appears: str = "every"  # "every" or "periodic"
    frequency: int = 4      # only used if appears == "periodic"


@dataclass
class Section:
    name: str
    clips: tuple[int, int]
    lyrics: str = ""
    mood: str = ""
    slow_motion: bool = False


@dataclass
class SeedConfig:
    prompt: str = ""
    image: str | None = None  # path to user-provided seed image


@dataclass
class ProjectConfig:
    title: str
    artist: str = ""
    album: str = ""
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    narrative: NarrativeConfig = field(default_factory=NarrativeConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    characters: list[Character] = field(default_factory=list)
    camera_moves: list[str] = field(default_factory=lambda: [
        "Steady tracking shot moving left to right, following the subject",
        "Slow dolly push-in from medium to close-up",
        "Crane shot rising upward to reveal the surroundings",
        "Handheld close-up with intimate proximity",
        "Wide locked-off shot, subject moves through frame",
        "Slow orbit circling around the subject",
        "Tracking shot from behind, following subject away from camera",
        "Low-angle dolly shot looking up at the subject",
    ])
    sections: list[Section] = field(default_factory=list)
    seed: SeedConfig = field(default_factory=SeedConfig)
    project_dir: Path = field(default_factory=lambda: Path("."))

    def get_section_for_clip(self, clip_num: int) -> Section:
        for section in self.sections:
            start, end = section.clips
            if start <= clip_num <= end:
                return section
        return self.sections[-1]

    def get_camera_move(self, clip_num: int) -> str:
        return self.camera_moves[clip_num % len(self.camera_moves)]

    def get_total_clips(self) -> int:
        if not self.sections:
            return 0
        return max(s.clips[1] for s in self.sections) + 1

    def get_character_descriptions(self) -> str:
        parts = []
        for c in self.characters:
            parts.append(c.description.strip())
        return "\n".join(parts)


def _build_default_narrative_prompt(config_dict: dict) -> str:
    """Build a default narrative system prompt from project metadata."""
    title = config_dict.get("project", {}).get("title", "a music video")
    chars = config_dict.get("characters", [])

    char_block = ""
    for c in chars:
        label = c.get("name", "character").upper()
        desc = c.get("description", "").strip()
        appears = c.get("appears", "every")
        if appears == "every":
            char_block += f"\n{label} (must appear in EVERY scene):\n{desc}\n"
        else:
            freq = c.get("frequency", 4)
            char_block += f"\n{label} (background, every {freq} clips):\n{desc}\n"

    return f"""You are directing a music video for "{title}".

You write scene descriptions for video generation. The video generator has NO memory
between clips — it does not know character names. You must DESCRIBE actors physically
every single time. NEVER use character names in scene descriptions.

CRITICAL — MOTION:
Every prompt MUST describe continuous physical movement. The subject is NEVER standing still.
You will be given a CAMERA DIRECTION for each clip. USE IT.
{char_block}
RULES:
- Keep prompts to 2-4 sentences of PLAIN PROSE — no markdown, no headers, no bold
- Do NOT include any text, writing, or symbols in the scene
- Scene changes happen naturally — costume and setting shift dreamlike"""


def load_config(yaml_path: Path) -> ProjectConfig:
    """Load a YAML project file into a ProjectConfig."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    project = raw.get("project", {})
    audio_raw = raw.get("audio", {})
    video_raw = raw.get("video", {})
    vision_raw = raw.get("vision", {})
    narrative_raw = raw.get("narrative", {})
    style_raw = raw.get("style", {})
    seed_raw = raw.get("seed", {})

    characters = [
        Character(
            name=c["name"],
            description=c.get("description", ""),
            appears=c.get("appears", "every"),
            frequency=c.get("frequency", 4),
        )
        for c in raw.get("characters", [])
    ]

    sections = [
        Section(
            name=s["name"],
            clips=tuple(s["clips"]),
            lyrics=s.get("lyrics", ""),
            mood=s.get("mood", ""),
            slow_motion=s.get("slow_motion", False),
        )
        for s in raw.get("sections", [])
    ]

    # Build default narrative prompt if none provided
    narrative_prompt = narrative_raw.get("system_prompt", "")
    if not narrative_prompt:
        narrative_prompt = _build_default_narrative_prompt(raw)

    config = ProjectConfig(
        title=project.get("title", "Untitled"),
        artist=project.get("artist", ""),
        album=project.get("album", ""),
        audio=AudioConfig(
            path=audio_raw.get("path", ""),
            duration=audio_raw.get("duration", 0.0),
        ),
        video=VideoConfig(
            clip_duration=video_raw.get("clip_duration", 8),
            size=video_raw.get("size", "1280x720"),
            model=video_raw.get("model", "sora-2"),
            slow_motion_factor=video_raw.get("slow_motion_factor", 1.0),
        ),
        vision=VisionConfig(
            model=vision_raw.get("model", "gpt-4o"),
            system_prompt=vision_raw.get("system_prompt", VisionConfig.system_prompt),
            max_tokens=vision_raw.get("max_tokens", 300),
        ),
        narrative=NarrativeConfig(
            model=narrative_raw.get("model", "anthropic/claude-sonnet-4"),
            system_prompt=narrative_prompt,
            max_tokens=narrative_raw.get("max_tokens", 200),
        ),
        style=StyleConfig(
            global_style=style_raw.get("global", ""),
            no_text=style_raw.get("no_text", True),
        ),
        characters=characters,
        camera_moves=raw.get("camera_moves", ProjectConfig.camera_moves),
        sections=sections,
        seed=SeedConfig(
            prompt=seed_raw.get("prompt", ""),
            image=seed_raw.get("image"),
        ),
        project_dir=yaml_path.parent,
    )

    return config


def save_config(config: ProjectConfig, yaml_path: Path) -> None:
    """Serialize a ProjectConfig to YAML."""
    data = {
        "project": {
            "title": config.title,
            "artist": config.artist,
            "album": config.album,
        },
        "audio": {
            "path": config.audio.path,
            "duration": config.audio.duration,
        },
        "video": {
            "clip_duration": config.video.clip_duration,
            "size": config.video.size,
            "model": config.video.model,
            "slow_motion_factor": config.video.slow_motion_factor,
        },
        "vision": {
            "model": config.vision.model,
            "system_prompt": config.vision.system_prompt,
            "max_tokens": config.vision.max_tokens,
        },
        "narrative": {
            "model": config.narrative.model,
            "system_prompt": config.narrative.system_prompt,
            "max_tokens": config.narrative.max_tokens,
        },
        "style": {
            "global": config.style.global_style,
            "no_text": config.style.no_text,
        },
        "characters": [
            {
                "name": c.name,
                "description": c.description,
                "appears": c.appears,
                "frequency": c.frequency,
            }
            for c in config.characters
        ],
        "camera_moves": config.camera_moves,
        "sections": [
            {
                "name": s.name,
                "clips": list(s.clips),
                "lyrics": s.lyrics,
                "mood": s.mood,
                "slow_motion": s.slow_motion,
            }
            for s in config.sections
        ],
        "seed": {
            "prompt": config.seed.prompt,
            "image": config.seed.image,
        },
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
