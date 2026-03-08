"""Core pipeline — the agentic generation loop."""

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv

from .config import ProjectConfig
from .frames import extract_last_frame
from .generators import VideoGenerator
from .narrative import generate_scene
from .vision import analyze_frame


def generate(
    config: ProjectConfig,
    generator: VideoGenerator,
    openai_client: openai.OpenAI,
    narrative_client: openai.OpenAI,
    start_from: int = 0,
    n_clips: int | None = None,
) -> Path:
    """Run the agentic music video generation loop.

    For each clip:
    1. Generate video from seed image + scene prompt (via VideoGenerator)
    2. Extract last frame of the generated clip
    3. Vision agent analyzes the frame
    4. Narrative agent writes the next scene description

    The narrative output becomes the prompt for the next clip, creating
    a chain of visually continuous clips that dream themselves forward.

    Args:
        config: Loaded project configuration.
        generator: VideoGenerator instance (e.g. Sora2Generator).
        openai_client: OpenAI client for vision (GPT-4o).
        narrative_client: Client for narrative agent (e.g. OpenRouter).
        start_from: Resume from this clip number.
        n_clips: Override total clip count (default: from config).

    Returns:
        Path to the manifest.json file.
    """
    total_clips = n_clips or config.get_total_clips()
    clips_dir = config.project_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = clips_dir / "manifest.json"

    print(f"\n{'='*60}")
    print(f"  {config.title}")
    if config.artist:
        print(f"  {config.artist}")
    print(f"  {total_clips} clips | {config.audio.duration:.0f}s")
    print(f"{'='*60}\n")

    # Load or initialize manifest
    if manifest_path.exists() and start_from > 0:
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["n_clips"] = total_clips
    else:
        manifest = {
            "song": config.title,
            "artist": config.artist,
            "album": config.album,
            "n_clips": total_clips,
            "video_model": config.video.model,
            "vision_model": config.vision.model,
            "narrative_model": config.narrative.model,
            "started": datetime.now(timezone.utc).isoformat(),
            "clips": [],
        }

    # Seed image — generate or use provided
    seed_path = clips_dir / "seed.png"
    if not seed_path.exists():
        if config.seed.image:
            src = config.project_dir / config.seed.image
            shutil.copy2(src, seed_path)
            print(f"  Copied seed image: {src.name}")
        elif config.seed.prompt:
            generator.generate_seed_image(
                prompt=config.seed.prompt,
                output_path=seed_path,
            )
        else:
            raise FileNotFoundError(
                "No seed image found. Provide seed.image or seed.prompt in config."
            )

    # Build style prefix for Sora prompts
    style_prefix = config.style.global_style.strip()
    char_desc = config.get_character_descriptions()
    if char_desc:
        style_prefix = f"{style_prefix} {char_desc}"

    # Reconstruct state for resumption
    story_so_far = [config.seed.prompt or "Opening scene."]
    current_seed = seed_path
    current_prompt = story_so_far[0]

    if start_from > 0 and manifest.get("clips"):
        for clip_data in manifest["clips"][:start_from]:
            story_so_far.append(clip_data.get("prompt", ""))
        prev_clip = clips_dir / f"clip_{start_from - 1:02d}.mp4"
        if prev_clip.exists():
            current_seed = extract_last_frame(prev_clip)
        if manifest["clips"]:
            current_prompt = manifest["clips"][-1].get("prompt", current_prompt)

    # Main generation loop
    for i in range(start_from, total_clips):
        section = config.get_section_for_clip(i)
        print(f"\n  --- Clip {i:02d} / {total_clips - 1:02d} [{section.name}] ---")
        clip_path = clips_dir / f"clip_{i:02d}.mp4"

        clip_data = {
            "clip": i,
            "prompt": current_prompt,
            "seed_image": str(current_seed),
            "section": section.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Prepend style + character descriptions (video generator has no memory)
            styled_prompt = f"{style_prefix} {current_prompt}"
            generator.generate(
                prompt=styled_prompt,
                seed_image=current_seed,
                output_path=clip_path,
                duration=config.video.clip_duration,
                size=config.video.size,
            )
            clip_data["video"] = clip_path.name
            clip_data["status"] = "ok"

            if i < total_clips - 1:
                # Extract last frame → vision → narrative → next prompt
                next_seed = extract_last_frame(clip_path)
                frame_description = analyze_frame(
                    image_path=next_seed,
                    system_prompt=config.vision.system_prompt,
                    model=config.vision.model,
                    max_tokens=config.vision.max_tokens,
                    client=openai_client,
                )
                clip_data["vision_analysis"] = frame_description

                next_prompt = generate_scene(
                    section=config.get_section_for_clip(i + 1),
                    story_so_far=story_so_far,
                    frame_description=frame_description,
                    clip_number=i + 1,
                    total_clips=total_clips,
                    camera_move=config.get_camera_move(i + 1),
                    characters=config.characters,
                    system_prompt=config.narrative.system_prompt,
                    model=config.narrative.model,
                    max_tokens=config.narrative.max_tokens,
                    client=narrative_client,
                )
                clip_data["next_prompt"] = next_prompt

                story_so_far.append(next_prompt)
                current_seed = next_seed
                current_prompt = next_prompt

        except Exception as e:
            print(f"    ERROR on clip {i}: {e}")
            clip_data["status"] = "error"
            clip_data["error"] = str(e)
            _save_manifest(manifest, clip_data, manifest_path)
            print(f"    Skipping clip {i}, continuing...")
            time.sleep(30)
            continue

        _save_manifest(manifest, clip_data, manifest_path)

        if i < total_clips - 1:
            time.sleep(2)

    manifest["completed"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  COMPLETE: {config.title}")
    print(f"  {total_clips} clips in {clips_dir}")
    print(f"{'='*60}\n")

    return manifest_path


def _save_manifest(manifest: dict, clip_data: dict, path: Path):
    """Save clip data to manifest, updating in place or appending."""
    clip_idx = clip_data["clip"]
    if clip_idx < len(manifest["clips"]):
        manifest["clips"][clip_idx] = clip_data
    else:
        manifest["clips"].append(clip_data)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
