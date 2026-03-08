"""Core pipeline — the agentic generation loop."""

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv

from .config import ProjectConfig
from .frames import extract_last_frame
from .generators import VideoGenerator
from .generators.flux import FluxGenerator
from .narrative import generate_scene
from .vision import analyze_frame


def _pick_seed_image(
    clip_num: int,
    freeze_frame: Path | None,
    seed_path: Path,
    refs_dir: Path,
    flux: FluxGenerator | None,
    reference_prompt: str,
    section_mood: str,
) -> tuple[Path, str]:
    """Pick seed image for a clip in interspersed mode.

    Even clips use the freeze frame (continuity). Odd clips use a
    Flux-generated character reference image (likeness anchoring).
    Clip 0 always uses the initial seed.
    """
    if clip_num == 0 or freeze_frame is None:
        return seed_path, "seed"

    if clip_num % 2 == 0:
        return freeze_frame, "freeze"

    # Odd clip: generate a character reference via Flux
    if flux and reference_prompt:
        ref_path = refs_dir / f"ref_{clip_num:02d}.png"
        if not ref_path.exists():
            # Combine character description with current section mood
            full_prompt = f"{reference_prompt} Scene context: {section_mood}"
            try:
                flux.generate(prompt=full_prompt, output_path=ref_path)
                return ref_path, f"flux-ref"
            except Exception as e:
                print(f"    Flux ref failed, falling back to freeze frame: {e}")
                return freeze_frame, "freeze-fallback"
        return ref_path, "flux-ref"

    return freeze_frame, "freeze"


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
    1. Pick seed image (freeze frame or character reference in interspersed mode)
    2. Generate video from seed image + scene prompt (via VideoGenerator)
    3. Extract last frame of the generated clip
    4. Vision agent analyzes the frame
    5. Narrative agent writes the next scene description

    The narrative output becomes the prompt for the next clip, creating
    a chain of visually continuous clips that dream themselves forward.

    In interspersed mode, odd-numbered clips use Flux-generated character
    reference images instead of freeze frames, producing a more dynamic
    cut-heavy result where character likeness stays anchored across jumps.

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
    refs_dir = config.project_dir / "refs"
    clips_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = clips_dir / "manifest.json"

    interspersed = config.video.seeding_mode == "interspersed"
    seeding_label = "interspersed (Flux refs)" if interspersed else "continuous"

    print(f"\n{'='*60}")
    print(f"  {config.title}")
    if config.artist:
        print(f"  {config.artist}")
    print(f"  {total_clips} clips | {config.audio.duration:.0f}s | {seeding_label}")
    print(f"{'='*60}\n")

    # Initialize Flux generator for interspersed mode
    flux = None
    if interspersed:
        load_dotenv()
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        if openrouter_key:
            flux = FluxGenerator(
                api_key=openrouter_key,
                model=config.video.reference_model,
            )
            print(f"  Flux reference model: {config.video.reference_model}")
        else:
            print("  WARNING: OPENROUTER_API_KEY not set, falling back to continuous seeding")
            interspersed = False

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
            "seeding_mode": config.video.seeding_mode,
            "video_model": config.video.model,
            "vision_model": config.vision.model,
            "narrative_model": config.narrative.model,
            "reference_model": config.video.reference_model if interspersed else None,
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

    # Build style prefix for video prompts
    style_prefix = config.style.global_style.strip()
    char_desc = config.get_character_descriptions()
    if char_desc:
        style_prefix = f"{style_prefix} {char_desc}"

    # Reconstruct state for resumption
    story_so_far = [config.seed.prompt or "Opening scene."]
    current_freeze = None  # last freeze frame (for interspersed alternation)
    current_seed = seed_path
    current_prompt = story_so_far[0]

    if start_from > 0 and manifest.get("clips"):
        for clip_data in manifest["clips"][:start_from]:
            story_so_far.append(clip_data.get("prompt", ""))
        prev_clip = clips_dir / f"clip_{start_from - 1:02d}.mp4"
        if prev_clip.exists():
            current_freeze = extract_last_frame(prev_clip)
            current_seed = current_freeze
        if manifest["clips"]:
            current_prompt = manifest["clips"][-1].get("prompt", current_prompt)

    # Main generation loop
    for i in range(start_from, total_clips):
        section = config.get_section_for_clip(i)
        print(f"\n  --- Clip {i:02d} / {total_clips - 1:02d} [{section.name}] ---")
        clip_path = clips_dir / f"clip_{i:02d}.mp4"

        # Pick seed image based on seeding mode
        if interspersed:
            seed_image, seed_type = _pick_seed_image(
                clip_num=i,
                freeze_frame=current_freeze,
                seed_path=seed_path,
                refs_dir=refs_dir,
                flux=flux,
                reference_prompt=config.video.reference_prompt,
                section_mood=section.mood,
            )
            print(f"    Seed: {seed_type} ({seed_image.name})")
        else:
            seed_image = current_seed
            seed_type = "freeze" if i > 0 else "seed"

        clip_data = {
            "clip": i,
            "prompt": current_prompt,
            "seed_image": str(seed_image),
            "seed_type": seed_type,
            "section": section.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Prepend style + character descriptions (video generator has no memory)
            styled_prompt = f"{style_prefix} {current_prompt}"
            generator.generate(
                prompt=styled_prompt,
                seed_image=seed_image,
                output_path=clip_path,
                duration=config.video.clip_duration,
                size=config.video.size,
            )
            clip_data["video"] = clip_path.name
            clip_data["status"] = "ok"

            if i < total_clips - 1:
                # Extract last frame → vision → narrative → next prompt
                next_freeze = extract_last_frame(clip_path)
                frame_description = analyze_frame(
                    image_path=next_freeze,
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
                current_freeze = next_freeze
                current_seed = next_freeze  # for continuous mode
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
