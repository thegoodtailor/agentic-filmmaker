"""CLI entry point — init, generate, assemble."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def cmd_init(args):
    """Initialize a project YAML from audio analysis + lyrics."""
    from .audio import detect_sections, get_duration, reconcile_with_lyrics
    from .config import (
        AudioConfig,
        NarrativeConfig,
        ProjectConfig,
        Section,
        SeedConfig,
        StyleConfig,
        VideoConfig,
        VisionConfig,
        save_config,
    )

    import openai

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    # Get duration
    duration = args.duration or get_duration(audio_path)
    clip_duration = args.clip_duration
    total_clips = int(duration / clip_duration)
    print(f"  Audio: {audio_path.name} ({duration:.1f}s, {total_clips} clips)")

    # Detect sections from audio
    sections_raw = detect_sections(audio_path)

    # Read lyrics if provided
    lyrics_text = None
    if args.lyrics:
        lyrics_path = Path(args.lyrics).resolve()
        if lyrics_path.exists():
            lyrics_text = lyrics_path.read_text()
            print(f"  Lyrics: {lyrics_path.name} ({len(lyrics_text)} chars)")
        else:
            print(f"Warning: lyrics file not found: {lyrics_path}")

    # Reconcile with LLM if we have lyrics
    if lyrics_text:
        load_dotenv()
        narrative_client = openai.OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
        reconciled = reconcile_with_lyrics(
            sections=sections_raw,
            lyrics_text=lyrics_text,
            duration=duration,
            clip_duration=clip_duration,
            title=args.title or "Untitled",
            client=narrative_client,
            model=args.narrative_model,
        )
        sections = [
            Section(
                name=s["name"],
                clips=tuple(s["clips"]),
                lyrics=s.get("lyrics", ""),
                mood=s.get("mood", ""),
                slow_motion=s.get("slow_motion", False),
            )
            for s in reconciled
        ]
    else:
        # No lyrics — generate sections from audio boundaries only
        sections = []
        clip_idx = 0
        for i, s in enumerate(sections_raw):
            start_clip = clip_idx
            n_section_clips = max(1, int((s["end_sec"] - s["start_sec"]) / clip_duration))
            end_clip = min(start_clip + n_section_clips - 1, total_clips - 1)
            sections.append(Section(
                name=f"Section {i + 1}",
                clips=(start_clip, end_clip),
                lyrics="(Instrumental)",
                mood=f"Energy level: {s['energy']:.4f}",
                slow_motion=s["energy"] < 0.02,
            ))
            clip_idx = end_clip + 1

    # Build config
    config = ProjectConfig(
        title=args.title or "Untitled",
        artist=args.artist or "",
        album=args.album or "",
        audio=AudioConfig(path=str(audio_path), duration=duration),
        video=VideoConfig(clip_duration=clip_duration),
        vision=VisionConfig(),
        narrative=NarrativeConfig(),
        style=StyleConfig(),
        characters=[],
        sections=sections,
        seed=SeedConfig(),
    )

    output_path = Path(args.output)
    save_config(config, output_path)
    print(f"\n  Project saved: {output_path}")
    print(f"  {len(sections)} sections, {total_clips} clips")
    print(f"\n  Next steps:")
    print(f"    1. Edit {output_path} — add style, characters, seed prompt, moods")
    print(f"    2. Run: filmmaker generate {output_path}")


def cmd_generate(args):
    """Generate video clips for a project."""
    import openai

    from .config import load_config
    from .generators import KlingGenerator, Sora2Generator
    from .pipeline import generate

    load_dotenv()

    config = load_config(Path(args.project))
    print(f"  Loaded project: {config.title}")

    video_model = getattr(config.video, "model", "sora-2")

    openai_client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "unused"),
    )

    narrative_client = openai.OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        base_url="https://openrouter.ai/api/v1",
    )

    if video_model.startswith("kling"):
        wavespeed_key = os.environ.get("WAVESPEED_API_KEY", "")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        # Map short names to WaveSpeed model paths
        kling_models = {
            "kling-2.6-pro": "kwaivgi/kling-v2.6-pro",
            "kling-3.0-pro": "kwaivgi/kling-v3.0-pro",
            "kling-3.0-std": "kwaivgi/kling-v3.0-std",
        }
        ws_model = kling_models.get(video_model, "kwaivgi/kling-v2.6-pro")
        generator = KlingGenerator(
            wavespeed_key=wavespeed_key,
            openrouter_key=openrouter_key,
            model=ws_model,
            no_text=config.style.no_text,
            clip_duration=config.video.clip_duration,
        )
    else:
        generator = Sora2Generator(
            client=openai_client,
            no_text=config.style.no_text,
        )

    generate(
        config=config,
        generator=generator,
        openai_client=openai_client,
        narrative_client=narrative_client,
        start_from=args.start_from,
        n_clips=args.clips,
    )


def cmd_assemble(args):
    """Assemble clips into final video with audio."""
    from .assembly import assemble
    from .config import load_config

    config = load_config(Path(args.project))
    clips_dir = config.project_dir / "clips"
    audio_path = Path(config.audio.path)

    if args.audio:
        audio_path = Path(args.audio)

    output_path = Path(args.output) if args.output else (
        config.project_dir / f"{config.title.lower().replace(' ', '_')}.mp4"
    )

    n_clips = config.get_total_clips()

    assemble(
        clips_dir=clips_dir,
        audio_path=audio_path,
        output_path=output_path,
        n_clips=n_clips,
        slow_motion_factor=config.video.slow_motion_factor,
    )

    print(f"\n  Final video: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="filmmaker",
        description="agentic-filmmaker — AI-powered music video pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- init ---
    p_init = subparsers.add_parser(
        "init",
        help="Analyze audio + lyrics to generate a project YAML",
    )
    p_init.add_argument("--audio", required=True, help="Path to audio file (WAV, MP3)")
    p_init.add_argument("--lyrics", help="Path to lyrics text file")
    p_init.add_argument("--title", help="Song title")
    p_init.add_argument("--artist", help="Artist name")
    p_init.add_argument("--album", help="Album name")
    p_init.add_argument("--duration", type=float, help="Override duration (seconds)")
    p_init.add_argument("--clip-duration", type=int, default=8, help="Seconds per clip (default: 8)")
    p_init.add_argument("--narrative-model", default="anthropic/claude-sonnet-4",
                        help="Model for lyrics reconciliation")
    p_init.add_argument("--output", default="project.yaml", help="Output YAML path")
    p_init.set_defaults(func=cmd_init)

    # --- generate ---
    p_gen = subparsers.add_parser(
        "generate",
        help="Generate video clips for a project",
    )
    p_gen.add_argument("project", help="Path to project YAML")
    p_gen.add_argument("--start-from", type=int, default=0, help="Resume from clip N")
    p_gen.add_argument("--clips", type=int, help="Override total clip count")
    p_gen.set_defaults(func=cmd_generate)

    # --- assemble ---
    p_asm = subparsers.add_parser(
        "assemble",
        help="Assemble clips into final video with audio",
    )
    p_asm.add_argument("project", help="Path to project YAML")
    p_asm.add_argument("--audio", help="Override audio path")
    p_asm.add_argument("--output", help="Output video path")
    p_asm.set_defaults(func=cmd_assemble)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
