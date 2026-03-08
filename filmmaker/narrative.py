"""Narrative agent — generates the next scene description from context."""

import re

import openai

from .config import Character, Section


def generate_scene(
    section: Section,
    story_so_far: list[str],
    frame_description: str,
    clip_number: int,
    total_clips: int,
    camera_move: str,
    characters: list[Character],
    system_prompt: str,
    model: str = "anthropic/claude-sonnet-4",
    max_tokens: int = 200,
    client: openai.OpenAI | None = None,
) -> str:
    """Generate the next scene prompt for video generation.

    Takes the current frame analysis, song section context, narrative
    history, and camera direction to write a 2-4 sentence scene
    description suitable for a video generation model.

    Args:
        section: Current song section (name, lyrics, mood).
        story_so_far: List of previous scene prompts (last 8 used).
        frame_description: Vision agent's description of the current frame.
        clip_number: Current clip index.
        total_clips: Total clips in the project.
        camera_move: Camera direction for this clip.
        characters: Character descriptions from project config.
        system_prompt: System prompt for the narrative model.
        model: Model name/slug (OpenRouter or direct).
        max_tokens: Max response length.
        client: OpenAI-compatible client instance.

    Returns:
        Scene description as plain prose (2-4 sentences).
    """
    print(f"    Narrative agent: clip {clip_number} [{section.name}]")

    story_lines = ""
    for i, prompt in enumerate(story_so_far[-8:]):
        summary = prompt[:100] + ("..." if len(prompt) > 100 else "")
        story_lines += f"  {i}. {summary}\n"

    slow_cue = ""
    if section.slow_motion:
        slow_cue = " Use slow, dreamlike pacing."

    # Build character reminder
    char_block = ""
    for c in characters:
        char_block += f"\n{c.name.upper()}: {c.description.strip()}"

    user_message = f"""SONG SECTION: {section.name}
LYRICS: {section.lyrics}
MOOD/ACTIVITY: {section.mood}

CAMERA DIRECTION FOR THIS CLIP: {camera_move}.{slow_cue}

CLIP: {clip_number} of {total_clips}

RECENT STORY:
{story_lines}
CURRENT FRAME (extracted from last clip):
{frame_description}

Write the next scene. What happens in the next 8 seconds of this music video?
REMEMBER: Describe characters physically — do NOT use names. Include the specified camera movement."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
    )
    prompt = resp.choices[0].message.content.strip()
    # Strip markdown formatting that LLMs sometimes add
    prompt = re.sub(r'\*\*[A-Z /:]+\*\*\s*', '', prompt)
    prompt = re.sub(r'^#+\s+.*$', '', prompt, flags=re.MULTILINE)
    prompt = prompt.strip()
    print(f"    Narrative: {prompt[:80]}...")
    return prompt
