"""Vision agent — analyzes frames to describe what's on screen."""

import re
from pathlib import Path

import openai

from .frames import encode_image_base64


def analyze_frame(
    image_path: Path,
    system_prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 300,
    client: openai.OpenAI | None = None,
) -> str:
    """Analyze a video frame and return a text description.

    Uses a vision-capable model to describe characters, environment,
    lighting, mood, and implied motion in the frame.

    Args:
        image_path: Path to the frame image.
        system_prompt: System prompt for the vision model.
        model: Vision model name (must support image input).
        max_tokens: Max response length.
        client: OpenAI client instance.

    Returns:
        Text description of the frame (3-4 sentences).
    """
    print(f"    Vision analyzing: {image_path.name}")
    b64 = encode_image_base64(image_path)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this frame from a music video."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
        max_tokens=max_tokens,
    )
    description = resp.choices[0].message.content.strip()
    print(f"    Vision: {description[:80]}...")
    return description
