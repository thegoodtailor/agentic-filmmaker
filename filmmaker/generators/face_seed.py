"""Face-conditioned seed image generation via Flux.2 Flex.

Takes a reference photo of a person + a scene prompt and generates
an image of THAT PERSON in the described scene. Critical for
multi-character films where each person needs their actual face
in the seed to prevent element confusion in video generation.

Usage:
    from filmmaker.generators.face_seed import generate_face_seed

    generate_face_seed(
        api_key="...",
        face_image=Path("iman.jpg"),
        prompt="Transform this person into a Byzantine emperor on a throne...",
        output_path=Path("seed_throne.png"),
    )
"""

import base64
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from .base import extract_openrouter_image


def generate_face_seed(
    api_key: str,
    face_image: Path,
    prompt: str,
    output_path: Path,
    model: str = "black-forest-labs/flux.2-flex",
    aspect_ratio: str = "16:9",
) -> Path:
    """Generate a seed image conditioned on a face reference photo.

    Uses Flux.2 Flex's image-conditioning capability to preserve the
    person's actual face while placing them in the described scene.

    Args:
        api_key: OpenRouter API key.
        face_image: Path to the reference photo (clear face, good lighting).
        prompt: Scene description — should start with "Transform this person into..."
        output_path: Where to save the generated seed.
        model: Flux model with image conditioning support.
        aspect_ratio: Output aspect ratio.

    Returns:
        Path to the generated seed image.
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"    Face seed exists: {output_path.name}")
        return output_path

    # Encode face reference
    with open(face_image, "rb") as f:
        face_b64 = base64.b64encode(f.read()).decode()

    ext = face_image.suffix.lower().strip(".")
    if ext == "jpg":
        ext = "jpeg"
    face_uri = f"data:image/{ext};base64,{face_b64}"

    print(f"    Generating face seed: {output_path.name}")

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": face_uri}},
                ],
            }],
            "modalities": ["image"],
            "image_config": {
                "aspect_ratio": aspect_ratio,
                "image_size": "2K",
            },
        },
        timeout=180,
    )

    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Face seed API error: {data['error']}")

    img_bytes = extract_openrouter_image(data)
    img = Image.open(BytesIO(img_bytes))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"    Face seed saved: {output_path.name} ({img.size})")
    return output_path
