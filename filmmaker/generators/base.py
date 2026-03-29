"""Abstract base class for video generators."""

import base64
from abc import ABC, abstractmethod
from pathlib import Path

import requests

NO_TEXT = (
    "Absolutely no text, letters, words, numbers, writing, subtitles, "
    "captions, or symbols of any kind."
)


def extract_openrouter_image(data: dict) -> bytes:
    """Extract image bytes from an OpenRouter image generation response.

    Handles both message.images[] and message.content[] formats.
    Returns decoded image bytes.
    """
    message = data["choices"][0]["message"]
    images = message.get("images", [])
    if not images and isinstance(message.get("content"), list):
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                images.append(part)

    if not images:
        raise RuntimeError("No images in OpenRouter response")

    url = images[0] if isinstance(images[0], str) else images[0].get("image_url", {}).get("url", "")
    if not url.startswith("data:image"):
        raise RuntimeError(f"Unexpected image format: {url[:80]}")

    _, b64data = url.split(",", 1)
    return base64.b64decode(b64data)


def generate_flux_image(
    api_key: str,
    prompt: str,
    output_path: Path,
    model: str = "black-forest-labs/flux.2-pro",
    aspect_ratio: str = "3:4",
) -> Path:
    """Generate an image via Flux on OpenRouter. Shared by KlingGenerator and FluxGenerator."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image"],
            "image_config": {"aspect_ratio": aspect_ratio, "image_size": "2K"},
        },
        timeout=180,
    )

    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Flux API error: {data['error']}")

    img_bytes = extract_openrouter_image(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(img_bytes)

    print(f"    Saved: {output_path.name} ({len(img_bytes) // 1024} KB)")
    return output_path


class VideoGenerator(ABC):
    """Base class for video generation backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        seed_image: Path,
        output_path: Path,
        duration: int = 8,
        size: str = "1280x720",
    ) -> Path:
        ...

    @abstractmethod
    def generate_seed_image(
        self,
        prompt: str,
        output_path: Path,
        size: str = "1792x1024",
    ) -> Path:
        ...
