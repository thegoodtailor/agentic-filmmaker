"""Flux image generator via OpenRouter — for character reference images."""

from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from .base import extract_openrouter_image


class FluxGenerator:
    """Generate character reference images via Black Forest Labs Flux on OpenRouter.

    Used in interspersed seeding mode: even clips use the previous clip's last
    frame (continuity), odd clips use a Flux-generated character reference
    (likeness anchoring). This produces a more dynamic, cut-heavy result
    where the character's appearance stays consistent across jump cuts.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "black-forest-labs/flux.2-pro",
        aspect_ratio: str = "16:9",
    ):
        self.api_key = api_key
        self.model = model
        self.aspect_ratio = aspect_ratio

    def generate(
        self,
        prompt: str,
        output_path: Path,
        no_text: bool = True,
    ) -> Path:
        """Generate a character reference image via Flux.

        Args:
            prompt: Character description + scene context.
            output_path: Where to save the generated image.
            no_text: Append no-text enforcement to prompt.

        Returns:
            Path to the generated image.
        """
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"    Flux ref exists: {output_path.name}")
            return output_path

        if no_text:
            prompt += (
                " No text, no words, no writing, no watermarks, "
                "no symbols, no letters, no numbers."
            )

        print(f"    Generating Flux reference: {output_path.name}")
        print(f"    Prompt: {prompt[:100]}...")

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "modalities": ["image"],
                "image_config": {
                    "aspect_ratio": self.aspect_ratio,
                    "image_size": "2K",
                },
            },
            timeout=180,
        )

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Flux API error: {data['error']}")

        img_bytes = extract_openrouter_image(data)
        img = Image.open(BytesIO(img_bytes))
        img.save(output_path, "PNG")
        print(f"    Flux ref saved: {output_path.name} ({img.size})")
        return output_path
