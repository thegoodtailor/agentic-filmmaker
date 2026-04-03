"""Sora 2 video generator + DALL-E 3 seed image generator."""

import time
import urllib.request
from pathlib import Path

import openai

from ..frames import resize_for_video
from .base import NO_TEXT, VideoGenerator


class Sora2Generator(VideoGenerator):
    """Video generation using OpenAI Sora 2, seed images via DALL-E 3."""

    def __init__(
        self,
        client: openai.OpenAI,
        max_retries: int = 6,
        no_text: bool = True,
    ):
        self.client = client
        self.max_retries = max_retries
        self.no_text = no_text

    def generate(
        self,
        prompt: str,
        seed_image: Path,
        output_path: Path,
        duration: int = 8,
        size: str = "1280x720",
    ) -> Path:
        if output_path.exists():
            print(f"    Video exists: {output_path.name}")
            return output_path

        w, h = (int(x) for x in size.split("x"))
        resized = resize_for_video(seed_image, w, h)

        full_prompt = f"{prompt} {NO_TEXT}" if self.no_text else prompt
        print(f"    Generating video: {output_path.name}")
        print(f"    Prompt: {full_prompt[:100]}...")

        last_err = None
        for attempt in range(self.max_retries):
            try:
                video = self.client.videos.create_and_poll(
                    model="sora-2",
                    input_reference=resized,
                    prompt=full_prompt,
                    size=size,
                    seconds=duration,
                )
                response = self.client.videos.download_content(video.id)
                response.write_to_file(str(output_path))
                print(f"    OK: {output_path.name}")
                return output_path
            except Exception as e:
                last_err = e
                wait = 10 * (attempt + 1)
                print(f"    Sora attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    print(f"    Waiting {wait}s before retry...")
                    time.sleep(wait)
        raise last_err

    def generate_seed_image(
        self,
        prompt: str,
        output_path: Path,
        size: str = "1792x1024",
    ) -> Path:
        if output_path.exists():
            print(f"    Seed image exists: {output_path.name}")
            return output_path

        full_prompt = f"{prompt} {NO_TEXT}" if self.no_text else prompt
        print(f"    Generating seed image via DALL-E 3...")

        resp = self.client.images.generate(
            model="dall-e-3",
            prompt=full_prompt,
            size=size,
            quality="hd",
            n=1,
        )

        url = resp.data[0].url
        urllib.request.urlretrieve(url, str(output_path))

        print(f"    Seed image saved: {output_path.name}")
        return output_path
