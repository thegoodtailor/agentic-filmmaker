"""Kling 3.0 Pro video generator via WaveSpeed API.

IMPORTANT: Only use kling-v3.0-pro. Older models (v1.x, v2.x, v2.6) produce
inferior quality. See session 29 notes.

Two production modes:
  Mode A (Long Scene): Freeze-frame chain for continuous 30s/60s+ scenes.
    - Generate clip → extract last frame → seed next clip
    - Elements keep faces locked across clips
  Mode B (Quick Cuts): multi_prompt for rapid scene changes up to 15s.
    - Single API call, up to 6 shots
    - Kling handles transitions internally

Confirmed working parameters on 3.0 Pro via WaveSpeed:
  - sound: true — native audio + dialogue (1.5x cost)
  - multi_prompt: [{prompt, duration}] — multi-shot transitions
  - end_image: URL — start→end frame morphing
  - element_list: [{"element_id": "..."}] — persistent character lock
  - Dialogue in prompts: [Character, voice_tone]: "text"
  - duration: 3-15 seconds
  - cfg_scale: 0.0-1.0
  - aspect_ratio: "16:9", "9:16", "1:1"
  - negative_prompt: exclusions

Element IDs (persistent on Kling servers):
  - Asel: 306696265837507
  - Iman: 306696672116506
"""

import os
import time
import base64
import requests
from pathlib import Path

from .base import VideoGenerator

NO_TEXT = (
    "Absolutely no text, letters, words, numbers, writing, subtitles, "
    "captions, or symbols of any kind."
)


class KlingGenerator(VideoGenerator):
    """Video generation using Kling 3.0 Pro via WaveSpeed API."""

    def __init__(
        self,
        wavespeed_key: str,
        openrouter_key: str | None = None,
        model: str = "kwaivgi/kling-v3.0-pro",
        max_retries: int = 3,
        no_text: bool = True,
        clip_duration: int = 5,
        sound: bool = False,
        element_list: list[dict] | None = None,
        aspect_ratio: str | None = None,
    ):
        self.wavespeed_key = wavespeed_key
        self.openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.max_retries = max_retries
        self.no_text = no_text
        self.clip_duration = clip_duration
        self.sound = sound
        self.element_list = element_list or []
        self.aspect_ratio = aspect_ratio
        self.api_base = "https://api.wavespeed.ai/api/v3"

    def _encode_local(self, path: Path) -> str:
        """Encode a local file as a data URI."""
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = path.suffix.lower().strip(".")
        if ext == "jpg":
            ext = "jpeg"
        if ext in ("mp4", "mov"):
            return f"data:video/{ext};base64,{b64}"
        return f"data:image/{ext};base64,{b64}"

    def _poll(self, poll_url: str, label: str, timeout: int = 360) -> list[str] | None:
        """Poll a WaveSpeed task until completion."""
        print(f"    Polling {label}...", end="", flush=True)
        for _ in range(timeout // 2):
            time.sleep(2)
            result = requests.get(
                poll_url,
                headers={"Authorization": f"Bearer {self.wavespeed_key}"},
            ).json()
            status = result.get("data", {}).get("status", "")
            if status == "completed":
                print(" done")
                return result["data"].get("outputs", [])
            elif status in ("failed", "error"):
                err = result.get("data", {}).get("error", "unknown")
                print(f" FAILED: {err}")
                return None
            print(".", end="", flush=True)
        print(" timeout")
        return None

    def _download(self, url: str, output_path: Path) -> Path:
        """Download a generated video."""
        video_resp = requests.get(url, timeout=120)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(video_resp.content)
        print(f"    Saved: {output_path.name} ({len(video_resp.content) // 1024} KB)")
        return output_path

    def _submit(self, payload: dict) -> dict:
        """Submit a generation task to WaveSpeed."""
        resp = requests.post(
            f"{self.api_base}/{self.model}/image-to-video",
            headers={
                "Authorization": f"Bearer {self.wavespeed_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        data = resp.json()
        if data.get("code") != 200:
            raise RuntimeError(f"API error: {data}")
        return data["data"]

    # ── Mode A: Standard generation (single clip) ──────────────────

    def generate(
        self,
        prompt: str,
        seed_image: Path,
        output_path: Path,
        duration: int = 5,
        size: str = "1280x720",
    ) -> Path:
        """Generate a single video clip from seed image + prompt.

        This is the standard Mode A building block. Chain multiple calls
        with freeze-frame extraction for long scenes (30s, 60s+).
        """
        if output_path.exists():
            print(f"    Video exists: {output_path.name}")
            return output_path

        full_prompt = f"{prompt} {NO_TEXT}" if self.no_text else prompt
        print(f"    Generating video: {output_path.name}")
        print(f"    Prompt: {full_prompt[:100]}...")

        image_uri = self._encode_local(seed_image)

        payload = {
            "image": image_uri,
            "prompt": full_prompt,
            "cfg_scale": 0.5,
            "duration": min(duration, 15),
            "sound": self.sound,
        }
        if self.element_list:
            payload["element_list"] = self.element_list
        if self.aspect_ratio:
            payload["aspect_ratio"] = self.aspect_ratio

        last_err = None
        for attempt in range(self.max_retries):
            try:
                task = self._submit(payload)
                outputs = self._poll(task["urls"]["get"], output_path.name)
                if outputs:
                    return self._download(outputs[0], output_path)
                raise RuntimeError("No outputs")
            except Exception as e:
                last_err = e
                print(f"\n    Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(5)

        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_err}")

    # ── Mode B: Multi-shot quick cuts (single API call) ────────────

    def generate_multishot(
        self,
        prompt: str,
        seed_image: Path,
        output_path: Path,
        shots: list[dict],
        duration: int = 15,
    ) -> Path:
        """Generate a multi-shot clip with scene transitions.

        Args:
            prompt: Master prompt (overall creative direction).
            seed_image: Starting image.
            shots: List of {"prompt": str, "duration": int} dicts.
                   Up to 6 shots, total duration <= 15s.
            duration: Total clip duration (should match sum of shot durations).
        """
        if output_path.exists():
            print(f"    Video exists: {output_path.name}")
            return output_path

        full_prompt = f"{prompt} {NO_TEXT}" if self.no_text else prompt
        print(f"    Generating multishot: {output_path.name} ({len(shots)} shots)")

        image_uri = self._encode_local(seed_image)

        payload = {
            "image": image_uri,
            "prompt": full_prompt,
            "multi_prompt": shots,
            "cfg_scale": 0.5,
            "duration": min(duration, 15),
            "sound": self.sound,
        }
        if self.element_list:
            payload["element_list"] = self.element_list
        if self.aspect_ratio:
            payload["aspect_ratio"] = self.aspect_ratio

        task = self._submit(payload)
        outputs = self._poll(task["urls"]["get"], output_path.name, timeout=480)
        if outputs:
            return self._download(outputs[0], output_path)
        raise RuntimeError("Multi-shot generation failed")

    # ── Start/End frame morphing ───────────────────────────────────

    def generate_morph(
        self,
        prompt: str,
        start_image: Path,
        end_image: Path,
        output_path: Path,
        duration: int = 10,
    ) -> Path:
        """Generate a video that morphs between start and end frames.

        Useful for smooth transitions between scenes or character states.
        """
        if output_path.exists():
            print(f"    Video exists: {output_path.name}")
            return output_path

        full_prompt = f"{prompt} {NO_TEXT}" if self.no_text else prompt
        print(f"    Generating morph: {output_path.name}")

        payload = {
            "image": self._encode_local(start_image),
            "end_image": self._encode_local(end_image),
            "prompt": full_prompt,
            "cfg_scale": 0.5,
            "duration": min(duration, 15),
            "sound": self.sound,
        }
        if self.element_list:
            payload["element_list"] = self.element_list

        task = self._submit(payload)
        outputs = self._poll(task["urls"]["get"], output_path.name)
        if outputs:
            return self._download(outputs[0], output_path)
        raise RuntimeError("Morph generation failed")

    # ── Element management ─────────────────────────────────────────

    def create_element(
        self,
        name: str,
        description: str,
        primary_image: str,
        reference_images: list[str],
    ) -> str:
        """Register a persistent character element on Kling's servers.

        Args:
            name: Character name (e.g. "Asel").
            description: Physical description, max 100 characters.
            primary_image: URL of the main reference photo.
            reference_images: List of additional photo URLs (min 1).

        Returns:
            element_id: Persistent ID for use in element_list.
        """
        print(f"    Creating element: {name}")
        resp = requests.post(
            f"{self.api_base}/kwaivgi/kling-elements",
            headers={
                "Authorization": f"Bearer {self.wavespeed_key}",
                "Content-Type": "application/json",
            },
            json={
                "image": primary_image,
                "name": name,
                "description": description[:100],
                "element_refer_list": reference_images,
            },
            timeout=60,
        )
        data = resp.json()
        if data.get("code") != 200:
            raise RuntimeError(f"Element creation failed: {data}")

        outputs = self._poll(data["data"]["urls"]["get"], f"element-{name}", timeout=60)
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            element_id = outputs[0].get("element_id")
            print(f"    Element registered: {name} → {element_id}")
            return element_id
        raise RuntimeError("Element creation returned no ID")

    # ── Seed image generation via Flux ─────────────────────────────

    def generate_seed_image(
        self,
        prompt: str,
        output_path: Path,
        size: str = "1792x1024",
    ) -> Path:
        """Generate a seed image via Flux 2 Pro on OpenRouter."""
        if output_path.exists():
            print(f"    Seed exists: {output_path.name}")
            return output_path

        print(f"    Generating seed: {output_path.name}")

        w, h = (int(x) for x in size.split("x"))
        ar = "16:9" if w > h else ("3:4" if h > w else "1:1")

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "black-forest-labs/flux.2-pro",
                "messages": [{"role": "user", "content": prompt}],
                "modalities": ["image"],
                "image_config": {"aspect_ratio": ar, "image_size": "2K"},
            },
            timeout=180,
        )

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Seed image error: {data['error']}")

        message = data["choices"][0]["message"]
        images = message.get("images", [])
        if not images and isinstance(message.get("content"), list):
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    images.append(part)

        if not images:
            raise RuntimeError("No images in response")

        url = images[0] if isinstance(images[0], str) else images[0].get("image_url", {}).get("url", "")
        _, b64data = url.split(",", 1)
        img_bytes = base64.b64decode(b64data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_bytes)

        print(f"    Saved: {output_path.name} ({len(img_bytes) // 1024} KB)")
        return output_path
