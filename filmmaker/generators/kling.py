"""Kling 3.0 Pro video generator via WaveSpeed API.

IMPORTANT: Only use kling-v3.0-pro. Older models produce inferior quality.

Two production modes:
  Mode A (Long Scene): Freeze-frame chain for 30s/60s+ continuous scenes.
  Mode B (Quick Cuts): multi_prompt for up to 6 shots in one API call.

Architecture: empty environment seeds + character elements.
  - Generate empty scenes (no people) as seed images
  - Characters enter via element_list + prompt descriptions
  - Dialogue format: [Character, voice_tone]: "text" with sound=True
"""

import os
import time
import base64
import requests
from pathlib import Path

from .base import NO_TEXT, VideoGenerator, generate_flux_image

KLING_MODELS = {
    "kling-3.0-pro": "kwaivgi/kling-v3.0-pro",
    "kling-3.0-std": "kwaivgi/kling-v3.0-std",
}


class KlingGenerator(VideoGenerator):
    """Video generation using Kling 3.0 Pro via WaveSpeed API."""

    def __init__(
        self,
        wavespeed_key: str,
        openrouter_key: str | None = None,
        model: str = "kling-3.0-pro",
        max_retries: int = 3,
        no_text: bool = True,
        clip_duration: int = 5,
        sound: bool = False,
        element_list: list[dict] | None = None,
        aspect_ratio: str | None = None,
    ):
        self.wavespeed_key = wavespeed_key
        self.openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = self._resolve_model(model)
        self.max_retries = max_retries
        self.no_text = no_text
        self.clip_duration = clip_duration
        self.sound = sound
        self.element_list = element_list or []
        self.aspect_ratio = aspect_ratio
        self.api_base = "https://api.wavespeed.ai/api/v3"

    @staticmethod
    def _resolve_model(model: str) -> str:
        """Resolve short model name to WaveSpeed path."""
        if model in KLING_MODELS:
            return KLING_MODELS[model]
        if model.startswith("kwaivgi/"):
            return model
        raise ValueError(
            f"Unknown Kling model: {model!r}. "
            f"Valid: {', '.join(KLING_MODELS)} or a full kwaivgi/ path."
        )

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
        """Poll a WaveSpeed task until completion with exponential backoff."""
        print(f"    Polling {label}...", end="", flush=True)
        elapsed = 0
        sleep = 2
        while elapsed < timeout:
            time.sleep(sleep)
            elapsed += sleep
            try:
                result = requests.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {self.wavespeed_key}"},
                    timeout=30,
                ).json()
            except requests.RequestException:
                print("x", end="", flush=True)
                sleep = min(sleep * 1.5, 20)
                continue

            status = result.get("data", {}).get("status", "")
            if status == "completed":
                print(" done")
                return result["data"].get("outputs", [])
            elif status in ("failed", "error"):
                err = result.get("data", {}).get("error", "unknown")
                print(f" FAILED: {err}")
                return None
            print(".", end="", flush=True)
            sleep = min(sleep * 1.3, 15)

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

    def _build_payload(self, image_uri: str, prompt: str, duration: int, **extra) -> dict:
        """Build a standard generation payload."""
        full_prompt = f"{prompt} {NO_TEXT}" if self.no_text else prompt
        payload = {
            "image": image_uri,
            "prompt": full_prompt,
            "cfg_scale": 0.5,
            "duration": min(duration, 15),
            "sound": self.sound,
            **extra,
        }
        if self.element_list:
            payload["element_list"] = self.element_list
        if self.aspect_ratio:
            payload["aspect_ratio"] = self.aspect_ratio
        return payload

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

    def _run(self, payload: dict, output_path: Path, poll_timeout: int = 360) -> Path:
        """Submit, poll, download with retries. Used by all generation methods."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                task = self._submit(payload)
                outputs = self._poll(task["urls"]["get"], output_path.name, timeout=poll_timeout)
                if outputs:
                    return self._download(outputs[0], output_path)
                raise RuntimeError("No outputs")
            except Exception as e:
                last_err = e
                print(f"\n    Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(5)

        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_err}")

    # ── Mode A: Standard generation (single clip) ──────────────────

    def generate(
        self,
        prompt: str,
        seed_image: Path,
        output_path: Path,
        duration: int = 5,
        size: str = "1280x720",
    ) -> Path:
        """Generate a single video clip from seed image + prompt."""
        if output_path.exists():
            print(f"    Video exists: {output_path.name}")
            return output_path

        print(f"    Generating video: {output_path.name}")
        payload = self._build_payload(self._encode_local(seed_image), prompt, duration)
        return self._run(payload, output_path)

    # ── Mode B: Multi-shot quick cuts ──────────────────────────────

    def generate_multishot(
        self,
        prompt: str,
        seed_image: Path,
        output_path: Path,
        shots: list[dict],
        duration: int = 15,
    ) -> Path:
        """Generate a multi-shot clip. Up to 6 shots, max 15s total."""
        if output_path.exists():
            print(f"    Video exists: {output_path.name}")
            return output_path

        print(f"    Generating multishot: {output_path.name} ({len(shots)} shots)")
        payload = self._build_payload(
            self._encode_local(seed_image), prompt, duration,
            multi_prompt=shots,
        )
        return self._run(payload, output_path, poll_timeout=480)

    # ── Start/End frame transition ─────────────────────────────────

    def generate_transition(
        self,
        prompt: str,
        start_image: Path,
        end_image: Path,
        output_path: Path,
        duration: int = 10,
    ) -> Path:
        """Generate a video transitioning between start and end frames."""
        if output_path.exists():
            print(f"    Video exists: {output_path.name}")
            return output_path

        print(f"    Generating transition: {output_path.name}")
        payload = self._build_payload(
            self._encode_local(start_image), prompt, duration,
            end_image=self._encode_local(end_image),
        )
        return self._run(payload, output_path)

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
            name: Character name.
            description: Physical description, max 100 characters.
            primary_image: URL of the main reference photo.
            reference_images: Additional photo URLs (min 1).

        Returns:
            Persistent element_id for use in element_list.
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
            print(f"    Element registered: {name} -> {element_id}")
            return element_id
        raise RuntimeError("Element creation returned no ID")

    # ── Seed image generation ──────────────────────────────────────

    def generate_seed_image(
        self,
        prompt: str,
        output_path: Path,
        size: str = "1792x1024",
    ) -> Path:
        """Generate a seed image via Flux 2 Pro on OpenRouter."""
        w, h = (int(x) for x in size.split("x"))
        ar = "16:9" if w > h else ("3:4" if h > w else "1:1")
        return generate_flux_image(
            api_key=self.openrouter_key,
            prompt=prompt,
            output_path=output_path,
            aspect_ratio=ar,
        )
