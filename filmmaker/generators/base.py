"""Abstract base class for video generators."""

from abc import ABC, abstractmethod
from pathlib import Path


class VideoGenerator(ABC):
    """Base class for video generation backends.

    Implementations must provide both video clip generation (from a seed
    image + text prompt) and seed image generation (from text prompt only).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        seed_image: Path,
        output_path: Path,
        duration: int = 8,
        size: str = "1280x720",
    ) -> Path:
        """Generate a video clip from a seed image and text prompt.

        Args:
            prompt: Scene description for the video.
            seed_image: Path to the seed image (will be resized if needed).
            output_path: Where to save the generated video.
            duration: Clip duration in seconds.
            size: Video dimensions as "WxH".

        Returns:
            Path to the generated video file.
        """
        ...

    @abstractmethod
    def generate_seed_image(
        self,
        prompt: str,
        output_path: Path,
        size: str = "1792x1024",
    ) -> Path:
        """Generate an initial seed image from a text prompt.

        Args:
            prompt: Scene description for the seed image.
            output_path: Where to save the generated image.
            size: Image dimensions as "WxH".

        Returns:
            Path to the generated image file.
        """
        ...
