"""Text-Image-to-Video (TI2V) generation interface.

Uses Veo 2.0 (veo-2.0-generate-001) via Gemini API.
"""

import io
import logging
import time
from pathlib import Path

from google.genai import types
from PIL import Image

from src.config import settings
from src.models.mllm import get_client

logger = logging.getLogger(__name__)

VIDEO_MODEL = "veo-2.0-generate-001"


def generate_video(
    prompt: str,
    begin_frame: Path | None = None,
    end_frame: Path | None = None,
    output_path: Path | None = None,
    duration_seconds: int = 8,
    aspect_ratio: str = "16:9",
    max_retries: int = 3,
    poll_interval: int = 10,
    max_poll_time: int = 300,
) -> Path:
    """Generate a video from text prompt + optional begin frame."""
    if output_path is None:
        output_path = settings.output_dir / f"segment_{int(time.time())}.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()

    for attempt in range(max_retries):
        try:
            # Prepare image conditioning
            image = _load_and_resize(begin_frame) if begin_frame else None
            end_image = _load_and_resize(end_frame) if end_frame else None

            # Build generation config
            gen_config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
            )

            # For interpolation (both begin + end frames), embed end frame info in prompt
            effective_prompt = prompt
            if end_image and image:
                effective_prompt = (
                    f"{prompt}\n\nIMPORTANT: The video must transition smoothly "
                    f"from the beginning frame to the ending frame provided."
                )

            # Use Veo API
            response = client.models.generate_videos(
                model=VIDEO_MODEL,
                prompt=effective_prompt,
                image=image,
                config=gen_config,
            )

            # Poll for completion
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed > max_poll_time:
                    logger.warning(f"Video generation timed out after {max_poll_time}s")
                    break

                result = client.operations.get(operation=response)

                if result.done:
                    if result.response and result.response.generated_videos:
                        video = result.response.generated_videos[0]
                        video.video.save(str(output_path))
                        logger.info(f"Generated video: {output_path}")
                        return output_path
                    else:
                        logger.warning("Video generation completed but no video returned")
                        break

                logger.debug(f"Video generating... ({elapsed:.0f}s elapsed)")
                time.sleep(poll_interval)

        except Exception as e:
            err_str = str(e)
            logger.warning(f"Video generation failed (attempt {attempt+1}): {e}")

            # If image conditioning fails, try without it
            if "image" in err_str.lower() and begin_frame:
                logger.info("  Retrying without image conditioning...")
                try:
                    response = client.models.generate_videos(
                        model=VIDEO_MODEL,
                        prompt=prompt,
                        config=types.GenerateVideosConfig(
                            aspect_ratio=aspect_ratio,
                            number_of_videos=1,
                        ),
                    )
                    start_time = time.time()
                    while True:
                        elapsed = time.time() - start_time
                        if elapsed > max_poll_time:
                            break
                        result = client.operations.get(operation=response)
                        if result.done:
                            if result.response and result.response.generated_videos:
                                video = result.response.generated_videos[0]
                                video.video.save(str(output_path))
                                logger.info(f"Generated video (no image): {output_path}")
                                return output_path
                            break
                        time.sleep(poll_interval)
                except Exception as e2:
                    logger.warning(f"  Text-only video also failed: {e2}")

            if attempt < max_retries - 1:
                wait = 10 if "429" in err_str else 5
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed to generate video after {max_retries} attempts")


def _load_and_resize(path: Path | None, max_dim: int = 1024) -> Image.Image | None:
    """Load image and resize if too large for Veo API."""
    if not path or not Path(path).exists():
        return None
    img = Image.open(path)
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img
