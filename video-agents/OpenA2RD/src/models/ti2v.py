"""Text-Image-to-Video (TI2V) generation interface.

Uses Veo 2.0 (veo-2.0-generate-001) via Gemini API.
Available models: veo-2.0, veo-3.0, veo-3.0-fast, veo-3.1, veo-3.1-fast, veo-3.1-lite
"""

import logging
import time
from pathlib import Path

from google.genai import types
from PIL import Image

from src.config import settings
from src.models.mllm import get_client

logger = logging.getLogger(__name__)

# Default video model
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
    """Generate a video from text prompt + boundary frames.

    Args:
        prompt: Detailed video prompt
        begin_frame: Starting frame image path
        end_frame: Ending frame image path (for interpolation mode)
        output_path: Where to save the generated video
        duration_seconds: Target video duration
        aspect_ratio: Aspect ratio
        max_retries: Retry count
        poll_interval: Seconds between status polls
        max_poll_time: Max seconds to wait for generation

    Returns:
        Path to the generated video (.mp4)
    """
    if output_path is None:
        output_path = settings.output_dir / f"segment_{int(time.time())}.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()

    for attempt in range(max_retries):
        try:
            # Build image conditioning
            image = None
            if begin_frame and begin_frame.exists():
                image = Image.open(begin_frame)

            # Use Veo API for video generation
            response = client.models.generate_videos(
                model=VIDEO_MODEL,
                prompt=prompt,
                image=image,
                config=types.GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    number_of_videos=1,
                ),
            )

            # Poll for completion
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed > max_poll_time:
                    logger.warning(f"Video generation timed out after {max_poll_time}s")
                    break

                # Check if operation is complete
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
            logger.warning(f"Video generation failed (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise

    raise RuntimeError(f"Failed to generate video after {max_retries} attempts")
