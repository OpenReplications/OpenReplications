"""Text-Image-to-Video (TI2V) generation interface.

Uses Veo 2.0 (veo-2.0-generate-001) via Gemini API.
Image must be passed as types.Image(image_bytes=..., mime_type=...).
"""

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
            # Convert image to types.Image format (bytes + mime_type)
            veo_image = _to_veo_image(begin_frame)

            # For interpolation, embed end frame context in prompt
            effective_prompt = prompt
            if end_frame and Path(end_frame).exists() and veo_image:
                effective_prompt = (
                    f"{prompt}\n\nIMPORTANT: The video must transition smoothly "
                    f"from the beginning frame to the ending frame provided."
                )

            response = client.models.generate_videos(
                model=VIDEO_MODEL,
                prompt=effective_prompt,
                image=veo_image,
                config=types.GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    number_of_videos=1,
                ),
            )

            # Poll for completion
            video = _poll_video(client, response, poll_interval, max_poll_time)
            if video:
                _save_video(video, output_path)
                logger.info(f"Generated video: {output_path}")
                return output_path

            logger.warning("Video generation completed but no video returned")

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
                    video = _poll_video(client, response, poll_interval, max_poll_time)
                    if video:
                        _save_video(video, output_path)
                        logger.info(f"Generated video (no image): {output_path}")
                        return output_path
                except Exception as e2:
                    logger.warning(f"  Text-only video also failed: {e2}")

            if attempt < max_retries - 1:
                wait = 10 if "429" in err_str else 5
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed to generate video after {max_retries} attempts")


def _to_veo_image(path: Path | None, max_dim: int = 1024) -> types.Image | None:
    """Convert image file to types.Image(image_bytes, mime_type) for Veo API."""
    if not path or not Path(path).exists():
        return None

    import io

    img = Image.open(path)

    # Resize if too large
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Convert to PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()

    return types.Image(image_bytes=raw, mime_type="image/png")


def _poll_video(client, operation, poll_interval: int, max_poll_time: int):
    """Poll a video generation operation until done."""
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_poll_time:
            logger.warning(f"Video generation timed out after {max_poll_time}s")
            return None

        result = client.operations.get(operation=operation)

        if result.done:
            if result.response and result.response.generated_videos:
                return result.response.generated_videos[0]
            if result.error:
                logger.warning(f"Video generation error: {result.error}")
            return None

        logger.debug(f"Video generating... ({elapsed:.0f}s elapsed)")
        time.sleep(poll_interval)


def _save_video(video, output_path: Path):
    """Save generated video to disk, handling both local and remote cases."""
    # Try direct save first (works for local/inline videos)
    try:
        video.video.save(str(output_path))
        return
    except Exception:
        pass

    # If video has a URI, download it with API key auth
    uri = getattr(video.video, 'uri', None)
    if uri:
        import httpx
        try:
            # Append API key for authentication
            sep = "&" if "?" in uri else "?"
            auth_uri = f"{uri}{sep}key={settings.gemini_api_key}"
            with httpx.Client(timeout=120, follow_redirects=True) as http:
                resp = http.get(auth_uri)
                resp.raise_for_status()
                with open(output_path, 'wb') as f:
                    f.write(resp.content)
            logger.info(f"Downloaded video from URI ({len(resp.content)//1024}KB)")
            return
        except Exception as e:
            logger.warning(f"URI download failed: {e}")

    # If video has bytes directly
    if video.video.video_bytes:
        with open(output_path, 'wb') as f:
            f.write(video.video.video_bytes)
        return

    raise RuntimeError("Cannot save video: no bytes, no URI")
