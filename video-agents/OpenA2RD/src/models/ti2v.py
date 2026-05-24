"""Text-Image-to-Video (TI2V) generation interface.

Uses Veo 3.1 (veo-3.1-generate-preview) via Gemini API.
Supports:
- image: starting frame
- lastFrame: ending frame (for interpolation)
- referenceImages: up to 3 reference images for consistency
"""

import logging
import time
from pathlib import Path

import httpx
from google.genai import types
from PIL import Image

from src.config import settings
from src.models.mllm import get_client

logger = logging.getLogger(__name__)

VIDEO_MODEL = "veo-3.1-generate-preview"


def generate_video(
    prompt: str,
    begin_frame: Path | None = None,
    end_frame: Path | None = None,
    reference_images: list[Path] | None = None,
    output_path: Path | None = None,
    duration_seconds: int = 8,
    aspect_ratio: str = "16:9",
    max_retries: int = 3,
    poll_interval: int = 10,
    max_poll_time: int = 300,
) -> Path:
    """Generate a video from text prompt + optional frames.

    Args:
        prompt: Video description
        begin_frame: Starting frame (image parameter)
        end_frame: Ending frame (lastFrame, for interpolation)
        reference_images: Up to 3 reference images for consistency
        output_path: Where to save
        duration_seconds: 4, 6, or 8
        aspect_ratio: 16:9 or 9:16
    """
    if output_path is None:
        output_path = settings.output_dir / f"segment_{int(time.time())}.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()

    for attempt in range(max_retries):
        try:
            # Build parameters
            veo_image = _to_veo_image(begin_frame)

            gen_config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
            )

            # Veo 3.1: set lastFrame for interpolation
            if end_frame:
                veo_last = _to_veo_image(end_frame)
                if veo_last:
                    gen_config.last_frame = veo_last

            # Veo 3.1: set reference images for consistency
            if reference_images:
                ref_imgs = []
                for i, ref_path in enumerate(reference_images[:3]):
                    ref_img = _to_veo_image(ref_path)
                    if ref_img:
                        ref_imgs.append(types.RawReferenceImage(
                            reference_id=i + 1,
                            reference_image=ref_img,
                        ))
                if ref_imgs:
                    gen_config.reference_images = ref_imgs

            response = client.models.generate_videos(
                model=VIDEO_MODEL,
                prompt=prompt,
                image=veo_image,
                config=gen_config,
            )

            video = _poll_video(client, response, poll_interval, max_poll_time)
            if video:
                _save_video(video, output_path)
                logger.info(f"Generated video: {output_path}")
                return output_path

            logger.warning("Video generation completed but no video returned")

        except Exception as e:
            err_str = str(e)
            logger.warning(f"Video generation failed (attempt {attempt+1}): {e}")

            # Fallback: try without image/lastFrame
            if attempt == 0 and ("image" in err_str.lower() or "lastFrame" in err_str.lower()):
                logger.info("  Retrying text-only...")
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
                        logger.info(f"Generated video (text-only fallback): {output_path}")
                        return output_path
                except Exception as e2:
                    logger.warning(f"  Text-only also failed: {e2}")

            if attempt < max_retries - 1:
                wait = 30 if "429" in err_str else 5
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
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return types.Image(image_bytes=buf.getvalue(), mime_type="image/png")


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
    """Save generated video, downloading from URI if needed."""
    # Try direct save
    try:
        video.video.save(str(output_path))
        return
    except Exception:
        pass

    # Download from URI with API key
    uri = getattr(video.video, 'uri', None)
    if uri:
        sep = "&" if "?" in uri else "?"
        auth_uri = f"{uri}{sep}key={settings.gemini_api_key}"
        with httpx.Client(timeout=120, follow_redirects=True) as http:
            resp = http.get(auth_uri)
            resp.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(resp.content)
        logger.info(f"Downloaded video ({len(resp.content)//1024}KB)")
        return

    # Use bytes directly
    if video.video.video_bytes:
        with open(output_path, 'wb') as f:
            f.write(video.video.video_bytes)
        return

    raise RuntimeError("Cannot save video: no bytes, no URI")
