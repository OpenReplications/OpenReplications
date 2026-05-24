"""Text-Image-to-Image (TI2I) generation interface.

Uses Nano Banana 2 (gemini-3.1-flash-image-preview) via Gemini API.
For reference-conditioned generation, uses multimodal generate_content
with response_modalities=["IMAGE", "TEXT"].
"""

import logging
import time
from pathlib import Path

from google.genai import types
from PIL import Image

from src.config import settings
from src.models.mllm import get_client

logger = logging.getLogger(__name__)


def _img_to_part(path):
    """Convert image file to Gemini Part."""
    import io
    p = Path(path)
    img = Image.open(p)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

# Available image models (try in order):
# 1. gemini-3.1-flash-image-preview (Nano Banana 2) - multimodal, supports references
# 2. imagen-4.0-fast-generate-001 (Imagen 4 Fast) - text-only, separate quota
# 3. gemini-2.5-flash (Gemini Flash) - multimodal fallback with IMAGE output
MULTIMODAL_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
IMAGEN_MODEL = "imagen-4.0-fast-generate-001"
FALLBACK_MODEL = "gemini-2.5-flash"


def generate_image(
    prompt: str,
    reference_images: list[Path | str] | None = None,
    output_path: Path | None = None,
    aspect_ratio: str = "16:9",
    max_retries: int = 5,
) -> Path:
    """Generate an image from text prompt + optional reference images.

    Tries multiple backends in order:
    1. Nano Banana 2 (multimodal, supports reference images)
    2. Imagen 4 Fast (text-only, separate quota pool)
    3. Gemini Flash with IMAGE modality (fallback)
    """
    if output_path is None:
        output_path = settings.output_dir / f"frame_{int(time.time())}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()

    # Strategy 1: Try Nano Banana 2 (multimodal - supports reference images)
    result = _try_multimodal_generation(client, prompt, reference_images, output_path)
    if result:
        return result

    # Strategy 2: Try Imagen 4 Fast (text-only, different quota)
    result = _try_imagen_generation(client, prompt, output_path, aspect_ratio)
    if result:
        return result

    # Strategy 3: Fallback to Gemini Flash with IMAGE output
    result = _try_gemini_flash_image(client, prompt, reference_images, output_path)
    if result:
        return result

    raise RuntimeError("All image generation backends failed")


def _try_multimodal_generation(
    client, prompt: str, reference_images: list | None, output_path: Path
) -> Path | None:
    """Try Nano Banana 2 multimodal generation."""
    try:
        time.sleep(5)
        contents = []
        if reference_images:
            for ref_path in reference_images[:3]:
                p = Path(ref_path)
                if p.exists():
                    contents.append(_img_to_part(p))
            contents.append(
                f"Generate a high-quality, cinematic image following this prompt, "
                f"maintaining strict visual consistency with the reference images:\n\n{prompt}"
            )
        else:
            contents.append(prompt)

        response = client.models.generate_content(
            model=MULTIMODAL_IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.8,
            ),
        )
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    logger.info(f"Generated image (Nano Banana 2): {output_path}")
                    return output_path
    except Exception as e:
        logger.warning(f"Nano Banana 2 failed: {e}")
    return None


def _try_imagen_generation(
    client, prompt: str, output_path: Path, aspect_ratio: str
) -> Path | None:
    """Try Imagen 4 Fast text-to-image."""
    try:
        time.sleep(5)
        response = client.models.generate_images(
            model=IMAGEN_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            ),
        )
        if response.generated_images:
            img = response.generated_images[0]
            img.image.save(str(output_path))
            logger.info(f"Generated image (Imagen 4 Fast): {output_path}")
            return output_path
    except Exception as e:
        logger.warning(f"Imagen 4 Fast failed: {e}")
    return None


def _try_gemini_flash_image(
    client, prompt: str, reference_images: list | None, output_path: Path
) -> Path | None:
    """Fallback: Gemini Flash with IMAGE response modality."""
    try:
        time.sleep(5)
        contents = []
        if reference_images:
            for ref_path in reference_images[:2]:
                p = Path(ref_path)
                if p.exists():
                    contents.append(_img_to_part(p))
        contents.append(f"Generate a high-quality cinematic image:\n\n{prompt}")

        response = client.models.generate_content(
            model=FALLBACK_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.8,
            ),
        )
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    logger.info(f"Generated image (Gemini Flash): {output_path}")
                    return output_path
    except Exception as e:
        logger.warning(f"Gemini Flash image failed: {e}")
    return None
